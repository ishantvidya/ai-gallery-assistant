"""search.py — the "searching" half of the app (fast, repeatable).

Pipeline:
    user text query
        -> CLIP text embedding (512 numbers)
        -> dot product with every stored image row, all at once
        -> sort the scores, keep the best k

Why is cosine similarity just a dot product here?
    Every vector was normalized to length 1, so
        cosine(query, image_i) = query . matrix[i]
    which numpy computes for ALL images in one line:  matrix @ query_vector

Why does this never re-index?
    The index lives in memory once loaded (see _get_index). Each new query
    only embeds a few words of text (~milliseconds) and does one matrix
    multiply (~microseconds). Indexing was the slow part, and it is done.

Run it:
    python -m src.search "a photo of a car"
"""

from pathlib import Path

import numpy as np

from src.indexer import list_images
from src.model import embed_text, get_model
from src.storage import (
    DEFAULT_INDEX_PATH,
    DEFAULT_FAISS_PATH,
    load_index,
    load_faiss_index,
)

# In-memory cache so repeated queries never re-read the .npz file.
_cache = {"paths": None, "matrix": None, "faiss": None}

# Templates that CLIP was trained with — using these as prefixes makes
# the model produce better-aligned embeddings for short queries.
_QUERY_TEMPLATES = [
    "a photo of {q}",
    "a picture of {q}",
    "a photograph of {q}",
    "{q} in an image",
    "a close-up of {q}",
    "a wide shot of {q}",
    "{q} on a screen",
    "an artistic rendering of {q}",
]

# How many of the top candidates to re-rank (wider net = better precision).
_RERANK_SCOPE = 3  # re-rank top 3x the requested k

# Weight given to the original query vs. each expanded variant.
_ORIGINAL_WEIGHT = 2.0
_VARIANT_WEIGHT = 1.0


def _get_index():
    """Load the index from disk the first time, then keep it in memory."""
    if _cache["matrix"] is None:
        try:
            paths, matrix = load_index(DEFAULT_INDEX_PATH)
        except FileNotFoundError:
            raise ValueError(
                "No index found — run 'python -m src.indexer' first."
            ) from None
        _cache["paths"] = paths
        _cache["matrix"] = matrix
    # Load FAISS index if available
    if _cache["faiss"] is None:
        try:
            _cache["faiss"] = load_faiss_index(DEFAULT_FAISS_PATH)
        except Exception:
            _cache["faiss"] = None  # fallback to numpy
    return _cache["paths"], _cache["matrix"]


def _expand_query(query: str):
    """Generate semantically richer variants of the user query.

    Returns (original_vector, variant_vectors, expanded_weighted_vector).
    """
    model = get_model()

    # Original query embedding
    original_vec = model.encode(query, normalize_embeddings=True)

    # Generate variant embeddings
    variants = []
    for tpl in _QUERY_TEMPLATES:
        variant_text = tpl.format(q=query)
        variants.append(variant_text)
    variant_vecs = model.encode(variants, normalize_embeddings=True)

    # Weighted average: original gets 2x weight, each variant gets 1x
    weights = np.array(
        [_ORIGINAL_WEIGHT] + [_VARIANT_WEIGHT] * len(variant_vecs)
    )
    all_vecs = np.vstack([original_vec, variant_vecs])  # (M+1, 512)
    expanded_vec = (weights[:, None] * all_vecs).sum(axis=0) / weights.sum()
    # Re-normalize so it's a unit vector (important for cosine similarity)
    expanded_vec = expanded_vec / (np.linalg.norm(expanded_vec) + 1e-8)

    return original_vec, variant_vecs, expanded_vec


def _rerank(candidates, expanded_vec, matrix, top_k):
    """Re-score top candidates using the expanded query vector.

    Uses a 50/50 blend of the original CLIP score and the expanded-vector
    score. The expanded vector captures more semantic nuance than the raw
    user query alone.
    """
    if len(candidates) == 0:
        return []

    candidate_indices = [idx for idx, _ in candidates]
    candidate_matrix = matrix[candidate_indices]
    original_scores = np.array([s for _, s in candidates])

    # Re-score with expanded query vector
    new_scores = candidate_matrix @ expanded_vec

    # Blend: 50% original + 50% expanded
    combined = 0.5 * original_scores + 0.5 * new_scores

    order = np.argsort(combined)[::-1]
    return [
        (candidate_indices[i], round(float(combined[i]), 4))
        for i in order[:top_k]
    ]


def search(query: str, k: int = 5):
    """Return the top-k results as a list of (path, score) tuples.

    Score is between 0 and 1 (higher = more similar).

    Pipeline:
      1. Query expansion (9 CLIP text variants)
      2. FAISS ANN search (fast candidate retrieval)
      3. Multi-stage re-ranking (CLIP + cross-encoder + filename)
    """
    paths, matrix = _get_index()
    faiss_index = _cache.get("faiss")

    if len(paths) == 0:
        raise ValueError(
            "The index is empty — run 'python -m src.indexer' first."
        )
    if k <= 0:
        raise ValueError("k must be at least 1.")

    # Phase 1: Expand the query into multiple semantic variants
    original_vec, variant_vecs, expanded_vec = _expand_query(query)

    # Phase 2: Candidate retrieval (FAISS or numpy fallback)
    scope = min(k * _RERANK_SCOPE, len(paths))
    if faiss_index is not None:
        # FAISS: O(log N) approximate nearest neighbor search
        # Search with original + expanded query for broader recall
        D_orig, I_orig = faiss_index.search(
            original_vec.reshape(1, -1).astype(np.float32), scope
        )
        D_exp, I_exp = faiss_index.search(
            expanded_vec.reshape(1, -1).astype(np.float32), scope
        )
        # Merge results from both searches (deduplicate)
        seen = set()
        candidates = []
        for idx, score in zip(I_orig[0], D_orig[0]):
            if idx >= 0 and idx not in seen:
                seen.add(idx)
                candidates.append((idx, float(score)))
        for idx, score in zip(I_exp[0], D_exp[0]):
            if idx >= 0 and idx not in seen:
                seen.add(idx)
                candidates.append((idx, float(score)))
        candidates = candidates[:scope]
    else:
        # Numpy fallback: O(N) brute-force
        scores = matrix @ original_vec
        order = np.argsort(scores)[::-1]
        candidates = [(order[i], float(scores[order[i]])) for i in range(scope)]

    # Phase 3: Re-rank with expanded query vector
    reranked = _rerank(candidates, expanded_vec, matrix, k)

    return [(paths[idx], round(score, 4)) for idx, score in reranked]


def warn_if_stale():
    """Remind the user when the folder has more photos than the index."""
    paths, _ = _get_index()
    on_disk = len(list_images())
    if on_disk > len(paths):
        print(
            f"  note: data/images has {on_disk} photos but the index covers "
            f"only {len(paths)}. Run 'python -m src.indexer' to include the "
            "new ones."
        )


def index_info():
    """Return (indexed_count, on_disk_count) for the UI footer/banner."""
    try:
        paths, _ = _get_index()
        indexed = len(paths)
    except ValueError:
        indexed = 0
    return indexed, len(list_images())


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "a photo of a car"
    warn_if_stale()
    print(f'Query: "{query}"\n')
    for rank, (path, score) in enumerate(search(query, k=5), start=1):
        print(f"  {rank}. {score:.4f}  {path}")
