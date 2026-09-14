"""storage.py — save and load the index to/from disk.

One numpy .npz file holds everything:
    embeddings/   the N x 512 matrix (one row per photo)
    paths/        the photo paths, in the SAME row order as the matrix

Why .npz instead of a database?
    For ~100 images this is fast (milliseconds), needs zero setup, and the
    file is tiny (~200 KB). The important part is that the rest of the app
    only talks to this file through save_index() / load_index(). When V2
    wants a real vector database, only THIS module changes — search.py and
    the UI won't notice.

Why relative paths?
    We store paths like "data/images/sunset.png" instead of
    "C:\\Users\\...\\data\\images\\sunset.png". Absolute paths break the
    moment the project folder moves or is re-synced (e.g. OneDrive).
    Relative paths work wherever the project lives.

Run it (demo):
    python -m src.storage
"""

from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = ROOT / "embeddings" / "index.npz"
DEFAULT_FAISS_PATH = ROOT / "embeddings" / "index.faiss"


def _to_relative(path):
    """Return the path as a project-relative string (safe to store).

    Always uses forward slashes (as_posix) so the stored text looks the same
    no matter which OS wrote it. If the path is outside the project folder,
    we keep it absolute rather than crash.
    """
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.relative_to(ROOT)
        except ValueError:
            pass  # outside the project — keep the absolute path
    return p.as_posix()


def save_index(paths, matrix, index_path=DEFAULT_INDEX_PATH):
    """Store the index: save (paths, matrix) as one .npz file."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # float32 keeps the file half the size of float64 with no loss for our
    # purposes. dtype=str stores the paths as plain text (no pickling needed).
    np.savez(
        index_path,
        embeddings=matrix.astype(np.float32),
        paths=np.array([_to_relative(p) for p in paths], dtype=str),
    )
    print(f"Saved index: {index_path}  ({len(paths)} images, {matrix.shape})")

    # Also build and save a FAISS index for fast ANN search.
    save_faiss_index(matrix.astype(np.float32))


def save_faiss_index(matrix, faiss_path=DEFAULT_FAISS_PATH):
    """Build a FAISS inner-product index from the embedding matrix.

    For normalized vectors, inner product = cosine similarity.
    IndexFlatIP is brute-force but still faster than raw numpy for large N
    because FAISS uses SIMD-optimized BLAS routines.
    For >10k images, switch to IndexIVFFlat for sub-linear search.
    """
    faiss_path = Path(faiss_path)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    dim = matrix.shape[1]
    n = matrix.shape[0]

    if n < 10000:
        # Small index: exact search, SIMD-optimized
        index = faiss.IndexFlatIP(dim)
    else:
        # Large index: IVF with 256 clusters for sub-linear search
        nlist = min(256, n // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)

    index.add(matrix)
    faiss.write_index(index, str(faiss_path))
    print(f"Saved FAISS index: {faiss_path}  ({n} vectors, dim={dim})")


def load_faiss_index(faiss_path=DEFAULT_FAISS_PATH):
    """Load a FAISS index from disk."""
    return faiss.read_index(str(faiss_path))


def load_index(index_path=DEFAULT_INDEX_PATH):
    """Load the index back; returns (paths, matrix) in matching row order."""
    data = np.load(index_path)
    return data["paths"].tolist(), data["embeddings"]


def resolve_image_path(rel_path):
    """Turn a stored (relative) path back into an absolute Path for reading."""
    p = Path(rel_path)
    return p if p.is_absolute() else ROOT / p


if __name__ == "__main__":
    # Demo with a throwaway file so the real index is never touched.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        demo_file = Path(tmp) / "demo.npz"
        # One relative path AND one absolute path, to exercise the conversion.
        demo_paths = ["data/images/sunset.png", str(ROOT / "data" / "images" / "robot.png")]
        demo_matrix = np.array([[0.1, 0.2], [0.9, 0.8]], dtype=np.float32)

        save_index(demo_paths, demo_matrix, demo_file)
        loaded_paths, loaded_matrix = load_index(demo_file)

        print("reloaded paths:   ", loaded_paths)
        print("reloaded matrix:  ", loaded_matrix)
        print("round trip equal: ", np.array_equal(loaded_matrix, demo_matrix))
