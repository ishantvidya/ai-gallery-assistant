"""benchmark_b32_vs_l14.py — M0 Step 2: is ViT-B/32 good enough?

What it does:
    1. Embeds every photo in data/images/ THREE ways:
         - ViT-L-14 (the desktop model, via sentence-transformers) — the baseline
         - ViT-B/32 ONNX fp32   (the candidate, float32)
         - ViT-B/32 ONNX int8   (the candidate, quantized — what ships on phone)
    2. Runs the text queries from data/images/labels.json through each encoder.
    3. Scores CORRECTNESS, not agreement: each image in labels.json declares
       "must_have" captions that are objectively true of it. A model scores
       a hit when a true image ranks in its top-3 for that caption.

    4. Sanity-gates the dataset itself first: if even the desktop ViT-L-14
       cannot beat 0.17 mean cosine on a caption, the images are too unlike
       photos for ANY verdict to be trusted, and the benchmark says so
       instead of producing an meaningless ranking comparison.

Why not "top-5 overlap with ViT-L-14"?
    Overlap measures agreement, not correctness. The v1 dataset (flat-color
    shapes) scored 0.06-0.15 cosine on the baseline — the baseline itself
    could not recognize anything, so its ranking was noise, and overlap with
    it was agreement-with-noise (67% in that run, proving nothing).

How to read the numbers:
    - must_have hit-rate >= 80% on both B/32 variants -> ship it.
    - fp32 fine but int8 drops -> tune the int8 export (separate lever).
    - BOTH models fail -> the model choice was never the problem: the
      dataset or the labels are. Fix the data before touching the model.
    - L-14 truth cosine < ~0.17 or margin < ~0.04 -> dataset sanity gate
      tripped; images are not photo-like enough to judge anything.

Note: this compares SINGLE-crop embeddings on both sides so the models are
judged fairly. The app's 7-crop augmentation is an accuracy lever we can add
on top of whichever model wins; using it here would slow the benchmark 7x
without changing which model is better.

Run:
    python scripts/benchmark_b32_vs_l14.py
    python scripts/benchmark_b32_vs_l14.py --queries "a dog" "a red car"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make 'src' importable no matter where the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images"
ONNX_DIR = ROOT / "models" / "onnx"
LABELS_PATH = IMAGES_DIR / "labels.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Fallback if labels.json is missing (rankings still print, accuracy does not).
FALLBACK_QUERIES = [
    "a sunset over the ocean",
    "a dog",
    "a car on a road",
    "food on a plate",
    "a group of people",
]

# Dataset sanity gate. A true photo/caption pair normally scores 0.25-0.35
# cosine; a mismatched pair 0.12-0.22. If even the desktop ViT-L-14 can't
# (a) score >= 0.17 on the TRUE image, or (b) beat the non-matching images by
# a >= 0.04 margin, the images are not photo-like enough for the benchmark to
# mean anything — no model comparison is trustworthy on such data.
L14_MEAN_GATE = 0.17
L14_MARGIN_GATE = 0.04


def list_images(folder=IMAGES_DIR):
    return sorted(
        (p for p in folder.iterdir()
         if p.suffix.lower() in IMAGE_EXTENSIONS and p.name != "labels.json"),
        key=lambda p: p.name,
    )


def load_labels():
    """Return ({filename: {"must_have": [...], ...}}, queries_in_label_order)."""
    if not LABELS_PATH.exists():
        return None, FALLBACK_QUERIES
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    queries = []
    for info in labels.values():
        for q in info.get("must_have", []):
            if q not in queries:
                queries.append(q)
    return labels, queries


# --------------------------------------------------------------------------
# Baseline: ViT-L-14 through sentence-transformers (same as the desktop app)
# --------------------------------------------------------------------------
def embed_with_l14(images, queries):
    from src.model import get_model  # reuses the app's model cache

    model = get_model()
    t0 = time.perf_counter()
    img_matrix = model.encode(images, normalize_embeddings=True, batch_size=8)
    txt_matrix = model.encode(queries, normalize_embeddings=True)
    dt = time.perf_counter() - t0
    return img_matrix.astype(np.float32), txt_matrix.astype(np.float32), dt


# --------------------------------------------------------------------------
# Candidate: ViT-B/32 via the ONNX files exported by export_clip_b32_onnx.py
# --------------------------------------------------------------------------
def embed_with_b32_onnx(images, queries, variant):
    """variant: 'fp32' or 'int8'. Returns (img_matrix, txt_matrix, seconds)."""
    import onnxruntime as ort
    from transformers import CLIPProcessor

    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    sess_img = ort.InferenceSession(
        str(ONNX_DIR / f"clip-b32-image-{variant}.onnx"),
        sess_options=so, providers=["CPUExecutionProvider"],
    )
    sess_txt = ort.InferenceSession(
        str(ONNX_DIR / f"clip-b32-text-{variant}.onnx"),
        sess_options=so, providers=["CPUExecutionProvider"],
    )

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    t0 = time.perf_counter()
    pixels = processor(images=images, return_tensors="np")["pixel_values"]
    img_matrix = sess_img.run(None, {"pixel_values": pixels})[0]
    ids = processor.tokenizer(
        queries, padding="max_length", max_length=77, truncation=True,
        return_tensors="np",
    )["input_ids"]
    txt_matrix = sess_txt.run(None, {"input_ids": ids})[0]
    dt = time.perf_counter() - t0

    # ONNX towers already normalize, but re-normalize defensively: cosine
    # similarity as a plain dot product is only valid for unit vectors.
    img_matrix /= np.linalg.norm(img_matrix, axis=1, keepdims=True) + 1e-8
    txt_matrix /= np.linalg.norm(txt_matrix, axis=1, keepdims=True) + 1e-8
    return img_matrix.astype(np.float32), txt_matrix.astype(np.float32), dt


def topk(matrix, txt_vec, k=5):
    scores = matrix @ txt_vec
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]


def fmt_ranking(paths, ranking):
    return "  ".join(f"{paths[i].name}({s:.2f})" for i, s in ranking)


def evaluate(results, paths, labels, queries, k=3):
    """Correctness against ground truth.

    For every must_have caption, the image(s) it is true of must rank in the
    model's top-k. Returns per-model (hits, total, per_query dict).
    """
    name_to_idx = {p.name: i for i, p in enumerate(paths)}
    per_model = {}
    for model_name, (img_m, txt_m) in results.items():
        hits = total = 0
        per_query = {}
        for fname, info in labels.items():
            for caption in info.get("must_have", []):
                if caption not in queries:
                    continue
                ci = queries.index(caption)
                ranking = topk(img_m, txt_m[ci], k=k)
                top_set = {i for i, _ in ranking}
                ok = name_to_idx[fname] in top_set
                hits += ok
                total += 1
                per_query[caption] = {
                    "hit": ok,
                    "truth_rank": (next(
                        (r for r, (i, _) in enumerate(ranking, 1)
                         if i == name_to_idx[fname]), None)),
                    "top": [paths[i].name for i, _ in ranking],
                    "mean_cos": float(np.mean(img_m @ txt_m[ci])),
                }
        per_model[model_name] = (hits, total, per_query)
    return per_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", nargs="*", default=None,
                        help="Override queries (disables accuracy scoring).")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--topk-truth", type=int, default=3,
                        help="How deep a TRUE image may rank and still count.")
    args = parser.parse_args()

    paths = list_images()
    if not paths:
        raise SystemExit(f"No images found in {IMAGES_DIR}")

    labels, label_queries = load_labels()
    use_labels = labels is not None and args.queries is None
    queries = args.queries if args.queries else label_queries
    print(f"Benchmarking on {len(paths)} photos, {len(queries)} queries\n")

    images = [Image.open(p).convert("RGB") for p in paths]

    print("Embedding with ViT-L-14 (desktop baseline)...")
    l14_img, l14_txt, t_l14 = embed_with_l14(images, queries)
    print(f"  done in {t_l14:.1f}s\n")

    results = {"ViT-L-14": (l14_img, l14_txt)}
    for variant in ("fp32", "int8"):
        print(f"Embedding with ViT-B/32 ONNX ({variant})...")
        b32_img, b32_txt, t_b32 = embed_with_b32_onnx(images, queries, variant)
        print(f"  done in {t_b32:.1f}s\n")
        results[f"B/32 {variant}"] = (b32_img, b32_txt)

    # ---- Dataset sanity gate ---------------------------------------------
    # If the desktop model can't recognize the TRUE captions either, the
    # images are the problem — no model comparison is meaningful.
    # truth_cos: how strongly L-14 matches the TRUE image to its caption.
    # margin:    how much that beats the average non-matching image.
    gate_truth = gate_margin = None
    if use_labels:
        name_to_idx = {p.name: i for i, p in enumerate(paths)}
        truths, margins = [], []
        for fname, info in labels.items():
            ti = name_to_idx[fname]
            others = [i for i in range(len(paths)) if i != ti]
            for caption in info.get("must_have", []):
                ci = queries.index(caption)
                scores = l14_img @ l14_txt[ci]
                truths.append(float(scores[ti]))
                margins.append(float(scores[ti] - np.mean(scores[others])))
        gate_truth = float(np.mean(truths))
        gate_margin = float(np.mean(margins))
        ok = gate_truth >= L14_MEAN_GATE and gate_margin >= L14_MARGIN_GATE
        print(f"Dataset sanity gate (ViT-L-14 on TRUE captions):\n"
              f"  truth-image cosine {gate_truth:.3f} "
              f"(gate >= {L14_MEAN_GATE})\n"
              f"  margin over non-truth images {gate_margin:.3f} "
              f"(gate >= {L14_MARGIN_GATE})\n"
              f"  -> {'OK' if ok else 'FAILED'}\n")
        if not ok:
            print("  The images are too unlike photos for any model verdict")
            print("  to be trusted. Fix the dataset before comparing models.\n")

    # ---- Side-by-side rankings ------------------------------------------
    for qi, query in enumerate(queries):
        print(f'Query: "{query}"')
        for name, (img_m, txt_m) in results.items():
            r = topk(img_m, txt_m[qi], k=args.k)
            print(f"  {name:12s} {fmt_ranking(paths, r)}")
        print()

    # ---- Correctness vs ground truth -------------------------------------
    if use_labels:
        evals = evaluate(results, paths, labels, queries, k=args.topk_truth)
        n_true = sum(len(i.get("must_have", [])) for i in labels.values())
        print("=" * 64)
        print(f"CORRECTNESS — is a TRUE image in the top-{args.topk_truth}? "
              f"({n_true} true image/caption pairs)")
        print("=" * 64)
        for name, (hits, total, per_query) in evals.items():
            pct = 100.0 * hits / total if total else 0.0
            print(f"  {name:12s}: {hits}/{total} ({pct:.0f}%)")
        misses = [q for q, r in evals["B/32 fp32"][2].items() if not r["hit"]]
        if misses:
            print("\n  fp32 missed (query -> truth rank, its top pick):")
            for q in misses:
                r = evals["B/32 fp32"][2][q]
                print(f'    "{q}" -> truth ranked #{r["truth_rank"]}, '
                      f'it picked {r["top"][0]}')
    else:
        print("(labels.json not found or --queries given: "
              "rankings only, no accuracy score)")

    # Sanity check: the two B/32 variants must agree with each other, else
    # the int8 export is broken and the numbers above mean nothing.
    i32, t32 = results["B/32 fp32"]
    i8, t8 = results["B/32 int8"]
    same = sum(
        1 for qi in range(len(queries))
        if {i for i, _ in topk(i32, t32[qi], k=args.k)}
        == {i for i, _ in topk(i8, t8[qi], k=args.k)}
    )
    print(f"\n  fp32 vs int8 identical top-{args.k}: {same}/{len(queries)} queries")


if __name__ == "__main__":
    main()
