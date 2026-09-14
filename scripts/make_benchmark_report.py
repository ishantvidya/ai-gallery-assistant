"""make_benchmark_report.py — M0 final step: a visual verdict page.

The accuracy numbers in benchmark_b32_vs_l14.py can't tell you *why* a model
missed — only your eyes can. This script re-runs the same comparison and
writes benchmark_report.html in the project root.

For every must_have caption from data/images/labels.json:
    one block with three rows of photo thumbnails, truth-marked:
      row 1: ViT-L-14  (desktop baseline — the reference)
      row 2: B/32 fp32 (candidate, float32)
      row 3: B/32 int8 (candidate, quantized — what ships on phone)

    The photo the caption is TRUE of (declared in labels.json) is marked
    "TRUTH" in gold. A row scores a HIT when the truth photo carries a
    green border inside its top-3. The overall verdict = hit-rate, i.e.
    how often each model ranks the objectively right image high.

Why thumbnails, not filenames?
    "1.png vs 2.png" means nothing to a human. The whole point of M0 is a
    decision a person can make in five minutes of looking.

The page also reports the dataset sanity gate: if even ViT-L-14 averages a
low cosine on TRUE captions, the images are too unlike photos and the page
says so instead of pretending to judge models.

Run:
    python scripts/make_benchmark_report.py
"""

import base64
import html
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_b32_vs_l14 import (
    IMAGES_DIR,
    LABELS_PATH,
    L14_MARGIN_GATE,
    L14_MEAN_GATE,
    embed_with_b32_onnx,
    embed_with_l14,
    list_images,
    load_labels,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "benchmark_report.html"
THUMB_SIZE = 160
TOPK = 3


def _thumb_b64(path: Path) -> str:
    """Return one photo as an inline JPEG thumbnail (base64 data URI).

    Inlining keeps benchmark_report.html a single self-contained file that
    you can send to someone (a tester, a teammate) and it just works.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _cell(rank, idx, score, paths, thumbs, is_truth, is_hit):
    border = ""
    badge = ""
    if is_truth:
        border = "border:3px solid #2e7d32;"
        badge = '<span class="truth">TRUTH</span>'
        if is_hit:
            badge = '<span class="truth hit">TRUTH ✓ HIT</span>'
    elif rank == 1:
        badge = '<span class="picked">model picked</span>'
    return (
        f'<div class="card" style="{border}">'
        f'<img src="{thumbs[idx]}">'
        f'<div class="cap">#{rank} {html.escape(paths[idx].name)}<br>'
        f'<span class="score">{score:.3f}</span> {badge}</div></div>'
    )


def _row(label, ranking, paths, thumbs, truth_idx):
    cells = []
    for rank, (idx, score) in enumerate(ranking, start=1):
        cells.append(_cell(
            rank, idx, score, paths, thumbs,
            is_truth=(idx == truth_idx),
            is_hit=(idx == truth_idx),
        ))
    return (
        f'<div class="row"><div class="label">{label}</div>'
        f'<div class="cards">{"".join(cells)}</div></div>'
    )


def main() -> None:
    paths = list_images()
    if not paths:
        raise SystemExit(f"No images found in {IMAGES_DIR}")
    if not LABELS_PATH.exists():
        raise SystemExit(
            f"{LABELS_PATH} not found — run scripts/make_test_images.py first"
        )
    labels, queries = load_labels()
    k = TOPK

    name_to_idx = {p.name: i for i, p in enumerate(paths)}

    print(f"Embedding {len(paths)} photos with all three models...")
    images = [Image.open(p).convert("RGB") for p in paths]

    l14_img, l14_txt, _ = embed_with_l14(images, queries)
    b32_img, b32_txt, _ = embed_with_b32_onnx(images, queries, "fp32")
    b8_img, b8_txt, _ = embed_with_b32_onnx(images, queries, "int8")

    print("Rendering thumbnails...")
    thumbs = [_thumb_b64(p) for p in paths]

    models = [
        ("ViT-L-14", l14_img, l14_txt, True),
        ("B/32 fp32", b32_img, b32_txt, False),
        ("B/32 int8", b8_img, b8_txt, False),
    ]

    # ---- Dataset sanity gate (truth-image cosine + margin) ---------------
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
    gate_ok = (gate_truth >= L14_MEAN_GATE and gate_margin >= L14_MARGIN_GATE)

    sections = []
    hits = {name: 0 for name, *_ in models}
    total = 0
    misses = {name: [] for name, *_ in models}

    for fname, info in labels.items():
        truth_idx = name_to_idx[fname]
        for caption in info.get("must_have", []):
            total += 1
            ci = queries.index(caption)
            rows = []
            for name, img_m, txt_m, is_baseline in models:
                scores = img_m @ txt_m[ci]
                order = np.argsort(scores)[::-1][:k]
                ranking = [(int(i), float(scores[i])) for i in order]
                hit = any(i == truth_idx for i, _ in ranking)
                if hit:
                    hits[name] += 1
                else:
                    misses[name].append((caption, fname))
                truth_rank = next(
                    (r for r, (i, _) in enumerate(ranking, 1) if i == truth_idx),
                    None)
                label = (
                    f"{name} "
                    f"<span class='ref'>baseline</span>"
                    if is_baseline else
                    f"{name} <span class='ov'>"
                    + ("HIT" if hit else f"MISS (truth #{truth_rank})")
                    + "</span>"
                )
                rows.append(_row(label, ranking, paths, thumbs, truth_idx))
            sections.append(
                f'<h2>"{html.escape(caption)}" '
                f'<span class="truthfile">truth: {html.escape(fname)}</span></h2>'
                + "".join(rows)
            )

    nq = len(queries)
    summary = (
        f"<p class='big'>VERDICT — truth photo ranked in the top-{k}: "
        + " · ".join(
            f"<b>{name}: {hits[name]}/{total}"
            f" ({100 * hits[name] / total:.0f}%)</b>"
            for name, *_ in models)
        + "</p>"
        + (
            f"<p class='gate ok'>Dataset sanity gate: ViT-L-14 truth-image "
            f"cosine = {gate_truth:.3f} (gate ≥ {L14_MEAN_GATE}), margin over "
            f"non-truth images = {gate_margin:.3f} (gate ≥ {L14_MARGIN_GATE}) "
            f"— the images are photo-like enough to judge models.</p>"
            if gate_ok else
            f"<p class='gate bad'>Dataset sanity gate FAILED: ViT-L-14 "
            f"truth-image cosine = {gate_truth:.3f} (gate ≥ {L14_MEAN_GATE}), "
            f"margin = {gate_margin:.3f} (gate ≥ {L14_MARGIN_GATE}). The "
            f"desktop model itself cannot recognize these images — the "
            f"dataset is too unlike photos, so NO model verdict on this page "
            f"is trustworthy. Fix the dataset first.</p>"
        )
        + "<p class='note'>Gold border = the photo the caption is true of "
        "(labels.json must_have). Green border + TRUTH ✓ HIT = the model "
        "ranked the right photo in its top-" f"{k}. Score = cosine "
        "similarity (0..1).</p>"
    )

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>M0 model verdict — ViT-L-14 vs ViT-B/32</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#fafafa;margin:24px}}
h1{{font-size:22px}} h2{{font-size:16px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.truthfile{{font-size:12px;color:#8a6d00;background:#fff3c4;border-radius:4px;padding:2px 6px;font-weight:600}}
.row{{display:flex;align-items:flex-start;margin:6px 0}}
.label{{width:170px;min-width:170px;font-weight:600;font-size:13px;padding-top:40px;color:#333}}
.ref{{color:#888;font-weight:400}} .ov{{color:#2e7d32;font-size:12px}}
.cards{{display:flex;gap:8px;flex-wrap:wrap}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px;text-align:center}}
.card img{{width:{THUMB_SIZE}px;height:{THUMB_SIZE}px;object-fit:cover;border-radius:4px;display:block}}
.cap{{font-size:11px;margin-top:3px}} .score{{color:#666}}
.truth{{color:#8a6d00;font-weight:700;font-size:10px}}
.truth.hit{{color:#2e7d32}}
.picked{{color:#555;font-size:10px}}
.big{{font-size:16px;background:#eef;border:1px solid #ccd;border-radius:6px;padding:10px}}
.gate{{font-size:14px;border-radius:6px;padding:10px;margin:10px 0}}
.gate.ok{{background:#e8f5e9;border:1px solid #a5d6a7}}
.gate.bad{{background:#fdecea;border:1px solid #ef9a9a}}
.note{{color:#555;font-size:13px}}
</style></head><body>
<h1>M0 verdict: is ViT-B/32 good enough to replace ViT-L-14 on the phone?</h1>
{summary}
{"".join(sections)}
</body></html>"""

    REPORT_PATH.write_text(page, encoding="utf-8")
    size_mb = os.path.getsize(REPORT_PATH) / 1e6
    print(f"\nWrote {REPORT_PATH}  ({size_mb:.1f} MB)")
    print("Open it in your browser: gold = truth photo, green = model got it "
          f"in its top-{k}.")
    print(f"\nQuick numbers (truth in top-{k}): "
          + ", ".join(f"{name} {hits[name]}/{total}" for name, *_ in models)
          + f" | sanity gate: truth {gate_truth:.3f}, margin {gate_margin:.3f} "
          f"({'OK' if gate_ok else 'FAILED'})")


if __name__ == "__main__":
    main()
