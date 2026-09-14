"""export_clip_b32_onnx.py — M0 Step 1: export CLIP ViT-B/32 to ONNX.

Why this exists:
    The Android app cannot run PyTorch — phones run ONNX Runtime. This script
    takes our candidate small model (ViT-B/32, ~151M params vs ViT-L-14's
    ~428M) and exports its two "towers" to models/onnx/:

        clip-b32-text-fp32.onnx    text tower, float32 (reference quality)
        clip-b32-image-fp32.onnx   image tower, float32
        clip-b32-text-int8.onnx    text tower, int8 weights (phone-sized)
        clip-b32-image-int8.onnx   image tower, int8 weights

What is a "tower"?
    CLIP is two encoders glued together: a TEXT tower (sentence -> 512-dim
    vector) and an IMAGE tower (photo -> 512-dim vector). We export them as
    separate ONNX files because on a phone you never run both at the same
    moment: indexing runs the image tower thousands of times, searching runs
    the text tower once per query.

What is int8 quantization?
    Storing the network weights as 8-bit integers instead of 32-bit floats
    makes the files ~4x smaller and inference faster on phone CPUs, at the
    cost of a tiny accuracy drop (M0 measures exactly how much).

Each tower normalizes its output (vector length = 1), so cosine similarity
stays a plain dot product — same math as the desktop app.

Run:
    python scripts/export_clip_b32_onnx.py
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "onnx"
MODEL_ID = "openai/clip-vit-base-patch32"  # same weights as ST "clip-ViT-B-32"
OPSET = 17


def _features(clip_out):
    """Pull the projected feature tensor out of get_*_features() output.

    transformers >= 5 returns a BaseModelOutputWithPooling whose
    pooler_output IS the projected features. transformers 4.x returned the
    tensor directly. Handling both keeps this script version-proof.
    """
    if hasattr(clip_out, "pooler_output"):  # transformers >= 5
        return clip_out.pooler_output
    if isinstance(clip_out, tuple):          # (last_hidden_state, projected)
        return clip_out[1]
    return clip_out                          # transformers 4.x: plain tensor


class TextEncoder(torch.nn.Module):
    """input_ids [batch, 77] -> normalized 512-dim embedding [batch, 512]."""

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, input_ids):
        features = _features(self.clip.get_text_features(input_ids=input_ids))
        return F.normalize(features, dim=-1)


class ImageEncoder(torch.nn.Module):
    """pixel_values [batch, 3, 224, 224] -> normalized 512-dim embedding."""

    def __init__(self, clip):
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values):
        features = _features(self.clip.get_image_features(pixel_values=pixel_values))
        return F.normalize(features, dim=-1)


def _export_onnx(module, dummy_args, out_path, input_name, output_name, dynamic_axes):
    """Export one module, handling torch version differences politely."""
    common = dict(
        f=str(out_path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: dynamic_axes, output_name: dynamic_axes},
        opset_version=OPSET,
        do_constant_folding=True,
    )
    try:
        # The classic TorchScript-based exporter — stable and well tested.
        torch.onnx.export(module, dummy_args, dynamo=False, **common)
    except TypeError:
        # Newer torch removed the `dynamo` kwarg — just use the default.
        torch.onnx.export(module, dummy_args, **common)


def _size_mb(path):
    return os.path.getsize(path) / 1e6


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} (first run downloads ~600 MB)...")
    clip = CLIPModel.from_pretrained(MODEL_ID).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    # ---- 1. Export fp32 towers -------------------------------------------
    text_fp32 = OUT_DIR / "clip-b32-text-fp32.onnx"
    image_fp32 = OUT_DIR / "clip-b32-image-fp32.onnx"

    print("Exporting TEXT tower (fp32)...")
    dummy_ids = torch.ones((1, 77), dtype=torch.int64)
    _export_onnx(
        TextEncoder(clip), (dummy_ids,), text_fp32,
        input_name="input_ids", output_name="embedding",
        dynamic_axes={0: "batch"},
    )

    print("Exporting IMAGE tower (fp32)...")
    dummy_pixels = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    _export_onnx(
        ImageEncoder(clip), (dummy_pixels,), image_fp32,
        input_name="pixel_values", output_name="embedding",
        dynamic_axes={0: "batch"},
    )

    # ---- 2. Quantize to int8 ---------------------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic

    text_int8 = OUT_DIR / "clip-b32-text-int8.onnx"
    image_int8 = OUT_DIR / "clip-b32-image-int8.onnx"

    print("Quantizing TEXT tower to int8...")
    # NOTE: per_channel=True made parity WORSE on this stack (image tower
    # cosine fell to 0.39 — dynamic per-channel MatMul quantization is
    # unreliable for transformers in ORT). Per-tensor is the stable choice:
    # parity ~0.91-0.92. If that hurts rankings, the next lever is static
    # QDQ quantization with a calibration set, not per-channel.
    #
    # Extra lever: keep the final PROJECTION layers (text_projection /
    # visual_projection) in fp32. They map into CLIP's shared semantic
    # space, so quantizing them damages the geometry the most.
    def _projection_nodes(onnx_path):
        import onnx

        return [
            n.name
            for n in onnx.load(str(onnx_path)).graph.node
            if "projection" in n.name.lower()
        ]

    text_excl = _projection_nodes(text_fp32)
    image_excl = _projection_nodes(image_fp32)
    print(f"  excluding from quantization: {text_excl + image_excl}")

    quantize_dynamic(
        str(text_fp32), str(text_int8),
        weight_type=QuantType.QInt8, nodes_to_exclude=text_excl,
    )
    print("Quantizing IMAGE tower to int8...")
    quantize_dynamic(
        str(image_fp32), str(image_int8),
        weight_type=QuantType.QInt8, nodes_to_exclude=image_excl,
    )

    # ---- 3. Self-check: ONNX must match PyTorch --------------------------
    # If the export broke something, we want to know NOW, not in M3.
    print("\nParity check (ONNX vs PyTorch, cosine should be ~1.0):")
    import onnxruntime as ort

    texts = ["a photo of a sunset over the ocean", "my dog sleeping on the sofa"]
    ids = tokenizer(
        texts, padding="max_length", max_length=77, truncation=True,
        return_tensors="pt",
    )["input_ids"]

    with torch.inference_mode():
        ref_text = F.normalize(
            _features(clip.get_text_features(input_ids=ids)), dim=-1
        ).numpy()

    sess_t32 = ort.InferenceSession(str(text_fp32), providers=["CPUExecutionProvider"])
    sess_t8 = ort.InferenceSession(str(text_int8), providers=["CPUExecutionProvider"])
    for name, sess in [("fp32", sess_t32), ("int8", sess_t8)]:
        out = sess.run(None, {"input_ids": ids.numpy()})[0]
        cos = float((out * ref_text).sum(axis=1).mean())
        diff = float(np.abs(out - ref_text).max())
        print(f"  text  {name}: mean cosine={cos:.5f}  max|diff|={diff:.4f}")

    # One real image as the image-tower probe.
    probe = Path(__file__).resolve().parent.parent / "data" / "images"
    probe_path = next(iter(sorted(probe.glob("*.png"))), None)
    if probe_path is not None:
        img = Image.open(probe_path).convert("RGB")
        pixels = processor(images=[img, img], return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            ref_img = F.normalize(
                _features(clip.get_image_features(pixel_values=pixels)), dim=-1
            ).numpy()
        sess_i32 = ort.InferenceSession(str(image_fp32), providers=["CPUExecutionProvider"])
        sess_i8 = ort.InferenceSession(str(image_int8), providers=["CPUExecutionProvider"])
        for name, sess in [("fp32", sess_i32), ("int8", sess_i8)]:
            out = sess.run(None, {"pixel_values": pixels.numpy()})[0]
            cos = float((out * ref_img).sum(axis=1).mean())
            diff = float(np.abs(out - ref_img).max())
            print(f"  image {name}: mean cosine={cos:.5f}  max|diff|={diff:.4f}")

    # ---- 4. Sizes ---------------------------------------------------------
    print("\nExported files:")
    for f in [text_fp32, image_fp32, text_int8, image_int8]:
        print(f"  {f.name:28s} {_size_mb(f):7.1f} MB")
    int8_total = _size_mb(text_int8) + _size_mb(image_int8)
    print(f"  int8 total: {int8_total:.1f} MB  (Android target is <= ~150 MB)")


if __name__ == "__main__":
    main()
