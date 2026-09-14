"""model.py — the only file that knows about the CLIP model.

Why this file exists:
    Everything in this project is about embeddings. CLIP produces them,
    but loading the model is slow (a few seconds) and takes ~600 MB of RAM,
    so we load it exactly ONCE and reuse it for every image and every query.
    This file hides that detail behind two small functions.

Concepts:
    - "Pretrained" model: trained once by OpenAI on ~400 million image/caption
      pairs. We only download and use it — we never train anything.
    - "Embedding": the 512 numbers that represent one image or one piece of
      text inside the model's shared semantic space.

Important detail:
    We ask for normalize_embeddings=True, so every vector has length 1.
    That makes cosine similarity between two vectors equal to their simple
    dot product — no square roots needed. This keeps the search math clean.

Run it:
    python -m src.model data/images/any_photo.png
"""

from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

# Module-level cache. None means "not loaded yet".
_model = None


def get_model() -> SentenceTransformer:
    """Load the CLIP model the first time it is needed, then reuse it."""
    global _model
    if _model is None:
        print("Loading CLIP model (first time only)...")
        _model = SentenceTransformer("clip-ViT-L-14")
    return _model


def _crop_image(image, top, left, bottom, right):
    """Return a cropped sub-image given fractional coordinates (0..1)."""
    w, h = image.size
    return image.crop((int(left * w), int(top * h), int(right * w), int(bottom * h)))


def _augment_image(image):
    """Generate multiple crops/flips of an image for multi-crop embedding.

    Returns a list of PIL images (center, 4 corners, horizontal flips).
    This mimics how CLIP was trained with random crops — averaging these
    embeddings produces a more robust representation of the image.
    """
    augmented = [
        image,                                               # original
        _crop_image(image, 0.0, 0.0, 0.7, 0.7),            # top-left
        _crop_image(image, 0.0, 0.3, 0.7, 1.0),            # top-right
        _crop_image(image, 0.3, 0.0, 1.0, 0.7),            # bottom-left
        _crop_image(image, 0.3, 0.3, 1.0, 1.0),            # bottom-right
        image.transpose(Image.FLIP_LEFT_RIGHT),             # horizontal flip
        _crop_image(image, 0.15, 0.15, 0.85, 0.85),        # center (70%)
    ]
    return augmented


def embed_image(image_path: str, use_augmentation=True):
    """Return the embedding (numpy array) of one image file.

    When use_augmentation=True, generates multiple crops and flips of the
    image and averages their embeddings for a more robust representation.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {image_path}\n"
            "  Did you mean a photo inside data/images/ ?\n"
            "  (You can create placeholder images with: "
            "python scripts/make_test_images.py)"
        )

    model = get_model()
    image = Image.open(path).convert("RGB")

    if use_augmentation:
        crops = _augment_image(image)
        embeddings = model.encode(crops, normalize_embeddings=True)
        # Average all crop embeddings, then re-normalize
        avg = embeddings.mean(axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)
    else:
        return model.encode(image, normalize_embeddings=True)


def embed_text(text: str):
    """Return the 512-dim embedding (numpy array) of one text query."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/images/sunset.png"
    v = embed_image(path)
    print(f"Image embedding of {path}:")
    print("  shape:", v.shape)
    print("  first 5 values:", v[:5])
    print("  length (should be ~1.0):", round(float(v @ v) ** 0.5, 4))
