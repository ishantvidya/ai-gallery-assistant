"""indexer.py — turn a folder of photos into embeddings.

This is the "indexing" half of the app (the expensive, one-time half).

  - list_images():  find every photo file in data/images/
  - embed_all():    run each photo through CLIP and collect the results
                    into one big matrix

What is a matrix here?
    With N photos we get N vectors of 512 numbers each. Stacking them gives
    an N x 512 matrix — a 2D table with one ROW per photo and one COLUMN per
    number. Stage 4 will save this matrix to disk, and Stage 6 will compare
    every row against a single query vector in one fast step.

Run it:
    python -m src.indexer

Note: only common formats are read (.jpg .jpeg .png .webp .bmp). iPhone
photos exported as .heic are ignored — convert them to .jpg first.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from src.model import get_model, _augment_image
from src.storage import DEFAULT_INDEX_PATH, save_index

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"


def list_images(folder=DEFAULT_IMAGES_DIR):
    """Return a sorted list of image file paths found in the folder."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_image(p):
    """Load one image file, return (path, PIL Image) or (path, None) on error."""
    try:
        return p, Image.open(p).convert("RGB")
    except Exception as e:
        print(f"  warning: skipping {p.name}: {e}")
        return p, None


def embed_all(image_paths, use_augmentation=True):
    """Embed every image.

    When use_augmentation=True, each image is cropped/flipped into 7 variants,
    each variant is embedded, and the embeddings are averaged. This produces
    more robust representations that match CLIP's training behavior.

    Returns (paths, matrix):
      - paths:  the list of files that were embedded successfully
      - matrix: numpy array of shape (N, 768), one row per image
    """
    model = get_model()

    # Concurrent image loading (I/O bound — benefits from threads)
    print(f"  Loading {len(image_paths)} images (concurrent)...")
    loaded = {}  # path -> Image
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_load_image, p): p for p in image_paths}
        for future in as_completed(futures):
            p, img = future.result()
            if img is not None:
                loaded[p] = img

    good_paths = [p for p in image_paths if p in loaded]
    images = [loaded[p] for p in good_paths]

    if not images:
        raise ValueError(
            "No readable images found. Add photos to data/images/ first "
            "(or run python scripts/make_test_images.py for placeholders)."
        )

    if use_augmentation:
        # Multi-crop augmentation: generate 7 variants per image, embed each,
        # and average the embeddings. This is slower but much more accurate.
        print(f"  Augmenting {len(images)} images (7 crops each)...")
        embeddings = []
        for i, img in enumerate(images):
            crops = _augment_image(img)
            crop_embeds = model.encode(crops, normalize_embeddings=True)
            avg = crop_embeds.mean(axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)  # re-normalize
            embeddings.append(avg)
            if (i + 1) % 5 == 0 or i == len(images) - 1:
                print(f"    [{i + 1}/{len(images)}] done")
        matrix = np.array(embeddings, dtype=np.float32)
    else:
        # Fast path: embed raw images without augmentation
        CHUNK = 16
        parts = []
        for i in range(0, len(images), CHUNK):
            parts.append(
                model.encode(images[i:i + CHUNK], normalize_embeddings=True)
            )
        matrix = np.concatenate(parts)

    return good_paths, matrix


def build_index(folder=DEFAULT_IMAGES_DIR, index_path=DEFAULT_INDEX_PATH):
    """The complete one-time job: scan folder -> embed -> save to disk."""
    paths = list_images(folder)
    if not paths:
        raise ValueError(
            f"No images found in {folder}. Add photos there first "
            "(or run python scripts/make_test_images.py for placeholders)."
        )
    good_paths, matrix = embed_all(paths)
    save_index(good_paths, matrix, index_path)
    return good_paths, matrix


if __name__ == "__main__":
    good_paths, matrix = build_index()
    print(f"Indexed {len(good_paths)} images -> matrix shape: {matrix.shape}")
