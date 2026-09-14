"""make_test_images.py — M0 test dataset generator (v2).

Why this exists
---------------
The M0 verdict ("is ViT-B/32 good enough to replace ViT-L-14 on the phone?")
is only as good as the dataset it is judged on. The first version of this
script drew FLAT-COLOR SHAPES: 2-4 solid fills per image, no texture. The
resulting images were so unlike photographs that:

  * ViT-L-14 (the desktop baseline!) scored only 0.06-0.15 against everyday
    captions — real photo/caption matches score 0.25-0.35. The baseline
    literally could not recognize anything, so its rankings were noise.
  * "Top-5 overlap with the baseline" was then agreement-with-noise, and the
    benchmark could not measure accuracy at all.
  * Laplacian variance (a blur metric) was 60-316 vs 1000+ for sharp photos —
    the images were objectively blurry.

What v2 does instead
--------------------
Draws 12 textured, high-frequency scenes with numpy (gradients + multi-octave
value noise + fine grain + strong edges), so they sit much closer to the
statistics of real photos. Each scene contains a distinctive object that CLIP
is expected to name (a dog, a car, a beach...), so the dataset can support a
*correctness* metric, not just an agreement metric.

Alongside the images it writes labels.json:

    {"1.png": {"scene": "...", "must_have": ["a dog"], "should_have": [...]}}

  must_have   — a caption that is objectively TRUE of the image. Any CLIP
                model should rank this image top-3 for it.
  should_have — related captions; useful for eyeballing near-misses, not
                scored automatically.

This file is the ground truth that benchmark_b32_vs_l14.py scores against.
When you replace these placeholders with real photos, write the same
labels.json for them (or at least keep must_have accurate) — the verdict is
only meaningful if the ground truth is true.

Run:
    python scripts/make_test_images.py
"""

import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "images"

# Portrait/landscape mix, like a real camera roll.
W, H = 768, 1024


# ----------------------------------------------------------------------------
# Texture toolkit: the difference between "flat drawing" and "photo-ish" is
# multi-scale random variation, so every scene composes these helpers.
# ----------------------------------------------------------------------------
def _rng(seed):
    return np.random.default_rng(seed)


def value_noise(shape, cell, rng):
    """Smooth random field: random values on a coarse grid, bilinearly
    interpolated. Multi-octave stacks of this read as natural texture."""
    gh, gw = shape[0] // cell + 2, shape[1] // cell + 2
    grid = rng.random((gh, gw))
    ys = np.linspace(0, gh - 1 - 1e-6, shape[0])
    xs = np.linspace(0, gw - 1 - 1e-6, shape[1])
    y0 = ys.astype(int)
    x0 = xs.astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x0 + 1)]
    g10 = grid[np.ix_(y0 + 1, x0)]
    g11 = grid[np.ix_(y0 + 1, x0 + 1)]
    top = g00 * (1 - fx) + g01 * fx
    bot = g10 * (1 - fx) + g11 * fx
    return top * (1 - fy) + bot * fy


def fbm(shape, rng, octaves=(64, 16, 4), weights=(0.6, 0.3, 0.1)):
    """Fractal (multi-octave) value noise in [0, 1]."""
    out = np.zeros(shape, dtype=np.float32)
    total = sum(weights)
    for cell, w in zip(octaves, weights):
        out += value_noise(shape, cell, rng) * (w / total)
    return out


def gradient_field(h, w, top_rgb, bottom_rgb):
    """Smooth vertical gradient (sky, water, backgrounds)."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.asarray(top_rgb, dtype=np.float32)[None, None, :]
    bot = np.asarray(bottom_rgb, dtype=np.float32)[None, None, :]
    col = top * (1 - t) + bot * t  # (h, 1, 3)
    return np.broadcast_to(col, (h, w, 3)).copy()


def grain(img, rng, amount=7.0):
    """Per-pixel sensor-like noise — kills the 'vector art' flatness."""
    n = rng.normal(0.0, amount, img.shape).astype(np.float32)
    return np.clip(img + n, 0, 255)


def vignette(img, strength=0.25):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = 1.0 - strength * np.clip(d - 0.55, 0, None) ** 2
    return np.clip(img * mask[:, :, None], 0, 255)


def sphere_shade(mask, light=(255, 255, 255), dark=None, depth=70):
    """Radial shading for round objects (fruit, sun, ball) — an object with
    uniform fill looks pasted-on; a shaded one reads as 3-D."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx, r = h / 2, w / 2, max(h, w) / 2
    d = np.sqrt(((yy - cy) / r) ** 2 + ((xx - cx) / r) ** 2)
    shade = np.clip(1.0 - d * 1.1, -1.0, 1.0)  # +1 center → -1 rim
    dark = dark if dark is not None else tuple(int(c * 0.45) for c in light)
    light = np.asarray(light, dtype=np.float32)
    dark = np.asarray(dark, dtype=np.float32)
    t = (shade + 1) / 2  # 0 rim → 1 center
    return light[None, None, :] * t[:, :, None] + dark[None, None, :] * (1 - t[:, :, None])


def paste(img, patch, mask):
    """Composite patch onto img using a float mask in [0, 1]."""
    return img * (1 - mask[:, :, None]) + patch * mask[:, :, None]


def circle_mask(shape, cx, cy, r, soft=2.0):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / soft, 0, 1)


def ellipse_mask(shape, box, soft=2.0):
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    rx = (box[2] - box[0]) / 2
    ry = (box[3] - box[1]) / 2
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return np.clip((1 - d) * max(rx, ry) / soft, 0, 1)


def rect_mask(shape, box, soft=2.0):
    h, w = shape
    m = np.zeros((h, w), dtype=np.float32)
    m[box[1]:box[3], box[0]:box[2]] = 1.0
    if soft > 0:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(soft)), dtype=np.float32) / 255.0
    return m


def finalize(arr, rng):
    """Last step for every scene: sensor grain, then an unsharp pass so the
    result has the high-frequency edge energy of a real photo. Without this
    the images measure as blurry (Laplacian variance < 300) and CLIP sees
    vector art, not photography."""
    img = to_img(grain(vignette(arr), rng))
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))


def to_img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------
# The 12 scenes. Each returns an (H, W, 3) float array.
# Every scene pairs a distinctive object (the must_have concept) with a
# textured background, so CLIP has both an object to recognize and photo-like
# statistics to work with.
# ----------------------------------------------------------------------------
SIZE = (H, W)


def scene_dog(rng):
    """A dog on grass — brown body, darker head, ears, tail; fbm lawn + sky."""
    img = gradient_field(H, W, (135, 185, 235), (200, 225, 245))  # sky
    horizon = int(H * 0.55)
    lawn = fbm(SIZE, rng, octaves=(48, 12, 3))
    grass = np.stack([
        60 + 70 * lawn, 130 + 60 * lawn, 45 + 45 * lawn], axis=-1)
    gm = rect_mask(SIZE, (0, horizon, W, H), soft=6)
    img = paste(img, grass, gm)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    body_c, body_r = (W * 0.5, H * 0.62), (150, 95)   # ellipse radii
    body = (((xx - body_c[0]) / body_r[0]) ** 2 + ((yy - body_c[1]) / body_r[1]) ** 2)
    body_m = np.clip((1 - body) * 40, 0, 1)
    dog_col = sphere_shade(body_m, light=(190, 135, 80), dark=(90, 55, 25))
    img = paste(img, dog_col, body_m)

    head_m = circle_mask(SIZE, W * 0.5 + 120, H * 0.52, 62, soft=2)
    img = paste(img, sphere_shade(head_m, light=(200, 145, 90), dark=(95, 60, 30)), head_m)
    # ears (two dark ellipses), legs (four rectangles), tail (diagonal)
    for ex in (-45, 45):
        ear = ellipse_mask(SIZE, (W * 0.5 + 120 + ex - 14, H * 0.52 - 80,
                                  W * 0.5 + 120 + ex + 14, H * 0.52 - 5), soft=2)
        img = paste(img, np.full_like(img, 70.0), ear * 0.9)
    for lx in (-95, -35, 35, 95):
        leg = rect_mask(SIZE, (int(W * 0.5 + lx - 12), int(H * 0.66),
                               int(W * 0.5 + lx + 12), int(H * 0.78)), soft=2)
        img = paste(img, dog_col, leg)
    tail = ellipse_mask(SIZE, (W * 0.5 - 190, H * 0.55, W * 0.5 - 120, H * 0.64), soft=3)
    img = paste(img, dog_col, tail)
    return finalize(img, rng)


def scene_cat(rng):
    """A cat on a sofa — grey body, stripes, pointy ears; warm room bg."""
    img = gradient_field(H, W, (120, 95, 80), (70, 52, 42))       # dim room
    sofa = fbm(SIZE, rng, octaves=(56, 14))
    sofa_col = np.stack([120 + 50 * sofa, 60 + 25 * sofa, 50 + 22 * sofa], axis=-1)
    img = paste(img, sofa_col, rect_mask(SIZE, (0, int(H * 0.45), W, H), soft=8))

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    body_r = (170, 100)
    body = (((xx - W * 0.48) / body_r[0]) ** 2 + ((yy - H * 0.66) / body_r[1]) ** 2)
    body_m = np.clip((1 - body) * 40, 0, 1)
    # tabby stripes: sinusoidal dark bands modulated by noise
    stripes = (np.sin(xx / 14.0 + 3 * fbm(SIZE, rng, octaves=(32,))) > 0.55).astype(np.float32)
    grey = np.stack([175 + 20 * fbm(SIZE, rng, octaves=(24, 6)),
                     175 + 20 * fbm(SIZE, rng, octaves=(24, 6)),
                     180 + 20 * fbm(SIZE, rng, octaves=(24, 6))], axis=-1)
    grey = grey * (1 - 0.45 * stripes[:, :, None])
    grey = sphere_shade(body_m, light=(200, 200, 205), dark=(90, 90, 95)) * body_m[:, :, None] \
        + grey * (1 - body_m[:, :, None])
    img = paste(img, grey, body_m)

    head_m = circle_mask(SIZE, W * 0.52, H * 0.5, 58, soft=2)
    img = paste(img, sphere_shade(head_m, light=(205, 205, 210), dark=(95, 95, 100)), head_m)
    # pointy ears (triangles via polygon masks)
    ear = np.zeros(SIZE, dtype=np.float32)
    ear_img = to_img(ear)
    d = ImageDraw.Draw(ear_img)
    hx, hy = W * 0.52, H * 0.5
    d.polygon([(hx - 52, hy - 28), (hx - 20, hy - 52), (hx - 8, hy - 12)], fill=255)
    d.polygon([(hx + 52, hy - 28), (hx + 20, hy - 52), (hx + 8, hy - 12)], fill=255)
    ear_m = np.asarray(ear_img, dtype=np.float32) / 255.0
    img = paste(img, np.full_like(img, 150.0), ear_m)
    # eyes: two bright green slits
    for ex in (-22, 22):
        eye = ellipse_mask(SIZE, (hx + ex - 9, hy - 12, hx + ex + 9, hy + 8), soft=1)
        img = paste(img, np.stack([120 * np.ones(SIZE), 230 * np.ones(SIZE),
                                   90 * np.ones(SIZE)], axis=-1), eye)
    return finalize(img, rng)


def scene_sunset(rng):
    """Sunset over the ocean — warm gradient sky, glowing sun, dark sea band."""
    img = gradient_field(H, W, (70, 60, 130), (250, 170, 90))
    horizon = int(H * 0.58)
    # sun with soft glow
    sun_m = circle_mask(SIZE, W * 0.5, horizon - 90, 55, soft=3)
    glow_m = circle_mask(SIZE, W * 0.5, horizon - 90, 170, soft=60)
    img = paste(img, np.full_like(img, 255.0), sun_m * 0.95)
    img = paste(img, np.stack([255 * np.ones(SIZE), 200 * np.ones(SIZE),
                               120 * np.ones(SIZE)], axis=-1), glow_m * 0.35)
    # sea: darker blue with horizontal wave shimmer near the horizon
    sea = gradient_field(H - horizon, W, (40, 60, 110), (25, 35, 70))
    shimmer = fbm((H - horizon, W), rng, octaves=(6, 2))
    band = np.exp(-np.abs(np.arange(H - horizon)[:, None] - 20) / 90.0)
    sea = sea * (1 - 0.35 * band[:, :, None] * shimmer[:, :, None]) \
        + np.stack([255 * shimmer, 190 * shimmer, 120 * shimmer], axis=-1) \
        * (0.5 * band[:, :, None])
    img[horizon:] = sea
    return finalize(img, rng)


def scene_car(rng):
    """A red car on a mountain road — body, cabin, wheels, dashed center line."""
    img = gradient_field(H, W, (120, 175, 225), (215, 230, 240))
    mtn = fbm(SIZE, rng, octaves=(90, 22))
    mtn_col = np.stack([90 + 60 * mtn, 110 + 50 * mtn, 130 + 45 * mtn], axis=-1)
    horizon = int(H * 0.42)
    img = paste(img, mtn_col, rect_mask(SIZE, (0, horizon, W, int(H * 0.55)), soft=4))
    road_col = np.stack([70 + 8 * fbm(SIZE, rng, octaves=(40,)),
                         70 + 8 * fbm(SIZE, rng, octaves=(40,)),
                         75 + 8 * fbm(SIZE, rng, octaves=(40,))], axis=-1)
    road_m = rect_mask(SIZE, (0, int(H * 0.55), W, H), soft=2)
    img = paste(img, road_col, road_m)
    # dashed center line, strong perspective cue
    for i, y in enumerate(range(int(H * 0.62), H, 60)):
        wln = 4 + i
        ln = rect_mask(SIZE, (W // 2 - wln, y, W // 2 + wln, y + 26 + i * 2), soft=1)
        img = paste(img, np.full_like(img, 235.0), ln)
    # car: red body + darker cabin + two wheels with hubcaps
    body_m = ellipse_mask(SIZE, (int(W * 0.22), int(H * 0.66), int(W * 0.78), int(H * 0.82)), soft=3)
    red = sphere_shade(body_m, light=(235, 60, 50), dark=(110, 15, 15))
    img = paste(img, red, body_m)
    cab_m = ellipse_mask(SIZE, (int(W * 0.33), int(H * 0.60), int(W * 0.66), int(H * 0.70)), soft=3)
    cab = sphere_shade(cab_m, light=(80, 130, 200), dark=(20, 40, 80))
    img = paste(img, cab, cab_m * 0.95)
    for cx in (int(W * 0.34), int(W * 0.66)):
        wheel = circle_mask(SIZE, cx, int(H * 0.83), 34, soft=1.5)
        img = paste(img, np.full_like(img, 25.0), wheel)
        hub = circle_mask(SIZE, cx, int(H * 0.83), 12, soft=1)
        img = paste(img, np.full_like(img, 150.0), hub)
    return finalize(img, rng)


def scene_food(rng):
    """Food on a plate — pasta texture on a white plate, wooden table."""
    table = fbm(SIZE, rng, octaves=(70, 18, 4))
    img = np.stack([150 + 55 * table, 105 + 40 * table, 60 + 25 * table], axis=-1)
    # plate: large off-white circle with rim shading
    plate_m = circle_mask(SIZE, W // 2, int(H * 0.55), 250, soft=3)
    plate = sphere_shade(plate_m, light=(250, 248, 242), dark=(180, 175, 165))
    img = paste(img, plate, plate_m)
    # inner well shadow
    well_m = circle_mask(SIZE, W // 2, int(H * 0.55), 195, soft=8)
    img = paste(img, plate * 0.94, well_m)
    # pasta: tangled high-frequency strands in warm yellow
    pasta = fbm(SIZE, rng, octaves=(9, 3), weights=(0.5, 0.5))
    strands = (np.sin(pasta * 40) > 0.4).astype(np.float32)
    pasta_m = circle_mask(SIZE, W // 2, int(H * 0.55), 180, soft=4) * (0.55 + 0.45 * strands)
    pasta_col = np.stack([215 + 30 * pasta, 170 + 30 * pasta, 70 + 25 * pasta], axis=-1)
    # a few red tomato dots and green basil flecks
    rr = _rng(7)
    for _ in range(9):
        cx, cy = W // 2 + int(rr.normal(0, 80)), int(H * 0.55) + int(rr.normal(0, 80))
        tom = circle_mask(SIZE, cx, cy, rr.integers(9, 16), soft=1.5)
        pasta_m = np.maximum(pasta_m, tom * 0.95)
        pasta_col = np.where(tom[:, :, None] > 0.5,
                             np.stack([200 * np.ones(SIZE), 45 * np.ones(SIZE),
                                       35 * np.ones(SIZE)], axis=-1), pasta_col)
    for _ in range(6):
        cx, cy = W // 2 + int(rr.normal(0, 100)), int(H * 0.55) + int(rr.normal(0, 100))
        bas = ellipse_mask(SIZE, (cx - 12, cy - 6, cx + 12, cy + 6), soft=1)
        pasta_m = np.maximum(pasta_m, bas * 0.95)
        pasta_col = np.where(bas[:, :, None] > 0.5,
                             np.stack([45 * np.ones(SIZE), 130 * np.ones(SIZE),
                                       50 * np.ones(SIZE)], axis=-1), pasta_col)
    img = paste(img, pasta_col, pasta_m)
    return finalize(img, rng)


def scene_city_night(rng):
    """City street at night — dark sky, lit building windows, road glow."""
    img = gradient_field(H, W, (15, 18, 40), (40, 35, 60))
    rr = _rng(3)
    horizon = int(H * 0.68)
    # skyline: rectangles of varying height with lit windows
    x = 0
    while x < W:
        bw = int(rr.integers(70, 150))
        bh = int(rr.integers(150, int(H * 0.52)))
        bm = rect_mask(SIZE, (x, horizon - bh, x + bw, horizon), soft=1)
        bld = np.full_like(img, 0.0)
        bld[:, :, 0] = 35 + 12 * fbm(SIZE, rng, octaves=(30,))
        bld[:, :, 1] = 38 + 12 * fbm(SIZE, rng, octaves=(30,))
        bld[:, :, 2] = 48 + 12 * fbm(SIZE, rng, octaves=(30,))
        img = paste(img, bld, bm)
        # windows: small bright grid, randomly lit
        for wy in range(horizon - bh + 14, horizon - 10, 22):
            for wx in range(x + 8, x + bw - 10, 18):
                if rr.random() < 0.45:
                    wm = rect_mask(SIZE, (wx, wy, wx + 8, wy + 10), soft=0.5)
                    warm = np.stack([255 * np.ones(SIZE), 220 * np.ones(SIZE),
                                     140 * np.ones(SIZE)], axis=-1)
                    img = paste(img, warm, wm * 0.9)
        x += bw + int(rr.integers(6, 20))
    # road below with faint reflections
    road_m = rect_mask(SIZE, (0, horizon, W, H), soft=2)
    road = np.stack([25 * np.ones(SIZE), 25 * np.ones(SIZE), 30 * np.ones(SIZE)], axis=-1)
    img = paste(img, road, road_m)
    for lx in range(int(W * 0.1), int(W * 0.95), int(W * 0.17)):
        lm = ellipse_mask(SIZE, (lx - 8, horizon, lx + 8, horizon + 60), soft=6)
        img = paste(img, np.stack([120 * np.ones(SIZE), 110 * np.ones(SIZE),
                                   80 * np.ones(SIZE)], axis=-1), lm * 0.5)
    # streetlamp glow
    lamp = circle_mask(SIZE, W * 0.5, horizon - 40, 120, soft=50)
    img = paste(img, np.stack([255 * np.ones(SIZE), 200 * np.ones(SIZE),
                               110 * np.ones(SIZE)], axis=-1), lamp * 0.25)
    return finalize(img, rng)


def scene_flower(rng):
    """Flower close-up — layered petals around a textured center, green bokeh."""
    img = gradient_field(H, W, (45, 110, 50), (25, 70, 35))
    leaf = fbm(SIZE, rng, octaves=(30, 8))
    img = img * (0.8 + 0.4 * leaf[:, :, None])
    # petals: 3 rings of rotated ellipses, pink with radial shading
    cx, cy = W // 2, int(H * 0.52)
    for ring, (n_pet, pr, col_l, col_d) in enumerate([
            (12, 300, (250, 190, 210), (190, 90, 130)),
            (9, 220, (255, 205, 220), (210, 110, 150)),
            (6, 140, (255, 220, 230), (225, 140, 170))]):
        for k in range(n_pet):
            ang = 2 * np.pi * k / n_pet + ring * 0.4
            px = cx + int(np.cos(ang) * (pr - 90))
            py = cy + int(np.sin(ang) * (pr - 90))
            pm = ellipse_mask(SIZE, (px - 95, py - 38, px + 95, py + 38), soft=6)
            pet = sphere_shade(pm, light=col_l, dark=col_d)
            img = paste(img, pet, pm * 0.96)
    # center: dense disc of tiny florets
    center_m = circle_mask(SIZE, cx, cy, 62, soft=2)
    florets = fbm(SIZE, rng, octaves=(3, 1), weights=(0.5, 0.5))
    center_col = np.stack([235 + 20 * florets, 160 + 40 * florets, 40 * np.ones(SIZE)], axis=-1)
    img = paste(img, center_col, center_m)
    return finalize(img, rng)


def scene_people(rng):
    """Three people outdoors — simple clothed figures, faces, park behind."""
    img = gradient_field(H, W, (150, 200, 235), (205, 225, 240))
    lawn = fbm(SIZE, rng, octaves=(48, 12))
    grass = np.stack([70 + 60 * lawn, 135 + 55 * lawn, 50 + 40 * lawn], axis=-1)
    img = paste(img, grass, rect_mask(SIZE, (0, int(H * 0.6), W, H), soft=6))
    rr = _rng(11)
    for i, (px, scale) in enumerate([(0.28, 1.0), (0.5, 1.12), (0.72, 0.95)]):
        cx = int(W * px)
        head_r = int(34 * scale)
        hy = int(H * (0.34 + 0.02 * (1 - scale)))
        head = circle_mask(SIZE, cx, hy, head_r, soft=2)
        img = paste(img, sphere_shade(head, light=(230, 185, 155), dark=(150, 105, 80)), head)
        # torso: colored shirt
        shirt_cols = [(70, 110, 200), (200, 80, 90), (240, 190, 70)]
        sc = shirt_cols[i]
        shirt = fbm(SIZE, rng, octaves=(26, 7))
        torso_m = ellipse_mask(SIZE, (cx - int(75 * scale), hy + head_r,
                                      cx + int(75 * scale), int(H * 0.62)), soft=8)
        torso = np.stack([(sc[0] + 25 * shirt) * np.ones(SIZE),
                          (sc[1] + 25 * shirt) * np.ones(SIZE),
                          (sc[2] + 25 * shirt) * np.ones(SIZE)], axis=-1)
        img = paste(img, torso, torso_m)
        # simple face: eyes + mouth
        for ex in (-12, 12):
            eye = circle_mask(SIZE, cx + ex, hy - 6, 3.5, soft=0.8)
            img = paste(img, np.full_like(img, 30.0), eye)
        mouth = ellipse_mask(SIZE, (cx - 10, hy + 12, cx + 10, hy + 18), soft=1.5)
        img = paste(img, np.stack([170 * np.ones(SIZE), 70 * np.ones(SIZE),
                                   60 * np.ones(SIZE)], axis=-1), mouth)
    return finalize(img, rng)


def scene_mountains(rng):
    """Snow-capped mountain landscape — jagged ridges, snow, forest below."""
    img = gradient_field(H, W, (110, 160, 215), (190, 215, 235))
    horizon = int(H * 0.55)
    # two ridge layers using smoothed random height profiles
    for layer, (base, amp, col_l, col_d, snowline) in enumerate([
            (int(H * 0.42), 90, (110, 120, 140), (70, 80, 100), 0.35),
            (int(H * 0.5), 60, (85, 100, 85), (50, 62, 52), 0.55)]):
        prof = fbm((1, W), rng, octaves=(70, 14))[0]
        prof = (prof - prof.min()) / (prof.max() - prof.min() + 1e-6)
        ridge_y = (base - prof * amp).astype(int)
        m = np.zeros(SIZE, dtype=np.float32)
        for x in range(W):
            m[ridge_y[x]:, x] = 1.0
        m = np.asarray(to_img(m * 255).filter(ImageFilter.GaussianBlur(1.2)),
                       dtype=np.float32) / 255.0
        slope = fbm(SIZE, rng, octaves=(18, 5))
        rock = np.stack([(col_l[0] + (col_d[0] - col_l[0]) * slope),
                         (col_l[1] + (col_d[1] - col_l[1]) * slope),
                         (col_l[2] + (col_d[2] - col_l[2]) * slope)], axis=-1)
        # snow on upper slopes
        yy = np.mgrid[0:H, 0:W][0]
        snow = np.clip((snowline * H - yy) / 120.0, 0, 1)
        rock = rock * (1 - snow[:, :, None] * 0.9) + np.full_like(rock, 245.0) * snow[:, :, None] * 0.9
        img = paste(img, rock, m)
    # forest strip at the bottom
    forest = fbm(SIZE, rng, octaves=(24, 6))
    fr = np.stack([30 + 25 * forest, 75 + 35 * forest, 35 + 25 * forest], axis=-1)
    img = paste(img, fr, rect_mask(SIZE, (0, int(H * 0.78), W, H), soft=4))
    return finalize(img, rng)


def scene_beach(rng):
    """Tropical beach — turquoise sea, sand, palm tree with fronds."""
    img = gradient_field(H, W, (110, 190, 230), (170, 225, 240))
    horizon = int(H * 0.45)
    sea = gradient_field(H - horizon, W, (30, 150, 170), (60, 190, 200))
    waves = fbm((H - horizon, W), rng, octaves=(10, 3))
    sea = sea * (0.9 + 0.25 * waves[:, :, None])
    img[horizon:] = sea[: H - horizon]
    sand_m = rect_mask(SIZE, (0, int(H * 0.72), W, H), soft=6)
    sand = fbm(SIZE, rng, octaves=(30, 8))
    sand_col = np.stack([225 + 20 * sand, 200 + 20 * sand, 155 + 20 * sand], axis=-1)
    img = paste(img, sand_col, sand_m)
    # foam line
    foam = rect_mask(SIZE, (0, int(H * 0.70), W, int(H * 0.73)), soft=4)
    img = paste(img, np.full_like(img, 250.0), foam * 0.8)
    # palm: curved trunk + radiating fronds
    base_x, base_y = int(W * 0.72), int(H * 0.86)
    trunk_img = np.zeros(SIZE, dtype=np.float32)
    d = ImageDraw.Draw(to_img(trunk_img))
    for t in np.linspace(0, 1, 40):
        tx = base_x + int(np.sin(t * 1.3) * 60 * t)
        ty = base_y - int(t * (H * 0.45))
        tk = ellipse_mask(SIZE, (tx - 10, ty - 12, tx + 10, ty + 12), soft=2)
        trunk_img = np.maximum(trunk_img, tk)
    trunk_col = np.stack([120 + 30 * fbm(SIZE, rng, octaves=(20, 5)),
                          85 + 25 * fbm(SIZE, rng, octaves=(20, 5)),
                          50 + 20 * fbm(SIZE, rng, octaves=(20, 5))], axis=-1)
    img = paste(img, trunk_col, trunk_img)
    top_x, top_y = base_x + int(np.sin(1.3) * 60), base_y - int(H * 0.45)
    for k in range(9):
        ang = -np.pi + np.pi * k / 8
        fx = top_x + int(np.cos(ang) * 170)
        fy = top_y + int(np.sin(ang) * 90) + 40
        frond = ellipse_mask(SIZE, (top_x - 10, top_y - 12, fx, fy), soft=8)
        frond_col = np.stack([40 + 20 * fbm(SIZE, rng, octaves=(24, 6)),
                              130 + 40 * fbm(SIZE, rng, octaves=(24, 6)),
                              50 + 20 * fbm(SIZE, rng, octaves=(24, 6))], axis=-1)
        img = paste(img, frond_col, frond * 0.95)
    return finalize(img, rng)


def scene_bicycle(rng):
    """Bicycle against a wall — two spoked wheels, frame tubes, handlebar."""
    wall = fbm(SIZE, rng, octaves=(60, 15))
    img = np.stack([190 + 40 * wall, 175 + 40 * wall, 150 + 35 * wall], axis=-1)
    # brick joints
    for y in range(0, H, 60):
        img = paste(img, img * 0.85, rect_mask(SIZE, (0, y, W, y + 4), soft=1))
    floor_m = rect_mask(SIZE, (0, int(H * 0.8), W, H), soft=3)
    img = paste(img, np.stack([120 + 15 * fbm(SIZE, rng, octaves=(40,)),
                               118 + 15 * fbm(SIZE, rng, octaves=(40,)),
                               115 + 15 * fbm(SIZE, rng, octaves=(40,))], axis=-1), floor_m)
    cx, cy, r = W // 2, int(H * 0.64), 170
    # wheels: rim ring + spokes + tire
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for wx in (cx - r, cx + r):
        d = np.sqrt((xx - wx) ** 2 + (yy - cy) ** 2)
        tire = np.clip(1 - np.abs(d - r) / 7, 0, 1)
        img = paste(img, np.full_like(img, 30.0), tire)
        ang = np.arctan2(yy - cy, xx - wx)
        spokes = (np.abs(np.sin(ang * 12)) < 0.06) & (d < r - 8)
        spoke_m = np.where(spokes, 1.0, 0.0).astype(np.float32)
        spoke_m = np.asarray(to_img(spoke_m * 255).filter(ImageFilter.GaussianBlur(0.6)),
                             dtype=np.float32) / 255.0
        img = paste(img, np.full_like(img, 140.0), spoke_m * 0.9)
        hub = circle_mask(SIZE, wx, cy, 9, soft=1)
        img = paste(img, np.full_like(img, 120.0), hub)
    # frame: blue tubes
    frame_img = np.zeros(SIZE, dtype=np.float32)
    fi = to_img(frame_img)
    d2 = ImageDraw.Draw(fi)
    A, B = (cx - r, cy), (cx, cy)          # rear axle, bottom bracket
    C, D = (cx + 20, cy - 200), (cx + r, cy)
    d2.line([A, C, D, B, A, B], fill=255, width=14)
    d2.line([C, (cx + 40, cy - 250), (cx + 90, cy - 250)], fill=255, width=10)  # handlebar
    d2.line([(cx + 65, cy - 250), C], fill=255, width=12)                        # seat tube top
    fm = np.asarray(fi, dtype=np.float32) / 255.0
    frame_col = np.stack([40 * np.ones(SIZE), 90 * np.ones(SIZE), 200 * np.ones(SIZE)], axis=-1)
    img = paste(img, frame_col, fm)
    return finalize(img, rng)


def scene_boat(rng):
    """Sailboat on a lake — white sails, hull, mast, water reflections."""
    img = gradient_field(H, W, (135, 185, 230), (195, 220, 240))
    horizon = int(H * 0.55)
    water = fbm((H - horizon, W), rng, octaves=(8, 2))
    lake = np.stack([40 + 20 * water, 110 + 40 * water, 150 + 50 * water], axis=-1)
    img[horizon:] = lake
    cx, cy = W // 2, int(H * 0.45)
    # main sail (right triangle) + jib (left)
    sails = np.zeros(SIZE, dtype=np.float32)
    si = to_img(sails)
    d = ImageDraw.Draw(si)
    d.polygon([(cx + 8, cy - 200), (cx + 8, cy + 10), (cx + 130, cy + 10)], fill=255)
    d.polygon([(cx - 8, cy - 170), (cx - 8, cy + 10), (cx - 105, cy + 10)], fill=255)
    sail_m = np.asarray(si, dtype=np.float32) / 255.0
    sail_m = np.asarray(to_img(sail_m * 255).filter(ImageFilter.GaussianBlur(1.5)),
                        dtype=np.float32) / 255.0
    sail_col = np.stack([245 * np.ones(SIZE), 243 * np.ones(SIZE), 235 * np.ones(SIZE)], axis=-1)
    img = paste(img, sail_col, sail_m * 0.97)
    # mast + hull
    mast = rect_mask(SIZE, (cx - 4, cy - 210, cx + 4, cy + 15), soft=1)
    img = paste(img, np.full_like(img, 90.0), mast)
    hull_img = np.zeros(SIZE, dtype=np.float32)
    hi = to_img(hull_img)
    dh = ImageDraw.Draw(hi)
    dh.polygon([(cx - 110, cy + 18), (cx + 110, cy + 18), (cx + 80, cy + 55), (cx - 80, cy + 55)],
               fill=255)
    hull_m = np.asarray(hi, dtype=np.float32) / 255.0
    hull_col = np.stack([180 * np.ones(SIZE), 50 * np.ones(SIZE), 45 * np.ones(SIZE)], axis=-1)
    img = paste(img, hull_col, hull_m)
    # reflection: dark smeared blob below the boat
    refl = ellipse_mask(SIZE, (cx - 70, cy + 60, cx + 70, cy + 95), soft=10)
    img = paste(img, img * 0.55, refl * 0.7)
    return finalize(img, rng)


SCENES = [
    ("dog.png", scene_dog, "a dog in a park",
     ["a dog"], ["grass", "a park", "an animal outdoors"]),
    ("cat.png", scene_cat, "a cat on a sofa",
     ["a cat"], ["a sofa", "a grey tabby cat", "an animal indoors"]),
    ("sunset.png", scene_sunset, "a sunset over the ocean",
     ["a sunset over the ocean", "a sunset"], ["the sun", "the sea", "an orange sky"]),
    ("car.png", scene_car, "a red car on a road",
     ["a car on a road", "a red car"], ["a road", "mountains", "wheels"]),
    ("food.png", scene_food, "food on a plate",
     ["food on a plate", "pasta on a plate"], ["a white plate", "a wooden table", "tomatoes"]),
    ("city_night.png", scene_city_night, "a city street at night",
     ["a city street at night", "city buildings at night"],
     ["lit windows", "a road", "a dark sky"]),
    ("flower.png", scene_flower, "a close-up of a flower",
     ["a close-up of a flower", "a pink flower"], ["petals", "a green background"]),
    ("people.png", scene_people, "a group of people outdoors",
     ["a group of people", "three people"], ["faces", "a park", "colorful shirts"]),
    ("mountains.png", scene_mountains, "snow-capped mountains",
     ["snowy mountains", "a mountain landscape"], ["snow", "a forest", "a blue sky"]),
    ("beach.png", scene_beach, "a tropical beach with a palm tree",
     ["a beach", "a palm tree on a beach"], ["the sea", "sand", "palm fronds"]),
    ("bicycle.png", scene_bicycle, "a bicycle against a wall",
     ["a bicycle", "a bicycle leaning against a wall"], ["wheels with spokes", "a brick wall"]),
    ("boat.png", scene_boat, "a sailboat on a lake",
     ["a sailboat on the water", "a sailboat"], ["white sails", "a lake", "a mast"]),
]


def sharpness_report(img):
    """Laplacian variance blur metric — the number that flagged v1 as blurry."""
    g = np.asarray(img.convert("L").filter(
        ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=0)),
        dtype=np.float32)
    return float(g.var())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    print(f"{'file':16s} {'sharpness':>10s}   (target: > 800; v1 placeholders were 60-316)")
    for name, fn, scene, must, should in SCENES:
        seed = zlib.crc32(name.encode())
        arr = fn(_rng(seed))
        img = to_img(arr)
        img.save(OUT / name)
        s = sharpness_report(img)
        print(f"{name:16s} {s:10.0f}")
        labels[name] = {
            "scene": scene,
            "must_have": must,
            "should_have": should,
        }
    (OUT / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"\nWrote {len(SCENES)} images + labels.json to {OUT}")
    print("Replace these with real photos (keeping labels.json accurate) for the final M0 call.")


if __name__ == "__main__":
    main()
