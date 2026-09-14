# Android Port Plan — AI Gallery Assistant

**Goal:** Ship the photo-text-search app ("AI Gallery Assistant") to the Google Play Store as a
native Android app, for the general public.

**Decisions locked in (planning session):**
- Audience: **public app** (anyone can install; no accounts in v1)
- App tech: **native Kotlin** (Jetpack Compose)
- AI location: **on-device** — locked. All AI runs on the phone; no backend server.

---

## Status update (Sept 2026)

- **M0 PC-side verdict: ViT-B/32 int8 wins.** On the labels.json benchmark
  (dataset sanity gate passed: L-14 truth cosine 0.218, margin 0.086),
  truth-in-top-3 was **B/32 int8 22/22 (100%)**, B/32 fp32 and ViT-L-14 both
  21/22 (95%). Ship model locked: **CLIP ViT-B/32 int8 via ONNX Runtime**
  (image tower ~90 MB + text tower ~65 MB). Remaining M0 exit criterion: the
  **phone smoke test** (load/embed latency) — hook shipped in the M1 onboarding
  screen.
- **Distribution pivot (Sept 2026):** GitHub Pages landing page + direct APK
  download instead of Play Store (no dev account, no 12-tester gate). CI
  (`.github/workflows/android-release.yml`) builds a signed APK on every push
  and attaches it to GitHub Releases on `v*` tags. Self-managed signing
  keystore lives outside the repo. Landing page: `docs/index.html`.
- **Model compression:** the current int8 pair (~155 MB) is to be compressed
  further (int4/weight-only quantization, target ~70 MB total) and bundled in
  the APK for fully-offline install.
- **M1 skeleton scaffolded** under `android/`: Gradle (AGP 9.1.1 built-in
  Kotlin, Gradle 9.3.1, Compose BOM 2026.08.00, ORT Android 1.29.0),
  onboarding + smoke-test screen, permission-free Photo Picker + grid,
  `ClipEncoder` (ORT int8 image tower, CLIP preprocessing). Not yet compiled —
  the first build happens in CI.
- Other M0 candidates (MobileCLIP, DFN/SigLIP) were **not** benchmarked —
  accepted risk: B/32 int8 already meets the ≥80% accuracy bar at phone size.

---

## 1. The core architectural decision: on-device AI (locked)

**Why on-device is right for a public app** (rather than hosting the Python/Flask backend in the
cloud):

1. **Privacy is the product.** A public photo-search app that uploads every user's personal
   photos to a server is (a) a huge trust barrier — nobody installs "send me all your photos"
   — and (b) a compliance/liability burden (GDPR data-deletion requests, breach risk, security
   review). On-device means photos **never leave the phone** — the best marketing line and the
   easiest Play Store data-safety review.
2. **Cost scales to zero per user.** No servers to pay for as users grow.
3. **Works offline.** Search keeps working without a network.
4. **Fits the tech choice.** Native Kotlin is the best platform for on-device ML (ONNX Runtime)
   and photo access (Photo Picker / MediaStore).

**The tradeoffs, acknowledged up front:**
- The entire search stack is reimplemented for mobile (no Python/torch/FAISS on a phone).
- The desktop model (768-dim ViT-L-14, ~600 MB) is too big for phones → a smaller model is
  required, which costs some accuracy (see §3.5 for how we win it back).
- First-time indexing is slower on phone hardware than on a desktop.

**Risk gate (Milestone 0):** verify on-device search accuracy before building any UI. If quality
is unacceptable after trying the accuracy levers, revisit (bigger on-device model, or hybrid)
before investing in the app UI.

---

## 2. Target architecture

```
┌──────────────────────────── Android app (Kotlin + Jetpack Compose) ───────────────────────────┐
│                                                                                                │
│  Photo Picker / MediaStore ──► Indexing pipeline (WorkManager, background)                     │
│                                    │  CLIP image encoder (ONNX Runtime)                        │
│                                    ▼                                                          │
│                              local index file  (embeddings matrix + paths)                     │
│                                                                                                │
│  Search bar ──► text encoder (ONNX) ──► query expansion ──► dot product ──► ranked grid        │
│                                                                                                │
│  All data stays in app-private storage. No network calls.                                      │
│                                                                                                │
│  v2: "teach the app" — concept learning in embedding space (see §3.5)                          │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Codebase mapping (existing Python → Android)

| Current file | What it does | Android equivalent |
|---|---|---|
| `src/model.py` | CLIP ViT-L-14 via sentence-transformers | ONNX Runtime + best benchmarked small CLIP variant (M0 decides; see §3.5) |
| `src/indexer.py` | scan folder → embed every photo | MediaStore/Photo Picker scan + WorkManager indexing job |
| `src/storage.py` | save/load `.npz` matrix + FAISS index | local binary index file (float32 matrix + path list); swap in HNSW later if >100k photos |
| `src/search.py` | query expansion + reranking + dot product | same algorithm, ported to Kotlin (see §3) |
| `src/ui.py` + `src/templates/index.html` | web UI + search page | Jetpack Compose screens (index.html is the design reference) |
| `data/images/` | local photo folder | device gallery (user-selected) |

### Model delivery
- Download the ONNX model on first run (with progress UI) into app-private storage, or bundle
  via Play Asset Delivery — avoids a 100+ MB APK. Decide in M0/M1.

---

## 3. Porting the search logic (from `src/search.py`)

Keep the tricks that make the desktop app accurate:

1. **Query expansion** — embed the raw query plus the 8 template variants
   (`"a photo of {q}"`, `"a picture of {q}"`, ...), weighted 2:1 original:variants,
   and re-normalize (this is `_expand_query`).
2. **Candidate retrieval** — brute-force normalized dot product over the local matrix
   (fine up to ~100k photos; the matrix is a few hundred KB per 1k photos). Add an ANN index
   (HNSW) only if users exceed ~100k photos.
3. **Re-ranking** — blend original score 50/50 with the expanded-vector score (`_rerank`).
4. **Multi-crop augmentation** during indexing (`_augment_image`: 7 crops/flips, average,
   re-normalize) — slower but meaningfully more accurate; make it a setting (default on).

All of the above is simple linear algebra + a text encoder — directly portable to Kotlin.

---

## 3.5 Accuracy strategy — from "how much do we lose?" to "can we win?"

**Reality check:** switching from ViT-L-14 to a phone-sized model costs real accuracy — a few
points on average, more on subtle distinctions (exact breeds, nuanced compound queries),
barely anything on everyday searches (cats, cars, sunsets). int8 quantization costs ~0–1% if
tuned properly. That's the honest baseline.

**The on-device app does not have to settle for "slightly worse than ViT-L-14".** These levers
can push it past the desktop baseline:

**Free — pick the best small base model (benchmark in M0, don't assume ViT-B/32):**
- MobileCLIP (Apple, 2024) — ~3% better than ViT-B/32 at similar size/speed.
- DFN-CLIP ViT-B/32 (OpenAI, better-trained) and SigLIP (Google) — same size, better accuracy.
- M0 decision: benchmark 2–3 candidates on the project's photo set; pick the winner.

**Free — hybrid search:** blend CLIP scores with filename text, OCR inside photos (ML Kit,
on-device), and EXIF date/location. Catches what a visual model can't read.

**Free — better reranking:** cross-encoder rerank of the top-k results (small matching model).

**Training — distillation:** train a phone-sized student to mimic the desktop ViT-L-14 on a
large image corpus → big-model accuracy at small-model cost. The desktop app is the teacher.

**Training — domain adaptation:** LoRA fine-tune on everyday-photo data (CLIP was trained on
generic internet images, not personal photo libraries). A few MB of adapter weights ship with
the app.

**v2 — on-device personalization (the biggest per-user lever):** let users teach the app
concepts ("these 12 photos are Rex"). What actually works on a phone:

- **Full CLIP fine-tuning on-device is NOT practical** (multi-GB training memory, battery/heat,
  minutes–hours, and photos have no captions to train on anyway).
- **The right approach: learn in embedding space** on top of the frozen model — trivial compute:
  1. *Concept groups* (zero training): average the embeddings of user-tagged photos → a concept
     vector; search = nearest neighbor.
  2. *Textual inversion* (train one 512-dim token per concept): makes the *text* "Rex" land
     near Rex's photos. Seconds on-device.
  3. *Linear probe* (tiny 1-layer head on frozen embeddings): background-safe training on
     user tags.
  4. *LoRA adapters* (~2–4% of weights): feasible but heavy (1–2 GB during training, minutes
     per concept) — defer to v2+.
- The user's tags ARE the training labels — data no generic model has. This keeps the
  "photos never leave the phone" promise intact.
- **Privacy boundary:** using user data to improve the *global* model requires explicit opt-in;
  v1 ships with no data leaving the device, period.

**M0 scope update:** benchmark candidate small models on the photo set (PC-side first, then a
phone smoke test); run one distillation experiment if time permits. Choose the winner before
building the app UI.

**M0 judging criteria (reworked after the v1 dataset proved unusable):** the first test set
(flat-color shapes, Laplacian variance 60–316) was so unlike photos that even ViT-L-14 scored
~0.1 cosine on true captions — its rankings were noise, and "top-5 overlap with the baseline"
measured agreement-with-noise, not accuracy. The benchmark now scores **correctness against
`data/images/labels.json` ground truth** (a hit = the objectively-true image ranks top-3 for
its caption), gated by a **dataset sanity check** (ViT-L-14 truth cosine ≥ 0.17 and margin over
non-truth images ≥ 0.04, plus sharpness > 800). If the gate fails, fix the dataset — the model
comparison is meaningless.

---

## 4. App structure (Android project)

- **Stack:** Kotlin, Jetpack Compose, MVVM, Room (or plain files) for the index + paths,
  WorkManager for indexing, Coil for image thumbnails, ONNX Runtime Android.
- **Screens (v1):**
  1. **Onboarding / model download** — first-run progress bar for the ONNX file.
  2. **Photo selection** — Android Photo Picker (no broad permission on modern Android);
     fall back to `READ_MEDIA_IMAGES` (API 33+) / `READ_EXTERNAL_STORAGE` (legacy) only if
     users want to index their whole library. If the full-library path is used, Play requires
     declaring the permission and justifying it as core functionality.
  3. **Indexing progress** — background job, notifications, incremental re-index of new photos.
  4. **Search** — search bar + ranked photo grid (mirror the desktop UI).
  5. **Settings** — re-index, augmentation toggle, storage usage, privacy note ("all processing
     is on-device").
- **Screens (v2):** "Teach the app" — tag photos into concepts ("Rex"), see learned concepts,
  manage/delete them (embedding-space learning from §3.5).

---

## 5. Distribution: GitHub Pages + direct APK (decision Sept 2026 — replaces Play Store)

**Decision:** ship as a free sideloaded APK with a GitHub Pages landing page. No Play
developer account, no 12-tester/14-day gate, no store review. (Play can still be added
later for $25 if wanted — the app itself is unaffected by distribution channel.)

### How it works
- **CI builds the APK** on every push (`.github/workflows/android-release.yml`): JDK 17 +
  SDK 36 in Actions, signed with the repo's secret keystore. No local Android SDK needed.
- **Releases:** pushing a tag `v*` attaches the signed `gallery-assist.apk` to a GitHub
  Release → stable URL `github.com/OWNER/REPO/releases/latest/download/gallery-assist.apk`.
- **Landing page:** `docs/index.html` served via GitHub Pages (repo Settings → Pages →
  “Deploy from branch”, `/docs` root). Hero + privacy pitch + download button + install steps.
- **Signing:** self-managed PKCS12 keystore (kept OUT of the repo; base64 + password go
  into GitHub Secrets). Losing it only blocks *updates* to installed apps — keep a backup.

### Tradeoffs vs Play Store (accepted)
- Users see the one-time “unknown sources” warning on install (landing page explains it).
- No automatic updates — users re-download the APK to update (landing page notes it).
- No store listing/search discovery — traffic comes from sharing the landing page link.
- Crash reporting must be opt-in and external (or none in v1) — no Play Console vitals.

### Checklist
- [ ] GitHub repo Secrets: `SIGNING_KEYSTORE_BASE64`, `SIGNING_STORE_PASSWORD`,
      `SIGNING_KEY_ALIAS` (values are in the local keystore folder's credentials file).
- [ ] Enable GitHub Pages from `/docs`; replace OWNER/REPO placeholders in `docs/index.html`.
- [ ] Tag `v0.1.0` to cut the first release; verify the landing page download works.
- [ ] Optional: privacy policy page (the app has no network permission — easy to write).

---

## 6. Milestones

| # | Milestone | Deliverable | Exit criteria |
|---|---|---|---|
| **M0** | **Model validation (the risk gate)** | PC-side first: export 2–3 candidate small models (ViT-B/32, MobileCLIP, DFN/SigLIP) to ONNX + int8; compare search accuracy vs desktop ViT-L-14 on the same photo set. Then a minimal Android smoke test (speed/memory on a real phone). Optional: one distillation experiment. | Model winner chosen; on-phone search accuracy ≈ desktop (or documented delta + compensation plan); model ≤ ~150 MB; indexing speed acceptable |
| **M1** | App skeleton | Compose app: onboarding, photo picker, search screen UI | App installs on a test phone, picks photos, shows a grid |
| **M2** | Indexing pipeline | WorkManager job, index file format, incremental re-index | Full library indexed in background with progress UI; survives app restarts |
| **M3** | Search | Port of expansion + reranking from `search.py`; ranked results grid | Text search returns good matches; latency feels instant |
| **M4** | Polish | Thumbnail cache, empty/error states, settings, privacy screen | Internal-testing-ready build |
| **M5** | Play Store readiness | Dev account, signing, privacy policy, data-safety, listing assets | App passes internal testing |
| **M6** | Closed testing & launch | 12 testers × 14 days, staged production rollout | App live on Play Store |
| **M7** | v2 — teachable concepts | "Teach the app" feature: concept groups + textual inversion in embedding space | Users can tag a concept, search by its name, manage/delete concepts |

---

## 7. Open questions (decide as we go)

- **Whole library vs. selected folders?** Photo Picker is permission-free but manual;
  full-library indexing needs `READ_MEDIA_IMAGES` + Play justification. v1 could do
  "pick folders/photos", add full-library later.
- **Model choice** — settled by M0 benchmark (candidates: ViT-B/32, MobileCLIP, DFN-CLIP,
  SigLIP; distillation as a follow-up).
- **Hybrid search in v1?** OCR + filename + EXIF scoring is a strong accuracy booster; decide
  whether it ships in v1 (M3–M4) or v2.
- **App name & branding** — affects listing assets and the privacy policy URL.
- **Do we keep the desktop app in sync?** The Python app stays as the development/test
  harness; the search math is shared conceptually, not in code.
- **Reach for help with the 12 testers?** The 14-day gate can't be skipped; plan the
  recruiting early.

---

## 8. What happens next

1. ~~M0 PC-side~~ **DONE** (see Status update): B/32 int8 locked. Remaining:
   phone smoke test once an APK is installed on a real phone.
2. ~~Build locally~~ **CI builds instead** (no local Android SDK needed): add the
   signing secrets, push, let `.github/workflows/android-release.yml` produce the APK,
   install it on a phone, run the in-app smoke test.
3. Compress the model further (int4/weight-only → target ~70 MB total) and ship it
   **bundled in the APK** so install works fully offline (landing page updated then).
4. Enable GitHub Pages + cut `v0.1.0` (see §5 checklist).
3. Then follow M1 → M6, with M7 (teachable concepts) after launch.