# AI Gallery Assistant — Android app

Native Kotlin + Jetpack Compose port of the desktop photo-text-search app.
All AI runs on-device (M0 decision: **CLIP ViT-B/32 int8 via ONNX Runtime**).

## Photo access architecture (how gallery access works)

- **Permissions (modern, least-privilege)** — `permissions/PhotoPermissions.kt`:
  - API 33+: `READ_MEDIA_IMAGES`; the system may grant **partial access**
    ("Select photos") via `READ_MEDIA_VISUAL_USER_SELECTED` — MediaStore then
    returns only the chosen items, and the app treats that as the library.
  - API ≤ 32: `READ_EXTERNAL_STORAGE` (manifest `maxSdkVersion="32"`).
  - Nothing else: no location/contacts/mic/file-system permissions.
- **Discovery** — `data/PhotoRepository.queryPhotos()`: a projection-limited
  MediaStore query (URI, display name, date taken/modified, bucket, size).
  No file paths, no file copies — images are always read through their
  `content://` URI.
- **Indexing** — `index/GalleryIndexer.kt`: scan → persist metadata JSON →
  optional embedding pass through the swappable `ml/EmbeddingEngine` interface
  (`ClipOnnxEngine` today; int4/newer models slot in without pipeline changes).
- **Lifecycle** — `GalleryViewModel`: grant → index → search; revoked or
  scope-changed permission re-indexes automatically; denial never crashes
  (the setup screen is shown instead).
- **Screens** — `PhotoAccessScreen` (rationale + "Allow Photo Access" +
  denied/partial states), `IndexingScreen` (progress + photo count),
  `SearchScreen` (indexed grid; ranking lands in M3).

## Building — via GitHub Actions (no local Android SDK needed)

`.github/workflows/android-release.yml` builds a signed release APK on every
push to `main` (artifact on the run) and attaches `gallery-assist.apk` to a
GitHub Release for every `v*` tag:

```
https://github.com/OWNER/REPO/releases/latest/download/gallery-assist.apk
```

One-time setup: repo secrets `SIGNING_KEYSTORE_BASE64`, `SIGNING_STORE_PASSWORD`,
`SIGNING_KEY_ALIAS` (values in the local keystore folder's credentials file);
enable Pages from `/docs`; replace `OWNER/REPO` in `docs/index.html`.
Without secrets, CI produces a debug-signed APK (fine for testing).

## Building locally (optional — needs JDK 17 + Android SDK 36)

```bash
cd android
gradle :app:assembleDebug
```

## Testing the permission flow on a real phone

1. Install the debug APK (`adb install app/build/outputs/apk/debug/*.apk` or
   the CI artifact) and launch → the setup screen explains photo access.
2. **Grant** → indexing runs (progress + count) → the gallery grid appears.
3. **Deny** (and "Don't ask again") → the screen stays on setup with an
   "Open system settings" path; the app must not crash.
4. **Partial access** (Android 14+): choose "Select photos" in the system
   dialog → the app indexes only the selected photos and says so.
5. **Revoke later**: system Settings → Apps → Permissions → Photos and videos
   → remove. Returning to the app (ON_RESUME) routes back to setup; re-grant
   triggers a fresh index. Kill + relaunch also re-checks (saved index state).
6. **Scope change**: switch full ↔ selected in system settings → next resume
   re-indexes with the new scope automatically.

## Models (bundled in the APK since v0.2.0)

Both int8 towers (image + text, ~155 MB) ship inside the APK under
`android/app/src/main/assets/`, together with the CLIP tokenizer
(`assets/tokenizer/clip-vocab.json`, `clip-merges.txt`). On first run they
are copied into app-private storage (`filesDir/`) — ONNX Runtime needs a
real file path — and both sessions load from there.

After an app update that changes the bundled models, bump the APK size
check: `ClipEncoder.extractAsset` re-extracts when the stored file size
differs from the asset.

`scripts/check_tokenizer_port.py` is the tokenizer parity guard: the Kotlin
`ClipTokenizer` must reproduce the reference ids for its golden texts
(run it whenever the export/tokenizer stack changes). Indexing still works
if the engine can't load — search then falls back to metadata matching.
`ml/SmokeTest.kt` times session load + warm image/text embed for a future
Settings screen.

## Project layout

```
.github/workflows/android-release.yml  # CI: signed APK on push, Release on v* tags
docs/index.html                        # GitHub Pages landing page + download link
android/
├── settings.gradle.kts / build.gradle.kts / gradle.properties
├── gradle/libs.versions.toml          # AGP 9.1.1 (built-in Kotlin), Compose BOM 2026.08.00, ORT 1.29.0
└── app/src/main/
    ├── AndroidManifest.xml            # granular photo permissions only
    └── java/com/example/galleryassist/
        ├── MainActivity.kt            # stage router (setup ⇄ indexing ⇄ search)
        ├── GalleryViewModel.kt        # permission→index→search lifecycle
        ├── data/PhotoMetadata.kt      # persisted per-photo metadata record
        ├── data/PhotoRepository.kt    # MediaStore query + JSON index store
        ├── permissions/PhotoPermissions.kt  # FULL/PARTIAL/NONE, version-aware
        ├── index/GalleryIndexer.kt    # scan + embed pipeline with progress
        └── ml/                        # EmbeddingEngine seam, ClipOnnxEngine, ClipEncoder, SmokeTest
```

## What's next (per ANDROID_PLAN.md)

- **M2:** incremental re-index (ContentObserver), embedding matrix persisted
  next to the metadata (port of `src/storage.py`).
- **M3:** text tower + Kotlin BPE tokenizer, query expansion + reranking
  (port of `src/search.py`) — real natural-language ranking in SearchScreen.
- **Model:** int4/weight-only compression (target ~70 MB) bundled in the APK.
