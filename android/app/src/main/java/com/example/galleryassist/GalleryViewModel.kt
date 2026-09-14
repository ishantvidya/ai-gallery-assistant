package com.example.galleryassist

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.galleryassist.data.PhotoMetadata
import com.example.galleryassist.data.PhotoRepository
import com.example.galleryassist.data.matchingMetadata
import com.example.galleryassist.diag.Diagnostics
import com.example.galleryassist.index.GalleryIndexer
import com.example.galleryassist.index.IndexingProgress
import com.example.galleryassist.ml.ClipOnnxEngine
import com.example.galleryassist.permissions.PhotoAccess
import com.example.galleryassist.permissions.PhotoPermissions
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Overall app stage shown by MainActivity's router. */
sealed interface AppStage {
    /** No photo permission (or revoked) → setup/rationale screen. */
    data object NeedAccess : AppStage

    /** Photos discovered; grid + search usable (embedding may still run). */
    data class Ready(
        val photos: List<PhotoMetadata>,
        /** Bumped on every checkpoint so the UI recomposes as embeddings
         *  fill in (StateFlow dedupes content-equal values). */
        val version: Int = 0,
    ) : AppStage

    /** First-run metadata scan in progress (seconds). */
    data object Indexing : AppStage
}

/**
 * Owns the permission → scan → search lifecycle, plus the CLIP engine for
 * indexing (image tower) and search (text tower).
 *
 * Design (v0.2.1): the app is usable IMMEDIATELY after a fast metadata scan.
 * CLIP embedding then runs as a background job with checkpoints persisted
 * every [GalleryIndexer.CHECKPOINT_EVERY] photos, so closing the app loses at
 * most one checkpoint's work — the next launch RESUMES from the saved index.
 *
 * Re-scan triggers: first grant, scope change (FULL ⇄ PARTIAL / revoke →
 * re-grant), and every app open (picks up new/removed photos; saved
 * embeddings are carried over by photo id, so completed work is never redone).
 *
 * DIAGNOSTIC BUILD (v0.2.2): every step above is recorded into
 * [Diagnostics] (rendered by the on-screen panel), and failures NEVER drop
 * the app back to NeedAccess — we always land in Ready so the panel with the
 * recorded root cause stays visible.
 */
class GalleryViewModel(app: Application) : AndroidViewModel(app) {

    private val repository = PhotoRepository(app)
    private val indexer = GalleryIndexer(repository)

    private val _stage = MutableStateFlow<AppStage>(AppStage.NeedAccess)
    val stage: StateFlow<AppStage> = _stage

    val indexingProgress: StateFlow<IndexingProgress> = indexer.progress

    /** DIAGNOSTIC BUILD: live event log, shown by the UI's diag panel. */
    val diag: StateFlow<List<Diagnostics.Event>> = Diagnostics.events

    /** Latest photo list (for the UI's ranking calls). */
    private var currentPhotos: List<PhotoMetadata> = emptyList()

    /** Bumped whenever a new Ready generation starts; guards stale updates. */
    private var readyGeneration = 0

    /** Shared engine for background embedding + query encoding. */
    private var engine: ClipOnnxEngine? = null
    private val engineMutex = Mutex()

    private var engineFailed = false
    private var indexJob: Job? = null
    private var queryJob: Job? = null

    /** Throttle for rank() logging — queries fire per keystroke. */
    private var lastRankLogMs = 0L

    /** Called by the UI after the user grants (or changes) photo access. */
    fun onPermissionChanged() {
        val access = PhotoPermissions.currentAccess(getApplication())
        Diagnostics.log("permission: $access")
        if (access == PhotoAccess.NONE) {
            indexJob?.cancel()
            queryJob?.cancel()
            closeEngine()
            _stage.value = AppStage.NeedAccess
            return
        }
        if (_stage.value is AppStage.Indexing) return // scan already running

        indexJob?.cancel()
        startLifecycle(access)
    }

    /**
     * Fast path: scan → save → Ready, then embed what's missing in the
     * background (resuming any interrupted session from the saved index).
     */
    private fun startLifecycle(access: PhotoAccess) {
        _stage.value = AppStage.Indexing

        var scanned: List<PhotoMetadata> = emptyList()
        indexJob = viewModelScope.launch {
            try {
                // 0. Load the saved index off the main thread (it can be tens
                //    of MB once embeddings are persisted).
                val saved = withContext(Dispatchers.IO) {
                    repository.loadIndex { detail ->
                        Diagnostics.log("saved index: PARSE FAILED — $detail")
                    }
                }
                val savedById = saved
                    ?.takeIf { it.accessScope == access.name }
                    ?.photos
                    ?.associateBy { it.id }
                    ?: emptyMap()
                Diagnostics.log(
                    "saved index: " + when {
                        saved == null ->
                            "none (first run)"
                        saved.accessScope != access.name ->
                            "${saved.photos.size} photos ignored (scope ${saved.accessScope} ≠ $access)"
                        else -> {
                            val n = saved.photos.count { it.embedding != null }
                            "${saved.photos.size} photos, $n with embeddings"
                        }
                    },
                )

                // 1. Metadata scan (seconds) — fresh objects, embeddings null.
                scanned = indexer.scan()
                Diagnostics.log("scan: ${scanned.size} photos found")

                // 2. Carry saved embeddings onto the fresh scan by photo id.
                val merged = scanned.map { fresh ->
                    savedById[fresh.id]?.embedding?.let { saved ->
                        fresh.embedding = saved
                    }
                    fresh
                }
                val ready = merged.count { it.embedding != null }
                Diagnostics.log(
                    "merge: ${merged.size} photos, $ready embeddings carried over, " +
                        "${merged.size - ready} to embed",
                )

                // 3. Persist immediately — from here on the app survives
                //    being killed without losing the scan.
                repository.saveIndex(indexFile(merged, access.name))
                Diagnostics.log("index: scan checkpoint saved")

                // 4. Usable NOW.
                enterReady(merged)

                // 5. Embed what's missing in the background, checkpointing:
                //    persist partial progress AND refresh the Ready state so
                //    the UI sees embeddings fill in live.
                if (merged.any { it.embedding == null }) {
                    Diagnostics.log("embed: starting background pass")
                    val active = acquireEngine()
                    if (active == null) {
                        Diagnostics.log("embed: SKIPPED — engine unavailable (see engine line above)")
                    } else {
                        indexer.embedMissing(merged, active) { checkpoint ->
                            val done = checkpoint.count { it.embedding != null }
                            repository.saveIndex(indexFile(checkpoint, access.name))
                            Diagnostics.log("checkpoint: $done/${merged.size} embedded — saved")
                            enterReady(checkpoint)
                        }
                        // Final persist (covers the tail after the last checkpoint).
                        repository.saveIndex(indexFile(merged, access.name))
                        val done = merged.count { it.embedding != null }
                        Diagnostics.log("embed: finished — $done/${merged.size} embedded (saved)")
                        enterReady(merged)
                    }
                } else {
                    Diagnostics.log("embed: nothing to do — all photos embedded")
                }
            } catch (e: CancellationException) {
                throw e // permission revoked / VM cleared mid-run
            } catch (e: Exception) {
                Diagnostics.failure("indexing", e)
                Log.e(TAG, "indexing failed", e)
                // DIAGNOSTIC BUILD: never drop to NeedAccess (that hides the
                // diag panel and looks like a permission problem). Land in
                // Ready with whatever we have — worst case an empty grid
                // plus the recorded failure.
                enterReady(scanned)
            }
        }
    }

    private fun indexFile(photos: List<PhotoMetadata>, scope: String) =
        PhotoRepository.IndexFile(
            lastIndexedEpochMs = System.currentTimeMillis(),
            photos = photos,
            accessScope = scope,
        )

    private fun enterReady(photos: List<PhotoMetadata>) {
        currentPhotos = photos
        readyGeneration++
        _stage.value = AppStage.Ready(photos, readyGeneration)
    }

    /**
     * Ranks [photos] for [query]: photos with embeddings are ordered by cosine
     * similarity to the query (on-device text tower); metadata matches without
     * embeddings are appended after (and serve as the only results while
     * embedding is still running). Metadata matches rank first when the query
     * looks like a filename/album/date anyway.
     */
    fun rank(query: String, photos: List<PhotoMetadata>, onResult: (List<PhotoMetadata>) -> Unit) {
        queryJob?.cancel()
        val trimmed = query.trim()
        if (trimmed.isEmpty()) {
            onResult(photos)
            return
        }
        val metadataMatches = photos.matchingMetadata(trimmed)
        val embedded = photos.filter { it.embedding != null }
        if (embedded.isEmpty()) {
            logRankThrottled(
                "rank \"$trimmed\": METADATA-ONLY — 0 of ${photos.size} photos embedded, " +
                    "${metadataMatches.size} metadata matches",
            )
            onResult(metadataMatches)
            return
        }
        queryJob = viewModelScope.launch(Dispatchers.Default) {
            val active = acquireEngine()
            if (active == null) {
                logRankThrottled(
                    "rank \"$trimmed\": METADATA-ONLY — engine unavailable, " +
                        "${metadataMatches.size} metadata matches",
                )
                onResult(metadataMatches)
                return@launch
            }
            val results = try {
                val qvec = active.encodeText(trimmed)
                val scored = embedded.map { p -> p to active.dot(qvec, p.embedding!!) }
                val ranked = scored.sortedByDescending { it.second }.map { it.first }
                val rankedIds = ranked.mapTo(HashSet()) { it.id }
                val appended = metadataMatches.filter { it.id !in rankedIds }
                logRankThrottled(
                    "rank \"$trimmed\": SEMANTIC over ${embedded.size} embedded " +
                        "(top score ${scored.maxOfOrNull { it.second }}, " +
                        "${appended.size} metadata appended)",
                )
                ranked + appended
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Diagnostics.failure("rank \"$trimmed\"", e)
                Log.w(TAG, "semantic ranking failed; metadata fallback", e)
                metadataMatches
            }
            onResult(results)
        }
    }

    /** Loads the engine once; later callers share the same instance. */
    private suspend fun acquireEngine(): ClipOnnxEngine? = engineMutex.withLock {
        engine?.let { return it }
        if (engineFailed) return null
        Diagnostics.log("engine: loading CLIP (extract ~155 MB assets + 2 sessions)…")
        val t0 = System.currentTimeMillis()
        withContext(Dispatchers.IO) {
            runCatching { ClipOnnxEngine.fromContext(getApplication()) }
                .onSuccess {
                    engine = it
                    Diagnostics.log(
                        "engine: OK in ${System.currentTimeMillis() - t0} ms (dim ${it.embedDim})",
                    )
                }
                .onFailure {
                    engineFailed = true
                    Diagnostics.failure("engine load", it)
                    Log.w(TAG, "CLIP engine unavailable", it)
                }
                .getOrNull()
        }
    }

    /** Latest indexed photo list, for the UI's ranking calls. */
    fun currentPhotosSnapshot(): List<PhotoMetadata> = currentPhotos

    private fun logRankThrottled(line: String) {
        val now = System.currentTimeMillis()
        if (now - lastRankLogMs >= RANK_LOG_INTERVAL_MS) {
            lastRankLogMs = now
            Diagnostics.log(line)
        }
    }

    private fun closeEngine() {
        engine?.close()
        engine = null
    }

    override fun onCleared() {
        indexJob?.cancel()
        queryJob?.cancel()
        closeEngine()
        super.onCleared()
    }

    private companion object {
        const val TAG = "GalleryViewModel"

        const val RANK_LOG_INTERVAL_MS = 1_500L
    }
}
