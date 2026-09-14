package com.example.galleryassist

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.galleryassist.data.PhotoMetadata
import com.example.galleryassist.data.PhotoRepository
import com.example.galleryassist.data.matchingMetadata
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
 */
class GalleryViewModel(app: Application) : AndroidViewModel(app) {

    private val repository = PhotoRepository(app)
    private val indexer = GalleryIndexer(repository)

    private val _stage = MutableStateFlow<AppStage>(AppStage.NeedAccess)
    val stage: StateFlow<AppStage> = _stage

    val indexingProgress: StateFlow<IndexingProgress> = indexer.progress

    /** Latest photo list (for the UI's ranking calls). */
    private var currentPhotos: List<PhotoMetadata> = emptyList()

    /** Bumped whenever a new Ready generation starts; guards stale updates. */
    private var readyGeneration = 0

    /** Shared engine for background embedding + query encoding. */
    private var engine: ClipOnnxEngine? = null
    private val engineMutex = Mutex()

    private var indexJob: Job? = null
    private var queryJob: Job? = null

    /** Called by the UI after the user grants (or changes) photo access. */
    fun onPermissionChanged() {
        val access = PhotoPermissions.currentAccess(getApplication())
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
        val savedById = repository.loadIndex()
            ?.takeIf { it.accessScope == access.name }
            ?.photos
            ?.associateBy { it.id }
            ?: emptyMap()

        indexJob = viewModelScope.launch {
            try {
                // 1. Metadata scan (seconds) — fresh objects, embeddings null.
                val scanned = indexer.scan()

                // 2. Carry saved embeddings onto the fresh scan by photo id.
                val merged = scanned.map { fresh ->
                    savedById[fresh.id]?.embedding?.let { saved ->
                        fresh.embedding = saved
                    }
                    fresh
                }

                // 3. Persist immediately — from here on the app survives
                //    being killed without losing the scan.
                repository.saveIndex(indexFile(merged, access.name))

                // 4. Usable NOW.
                enterReady(merged)

                // 5. Embed what's missing in the background, checkpointing:
                //    persist partial progress AND refresh the Ready state so
                //    the UI sees embeddings fill in live.
                if (merged.any { it.embedding == null }) {
                    val active = acquireEngine() ?: return@launch
                    indexer.embedMissing(merged, active) { checkpoint ->
                        repository.saveIndex(indexFile(checkpoint, access.name))
                        enterReady(checkpoint)
                    }
                    // Final persist (covers the tail after the last checkpoint).
                    repository.saveIndex(indexFile(merged, access.name))
                    enterReady(merged)
                }
            } catch (e: CancellationException) {
                throw e // permission revoked / VM cleared mid-run
            } catch (e: Exception) {
                Log.e(TAG, "indexing failed", e)
                _stage.value = AppStage.NeedAccess
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
            onResult(metadataMatches)
            return
        }
        queryJob = viewModelScope.launch(Dispatchers.Default) {
            val active = acquireEngine()
            val results = if (active == null) {
                metadataMatches
            } else {
                try {
                    val qvec = active.encodeText(trimmed)
                    val ranked = embedded
                        .map { p -> p to active.dot(qvec, p.embedding!!) }
                        .sortedByDescending { it.second }
                        .map { it.first }
                    val rankedIds = ranked.mapTo(HashSet()) { it.id }
                    ranked + metadataMatches.filter { it.id !in rankedIds }
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.w(TAG, "semantic ranking failed; metadata fallback", e)
                    metadataMatches
                }
            }
            onResult(results)
        }
    }

    /** Loads the engine once; later callers share the same instance. */
    private suspend fun acquireEngine(): ClipOnnxEngine? = engineMutex.withLock {
        engine?.let { return it }
        if (engineFailed) return null
        withContext(Dispatchers.IO) {
            runCatching { ClipOnnxEngine.fromContext(getApplication()) }
                .onSuccess { engine = it }
                .onFailure {
                    engineFailed = true
                    Log.w(TAG, "CLIP engine unavailable: ${it.message}")
                }
                .getOrNull()
        }
    }

    private var engineFailed = false

    /** Latest indexed photo list, for the UI's ranking calls. */
    fun currentPhotosSnapshot(): List<PhotoMetadata> = currentPhotos

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
    }
}
