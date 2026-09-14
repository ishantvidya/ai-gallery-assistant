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
import com.example.galleryassist.ml.EmbeddingEngine
import com.example.galleryassist.permissions.PhotoAccess
import com.example.galleryassist.permissions.PhotoPermissions
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Overall app state shown by MainActivity's router. */
sealed interface AppStage {
    /** No photo permission (or revoked) → setup/rationale screen. */
    data object NeedAccess : AppStage

    /** Permission granted; photos discovered; ready to search. */
    data class Ready(val photos: List<PhotoMetadata>) : AppStage

    /** Indexing in progress. */
    data object Indexing : AppStage
}

/**
 * Owns the permission → index → search lifecycle, and the CLIP engine used
 * for both indexing (image tower) and search (text tower).
 *
 * Re-index triggers:
 *  - first grant,
 *  - app restart with no saved index,
 *  - access-scope change (FULL ⇄ PARTIAL, or revoked → re-grant) detected by
 *    comparing the saved index's scope with the current one,
 *  - saved index lacking embeddings (older build) → backfill pass only.
 */
class GalleryViewModel(app: Application) : AndroidViewModel(app) {

    private val repository = PhotoRepository(app)
    private val indexer = GalleryIndexer(repository)

    private val _stage = MutableStateFlow<AppStage>(AppStage.NeedAccess)
    val stage: StateFlow<AppStage> = _stage

    val indexingProgress: StateFlow<IndexingProgress> = indexer.progress

    /** Current [PhotoMetadata] list (kept for ranking on query changes). */
    private var currentPhotos: List<PhotoMetadata> = emptyList()

    /**
     * The CLIP engine is kept alive while in [AppStage.Ready] so text queries
     * can be encoded without paying the ~1-2 s session load per keystroke.
     * Closed when leaving Ready (permission revoked, VM cleared).
     */
    private var engine: ClipOnnxEngine? = null
    private var queryJob: Job? = null

    /** Called by the UI after the user grants (or changes) photo access. */
    fun onPermissionChanged() {
        val access = PhotoPermissions.currentAccess(getApplication())
        if (access == PhotoAccess.NONE) {
            indexJob?.cancel()
            closeEngine()
            _stage.value = AppStage.NeedAccess
            return
        }
        if (_stage.value is AppStage.Indexing) return // already running

        val saved = repository.loadIndex()
        val scopeChanged = saved != null && saved.accessScope != access.name
        val hasEmbeddings = saved?.photos?.isNotEmpty() == true &&
            saved.photos.all { it.embedding != null }
        val needsIndex = saved == null || scopeChanged

        when {
            needsIndex -> startIndexing(access)
            !hasEmbeddings -> backfillEmbeddings(access, saved!!.photos)
            else -> enterReady(saved!!.photos)
        }
        // TODO(M2): also re-index when MediaStore content changes (ContentObserver).
    }

    /**
     * Ranks photos for [query] against their stored CLIP embeddings
     * (dot product == cosine; both towers emit unit vectors), descending.
     * Falls back to metadata matching when embeddings aren't available yet,
     * and degrades gracefully to metadata if the engine can't be loaded.
     */
    fun rank(query: String, photos: List<PhotoMetadata>, onResult: (List<PhotoMetadata>) -> Unit) {
        queryJob?.cancel()
        val trimmed = query.trim()
        if (trimmed.isEmpty()) {
            onResult(photos)
            return
        }
        val embedded = photos.filter { it.embedding != null }
        if (embedded.isEmpty()) {
            onResult(photos.matchingMetadata(trimmed))
            return
        }
        queryJob = viewModelScope.launch(Dispatchers.Default) {
            val active = engine ?: createEngine()
            val results = if (active == null) {
                photos.matchingMetadata(trimmed)
            } else {
                withContext(Dispatchers.Default) {
                    try {
                        val qvec = active.encodeText(trimmed)
                        embedded
                            .map { p -> p to active.dot(qvec, p.embedding!!) }
                            .sortedByDescending { it.second }
                            .map { it.first }
                    } catch (e: CancellationException) {
                        throw e
                    } catch (e: Exception) {
                        Log.w(TAG, "semantic ranking failed; falling back to metadata", e)
                        photos.matchingMetadata(trimmed)
                    }
                }
            }
            onResult(results)
        }
    }

    private suspend fun createEngine(): ClipOnnxEngine? =
        withContext(Dispatchers.IO) {
            runCatching { ClipOnnxEngine.fromContext(getApplication()) }
                .onSuccess { engine = it }
                .onFailure { Log.w(TAG, "CLIP engine unavailable: ${it.message}") }
                .getOrNull()
        }

    private fun enterReady(photos: List<PhotoMetadata>) {
        currentPhotos = photos
        _stage.value = AppStage.Ready(photos)
    }

    /** Latest indexed photo list, for the UI's ranking calls. */
    fun currentPhotosSnapshot(): List<PhotoMetadata> = currentPhotos

    private fun startIndexing(access: PhotoAccess) {
        _stage.value = AppStage.Indexing
        indexJob = viewModelScope.launch {
            val active = createEngine()
            try {
                val photos = indexer.indexAll(active)
                repository.saveIndex(
                    PhotoRepository.IndexFile(
                        lastIndexedEpochMs = System.currentTimeMillis(),
                        photos = photos,
                        accessScope = access.name,
                    ),
                )
                enterReady(photos)
            } catch (e: CancellationException) {
                // Indexing cancelled (permission revoked mid-run / VM cleared):
                // rethrow so structured concurrency stays intact.
                throw e
            } catch (e: Exception) {
                _stage.value = AppStage.NeedAccess
            } finally {
                if (active != null && _stage.value !is AppStage.Ready) active.close()
            }
        }
    }

    /**
     * Older index (v0.1.x) has metadata but no embeddings: scan the same
     * photo list, embed what's still visible, keep everything else.
     */
    private fun backfillEmbeddings(access: PhotoAccess, saved: List<PhotoMetadata>) {
        _stage.value = AppStage.Indexing
        indexJob = viewModelScope.launch {
            val active = createEngine()
            try {
                if (active == null) {
                    enterReady(saved)
                    return@launch
                }
                val byId = saved.associateBy { it.id }
                val fresh = indexer.indexAll(active).map { hit ->
                    // Newly embedded photos win; photos that dropped out of
                    // MediaStore since last time keep their old record.
                    if (hit.embedding != null) hit else byId[hit.id] ?: hit
                }
                repository.saveIndex(
                    PhotoRepository.IndexFile(
                        lastIndexedEpochMs = System.currentTimeMillis(),
                        photos = fresh,
                        accessScope = access.name,
                    ),
                )
                enterReady(fresh)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _stage.value = AppStage.Ready(saved)
            }
        }
    }

    private fun closeEngine() {
        engine?.close()
        engine = null
    }

    override fun onCleared() {
        queryJob?.cancel()
        closeEngine()
        super.onCleared()
    }

    private companion object {
        const val TAG = "GalleryViewModel"
    }
}
