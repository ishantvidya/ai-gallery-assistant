package com.example.galleryassist

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.galleryassist.data.PhotoRepository
import com.example.galleryassist.index.GalleryIndexer
import com.example.galleryassist.index.IndexingProgress
import com.example.galleryassist.ml.ClipOnnxEngine
import com.example.galleryassist.permissions.PhotoAccess
import com.example.galleryassist.permissions.PhotoPermissions
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/** Overall app state shown by MainActivity's router. */
sealed interface AppStage {
    /** No photo permission (or revoked) → setup/rationale screen. */
    data object NeedAccess : AppStage

    /** Permission granted; photos discovered; ready to search. */
    data class Ready(val photos: List<com.example.galleryassist.data.PhotoMetadata>) : AppStage

    /** Indexing in progress. */
    data object Indexing : AppStage
}

/**
 * Owns the permission → index → search lifecycle.
 *
 * Re-index triggers:
 *  - first grant,
 *  - app restart with no saved index,
 *  - access-scope change (FULL ⇄ PARTIAL, or revoked → re-grant) detected by
 *    comparing the saved index's scope with the current one.
 */
class GalleryViewModel(app: Application) : AndroidViewModel(app) {

    private val repository = PhotoRepository(app)
    private val indexer = GalleryIndexer(repository)

    private val _stage = MutableStateFlow<AppStage>(AppStage.NeedAccess)
    val stage: StateFlow<AppStage> = _stage

    val indexingProgress: StateFlow<IndexingProgress> = indexer.progress

    private var indexJob: Job? = null

    /** Called by the UI after the user grants (or changes) photo access. */
    fun onPermissionChanged() {
        val access = PhotoPermissions.currentAccess(getApplication())
        if (access == PhotoAccess.NONE) {
            indexJob?.cancel()
            _stage.value = AppStage.NeedAccess
            return
        }
        if (_stage.value is AppStage.Indexing) return // already running

        val saved = repository.loadIndex()
        val scopeChanged = saved != null && saved.accessScope != access.name
        val needsIndex = saved == null || scopeChanged

        if (needsIndex) {
            startIndexing(access)
        } else {
            _stage.value = AppStage.Ready(saved!!.photos)
        }
        // TODO(M2): also re-index when MediaStore content changes (ContentObserver).
    }

    private fun startIndexing(access: PhotoAccess) {
        _stage.value = AppStage.Indexing
        indexJob = viewModelScope.launch {
            val engine = runCatching { ClipOnnxEngine.fromFiles(getApplication()) }.getOrNull()
            try {
                val photos = indexer.indexAll(engine)
                repository.saveIndex(
                    PhotoRepository.IndexFile(
                        lastIndexedEpochMs = System.currentTimeMillis(),
                        photos = photos,
                        accessScope = access.name,
                    ),
                )
                _stage.value = AppStage.Ready(photos)
            } catch (e: kotlinx.coroutines.CancellationException) {
                // Indexing cancelled (permission revoked mid-run / VM cleared):
                // rethrow so structured concurrency stays intact.
                throw e
            } catch (e: Exception) {
                _stage.value = AppStage.NeedAccess
            } finally {
                engine?.close()
            }
        }
    }

}
