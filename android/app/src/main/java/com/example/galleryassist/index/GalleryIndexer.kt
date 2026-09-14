package com.example.galleryassist.index

import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import com.example.galleryassist.data.PhotoMetadata
import com.example.galleryassist.data.PhotoRepository
import com.example.galleryassist.ml.EmbeddingEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

/** User-visible indexing progress. */
data class IndexingProgress(
    val phase: Phase = Phase.SCANNING,
    val processed: Int = 0,
    val total: Int = 0,
) {
    enum class Phase { SCANNING, EMBEDDING, DONE }

    val photoCount: Int get() = total
}

/**
 * Builds the searchable index: MediaStore scan → metadata persisted →
 * (optional) embedding pass.
 *
 * Deliberately modular per the plan: the embed pass depends only on
 * [EmbeddingEngine]; if no model is available yet (M1 state), indexing still
 * completes and persists metadata, so M2/M3 work slots in cleanly.
 */
class GalleryIndexer(
    private val repository: PhotoRepository,
) {

    private val _progress = MutableStateFlow(IndexingProgress())
    val progress: StateFlow<IndexingProgress> = _progress

    private val cancelled = AtomicBoolean(false)
    private val processed = AtomicInteger(0)

    /**
     * Runs a full scan (+ embed pass when [engine] != null) off the main thread.
     * Returns the discovered metadata; safe to call from a coroutine only.
     */
    suspend fun indexAll(engine: EmbeddingEngine?): List<PhotoMetadata> = withContext(Dispatchers.IO) {
        cancelled.set(false)
        processed.set(0)
        _progress.value = IndexingProgress(phase = IndexingProgress.Phase.SCANNING)

        // 1. Discover what MediaStore exposes under the current permission scope.
        val photos = repository.queryPhotos()
        _progress.value = _progress.value.copy(
            phase = IndexingProgress.Phase.EMBEDDING,
            total = photos.size,
        )

        // 2. (Persistence is the caller's job: it knows the permission scope.)

        // 3. Embedding pass — skipped entirely when no engine is provided.
        //    Results are written back into each PhotoMetadata.embedding so the
        //    caller persists them with the metadata in one file.
        if (engine != null) {
            for (photo in photos) {
                currentCoroutineContext().ensureActive() // cooperative cancel on revoke/exit
                if (cancelled.get()) break
                val bmp: Bitmap? = repository.decodeThumbnail(Uri.parse(photo.contentUri))
                if (bmp != null) {
                    runCatching { engine.embedImage(bmp) }
                        .onSuccess { photo.embedding = it }
                        .onFailure { Log.w(TAG, "embed failed for ${photo.id}", it) }
                    bmp.recycle()
                }
                val n = processed.incrementAndGet()
                _progress.value = _progress.value.copy(processed = n)
            }
        }

        _progress.value = _progress.value.copy(phase = IndexingProgress.Phase.DONE)
        photos
    }

    fun cancel() {
        cancelled.set(true)
    }

    private companion object {
        const val TAG = "GalleryIndexer"
    }
}
