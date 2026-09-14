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

/** User-visible indexing/embedding progress. */
data class IndexingProgress(
    val phase: Phase = Phase.SCANNING,
    val processed: Int = 0,
    val total: Int = 0,
) {
    enum class Phase { SCANNING, EMBEDDING, DONE }
}

/**
 * Builds the searchable index in two independent stages:
 *
 *  1. [scan] — MediaStore metadata (fast, seconds) → UI is usable immediately.
 *  2. [embedMissing] — CLIP pass over photos lacking embeddings, with
 *     **checkpointing**: [onCheckpoint] fires every [CHECKPOINT_EVERY] photos
 *     so the ViewModel can persist partial progress. An interrupted pass
 *     resumes where it stopped instead of starting over.
 */
class GalleryIndexer(
    private val repository: PhotoRepository,
) {

    private val _progress = MutableStateFlow(IndexingProgress())
    val progress: StateFlow<IndexingProgress> = _progress

    private val cancelled = AtomicBoolean(false)
    private val processed = AtomicInteger(0)

    /**
     * Discovers what MediaStore exposes under the current permission scope.
     * Cheap (metadata only) — call before/while embedding.
     */
    suspend fun scan(): List<PhotoMetadata> = withContext(Dispatchers.IO) {
        cancelled.set(false)
        processed.set(0)
        _progress.value = IndexingProgress(phase = IndexingProgress.Phase.SCANNING)
        repository.queryPhotos()
    }

    /**
     * Embeds every photo in [photos] whose embedding is null, writing results
     * back into the list's [PhotoMetadata.embedding] (objects are mutated in
     * place; [photos] order is preserved).
     *
     * [onCheckpoint] fires after every [CHECKPOINT_EVERY] embedded photo and
     * once more at the end, so callers can persist incremental progress.
     * Cancellation (permission revoked / app backgrounded) stops cleanly at
     * the next photo boundary — completed work is kept by the caller.
     */
    suspend fun embedMissing(
        photos: List<PhotoMetadata>,
        engine: EmbeddingEngine,
        onCheckpoint: suspend (List<PhotoMetadata>) -> Unit = {},
    ): List<PhotoMetadata> = withContext(Dispatchers.IO) {
        val pending = photos.filter { it.embedding == null }
        _progress.value = IndexingProgress(
            phase = IndexingProgress.Phase.EMBEDDING,
            processed = 0,
            total = pending.size,
        )
        if (pending.isEmpty()) {
            _progress.value = _progress.value.copy(phase = IndexingProgress.Phase.DONE)
            return@withContext photos
        }

        processed.set(0)
        var sinceCheckpoint = 0
        for (photo in pending) {
            currentCoroutineContext().ensureActive() // cooperative cancel
            if (cancelled.get()) break

            val bmp: Bitmap? = runCatching {
                repository.decodeThumbnail(Uri.parse(photo.contentUri))
            }.getOrNull()
            if (bmp != null) {
                runCatching { engine.embedImage(bmp) }
                    .onSuccess { photo.embedding = it }
                    .onFailure { Log.w(TAG, "embed failed for ${photo.id}", it) }
                bmp.recycle()
            }

            val n = processed.incrementAndGet()
            sinceCheckpoint++
            _progress.value = _progress.value.copy(processed = n)
            if (sinceCheckpoint >= CHECKPOINT_EVERY) {
                sinceCheckpoint = 0
                onCheckpoint(photos)
            }
        }
        if (sinceCheckpoint > 0) onCheckpoint(photos)

        _progress.value = _progress.value.copy(phase = IndexingProgress.Phase.DONE)
        photos
    }

    fun cancel() {
        cancelled.set(true)
    }

    private companion object {
        const val TAG = "GalleryIndexer"

        /** Persist partial progress every N embedded photos. */
        const val CHECKPOINT_EVERY = 50
    }
}
