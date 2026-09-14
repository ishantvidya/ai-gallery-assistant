package com.example.galleryassist.index

import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import com.example.galleryassist.data.PhotoMetadata
import com.example.galleryassist.data.PhotoRepository
import com.example.galleryassist.diag.Diagnostics
import com.example.galleryassist.ml.EmbeddingEngine
import kotlinx.coroutines.CancellationException
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
     * [onCheckpoint] fires after every [CHECKPOINT_EVERY] embedded photos and
     * once more at the end. Checkpoints are SKIPPED while nothing new has been
     * embedded — the v0.2.2 diag log showed identical ~26 MB saves firing every
     * 50 photos when all decodes failed, which was pure I/O churn.
     *
     * Decode failures are counted with per-pipeline reasons (v0.2.3: the
     * repository tries ImageDecoder → BitmapFactory → loadThumbnail and
     * reports each failure); the first N details go to [Diagnostics].
     */
    suspend fun embedMissing(
        photos: List<PhotoMetadata>,
        engine: EmbeddingEngine,
        onCheckpoint: suspend (List<PhotoMetadata>) -> Unit = {},
    ): List<PhotoMetadata> = withContext(Dispatchers.IO) {
        val pending = photos.filter { it.embedding == null }
        Diagnostics.log("embed pass: ${pending.size} photos pending")
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
        var decodeFailures = 0
        var embeddedThisPass = 0
        var lastCheckpointEmbedded = 0
        var reasonSamples = 0
        var firstFailure: String? = null
        for (photo in pending) {
            currentCoroutineContext().ensureActive() // cooperative cancel
            if (cancelled.get()) break

            val reasons = mutableListOf<String>()
            val bmp: Bitmap? = try {
                repository.decodeThumbnail(Uri.parse(photo.contentUri)) { reason ->
                    reasons += reason
                }
            } catch (e: Exception) {
                reasons += "decode threw: ${e::class.java.simpleName}: ${e.message}"
                null
            }
            if (bmp == null) {
                decodeFailures++
                if (firstFailure == null) {
                    firstFailure = "decode id=${photo.id} (${photo.displayName}): " +
                        reasons.joinToString(" | ").ifEmpty { "decoder returned null" }
                } else if (reasonSamples < MAX_REASON_SAMPLES) {
                    // Sample a few more distinct failures across the pass.
                    reasonSamples++
                    Diagnostics.log(
                        "decode fail id=${photo.id}: " +
                            reasons.joinToString(" | ").ifEmpty { "decoder returned null" },
                    )
                }
            }
            if (bmp != null) {
                try {
                    photo.embedding = engine.embedImage(bmp)
                    embeddedThisPass++
                } catch (e: CancellationException) {
                    bmp.recycle()
                    throw e
                } catch (e: Exception) {
                    Log.w(TAG, "embed failed for ${photo.id}", e)
                }
                bmp.recycle()
            }

            val n = processed.incrementAndGet()
            sinceCheckpoint++
            _progress.value = _progress.value.copy(processed = n)
            if (sinceCheckpoint >= CHECKPOINT_EVERY) {
                sinceCheckpoint = 0
                if (embeddedThisPass > lastCheckpointEmbedded) {
                    lastCheckpointEmbedded = embeddedThisPass
                    Diagnostics.log(
                        "checkpoint: ${processed.get()}/${pending.size} processed, " +
                            "$embeddedThisPass embedded, $decodeFailures decode fails — saving",
                    )
                    onCheckpoint(photos)
                } else {
                    Diagnostics.log(
                        "checkpoint: ${processed.get()}/${pending.size} processed, " +
                            "$decodeFailures decode fails — skip save (nothing new)",
                    )
                }
            }
        }
        if (sinceCheckpoint > 0 && embeddedThisPass > lastCheckpointEmbedded) {
            onCheckpoint(photos)
        }

        Diagnostics.log(
            "embed pass done: ${pending.size} attempted, $embeddedThisPass embedded, " +
                "$decodeFailures decode failures" +
                (firstFailure?.let { " — first: $it" } ?: ""),
        )

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

        /** Cap for per-photo decode-failure detail lines in the diag panel. */
        const val MAX_REASON_SAMPLES = 5
    }
}
