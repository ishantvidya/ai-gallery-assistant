package com.example.galleryassist.ml

import android.content.Context
import android.graphics.Bitmap
import java.io.File

/**
 * The seam between gallery indexing and the AI model.
 *
 * The indexer only knows this interface — swap/upgrade the model (int4 CLIP,
 * a newer encoder, captions) without touching the pipeline. Implementations
 * must be closed after use to free native model memory.
 */
interface EmbeddingEngine : AutoCloseable {
    /** Dimensionality of the produced vectors (CLIP B/32 = 512). */
    val embedDim: Int

    /** Embeds one image into a unit-normalized vector. */
    fun embedImage(bitmap: Bitmap): FloatArray

    companion object {
        /** True when the on-device model file is present in app storage. */
        fun modelAvailable(context: Context): Boolean =
            File(context.filesDir, ClipEncoder.IMAGE_MODEL_FILE).exists()
    }
}
