package com.example.galleryassist.ml

import android.content.Context
import android.graphics.Bitmap

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
        /** True once both bundled model files are extracted in app storage. */
        fun modelAvailable(context: Context): Boolean =
            ClipEncoder.assetsExtracted(context)
    }
}
