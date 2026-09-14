package com.example.galleryassist.ml

import android.content.Context
import android.graphics.Bitmap

/**
 * CLIP ViT-B/32 (int8, ONNX Runtime) implementation of [EmbeddingEngine] —
 * the M0 winner. Wraps the existing [ClipEncoder] so the indexer stays
 * model-agnostic (swap here when a compressed/int4 model lands).
 */
class ClipOnnxEngine private constructor(
    private val encoder: ClipEncoder,
) : EmbeddingEngine {

    override val embedDim: Int = ClipEncoder.EMBED_DIM

    override fun embedImage(bitmap: Bitmap): FloatArray = encoder.encodeImage(bitmap)

    override fun close() = encoder.close()

    companion object {
        fun fromFiles(context: Context): ClipOnnxEngine =
            ClipOnnxEngine(ClipEncoder.fromFiles(context))
    }
}
