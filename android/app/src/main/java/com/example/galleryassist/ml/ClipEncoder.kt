package com.example.galleryassist.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device CLIP image encoder — the M0 winner (ViT-B/32 int8) via ONNX Runtime.
 *
 * Port of the desktop app's embedding step (src/model.py). Preprocessing mirrors
 * the CLIPProcessor used in the PC benchmark: resize to 224, center crop, RGB,
 * scale to [0,1], normalize with the OpenAI CLIP mean/std.
 *
 * The text tower is stubbed until M3 (needs a Kotlin BPE tokenizer); the
 * smoke-test hook lets us time image embedding on real phone hardware, which
 * is the M0 exit criterion that still needs phone numbers.
 */
class ClipEncoder private constructor(
    private val env: OrtEnvironment,
    private val imageSession: OrtSession,
) : AutoCloseable {

    /** Embeds one image into a 512-dim unit vector (float array). */
    fun encodeImage(bitmap: Bitmap): FloatArray {
        val input = preprocess(bitmap)
        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
            imageSession.run(mapOf(IMAGE_INPUT_NAME to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val vec = (results[0].value as Array<FloatArray>)[0]
                l2Normalize(vec)
                return vec
            }
        }
    }

    override fun close() {
        imageSession.close()
    }

    // --- Preprocessing: bitmap -> CHW float32 normalized tensor -------------

    private fun preprocess(src: Bitmap): FloatArray {
        // Resize so the shorter edge is IMAGE_SIZE, then center crop.
        val scale = IMAGE_SIZE.toFloat() / minOf(src.width, src.height)
        val scaled = Bitmap.createScaledBitmap(
            src,
            Math.round(src.width * scale),
            Math.round(src.height * scale),
            true,
        )
        val x0 = (scaled.width - IMAGE_SIZE) / 2
        val y0 = (scaled.height - IMAGE_SIZE) / 2
        val cropped = Bitmap.createBitmap(scaled, x0, y0, IMAGE_SIZE, IMAGE_SIZE)

        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        cropped.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val out = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        val plane = IMAGE_SIZE * IMAGE_SIZE
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            out[i] = (r - MEAN[0]) / STD[0]
            out[plane + i] = (g - MEAN[1]) / STD[1]
            out[2 * plane + i] = (b - MEAN[2]) / STD[2]
        }
        return out
    }

    companion object {
        const val IMAGE_SIZE = 224
        const val EMBED_DIM = 512

        // Matches openai/clip-vit-base-patch32 processor defaults.
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        // Input name of models/onnx/clip-b32-image-*.onnx as exported.
        private const val IMAGE_INPUT_NAME = "pixel_values"

        /** Expected file names inside app-private filesDir (adb-pushed for now). */
        const val IMAGE_MODEL_FILE = "clip-b32-image-int8.onnx"

        /**
         * Loads the image tower from app-private storage.
         *
         * For M1 the ~90 MB int8 model is pushed with adb (see android/README.md);
         * first-run download vs Play Asset Delivery is an M0/M1 open decision.
         */
        fun fromFiles(context: Context): ClipEncoder {
            val env = OrtEnvironment.getEnvironment()
            val modelFile = File(context.filesDir, IMAGE_MODEL_FILE)
            require(modelFile.exists()) {
                "${modelFile.absolutePath} not found — push it via adb (android/README.md)"
            }
            val opts = OrtSession.SessionOptions().apply {
                // Leave thread count at ORT defaults tuned for big.LITTLE phones.
                setCPUArenaAllocator(true)
            }
            val session = env.createSession(modelFile.absolutePath, opts)
            return ClipEncoder(env, session)
        }

        private fun l2Normalize(v: FloatArray) {
            var s = 0f
            for (x in v) s += x * x
            val inv = 1f / (Math.sqrt(s.toDouble()).toFloat() + 1e-8f)
            for (i in v.indices) v[i] *= inv
        }
    }
}
