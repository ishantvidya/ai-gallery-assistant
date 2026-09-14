package com.example.galleryassist.ml

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * On-device CLIP encoder (ViT-B/32 int8) via ONNX Runtime — both towers.
 *
 * Port of the desktop app's embedding step (src/model.py). Preprocessing mirrors
 * the CLIPProcessor used in the PC benchmark: resize to 224, center crop, RGB,
 * scale to [0,1], normalize with the OpenAI CLIP mean/std.
 *
 * Models are bundled in the APK (assets/) and copied into app-private storage
 * on first run — ONNX Runtime needs a real file path. The text tower turns a
 * query into a 512-dim vector; the image tower does the same for photos during
 * indexing. Both outputs are unit-normalized, so cosine similarity is a dot
 * product.
 */
class ClipEncoder private constructor(
    private val env: OrtEnvironment,
    private val imageSession: OrtSession,
    private val textSession: OrtSession,
    private val tokenizer: ClipTokenizer,
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

    /** Encodes one text query into a 512-dim unit vector (float array). */
    fun encodeText(text: String): FloatArray {
        val ids = tokenizer.encode(text)
        val shape = longArrayOf(1, ClipTokenizer.MAX_LEN.toLong())
        val buffer = LongBuffer.wrap(ids.map { it.toLong() }.toLongArray())
        OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
            textSession.run(mapOf(TEXT_INPUT_NAME to tensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val vec = (results[0].value as Array<FloatArray>)[0]
                l2Normalize(vec)
                return vec
            }
        }
    }

    /** Dot product of two unit vectors == cosine similarity. */
    fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        val n = minOf(a.size, b.size)
        for (i in 0 until n) s += a[i] * b[i]
        return s
    }

    override fun close() {
        imageSession.close()
        textSession.close()
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

        // Input names of the exported towers (scripts/export_clip_b32_onnx.py).
        private const val IMAGE_INPUT_NAME = "pixel_values"
        private const val TEXT_INPUT_NAME = "input_ids"

        /** Model file names, bundled in APK assets and copied to filesDir. */
        const val IMAGE_MODEL_ASSET = "clip-b32-image-int8.onnx"
        const val TEXT_MODEL_ASSET = "clip-b32-text-int8.onnx"
        private const val TOKENIZER_ASSET_DIR = "tokenizer"

        /** Model file, extracted into app-private storage on first run. */
        fun imageModelFile(context: Context): File =
            File(context.filesDir, IMAGE_MODEL_ASSET)

        /** True once both model files have been extracted and exist. */
        fun assetsExtracted(context: Context): Boolean =
            imageModelFile(context).exists() &&
                File(context.filesDir, TEXT_MODEL_ASSET).exists()

        /** Copies one asset to filesDir if missing or stale (size mismatch). */
        private fun extractAsset(context: Context, assetName: String): File {
            val out = File(context.filesDir, assetName)
            val expectedLen = try {
                context.assets.openFd(assetName).use { it.length }
            } catch (e: Exception) {
                -1L // compressed assets have no fd; size check unavailable
            }
            if (!out.exists() || out.length() == 0L ||
                (expectedLen > 0 && out.length() != expectedLen)
            ) {
                context.assets.open(assetName).use { input ->
                    val tmp = File(context.filesDir, "$assetName.tmp")
                    tmp.outputStream().use { output -> input.copyTo(output, COPY_BUFFER) }
                    if (tmp.renameTo(out)) {
                        // fast path done
                    } else {
                        // rename can fail across mount points; copy explicitly
                        tmp.inputStream().use { input2 ->
                            out.outputStream().use { input2.copyTo(it, COPY_BUFFER) }
                        }
                        tmp.delete()
                    }
                }
            }
            return out
        }

        /**
         * Extracts bundled models + tokenizer to app storage and loads both
         * towers. First call costs a one-time ~155 MB copy + session load.
         */
        fun fromContext(context: Context): ClipEncoder {
            val env = OrtEnvironment.getEnvironment()
            val imageFile = extractAsset(context, IMAGE_MODEL_ASSET)
            val textFile = extractAsset(context, TEXT_MODEL_ASSET)
            val opts = OrtSession.SessionOptions().apply {
                // Leave thread count at ORT defaults tuned for big.LITTLE phones.
                setCPUArenaAllocator(true)
            }
            val imageSession = env.createSession(imageFile.absolutePath, opts)
            val textSession = env.createSession(textFile.absolutePath, opts)
            val tokenizer = ClipTokenizer.fromAssets(context, TOKENIZER_ASSET_DIR)
            return ClipEncoder(env, imageSession, textSession, tokenizer)
        }

        private const val COPY_BUFFER = 1 shl 16

        private fun l2Normalize(v: FloatArray) {
            var s = 0f
            for (x in v) s += x * x
            val inv = 1f / (Math.sqrt(s.toDouble()).toFloat() + 1e-8f)
            for (i in v.indices) v[i] *= inv
        }
    }
}
