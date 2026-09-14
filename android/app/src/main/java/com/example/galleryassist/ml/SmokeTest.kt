package com.example.galleryassist.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * M0 phone smoke test — the last open M0 exit criterion.
 *
 * Times ONNX session load, then embeds a synthetic image twice (warm-up +
 * measured run) and reports per-image latency. Kept as a reusable util so a
 * Settings screen (M4) can surface it; call from any coroutine scope.
 */
suspend fun runModelSmokeTest(context: Context): String = withContext(Dispatchers.Default) {
    runCatching {
        val t0 = System.nanoTime()
        val engine = ClipOnnxEngine.fromContext(context)
        val loadMs = (System.nanoTime() - t0) / 1_000_000
        val result = try {
            val bmp = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888).apply {
                eraseColor(Color.rgb(120, 90, 200))
            }
            engine.embedImage(bmp) // warm-up (first inference allocates arenas)
            val t1 = System.nanoTime()
            engine.embedImage(bmp)
            val embedMs = (System.nanoTime() - t1) / 1_000_000
            val t2 = System.nanoTime()
            val textVec = engine.encodeText("a photo of a bed")
            val textMs = (System.nanoTime() - t2) / 1_000_000
            "session load: $loadMs ms\nimage embed (warm): $embedMs ms\n" +
                "text embed: $textMs ms\ntext dim: ${textVec.size}"
        } finally {
            engine.close()
        }
        result
    }.fold(
        onSuccess = { it },
        onFailure = { "smoke test FAILED: ${it.message}" },
    )
}
