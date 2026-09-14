package com.example.galleryassist.data

import android.content.ContentUris
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.IOException

/**
 * Discovers gallery photos through MediaStore and persists their metadata.
 *
 * Media access model (modern best practice):
 *  - Discovery = a projection-limited MediaStore query. We read only the
 *    columns in [PROJECTION] — never file paths — so we work identically
 *    under scoped storage (API 29+) and granular media permissions (33+).
 *  - Access = through the [PhotoMetadata.contentUri] via ContentResolver
 *    (decode a downscaled bitmap for embedding/thumbnails). Original files
 *    are never copied.
 *  - Persistence = a small JSON record of [PhotoMetadata] in app-private
 *    storage ([INDEX_FILE]); it survives restarts and is refreshed on re-index.
 *
 * Under partial access ("Select photos" on Android 14+), MediaStore itself
 * only returns the user-selected items, so this repository naturally respects
 * the user's choice without extra code paths.
 */
class PhotoRepository(private val context: Context) {

    @Serializable
    data class IndexFile(
        val lastIndexedEpochMs: Long,
        val photos: List<PhotoMetadata>,
        /** PhotoAccess name the index was built under (FULL/PARTIAL) — lets the
         *  app detect a scope change (e.g. revoked to "selected only") and
         *  re-index automatically. Empty on indexes from older builds. */
        val accessScope: String = "",
    )

    /** Returns the photos visible to this app right now (permission-scoped). */
    fun queryPhotos(): List<PhotoMetadata> {
        val out = mutableListOf<PhotoMetadata>()
        context.contentResolver.query(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            PROJECTION,
            null,
            null,
            "${MediaStore.Images.Media.DATE_ADDED} DESC",
        )?.use { c ->
            val idCol = c.getColumnIndexOrThrow(MediaStore.Images.Media._ID)
            val nameCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DISPLAY_NAME)
            val takenCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_TAKEN)
            val modCol = c.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_MODIFIED)
            val bucketCol = c.getColumnIndex(MediaStore.Images.Media.BUCKET_DISPLAY_NAME)
            val wCol = c.getColumnIndex(MediaStore.Images.Media.WIDTH)
            val hCol = c.getColumnIndex(MediaStore.Images.Media.HEIGHT)
            while (c.moveToNext()) {
                val id = c.getLong(idCol)
                out += PhotoMetadata(
                    id = id,
                    contentUri = ContentUris.withAppendedId(
                        MediaStore.Images.Media.EXTERNAL_CONTENT_URI, id,
                    ).toString(),
                    displayName = c.getString(nameCol) ?: "",
                    dateTakenSec = if (c.isNull(takenCol)) null else c.getLong(takenCol) / 1000,
                    dateModifiedSec = if (c.isNull(modCol)) null else c.getLong(modCol),
                    bucketDisplayName = if (bucketCol >= 0 && !c.isNull(bucketCol)) c.getString(bucketCol) else null,
                    width = if (wCol >= 0 && !c.isNull(wCol)) c.getInt(wCol) else null,
                    height = if (hCol >= 0 && !c.isNull(hCol)) c.getInt(hCol) else null,
                )
            }
        }
        return out
    }

    /**
     * Decodes a bitmap sized for embedding/display (long edge ≤ [maxDim]) from
     * its content URI. Tries three independent pipelines (v0.2.3):
     *
     *  1. [android.graphics.ImageDecoder] — the modern path, handles HEIF
     *     and vendor quirks; used by the system picker/gallery apps.
     *  2. [BitmapFactory] via a content stream (the v0.2.2 pipeline).
     *  3. [ContentResolver.loadThumbnail] — the system thumbnail cache.
     *
     * Returns null only if ALL fail. [onFail] receives one line per failed
     * pipeline ("pipeline: ExceptionType: message") so the diag panel shows
     * exactly why decoding fails on a given device. No file copies anywhere.
     */
    fun decodeThumbnail(uri: Uri, maxDim: Int = 512, onFail: ((String) -> Unit)? = null): Bitmap? {
        fun fail(pipe: String, e: Throwable): Bitmap? {
            onFail?.invoke("$pipe: ${e::class.java.simpleName}: ${e.message ?: "(no message)"}")
            return null
        }

        // Pipeline 1: ImageDecoder (API 28+; minSdk 26 → API 27 falls through).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            try {
                val source = ImageDecoder.createSource(context.contentResolver, uri)
                return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    // Force software config: the CLIP preprocessor reads
                    // pixels directly (getPixels), which hardware bitmaps
                    // (the ImageDecoder default) do not support.
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                    // Downscale to ≤ maxDim on the long edge during decode.
                    val w = info.size.width
                    val h = info.size.height
                    if (maxOf(w, h) > maxDim) {
                        decoder.setTargetSize(w * maxDim / maxOf(w, h), h * maxDim / maxOf(w, h))
                    }
                }
            } catch (e: Exception) {
                fail("p1 ImageDecoder", e)
            }
        }

        // Pipeline 2: BitmapFactory via content stream (v0.2.2 pipeline).
        // All failure paths inside run{} fall through to pipeline 3.
        try {
            val p2: Bitmap? = run {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                val boundsStream = context.contentResolver.openInputStream(uri)
                if (boundsStream == null) {
                    fail("p2 stream open", IOException("openInputStream returned null"))
                    return@run null
                }
                boundsStream.use { BitmapFactory.decodeStream(it, null, bounds) }
                if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
                    fail("p2 bounds", IOException("bounds ${bounds.outWidth}x${bounds.outHeight}"))
                    return@run null
                }

                var sample = 1
                while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) sample *= 2
                val dataStream = context.contentResolver.openInputStream(uri)
                if (dataStream == null) {
                    fail("p2 stream open(2)", IOException("openInputStream returned null"))
                    return@run null
                }
                val decoded = dataStream.use {
                    BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
                }
                if (decoded == null) {
                    fail("p2 decode", IOException("decodeStream returned null"))
                    return@run null
                }

                val scale = maxDim.toFloat() / maxOf(decoded.width, decoded.height)
                if (scale < 1f) {
                    Bitmap.createScaledBitmap(
                        decoded,
                        (decoded.width * scale).toInt().coerceAtLeast(1),
                        (decoded.height * scale).toInt().coerceAtLeast(1),
                        true,
                    ).also { if (it !== decoded) decoded.recycle() }
                } else {
                    decoded
                }
            }
            if (p2 != null) return p2
        } catch (e: Exception) {
            fail("p2 BitmapFactory", e)
        }

        // Pipeline 3: system thumbnail cache (ContentResolver.loadThumbnail).
        try {
            return context.contentResolver.loadThumbnail(uri, android.util.Size(maxDim, maxDim), null)
        } catch (e: Exception) {
            return fail("p3 loadThumbnail", e)
        }
    }

    // --- Persistence of metadata (never images) ----------------------------

    fun saveIndex(index: IndexFile) {
        val dir = File(context.filesDir, "index")
        dir.mkdirs()
        val file = File(dir, "photo_index.json")
        val tmp = File(dir, file.name + ".tmp")
        tmp.writeText(json.encodeToString(IndexFile.serializer(), index))
        if (!tmp.renameTo(file)) {
            file.writeText(tmp.readText())
            tmp.delete()
        }
    }

    /**
     * Loads the persisted index, or null when absent/unusable. [onError]
     * receives full parse/IO detail (DIAGNOSTIC BUILD) so a corrupt index is
     * distinguishable from "first run" — previously both were a silent null.
     */
    fun loadIndex(onError: (String) -> Unit = {}): IndexFile? {
        val file = File(File(context.filesDir, "index"), "photo_index.json")
        if (!file.exists()) return null
        return try {
            json.decodeFromString(IndexFile.serializer(), file.readText())
        } catch (e: Exception) {
            onError("${e::class.java.simpleName}: ${e.message} (file ${file.length()} bytes)")
            null
        }
    }

    private val json = Json { ignoreUnknownKeys = true }

    private companion object {
        val PROJECTION = arrayOf(
            MediaStore.Images.Media._ID,
            MediaStore.Images.Media.DISPLAY_NAME,
            MediaStore.Images.Media.DATE_TAKEN,
            MediaStore.Images.Media.DATE_MODIFIED,
            MediaStore.Images.Media.BUCKET_DISPLAY_NAME,
            MediaStore.Images.Media.WIDTH,
            MediaStore.Images.Media.HEIGHT,
        )
    }
}
