package com.example.galleryassist.permissions

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.content.ContextCompat

/** How much of the gallery the app may see right now. */
enum class PhotoAccess { FULL, PARTIAL, NONE }

/**
 * Central, version-aware photo permission logic.
 *
 * - API 33+: granular [Manifest.permission.READ_MEDIA_IMAGES]; the user may
 *   grant it fully or choose "Select photos" (partial), which grants
 *   [Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED] instead. Under
 *   partial access MediaStore only returns the chosen items — the rest of the
 *   app needs no special-casing.
 * - API <= 32: legacy [Manifest.permission.READ_EXTERNAL_STORAGE] (manifest
 *   caps it with maxSdkVersion=32). There is no partial concept there.
 *
 * Never any contact/location/mic/file-system permissions — see the manifest.
 */
object PhotoPermissions {

    private val readImages: String =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

    private val readImagesSelected: String? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED
        } else null

    /** Permissions to request in one launcher call for this Android version. */
    fun permissionsToRequest(): Array<String> =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            arrayOf(Manifest.permission.READ_MEDIA_IMAGES, Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED)
        } else {
            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
        }

    fun currentAccess(context: Context): PhotoAccess {
        val full = isGranted(context, readImages)
        if (full) return PhotoAccess.FULL
        val partial = readImagesSelected?.let { isGranted(context, it) } == true
        return if (partial) PhotoAccess.PARTIAL else PhotoAccess.NONE
    }

    private fun isGranted(context: Context, permission: String): Boolean =
        ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
}
