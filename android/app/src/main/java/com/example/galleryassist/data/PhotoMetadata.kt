package com.example.galleryassist.data

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Metadata for one gallery photo, discovered via MediaStore.
 *
 * We never copy image bytes: indexing and display always go through the
 * [contentUri], so the gallery remains the single source of truth (no
 * duplication of files). What we persist is just this small record.
 */
@Serializable
data class PhotoMetadata(
    /** MediaStore _ID, stable across queries on the same device. */
    val id: Long,
    /** content:// URI — the only handle we keep to the image bytes. */
    @SerialName("contentUri") val contentUri: String,
    /** MediaStore DISPLAY_NAME, e.g. "IMG_20250714_123456.jpg". */
    @SerialName("displayName") val displayName: String,
    @SerialName("dateTakenSec") val dateTakenSec: Long?,
    /** MediaStore DATE_MODIFIED (epoch seconds). */
    @SerialName("dateModifiedSec") val dateModifiedSec: Long?,
    /** Album/bucket (folder) display name, where available. */
    @SerialName("bucketDisplayName") val bucketDisplayName: String?,
    @SerialName("width") val width: Int?,
    @SerialName("height") val height: Int?,
)
