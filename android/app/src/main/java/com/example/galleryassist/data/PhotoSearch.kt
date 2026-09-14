package com.example.galleryassist.data

/**
 * Metadata-only photo matching, shared by the search UI (pre-model fallback)
 * and the ViewModel (when embeddings are unavailable).
 *
 * Every whitespace-separated token must match somewhere (AND semantics) in
 * the filename, album/bucket name, or the photo's date (year, full and short
 * month names) — so "beach 2024" narrows step by step.
 */
fun List<PhotoMetadata>.matchingMetadata(query: String): List<PhotoMetadata> {
    val tokens = query.lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
    if (tokens.isEmpty()) return this
    return filter { p ->
        val fields = buildList {
            add(p.displayName.lowercase())
            p.bucketDisplayName?.let { add(it.lowercase()) }
            val sec = p.dateTakenSec ?: p.dateModifiedSec
            if (sec != null && sec > 0) {
                val date = java.time.Instant.ofEpochSecond(sec)
                    .atZone(java.time.ZoneId.systemDefault())
                add(date.year.toString())
                add(date.month.getDisplayName(java.time.format.TextStyle.FULL, java.util.Locale.ROOT)
                    .lowercase(java.util.Locale.ROOT))
                add(date.month.getDisplayName(java.time.format.TextStyle.SHORT, java.util.Locale.ROOT)
                    .lowercase(java.util.Locale.ROOT))
            }
        }
        tokens.all { t -> fields.any { it.contains(t) } }
    }
}
