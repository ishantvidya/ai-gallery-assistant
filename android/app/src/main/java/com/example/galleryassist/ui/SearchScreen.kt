package com.example.galleryassist.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.galleryassist.data.PhotoMetadata
import java.time.Instant
import java.time.ZoneId
import java.time.format.TextStyle
import java.util.Locale

/**
 * Main gallery/search interface. Shows the photos discovered via MediaStore
 * (metadata only — images load straight from their content URIs, no copies)
 * and the search bar.
 *
 * Search today matches **metadata**: filename, album/bucket name, and the
 * photo's date (year, full and abbreviated month names). All tokens must
 * match somewhere (AND semantics), so "beach 2024" narrows step by step.
 * Semantic "describe it to find it" ranking with the on-device CLIP model
 * lands in M3 — it needs the model + stored image embeddings.
 */
@Composable
fun SearchScreen(
    photos: List<PhotoMetadata>,
    modifier: Modifier = Modifier,
) {
    var query by remember { mutableStateOf("") }
    val filtered = remember(photos, query) { photos.matching(query) }

    Column(modifier = modifier.fillMaxSize().padding(16.dp)) {
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("Search your photos… (\"a dog at the beach\")") },
            singleLine = true,
        )
        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            val count = if (query.isBlank()) "${photos.size} photos indexed"
            else "${filtered.size} of ${photos.size} photos match"
            Text(count, style = MaterialTheme.typography.bodySmall)
            Spacer(Modifier.width(12.dp))
        }
        Spacer(Modifier.height(8.dp))
        Card {
            Column(Modifier.padding(12.dp)) {
                Text(
                    "Search matches filenames, albums and dates — all on this phone, nothing uploaded.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "\"Describe it to find it\" search arrives with the on-device AI model (M3).",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        if (filtered.isEmpty() && query.isNotBlank()) {
            Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text(
                    "No photos match \u201C$query\u201D",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            LazyVerticalGrid(
                columns = GridCells.Adaptive(minSize = 110.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.weight(1f),
            ) {
                items(filtered, key = { it.id }) { photo ->
                    AsyncImage(
                        model = photo.contentUri,
                        contentDescription = photo.displayName,
                        modifier = Modifier.fillMaxWidth().height(110.dp),
                        contentScale = ContentScale.Crop,
                    )
                }
            }
        }
    }
}

/** Photos matching every whitespace-separated token in [query] (case-insensitive). */
private fun List<PhotoMetadata>.matching(query: String): List<PhotoMetadata> {
    val tokens = query.lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
    if (tokens.isEmpty()) return this
    return filter { p ->
        val fields = buildList {
            add(p.displayName.lowercase())
            p.bucketDisplayName?.let { add(it.lowercase()) }
            val sec = p.dateTakenSec ?: p.dateModifiedSec
            if (sec != null && sec > 0) {
                val date = Instant.ofEpochSecond(sec).atZone(ZoneId.systemDefault())
                add(date.year.toString())
                add(date.month.getDisplayName(TextStyle.FULL, Locale.ROOT).lowercase(Locale.ROOT))
                add(date.month.getDisplayName(TextStyle.SHORT, Locale.ROOT).lowercase(Locale.ROOT))
            }
        }
        tokens.all { t -> fields.any { it.contains(t) } }
    }
}
