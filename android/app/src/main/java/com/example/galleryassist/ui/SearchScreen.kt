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
import androidx.compose.runtime.LaunchedEffect
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
import com.example.galleryassist.data.matchingMetadata
import kotlinx.coroutines.delay

/**
 * Main gallery/search interface over the indexed photos.
 *
 * How a query is answered (best available wins, all on-device):
 *  1. **Semantic** (when the photo has CLIP embeddings): the query is encoded
 *     by the on-device text tower and photos are ranked by cosine similarity
 *     — "bed", "dog at the beach", "two people laughing" work.
 *  2. **Metadata fallback**: filename / album / date token matching, shown
 *     instantly while the semantic ranking is computed (and used exclusively
 *     when embeddings aren't available yet).
 */
@Composable
fun SearchScreen(
    photos: List<PhotoMetadata>,
    onRank: (query: String, onResult: (List<PhotoMetadata>) -> Unit) -> Unit,
    modifier: Modifier = Modifier,
) {
    var query by remember { mutableStateOf("") }
    var semanticResults by remember { mutableStateOf<List<PhotoMetadata>?>(null) }
    var searching by remember { mutableStateOf(false) }

    val metaResults = remember(photos, query) {
        if (query.isBlank()) photos else photos.matchingMetadata(query)
    }

    // Debounced semantic ranking; metadata results show immediately.
    LaunchedEffect(photos, query) {
        semanticResults = null
        if (query.isBlank()) {
            searching = false
        } else {
            searching = true
            delay(SEARCH_DEBOUNCE_MS)
            onRank(query) { ranked ->
                semanticResults = ranked
                searching = false
            }
        }
    }

    val display = semanticResults ?: metaResults
    val embeddedCount = remember(photos) { photos.count { it.embedding != null } }

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
            else "${display.size} of ${photos.size} match"
            Text(count, style = MaterialTheme.typography.bodySmall)
            if (searching) {
                Spacer(Modifier.width(8.dp))
                Text("· ranking by meaning…", style = MaterialTheme.typography.bodySmall)
            }
        }
        Spacer(Modifier.height(8.dp))
        Card {
            Column(Modifier.padding(12.dp)) {
                if (embeddedCount > 0) {
                    Text(
                        "AI search is on — describe what's in the photo (\"bed\", \"human\", \"sunset\").",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "$embeddedCount/${photos.size} photos have AI embeddings · filenames/albums/dates also match · nothing leaves this phone.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                } else {
                    Text(
                        "Search matches filenames, albums and dates — all on this phone, nothing uploaded.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "AI \"describe it to find it\" search activates after the next re-index (photos are being embedded).",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }
        Spacer(Modifier.height(12.dp))
        if (display.isEmpty() && query.isNotBlank()) {
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
                items(display, key = { it.id }) { photo ->
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

private const val SEARCH_DEBOUNCE_MS = 350L
