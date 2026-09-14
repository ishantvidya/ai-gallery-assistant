package com.example.galleryassist.ui

import androidx.compose.foundation.layout.Arrangement
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

/**
 * Main gallery/search interface. Shows the photos discovered via MediaStore
 * (metadata only — images load straight from their content URIs, no copies)
 * and the search bar. Ranking with the on-device CLIP model arrives in M3;
 * the grid below is the real indexed gallery, not a picker demo.
 */
@Composable
fun SearchScreen(
    photos: List<PhotoMetadata>,
    modifier: Modifier = Modifier,
) {
    var query by remember { mutableStateOf("") }

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
            Text(
                "${photos.size} photos indexed",
                style = MaterialTheme.typography.bodySmall,
            )
            Spacer(Modifier.width(12.dp))
        }
        Spacer(Modifier.height(8.dp))
        Card {
            Column(Modifier.padding(12.dp)) {
                Text(
                    "All processing happens on this phone — nothing is uploaded.",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "Natural-language ranking lands with the on-device model (M3).",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Spacer(Modifier.height(12.dp))
        LazyVerticalGrid(
            columns = GridCells.Adaptive(minSize = 110.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp),
            modifier = Modifier.weight(1f),
        ) {
            items(photos, key = { it.id }) { photo ->
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
