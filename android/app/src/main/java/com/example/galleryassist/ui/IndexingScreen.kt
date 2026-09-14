package com.example.galleryassist.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.galleryassist.index.IndexingProgress

/**
 * Indexing progress (spec §5): shows the discovery/embedding phase and the
 * number of photos found so far. Stays visible until the router switches to
 * search when the ViewModel emits Ready.
 */
@Composable
fun IndexingScreen(progress: IndexingProgress, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Preparing your gallery", style = MaterialTheme.typography.headlineSmall)
        Spacer(Modifier.height(8.dp))
        Text(
            when (progress.phase) {
                IndexingProgress.Phase.SCANNING -> "Looking for photos…"
                IndexingProgress.Phase.EMBEDDING ->
                    if (progress.total > 0) {
                        "Found ${progress.total} photos — processing ${progress.processed}/${progress.total}"
                    } else {
                        "Found 0 photos so far…"
                    }
                IndexingProgress.Phase.DONE -> "Done"
            },
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(16.dp))
        if (progress.total > 0) {
            LinearProgressIndicator(
                progress = { progress.processed.toFloat() / progress.total },
                modifier = Modifier.fillMaxWidth(),
            )
        } else {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth()) // indeterminate
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "This happens on your phone. Photos are never uploaded.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
