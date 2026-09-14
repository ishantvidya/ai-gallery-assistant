package com.example.galleryassist.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.example.galleryassist.diag.Diagnostics
import kotlinx.coroutines.flow.StateFlow

/**
 * DIAGNOSTIC BUILD (v0.2.2): small "Diag" toggle (top-right, above the UI)
 * opening a monospace panel with the timestamped event log from
 * [Diagnostics] — permission state, scan/index counts, engine load result
 * with full exception detail, embedding pass/failures, and the rank() path
 * taken per query. Includes a one-tap "Copy" so the user can paste the whole
 * log back to us.
 */
@Composable
fun DiagnosticsPanel(diag: StateFlow<List<Diagnostics.Event>>, modifier: Modifier = Modifier) {
    val events by diag.collectAsStateWithLifecycle()
    var open by remember { mutableStateOf(false) }
    val context = LocalContext.current

    Column(modifier = modifier) {
        Row(Modifier.fillMaxWidth()) {
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { open = !open }) {
                Text(
                    if (open) "Hide diag" else "Diag",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
        }
        if (open) {
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xF0101010),
                    contentColor = Color(0xFFD0FFD0),
                ),
            ) {
                Column(Modifier.padding(8.dp)) {
                    Text(
                        "v0.2.2 diagnostics — ${events.size} events",
                        style = MaterialTheme.typography.labelSmall,
                    )
                    TextButton(onClick = {
                        val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                        cm.setPrimaryClip(
                            ClipData.newPlainText(
                                "diag",
                                events.joinToString("\n") { e ->
                                    "+%.1fs %s".format(e.tSec, e.line)
                                },
                            ),
                        )
                    }) { Text("Copy all", style = MaterialTheme.typography.labelSmall) }
                    Spacer(Modifier.height(4.dp))
                    Column(
                        Modifier
                            .heightIn(max = 360.dp)
                            .verticalScroll(rememberScrollState()),
                    ) {
                        events.forEach { e ->
                            Text(
                                "+%.1fs %s".format(e.tSec, e.line),
                                style = MaterialTheme.typography.bodySmall
                                    .copy(fontFeatureSettings = "tnum"),
                            )
                        }
                    }
                }
            }
        }
    }
}
