package com.example.galleryassist

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.galleryassist.ui.DiagnosticsPanel
import com.example.galleryassist.ui.IndexingScreen
import com.example.galleryassist.ui.PhotoAccessScreen
import com.example.galleryassist.ui.SearchScreen
import com.example.galleryassist.ui.theme.GalleryAssistTheme
import androidx.lifecycle.compose.collectAsStateWithLifecycle

/**
 * Single-activity router over [GalleryViewModel]:
 *   NeedAccess → PhotoAccessScreen (rationale + grant + denied/partial states)
 *   Indexing   → IndexingScreen (progress + photo count)
 *   Ready      → SearchScreen (indexed gallery grid; search lands in M3)
 *
 * Permission checks re-run on every ON_RESUME, so grants, revocations and
 * partial-access changes made in system settings are picked up without
 * restarting the app. The app never crashes on denial — without permission
 * it simply shows the setup screen.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: GalleryViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            GalleryAssistTheme {
                AppRoot(viewModel)
            }
        }
    }
}

@Composable
private fun AppRoot(viewModel: GalleryViewModel) {
    val stage by viewModel.stage.collectAsStateWithLifecycle()

    // Re-evaluate permission state whenever the app comes to the foreground
    // (covers first launch, return from system settings, and revocations).
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                viewModel.onPermissionChanged()
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    // DIAGNOSTIC BUILD (v0.2.2): the diag panel overlays EVERY stage —
    // including NeedAccess and Indexing — so a failure that would previously
    // hide the app behind the permission screen is still readable in-app.
    Box(Modifier.fillMaxSize()) {
        when (val s = stage) {
            AppStage.NeedAccess -> PhotoAccessScreen(onGranted = { viewModel.onPermissionChanged() })
            AppStage.Indexing -> IndexingScreen(progress = viewModel.indexingProgress.collectAsStateWithLifecycle().value)
            is AppStage.Ready -> SearchScreen(
                photos = s.photos,
                aiProgress = viewModel.indexingProgress.collectAsStateWithLifecycle().value,
                onRank = { query, onResult -> viewModel.rank(query, viewModel.currentPhotosSnapshot(), onResult) },
            )
        }
        DiagnosticsPanel(
            diag = viewModel.diag,
            modifier = Modifier.fillMaxSize().padding(top = 48.dp),
        )
    }
}
