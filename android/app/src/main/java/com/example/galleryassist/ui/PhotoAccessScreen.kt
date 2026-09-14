package com.example.galleryassist.ui

import android.Manifest
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.galleryassist.permissions.PhotoAccess
import com.example.galleryassist.permissions.PhotoPermissions

/**
 * First-time setup + permission flow (spec §2, §5).
 *
 * States handled:
 *  - initial: privacy rationale + "Allow Photo Access"
 *  - denied once: rationale again + "try again" + open app settings
 *  - partial access (Android 14+ "Select photos"): acknowledged, offer to
 *    broaden via the system photo picker dialog
 *  - granted: calls [onGranted] (router moves on to indexing/search)
 *
 * The app never crashes on denial: without permission we simply stay here.
 */
@Composable
fun PhotoAccessScreen(onGranted: () -> Unit) {
    val context = LocalContext.current
    var access by remember { mutableStateOf(PhotoPermissions.currentAccess(context)) }
    var deniedBefore by remember { mutableStateOf(false) }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { result ->
        access = PhotoPermissions.currentAccess(context)
        if (access == PhotoAccess.NONE) {
            deniedBefore = true
        } else {
            onGranted()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("AI Gallery Assistant", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text(
            "Give AI Gallery Assistant access to your photos",
            style = MaterialTheme.typography.titleMedium,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            "The app looks at your photos on this phone to make them searchable by "
                + "description — \"a dog at the beach\". Nothing is uploaded: there is "
                + "no server, and the app has no internet permission. "
                + "You can revoke access at any time in system settings.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Spacer(Modifier.height(24.dp))

        when (access) {
            PhotoAccess.NONE -> {
                if (deniedBefore) {
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp)) {
                            Text("Photo access is off", style = MaterialTheme.typography.titleSmall)
                            Spacer(Modifier.height(4.dp))
                            Text(
                                "Without it the app cannot see any photos. You can grant "
                                    + "access to all photos, or select just some.",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                }
                Button(modifier = Modifier.fillMaxWidth(), onClick = {
                    launcher.launch(PhotoPermissions.permissionsToRequest())
                }) { Text("Allow Photo Access") }
                if (deniedBefore) {
                    TextButton(onClick = {
                        runCatching {
                            context.startActivity(
                                android.content.Intent(
                                    android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                    android.net.Uri.fromParts("package", context.packageName, null),
                                ),
                            )
                        }
                    }) { Text("Open system settings") }
                }
            }

            PhotoAccess.PARTIAL -> {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Selected photos only", style = MaterialTheme.typography.titleSmall)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "You've shared part of your library. Search only covers those "
                                + "photos. You can add more at any time.",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
                Spacer(Modifier.height(12.dp))
                Button(modifier = Modifier.fillMaxWidth(), onClick = onGranted) {
                    Text("Continue with selected photos")
                }
                OutlinedButton(modifier = Modifier.fillMaxWidth(), onClick = {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        launcher.launch(
                            arrayOf(
                                Manifest.permission.READ_MEDIA_IMAGES,
                                Manifest.permission.READ_MEDIA_VISUAL_USER_SELECTED,
                            ),
                        )
                    }
                }) { Text("Select more photos") }
            }

            PhotoAccess.FULL -> onGranted()
        }
    }
}
