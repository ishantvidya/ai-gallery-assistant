plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.serialization)
}

android {
    namespace = "com.example.galleryassist"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.galleryassist"
        minSdk = 26
        // Play Store requirement (Aug 2026): new apps must target API 36.
        targetSdk = 36
        versionCode = 4
        versionName = "0.2.1"
    }

    // Release signing comes from CI secrets (see .github/workflows/android-release.yml).
    // Local/PR builds without those env vars fall back to debug signing so the
    // build always succeeds.
    signingConfigs {
        create("release") {
            val storePath = System.getenv("SIGNING_STORE_FILE")
            if (storePath != null && file(storePath).exists()) {
                storeFile = file(storePath)
                storePassword = System.getenv("SIGNING_STORE_PASSWORD")
                keyAlias = System.getenv("SIGNING_KEY_ALIAS")
                keyPassword = System.getenv("SIGNING_STORE_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            val signingConfigured = System.getenv("SIGNING_STORE_FILE")
                ?.let { file(it).exists() } == true
            signingConfig = if (signingConfigured) {
                signingConfigs.getByName("release")
            } else {
                signingConfigs.getByName("debug")
            }
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // Built-in Kotlin: jvmTarget defaults to compileOptions.targetCompatibility,
    // so no kotlin { compilerOptions { } } block is needed.
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.kotlinx.serialization.json)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    implementation(libs.coil.compose)

    // On-device CLIP inference (M0 winner: ViT-B/32 int8).
    implementation(libs.onnxruntime.android)

    debugImplementation(libs.androidx.compose.ui.tooling)
}
