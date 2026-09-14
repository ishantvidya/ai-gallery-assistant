// AGP 9 has built-in Kotlin support: do NOT apply org.jetbrains.kotlin.android.
// The Compose compiler plugin is still applied per-module (see app/build.gradle.kts).
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.compose.compiler) apply false
}
