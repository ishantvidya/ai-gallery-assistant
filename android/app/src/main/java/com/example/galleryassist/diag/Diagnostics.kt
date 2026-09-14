package com.example.galleryassist.diag

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * DIAGNOSTIC BUILD (v0.2.2): in-memory, timestamped event log of every step
 * that can fail silently on-device (permission, scan, index load/save, engine
 * load, embedding pass, ranking path). Rendered by the UI's diagnostics panel
 * so a user can read the lines back and we can pinpoint the root cause.
 *
 * Deliberately process-wide (one gallery pipeline per process) and bounded
 * (oldest lines dropped past [MAX_LINES]) — this ships only in v0.2.2.
 */
object Diagnostics {

    data class Event(val tSec: Double, val line: String)

    private val _events = MutableStateFlow<List<Event>>(emptyList())
    val events: StateFlow<List<Event>> = _events

    private val startMs = System.currentTimeMillis()

    /** Appends one timestamped line: `+12.3s scan: 2431 photos`. */
    fun log(line: String) {
        val t = (System.currentTimeMillis() - startMs) / 1000.0
        val updated = _events.value + Event(t, line)
        _events.value = if (updated.size > MAX_LINES) {
            updated.takeLast(MAX_LINES)
        } else {
            updated
        }
    }

    /** `step: message (ExceptionType: detail)` — full detail, never just .message. */
    fun failure(step: String, e: Throwable) {
        val cls = e::class.java.simpleName.ifEmpty { e::class.java.name }
        val msg = e.message ?: "(no message)"
        val cause = e.cause?.let { " | caused by ${it::class.java.simpleName}: ${it.message}" } ?: ""
        log("FAIL $step: $cls: $msg$cause")
    }

    private const val MAX_LINES = 200
}
