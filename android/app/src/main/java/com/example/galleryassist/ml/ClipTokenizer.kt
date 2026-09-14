package com.example.galleryassist.ml

import android.content.Context
import kotlinx.serialization.json.Json
import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.Locale

/**
 * CLIP (ViT-B/32) byte-level BPE tokenizer — a faithful Kotlin port of the
 * exact algorithm used at export time (HF CLIPTokenizer,
 * openai/clip-vit-base-patch32), producing identical token ids.
 *
 * Pipeline: NFC normalize → collapse whitespace → lowercase → regex pre-split
 * → byte-encode (printable bytes identity, others → U+0100+) → append the
 * "</w>" end-of-word suffix to each chunk's last char → greedy BPE merges by
 * rank → vocab lookup.
 *
 * Verified by scripts/check_tokenizer_port.py against the reference
 * tokenizer on: "a photo of a bed", "human", "dog at the beach",
 * "sunset over the mountains", "IMG_20240714_123456" — all identical ids.
 *
 * @param vocab    token string -> id
 * @param mergeRanks merge pair -> rank (lower rank merges first)
 */
class ClipTokenizer private constructor(
    private val vocab: Map<String, Int>,
    private val mergeRanks: Map<Pair<String, String>, Int>,
) {
    /**
     * Encodes [text] into a fixed [IntArray] of length [MAX_LEN] (77):
     * SOT, text tokens, EOT, then EOT padding.
     */
    fun encode(text: String): IntArray {
        val ids = IntArray(MAX_LEN)
        var n = 0
        fun push(id: Int) {
            if (n < MAX_LEN) ids[n++] = id
        }

        push(vocab.getValue(SOT))
        for (chunk in chunker.findall(normalize(text))) {
            val mapped = byteEncode(chunk)
            val chars = mapped.toCharArray().map { it.toString() }.toTypedArray()
            chars[chars.size - 1] += END_OF_WORD
            for (piece in bpe(chars)) push(vocab.getValue(piece))
        }
        push(vocab.getValue(EOT))
        while (n < MAX_LEN) push(vocab.getValue(EOT))
        return ids
    }

    // --- Normalization -----------------------------------------------------

    private fun normalize(text: String): String =
        java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC)
            .replace(WS_REGEX, " ")
            .lowercase(Locale.ROOT)

    // --- Pre-tokenization ----------------------------------------------------

    /**
     * Mirrors CLIP's pre-split regex:
     * "<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
     * Emulated with Unicode-aware Kotlin: iterate code points, group letters,
     * single digits, symbol runs; pull out the special tokens and 'contractions
     * first. Behavior verified against the reference via golden vectors.
     */
    private object chunker {
        private val CONTRACTIONS =
            listOf("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

        fun findall(text: String): List<String> {
            val out = mutableListOf<String>()
            var i = 0
            val n = text.length
            while (i < n) {
                // Special tokens (only at this position, case already lowered)
                if (text.startsWith(SOT, i)) {
                    out.add(SOT); i += SOT.length; continue
                }
                if (text.startsWith(EOT, i)) {
                    out.add(EOT); i += EOT.length; continue
                }
                val c = text[i]
                // Contractions
                val contraction = CONTRACTIONS.firstOrNull { text.startsWith(it, i) }
                if (contraction != null) {
                    out.add(contraction); i += contraction.length; continue
                }
                // Letter runs [\\p{L}]+
                if (c.isLetter()) {
                    val start = i
                    while (i < n && text[i].isLetter()) i++
                    out.add(text.substring(start, i)); continue
                }
                // Single digits [\\p{N}]
                if (c.isDigit()) {
                    out.add(c.toString()); i++; continue
                }
                // Symbol runs [^\\s\\p{L}\\p{N}]+
                if (!c.isWhitespace()) {
                    val start = i
                    while (i < n && !text[i].isWhitespace() && !text[i].isLetter() && !text[i].isDigit()) i++
                    out.add(text.substring(start, i)); continue
                }
                i++ // whitespace: dropped
            }
            return out
        }
    }

    // --- Byte encoder (CLIP's reversible byte<->unicode table) ---------------

    private val byteToUnicode: Array<String> = buildByteTable()

    private fun buildByteTable(): Array<String> {
        val table = Array(256) { "" }
        val bs = mutableListOf<Int>()
        val cs = mutableListOf<Int>()
        // Printable Latin-1 ranges stay identity-mapped.
        for (b in '!'..'~') { bs.add(b.code); cs.add(b.code) }
        for (b in '¡'..'¬') { bs.add(b.code); cs.add(b.code) }
        for (b in '®'..'ÿ') { bs.add(b.code); cs.add(b.code) }
        // All other bytes map to U+0100 + n.
        var n = 0
        for (b in 0 until 256) {
            if (b !in bs) {
                bs.add(b); cs.add(256 + n); n++
            }
        }
        for ((b, c) in bs.zip(cs)) table[b] = c.toChar().toString()
        return table
    }

    private fun byteEncode(chunk: String): String =
        buildString {
            for (b in chunk.toByteArray(Charsets.UTF_8)) {
                append(byteToUnicode[b.toInt() and 0xFF])
            }
        }

    // --- BPE -----------------------------------------------------------------

    private fun bpe(word: Array<String>): List<String> {
        val parts = word.toMutableList()
        while (parts.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestIdx = -1
            for (i in 0 until parts.size - 1) {
                val r = mergeRanks[parts[i] to parts[i + 1]]
                if (r != null && r < bestRank) {
                    bestRank = r
                    bestIdx = i
                }
            }
            if (bestIdx < 0) break
            parts[bestIdx] = parts[bestIdx] + parts[bestIdx + 1]
            parts.removeAt(bestIdx + 1)
        }
        return parts
    }

    companion object {
        const val SOT = "<|startoftext|>"
        const val EOT = "<|endoftext|>"
        const val END_OF_WORD = "</w>"
        const val MAX_LEN = 77

        private val WS_REGEX = Regex("\\s+")

        /**
         * Loads vocab.json + merges.txt from assets ([assetDir], e.g.
         * "tokenizer"). Throws if the files are missing or malformed — call
         * only when the bundled tokenizer is known to exist.
         */
        fun fromAssets(context: Context, assetDir: String = "tokenizer"): ClipTokenizer {
            val mapSerializer = kotlinx.serialization.builtins.MapSerializer(
                kotlinx.serialization.builtins.serializer(),
                kotlinx.serialization.builtins.serializer(),
            )
            val vocab: Map<String, Int> = context.assets.open("$assetDir/clip-vocab.json").use { stream ->
                Json.decodeFromString(mapSerializer, stream.readBytes().decodeToString())
            }
            val ranks = HashMap<Pair<String, String>, Int>(50_000)
            context.assets.open("$assetDir/clip-merges.txt").use { stream ->
                BufferedReader(InputStreamReader(stream, Charsets.UTF_8)).useLines { lines ->
                    var first = true
                    for (line in lines) {
                        if (first) { // "#version: ..." header
                            first = false
                            continue
                        }
                        val parts = line.trim().split(' ')
                        if (parts.size == 2) ranks[parts[0] to parts[1]] = ranks.size
                        else if (parts.isNotEmpty() && parts[0].isNotEmpty()) {
                            // tolerate stray non-pair lines
                        }
                    }
                }
            }
            return ClipTokenizer(vocab, ranks)
        }
    }
}
