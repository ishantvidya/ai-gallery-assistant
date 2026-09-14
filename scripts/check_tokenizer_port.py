"""Sanity-check: pure-Python reimplementation of CLIP byte-level BPE.

This mirrors exactly the algorithm we will port to Kotlin (ClipTokenizer.kt).
If it reproduces HF CLIPTokenizer ids for the golden texts, the Kotlin port is
safe to write.
"""

import json

VOCAB_PATH = "android/app/src/main/assets/tokenizer/clip-vocab.json"
MERGES_PATH = "android/app/src/main/assets/tokenizer/clip-merges.txt"

SOT = "<|startoftext|>"
EOT = "<|endoftext|>"
MAX_LEN = 77

try:
    import regex as re  # CLIP's pattern needs \p{L}/\p{N}
    PAT = re.compile(
        r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        re.IGNORECASE,
    )
except ImportError:
    import re
    PAT = re.compile(
        r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|[^\s\w]+|_+",
        re.IGNORECASE | re.UNICODE,
    )

def _bytes_to_unicode():
    """CLIP's reversible byte<->unicode table (printable bytes stay identity)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTES_TO_UNICODE = _bytes_to_unicode()


def byte_encode(text: str) -> str:
    return "".join(BYTES_TO_UNICODE[b] for b in text.encode("utf-8"))


def load_assets():
    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)
    merges = []
    with open(MERGES_PATH, encoding="utf-8") as f:
        for line in f.read().split("\n")[1:]:
            parts = line.split()
            if len(parts) == 2:
                merges.append(tuple(parts))
    ranks = {pair: i for i, pair in enumerate(merges)}
    return vocab, ranks


def bpe(token: str, ranks: dict) -> list:
    word = list(token)
    while len(word) > 1:
        pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
        best = min(pairs, key=lambda p: ranks.get(p, 1 << 30))
        if best not in ranks:
            break
        first, second = best
        merged, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged.append(first + second)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        word = merged
    return word


def tokenize(text: str, vocab: dict, ranks: dict) -> list:
    ids = [vocab[SOT]]
    for chunk in PAT.findall(text.lower()):
        mapped = byte_encode(chunk)
        chars = list(mapped)
        chars[-1] += "</w>"  # HF port: end_of_word_suffix on the last char
        for piece in bpe(chars, ranks):
            ids.append(vocab[piece])  # KeyError here = vocab gap: stop, don't guess
    ids.append(vocab[EOT])
    return ids[:MAX_LEN] + [vocab[EOT]] * (MAX_LEN - len(ids[:MAX_LEN]))


GOLDEN = {
    "a photo of a bed": [49406, 320, 1125, 539, 320, 2722, 49407],
    "human": [49406, 2751, 49407],
    "dog at the beach": [49406, 1929, 536, 518, 2117, 49407],
    "sunset over the mountains": [49406, 3424, 962, 518, 5873, 49407],
    "IMG_20240714_123456": [
        49406, 24157, 318, 273, 271, 273, 275, 271, 278, 272, 275, 318,
        272, 273, 274, 275, 276, 277, 49407,
    ],
}


def main():
    vocab, ranks = load_assets()
    ok = True
    for text, expected in GOLDEN.items():
        got = tokenize(text, vocab, ranks)
        exp = expected + [49407] * (MAX_LEN - len(expected))
        match = got == exp
        ok &= match
        print(("PASS" if match else "FAIL"), repr(text))
        if not match:
            print("  expected:", exp[: len(expected) + 4])
            print("  got     :", got[: len(expected) + 4])
    print("\nALL PASS" if ok else "\nMISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
