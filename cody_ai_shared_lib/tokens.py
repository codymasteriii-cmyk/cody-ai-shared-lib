"""Text token estimation shared by the model modules and the RAG engine (§6.1)."""
from __future__ import annotations

import math


def is_cjk_char(ch: str) -> bool:
    """Return True if character falls in CJK Unified Ideographs, Kana, Hangul, or related ranges."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF          # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF      # CJK Unified Ideographs Extension A
        or 0x20000 <= cp <= 0x2A6DF    # Extension B
        or 0x2A700 <= cp <= 0x2B73F    # Extension C
        or 0x2B740 <= cp <= 0x2B81F    # Extension D
        or 0x2B820 <= cp <= 0x2CEAF    # Extension E
        or 0xF900 <= cp <= 0xFAFF      # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F    # Compatibility Supplement
        or 0x3040 <= cp <= 0x309F      # Hiragana
        or 0x30A0 <= cp <= 0x30FF      # Katakana
        or 0xAC00 <= cp <= 0xD7AF      # Hangul Syllables
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    CJK characters count 1.0 token each; other characters count 0.25 each; then +10%.
    Returns an integer ceiling.
    """
    if not text:
        return 0
    cjk_count = 0
    other_count = 0
    for ch in text:
        if is_cjk_char(ch):
            cjk_count += 1
        else:
            other_count += 1
    raw = cjk_count * 1.0 + other_count * 0.25
    return math.ceil(raw * 1.10)
