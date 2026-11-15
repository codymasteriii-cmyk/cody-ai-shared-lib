"""Text token estimation shared by the model modules and the RAG engine (§6.1).

The library budgets prompts in tokens but does not load a tokenizer (it would be slow to start and tied to one model's
vocabulary). Instead it estimates: roughly four Latin characters per token, and about one token per Chinese, Japanese or Korean
character, plus a 10% safety margin. The estimate errs high on purpose, so a prompt built to fit the budget really does.
"""
from __future__ import annotations

import math


# The one list of CJK code point ranges; every CJK check in the library derives from it.
CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x4E00, 0x9FFF),      # CJK Unified Ideographs
    (0x3400, 0x4DBF),      # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),    # Extension B
    (0x2A700, 0x2B73F),    # Extension C
    (0x2B740, 0x2B81F),    # Extension D
    (0x2B820, 0x2CEAF),    # Extension E
    (0xF900, 0xFAFF),      # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),    # Compatibility Supplement
    (0x3040, 0x309F),      # Hiragana
    (0x30A0, 0x30FF),      # Katakana
    (0xAC00, 0xD7AF),      # Hangul Syllables
)

# The same ranges as the inside of a regular-expression character class, e.g. for `[^\W_{CJK_CHAR_CLASS}]`.
CJK_CHAR_CLASS = "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_RANGES)


def is_cjk_char(ch: str) -> bool:
    """Return True if character falls in CJK Unified Ideographs, Kana, Hangul, or related ranges.

    >>> is_cjk_char("你"), is_cjk_char("a")
    (True, False)
    """
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    CJK characters count 1.0 token each; other characters count 0.25 each; then +10%.
    Returns an integer ceiling.

    >>> estimate_tokens("abcd"), estimate_tokens("你好"), estimate_tokens("")
    (2, 3, 0)
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
