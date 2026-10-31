"""Edge-TTS Speech Synthesizer — Neural speech synthesis, cue extraction, and cache.

Shared across Cody AI projects (MyCompanion, AudioReading, Video generation).
Provides Microsoft Edge neural voice synthesis with zero API cost, markdown/symbol
cleaning, word-boundary timestamp extraction, language auto-detection, and local disk LRU cache.

Usage:
    from cody_ai_shared_lib.tts import (
        synthesize,
        align_cues,
        clean_text_for_tts,
        default_voice,
        VOICES,
        TTSCache,
    )

    # Clean text
    clean_text = clean_text_for_tts("Hello **world**!")

    # Synthesize with word-boundary cues
    result = await synthesize(clean_text, voice="en-US-GuyNeural", with_cues=True)
    # result.audio: bytes (MP3)
    # result.cues: list of {text, start, end, char_start, char_end}
    # result.duration: float (seconds)
"""

import asyncio
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import edge_tts

# Public API of this module. The package __init__ re-exports exactly these names, so it never leaks
# helper imports (asyncio, os, re, ...) into the package namespace.
__all__ = [
    "DEFAULT_CACHE_DIR",
    "SynthResult",
    "TTSCache",
    "TTSError",
    "UnknownVoiceError",
    "VOICES",
    "VOICE_MAP",
    "align_cues",
    "clean_text_for_tts",
    "default_voice",
    "detect_language",
    "is_mostly_english",
    "synthesize",
    "verify_voices_available",
]

logger = logging.getLogger("cody-ai-shared-tts")

DEFAULT_CACHE_DIR = os.path.join("data", "tts_cache")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TTSError(Exception):
    """Base exception for all TTS synthesis errors."""
    pass


class UnknownVoiceError(TTSError):
    """Raised when an unrecognized voice ID is requested."""
    pass



# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynthResult:
    """Result of a speech synthesis operation."""
    audio: bytes
    cues: Optional[List[Dict[str, Any]]]
    duration: float


# ---------------------------------------------------------------------------
# Voice Catalog & Registry
# ---------------------------------------------------------------------------

VOICES: List[Dict[str, Any]] = [
    {
        "id": "en-US-GuyNeural",
        "lang": "en-US",
        "gender": "male",
        "label": "Guy",
    },
    {
        "id": "en-US-JennyNeural",
        "lang": "en-US",
        "gender": "female",
        "label": "Jenny",
    },
    {
        "id": "zh-CN-XiaoxiaoNeural",
        "lang": "zh-CN",
        "gender": "female",
        "label": "晓晓 (Xiaoxiao)",
    },
    {
        "id": "zh-CN-YunxiNeural",
        "lang": "zh-CN",
        "gender": "male",
        "label": "云希 (Yunxi)",
    },
    {
        "id": "zh-CN-XiaoyiNeural",
        "lang": "zh-CN",
        "gender": "female",
        "label": "晓伊 (Xiaoyi)",
    },
    {
        "id": "zh-CN-YunjianNeural",
        "lang": "zh-CN",
        "gender": "male",
        "label": "云健 (Yunjian)",
    },
    {
        "id": "zh-CN-YunyangNeural",
        "lang": "zh-CN",
        "gender": "male",
        "label": "云扬 (Yunyang)",
    },
    {
        "id": "en-US-ChristopherNeural",
        "lang": "en-US",
        "gender": "male",
        "label": "Christopher",
    },
    {
        "id": "en-US-AndrewNeural",
        "lang": "en-US",
        "gender": "male",
        "label": "Andrew",
    },
    {
        "id": "en-US-AriaNeural",
        "lang": "en-US",
        "gender": "female",
        "label": "Aria",
    },
    {
        "id": "en-US-AvaNeural",
        "lang": "en-US",
        "gender": "female",
        "label": "Ava",
    },
]

VOICE_MAP: Dict[str, Dict[str, Any]] = {v["id"]: v for v in VOICES}

def default_voice(lang: str, enabled: Optional[List[str]] = None) -> str:
    """Return the default voice ID for the given language.

    If `enabled` list is provided, returns the first enabled voice matching the language.
    Otherwise returns the catalog default: 'en-US-GuyNeural' for English, 'zh-CN-XiaoxiaoNeural' for Chinese.
    """
    is_en = lang.lower().startswith("en")
    target_prefix = "en" if is_en else "zh"

    if enabled:
        for vid in enabled:
            v_info = VOICE_MAP.get(vid)
            if v_info and v_info["lang"].lower().startswith(target_prefix):
                return vid
            elif not v_info and vid.lower().startswith(target_prefix):
                return vid

    return "en-US-GuyNeural" if is_en else "zh-CN-XiaoxiaoNeural"


# ---------------------------------------------------------------------------
# Language & Text Cleaning
# ---------------------------------------------------------------------------


def is_mostly_english(text: str) -> bool:
    """Detect whether text is primarily English based on Latin vs Chinese character frequency."""
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return ascii_letters > (chinese_chars * 2)


def detect_language(text: str) -> str:
    """Return 'en' if text is predominantly English, otherwise 'zh'."""
    return "en" if is_mostly_english(text) else "zh"


# Symbols in the otherwise-stripped categories that carry meaning when read aloud ("25℃").
_SPOKEN_SYMBOLS = frozenset("°℃℉")


def clean_text_for_tts(text: str, *, asterisk_spans: str = "drop") -> str:
    """Preprocess text for TTS following the strict 5-stage pipeline (§7.7).

    Rules:
    1. Remove fenced and inline code, images; replace [text](url) with text;
       remove heading, blockquote and horizontal-rule markers; remove bare URLs.
    2. Unwrap **bold** and __bold__ to their text.
    3. Handle single-asterisk spans by mode:
       - 'drop' (default): delete the span including words (roleplay actions).
       - 'keep_text': unwrap and keep words.
       Non-spans like '5 * 3 and 2 * 4' are preserved untouched.
    4. Unwrap _italic_ only when word-boundary flanked; replace remaining
       underscores between letters/digits with a space (my_var_name -> my var name).
    5. Remove ~, ～, emoji and symbol categories (So, Sk, Cn, Cs, U+FE0F, U+200D) except the
       measurement signs in _SPOKEN_SYMBOLS (°, ℃, ℉) that change the meaning of a number;
       collapse whitespace; remove space before punctuation marks (. , ! ? ; :).
    """
    if not text:
        return ""

    t = text

    # 1. Code blocks, inline code, images, links, markdown structural markers, bare URLs
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"`[^`]*`", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^>\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-*_]{3,}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"https?://\S+", "", t)

    # 2. Unwrap bold
    t = re.sub(r"\*\*([^*]+?)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+?)__", r"\1", t)

    # 3. Handle single-asterisk spans
    # Opening * must be followed by a non-space, closing * preceded by non-space
    asterisk_pattern = r"\*(?!\s)([^*]+?)(?<!\s)\*"
    if asterisk_spans == "drop":
        t = re.sub(asterisk_pattern, "", t)
    elif asterisk_spans == "keep_text":
        t = re.sub(asterisk_pattern, r"\1", t)
    else:
        raise ValueError(f"Invalid asterisk_spans mode: {asterisk_spans}")

    # 4. Unwrap _italic_ when word-boundary flanked; replace remaining underscores
    t = re.sub(r"(?:\b|(?<=\s))_([^_]+?)_(?:\b|(?=\s|[.,!?;:]))", r"\1", t)
    t = re.sub(r"(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])", " ", t)
    t = t.replace("_", " ")

    # 5. Remove ~, ～, emoji/symbol categories, collapse whitespace, space before punctuation
    t = t.replace("~", "").replace("～", "")
    cleaned_chars = [
        c for c in t
        if (c in _SPOKEN_SYMBOLS or unicodedata.category(c) not in ("So", "Sk", "Cn", "Cs"))
        and c not in ("\ufe0f", "\u200d")
    ]
    t = "".join(cleaned_chars)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Cue Extraction and Alignment (§7.2-C)
# ---------------------------------------------------------------------------


def align_cues(text: str, boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Align raw WordBoundary events from Edge-TTS to UTF-16 code-unit offsets into text.

    Pure function.
    - Forward-only scan into text with up to 32 characters skipped (punctuation/whitespace/markup).
    - Converts character offsets into UTF-16 code units (for JavaScript consumer).
    - Enforces monotonic media time (start <= end).
    - Degrades to [] if >10% of tokens missed or cues end before 70% of text.
    """
    if not text or not boundaries:
        return []

    cues: List[Dict[str, Any]] = []
    cursor = 0
    miss_count = 0
    text_len_cp = len(text)
    text_utf16_len = len(text.encode("utf-16-le")) // 2

    def is_skippable(skipped: str) -> bool:
        if len(skipped) > 32:
            return False
        for c in skipped:
            if c.isspace():
                continue
            cat = unicodedata.category(c)
            if cat.startswith("P"):  # Punctuation
                continue
            if c in '<>*_~`"\'[](){}#=-+$/\\':
                continue
            return False
        return True

    for b in boundaries:
        token = b.get("text", "").strip()
        if not token:
            continue

        raw_offset = b.get("offset", 0)
        raw_dur = b.get("duration", 0)
        start = round(raw_offset / 1e7, 3)
        end = round((raw_offset + raw_dur) / 1e7, 3)
        if end < start:
            end = start

        # 1. Exact match forward search
        idx = text.find(token, cursor)
        matched = False

        if idx != -1:
            skipped = text[cursor:idx]
            if is_skippable(skipped):
                matched = True

        # 2. Try NFKC normalized search if not matched
        if not matched:
            norm_token = unicodedata.normalize("NFKC", token)
            search_window = text[cursor:min(text_len_cp, cursor + len(token) + 40)]
            norm_window = unicodedata.normalize("NFKC", search_window)
            w_idx = norm_window.find(norm_token)
            if w_idx != -1:
                skipped = search_window[:w_idx]
                if is_skippable(skipped):
                    idx = cursor + w_idx
                    matched = True

        if not matched:
            miss_count += 1
            continue

        char_start_cp = idx
        char_end_cp = idx + len(token)
        cursor = char_end_cp

        char_start_u16 = len(text[:char_start_cp].encode("utf-16-le")) // 2
        char_end_u16 = len(text[:char_end_cp].encode("utf-16-le")) // 2

        # Monotonic time check
        if cues:
            prev_end = cues[-1]["end"]
            if start < prev_end:
                start = prev_end
            if end < start:
                end = start

        cues.append({
            "text": token,
            "start": start,
            "end": end,
            "char_start": char_start_u16,
            "char_end": char_end_u16,
        })

    # Degrade criteria (§7.2-C: >10% misses or <70% coverage -> return [])
    total = len(boundaries)
    if total == 0 or (miss_count / total) > 0.10:
        return []

    if not cues or cues[-1]["char_end"] < int(0.70 * text_utf16_len):
        return []

    return cues


# ---------------------------------------------------------------------------
# TTS Disk LRU Cache (§7.4)
# ---------------------------------------------------------------------------


class TTSCache:
    """Disk-backed LRU cache for TTS MP3 audio and JSON word-boundary cues.

    Key: sha256(text + "\x1f" + voice).
    Storage: {key}.mp3 and {key}.json in cache_dir.
    Writes: Atomic (temp file then rename).
    Eviction: Byte-bounded LRU based on file mtime.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_bytes: int = 200 * 1024 * 1024,
    ):
        self.cache_dir = cache_dir
        self.max_bytes = max_bytes

    @staticmethod
    def get_key(text: str, voice: str) -> str:
        """Compute deterministic SHA-256 cache key."""
        payload = f"{text}\x1f{voice}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Tuple[bytes, Optional[List[Dict[str, Any]]], float]]:
        """Return (audio_bytes, cues, duration) if cached, else None."""
        mp3_path = os.path.join(self.cache_dir, f"{key}.mp3")
        json_path = os.path.join(self.cache_dir, f"{key}.json")

        if not (os.path.exists(mp3_path) and os.path.exists(json_path)):
            return None

        try:
            with open(mp3_path, "rb") as f:
                audio = f.read()
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Update mtime on access for LRU
            now = time.time()
            try:
                os.utime(mp3_path, (now, now))
                os.utime(json_path, (now, now))
            except OSError:
                pass

            return audio, meta.get("cues"), float(meta.get("duration", 0.0))
        except Exception as e:
            logger.warning(f"[TTSCache] Read error for {key}: {e}")
            return None

    def put(
        self,
        key: str,
        audio: bytes,
        cues: Optional[List[Dict[str, Any]]],
        duration: float,
        voice: str = "",
    ) -> None:
        """Atomically persist {key}.mp3 and {key}.json, then prune if over budget."""
        if not audio:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        mp3_path = os.path.join(self.cache_dir, f"{key}.mp3")
        json_path = os.path.join(self.cache_dir, f"{key}.json")
        pid = os.getpid()
        mp3_tmp = os.path.join(self.cache_dir, f"{key}.mp3.tmp.{pid}")
        json_tmp = os.path.join(self.cache_dir, f"{key}.json.tmp.{pid}")

        try:
            with open(mp3_tmp, "wb") as f:
                f.write(audio)
            with open(json_tmp, "w", encoding="utf-8") as f:
                json.dump({"duration": duration, "cues": cues, "voice": voice}, f)

            os.replace(mp3_tmp, mp3_path)
            os.replace(json_tmp, json_path)
            self.prune()
        except Exception as e:
            logger.warning(f"[TTSCache] Write error for {key}: {e}")
            for tmp in (mp3_tmp, json_tmp):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def prune(self) -> None:
        """Prune cache directory to maintain total size <= max_bytes."""
        if not os.path.exists(self.cache_dir):
            return

        try:
            entries: List[Tuple[str, float, int]] = []
            total_bytes = 0
            for fname in os.listdir(self.cache_dir):
                if fname.endswith(".tmp") or ".tmp." in fname:
                    continue
                fpath = os.path.join(self.cache_dir, fname)
                try:
                    stat = os.stat(fpath)
                    entries.append((fpath, stat.st_mtime, stat.st_size))
                    total_bytes += stat.st_size
                except OSError:
                    continue

            if total_bytes <= self.max_bytes:
                return

            # Sort oldest first by mtime
            entries.sort(key=lambda x: x[1])
            for fpath, _, fsize in entries:
                if total_bytes <= self.max_bytes:
                    break
                try:
                    os.remove(fpath)
                    total_bytes -= fsize
                except OSError:
                    pass
        except Exception as e:
            logger.warning(f"[TTSCache] Prune warning: {e}")


# ---------------------------------------------------------------------------
# Core Synthesis Function (§7.7)
# ---------------------------------------------------------------------------


async def synthesize(
    text: str,
    voice: str,
    *,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    with_cues: bool = False,
) -> SynthResult:
    """The single entry point for Edge-TTS synthesis.

    - Calls edge_tts.Communicate with boundary="WordBoundary" if with_cues=True.
    - At most one retry on transient connection errors, with the SAME voice.
    - Never substitutes another voice on failure.
    - Returns SynthResult(audio: bytes, cues: list | None, duration: float).
    """
    if not text or not text.strip():
        raise TTSError("Cannot synthesize empty text")

    # Validate voice ID pattern
    if voice not in VOICE_MAP and not re.match(r"^[a-z]{2,3}-[A-Z]{2}-.+Neural$", voice):
        raise UnknownVoiceError(f"Unknown voice ID: {voice}")

    actual_rate = rate or "+0%"
    actual_pitch = pitch or "+0Hz"
    actual_volume = volume or "+0%"
    boundary_mode = "WordBoundary" if with_cues else "SentenceBoundary"

    last_err: Optional[Exception] = None

    for attempt in range(2):
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=actual_rate,
                pitch=actual_pitch,
                volume=actual_volume,
                boundary=boundary_mode,
            )

            audio_chunks: List[bytes] = []
            boundaries: List[Dict[str, Any]] = []

            async for chunk in communicate.stream():
                chunk_type = chunk.get("type")
                if chunk_type == "audio":
                    audio_chunks.append(chunk["data"])
                elif chunk_type == "WordBoundary":
                    boundaries.append(chunk)

            audio_bytes = b"".join(audio_chunks)
            if not audio_bytes:
                raise TTSError("Edge-TTS returned empty audio")

            # Calculate duration
            if boundaries:
                max_tick = max(b.get("offset", 0) + b.get("duration", 0) for b in boundaries)
                duration = round(max_tick / 1e7, 3)
            else:
                # Estimate duration from 48kbps MP3 data (~6000 bytes/sec)
                duration = round(len(audio_bytes) / 6000.0, 3)

            cues = align_cues(text, boundaries) if with_cues else None
            return SynthResult(audio=audio_bytes, cues=cues, duration=duration)

        except Exception as e:
            last_err = e
            # Only retry once on transient connection issues
            if attempt == 0:
                logger.warning(f"[TTS] Transient error on attempt 1 for voice {voice}: {e}. Retrying...")
                await asyncio.sleep(0.2)
                continue
            break

    raise TTSError(f"TTS synthesis failed for voice {voice}: {last_err}") from last_err


# ---------------------------------------------------------------------------
# Voice Drift Check (§7.1-E)
# ---------------------------------------------------------------------------


async def verify_voices_available(ids: List[str]) -> List[str]:
    """Check whether given voice IDs are present and GA in Edge-TTS.

    Returns a list of warning messages for any voice ID that is missing or not GA.
    """
    warnings: List[str] = []
    try:
        voices = await edge_tts.list_voices()
        voice_dict = {v["ShortName"]: v for v in voices}
        for voice_id in ids:
            v_info = voice_dict.get(voice_id)
            if not v_info:
                warnings.append(f"Voice {voice_id} is missing from Edge-TTS voice list")
            elif v_info.get("Status") != "GA":
                warnings.append(f"Voice {voice_id} has status '{v_info.get('Status')}', not GA")
    except Exception as e:
        warnings.append(f"Failed to check Edge-TTS voice list: {e}")
    return warnings
