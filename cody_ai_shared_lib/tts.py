"""Edge-TTS Speech Synthesizer — Neural speech synthesis and text preprocessing.

Shared across Cody AI projects (MyCompanion, AudioReading, Video generation).
Provides Microsoft Edge neural voice synthesis with zero API cost, markdown/symbol
cleaning, language auto-detection, and local disk cache management.

Usage:
    from cody_ai_shared_lib.tts import (
        synthesize_speech,
        clean_text_for_tts,
        detect_language,
        POPULAR_VOICES,
    )

    # Clean raw markdown/document text before synthesis
    clean_text = clean_text_for_tts("# Header\nThis is **bold** text ~!")

    # Synthesize speech to a cached mp3 file
    audio_path = await synthesize_speech(clean_text, voice="en-US-JennyNeural")
"""

import hashlib
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional
import edge_tts

logger = logging.getLogger("cody-ai-shared-tts")

DEFAULT_CACHE_DIR = os.path.join("data", "tts_cache")

DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"

POPULAR_VOICES: List[Dict[str, str]] = [
    {
        "id": "zh-CN-XiaoxiaoNeural",
        "name": "晓晓 (女性 - 温暖温柔)",
        "lang": "zh-CN",
        "gender": "female",
        "persona": "zh-female",
    },
    {
        "id": "zh-CN-YunxiNeural",
        "name": "云希 (男性 - 自然沉稳)",
        "lang": "zh-CN",
        "gender": "male",
        "persona": "zh-male",
    },
    {
        "id": "en-US-JennyNeural",
        "name": "Jenny (Female - English)",
        "lang": "en-US",
        "gender": "female",
        "persona": "en-female",
    },
    {
        "id": "en-US-GuyNeural",
        "name": "Guy (Male - English)",
        "lang": "en-US",
        "gender": "male",
        "persona": "en-male",
    },
]


def is_mostly_english(text: str) -> bool:
    """Detect whether text is primarily English based on Latin vs Chinese character frequency."""
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return ascii_letters > (chinese_chars * 2)


def detect_language(text: str) -> str:
    """Return 'en' if text is predominantly English, otherwise 'zh'."""
    return "en" if is_mostly_english(text) else "zh"


def clean_text_for_tts(text: str) -> str:
    """Preprocess text for TTS by stripping Markdown, URLs, roleplay asterisks, and emojis.

    Ensures the speech engine reads natural prose without pronouncing symbols like
    'hashtag', 'asterisk', 'bracket', or tildes aloud.
    """
    if not text:
        return ""

    t = text

    # Remove Markdown code blocks ```code``` and inline code `code`
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"`[^`]*`", "", t)

    # Remove Markdown images ![alt](url) and extract text from links [text](url) -> text
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)

    # Remove Markdown headings (# Heading), blockquotes (> quote), and horizontal rules
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^>\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^[-*_]{3,}\s*$", "", t, flags=re.MULTILINE)

    # Strip common conversational roleplay actions (e.g. *sigh*, *smiles*, *laughs*, *action*)
    action_pattern = r"\*(?:sigh|smiles?|laughs?|giggles?|waves?|chuckles?|nods?|pauses?|snickers?|gasps?|whispers?|coughs?|action|groans?|shrugs?|blushes?)\*"
    t = re.sub(action_pattern, "", t, flags=re.IGNORECASE)

    # Remove Markdown bold syntax (**bold**, __bold__) and italics (*italic*, _italic_)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"_([^_]+)_", r"\1", t)

    # Remove tildes (~ or ～)
    t = t.replace("~", "").replace("～", "")

    # Remove standalone URLs
    t = re.sub(r"https?://\S+", "", t)

    # Strip emojis and symbols (Unicode categories: So=Other Symbol, Sk=Modifier Symbol, Cn/Cs)
    cleaned_chars = [
        c for c in t
        if unicodedata.category(c) not in ("So", "Sk", "Cn", "Cs")
        and c not in ("\ufe0f", "\u200d")
    ]
    t = "".join(cleaned_chars)

    # Normalize multiple whitespace characters and remove space before punctuation marks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def get_audio_hash(
    text: str,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> str:
    """Compute a deterministic MD5 hash for cached audio files."""
    key = f"{text}_{voice}_{rate}_{pitch}_{volume}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def prune_tts_cache(cache_dir: str = DEFAULT_CACHE_DIR, max_files: int = 100) -> None:
    """Keep local disk audio cache strictly bounded by removing oldest files."""
    try:
        if not os.path.exists(cache_dir):
            return
        files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".mp3")
        ]
        if len(files) > max_files:
            files.sort(key=lambda x: os.path.getmtime(x))
            excess = len(files) - max_files
            for f in files[:excess]:
                try:
                    os.remove(f)
                except OSError:
                    pass
    except Exception as e:
        logger.warning(f"[TTS Cache] Pruning warning: {e}")


async def synthesize_speech(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
    cache_dir: str = DEFAULT_CACHE_DIR,
    voice_override: Optional[str] = None,
) -> str:
    """Synthesize speech using edge-tts with prosody modulation (pitch, rate, volume).

    Returns:
        The relative or absolute file path to the cached MP3 audio file.
    """
    actual_voice = voice_override or voice or DEFAULT_VOICE
    tts_text = clean_text_for_tts(text)
    if not tts_text:
        return ""

    actual_rate = rate or "+0%"
    actual_pitch = pitch or "+0Hz"
    actual_volume = volume or "+0%"

    os.makedirs(cache_dir, exist_ok=True)
    audio_hash = get_audio_hash(tts_text, actual_voice, actual_rate, actual_pitch, actual_volume)
    filename = f"{audio_hash}.mp3"
    filepath = os.path.join(cache_dir, filename)

    if not os.path.exists(filepath) or os.path.getsize(filepath) < 100:
        last_err = None
        for _ in range(2):
            try:
                communicate = edge_tts.Communicate(
                    text=tts_text,
                    voice=actual_voice,
                    rate=actual_rate,
                    pitch=actual_pitch,
                    volume=actual_volume,
                )
                await communicate.save(filepath)
                if os.path.exists(filepath) and os.path.getsize(filepath) >= 100:
                    last_err = None
                    prune_tts_cache(cache_dir=cache_dir, max_files=100)
                    break
            except Exception as e:
                last_err = e
                actual_voice = DEFAULT_VOICE

        if last_err and (not os.path.exists(filepath) or os.path.getsize(filepath) < 100):
            logger.error(f"[TTS] synthesize_speech failed: {last_err}")
            return ""

    return filepath


async def synthesize_chunk_bytes(
    text: str,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    volume: str = "+0%",
) -> bytes:
    """Synthesize speech directly into raw audio bytes (suitable for in-memory streaming)."""
    tts_text = clean_text_for_tts(text)
    if not tts_text:
        return b""

    actual_rate = rate or "+0%"
    actual_pitch = pitch or "+0Hz"
    actual_volume = volume or "+0%"

    communicate = edge_tts.Communicate(
        text=tts_text,
        voice=voice,
        rate=actual_rate,
        pitch=actual_pitch,
        volume=actual_volume,
    )

    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])

    return b"".join(audio_chunks)


def get_available_voices() -> List[Dict[str, str]]:
    """Return list of popular/supported voice personas."""
    return POPULAR_VOICES
