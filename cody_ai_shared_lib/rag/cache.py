"""Caching primitives for the RAG engine (§13.5).

Provides:
  - Cache Protocol (`get(key) -> bytes | None`, `set(key, value, ttl_seconds) -> None`)
  - InMemoryLRUCache: thread-safe, byte-bounded LRU cache (default 32 MiB)
  - make_embedding_cache_key: `sha256(model | task_type | normalized_text)`
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Protocol for byte-keyed caches (§13.5)."""

    def get(self, key: str) -> bytes | None:
        ...

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        ...


class InMemoryLRUCache:
    """Thread-safe, byte-bounded in-process LRU cache (§13.5)."""

    def __init__(self, max_bytes: int = 32 * 1024 * 1024) -> None:
        self.max_bytes = max(0, max_bytes)
        self._current_bytes = 0
        self._store: OrderedDict[str, tuple[bytes, float | None]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    def get(self, key: str) -> bytes | None:
        if self.max_bytes <= 0:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            val, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                self._current_bytes -= len(val) + len(key.encode("utf-8"))
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return val

    def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        if self.max_bytes <= 0:
            return
        entry_size = len(value) + len(key.encode("utf-8"))
        if entry_size > self.max_bytes:
            return
        expires_at = (time.monotonic() + ttl_seconds) if ttl_seconds is not None else None
        with self._lock:
            if key in self._store:
                old_val, _ = self._store.pop(key)
                self._current_bytes -= len(old_val) + len(key.encode("utf-8"))
            while self._store and (self._current_bytes + entry_size > self.max_bytes):
                evicted_key, (evicted_val, _) = self._store.popitem(last=False)
                self._current_bytes -= len(evicted_val) + len(evicted_key.encode("utf-8"))
            self._store[key] = (value, expires_at)
            self._current_bytes += entry_size

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._current_bytes = 0


def make_embedding_cache_key(model: str, task_type: str, normalized_text: str) -> str:
    """Compute query embedding cache key: sha256(model | task_type | normalized_text) (§13.5)."""
    raw = f"{model}|{task_type}|{normalized_text}"
    return "emb:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def serialize_vector(vec: list[float]) -> bytes:
    """Serialize a float vector to compact JSON bytes."""
    return json.dumps(vec, separators=(",", ":")).encode("utf-8")


def deserialize_vector(data: bytes) -> list[float]:
    """Deserialize compact JSON bytes back to a float vector."""
    loaded = json.loads(data.decode("utf-8"))
    return [float(x) for x in loaded]
