"""Runtime determinism controls, Clock, Rng, and Deadlines (§13.4, RAG-EVAL-4)."""
from __future__ import annotations

import random
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from .errors import DeadlineExceeded, RequestCancelled

T = TypeVar("T")


@runtime_checkable
class Clock(Protocol):
    """Time source protocol allowing injected fixed clocks for deterministic evaluation."""

    def now_utc(self) -> datetime:
        ...

    def monotonic(self) -> float:
        ...


class SystemClock:
    """Standard system wall-clock and monotonic time source."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


class FrozenClock:
    """Fixed clock for tests and evaluation replays (AC-RAG-17, AC-RAG-35, AC-RAG-52)."""

    def __init__(self, fixed_dt: datetime, fixed_monotonic: float = 0.0):
        if fixed_dt.tzinfo is None:
            fixed_dt = fixed_dt.replace(tzinfo=timezone.utc)
        self._dt = fixed_dt
        self._mono = fixed_monotonic

    def now_utc(self) -> datetime:
        return self._dt

    def monotonic(self) -> float:
        return self._mono

    def advance(self, seconds: float) -> None:
        from datetime import timedelta
        self._dt += timedelta(seconds=seconds)
        self._mono += seconds


@runtime_checkable
class Rng(Protocol):
    """Random source protocol for fence ids and sampling."""

    def hex(self, nbytes: int) -> str:
        ...

    def choice(self, seq: Sequence[T]) -> T:
        ...


class SystemRng:
    """Standard cryptographically strong system random source."""

    def hex(self, nbytes: int = 8) -> str:
        return secrets.token_hex(nbytes)

    def choice(self, seq: Sequence[T]) -> T:
        return secrets.choice(seq)


class DeterministicRng:
    """Seeded pseudorandom generator for deterministic evaluation."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def hex(self, nbytes: int = 8) -> str:
        # Generate nbytes bytes deterministically
        raw_bytes = bytes(self._rng.randint(0, 255) for _ in range(nbytes))
        return raw_bytes.hex()

    def choice(self, seq: Sequence[T]) -> T:
        return self._rng.choice(seq)


class CancelToken:
    """A thread-safe "stop now" flag the app hands to a request and sets from any thread (§13.4).

    The engine looks at it at its stage boundaries: around routing, before each query embedding and
    post-stage, between relaxation steps, before generation and between streamed chunks. It cannot
    interrupt a provider call that is already in flight, so a request stops at the next boundary.
    One token covers one request; do not reuse it.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Ask the request to stop. Safe to call from any thread, and more than once."""
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


class Deadline:
    """Tracks remaining execution time budget (§13.4) and carries the request's CancelToken."""

    def __init__(self, deadline_seconds: float, clock: Clock | None = None, cancel: CancelToken | None = None):
        self.deadline_seconds = deadline_seconds
        self.clock = clock or SystemClock()
        self.start_mono = self.clock.monotonic()
        self.cancel = cancel

    def raise_if_cancelled(self, stage: str) -> None:
        """Raise RequestCancelled when the caller has set the request's CancelToken."""
        if self.cancel is not None and self.cancel.is_cancelled:
            reason = f": {self.cancel.reason}" if self.cancel.reason else ""
            raise RequestCancelled(f"Request cancelled before stage '{stage}'{reason}", stage=stage)

    def elapsed_seconds(self) -> float:
        return self.clock.monotonic() - self.start_mono

    def remaining_seconds(self) -> float:
        rem = self.deadline_seconds - self.elapsed_seconds()
        return max(0.0, rem)

    def is_expired(self) -> bool:
        return self.remaining_seconds() <= 0.0

    def fits(self, stage_budget_seconds: float) -> bool:
        """True when an optional stage of this cost still fits in the remaining time."""
        return self.remaining_seconds() >= stage_budget_seconds

    def check(self, stage: str) -> None:
        """Raise RequestCancelled when cancelled, else DeadlineExceeded when no time is left for a required stage."""
        self.raise_if_cancelled(stage)
        if self.is_expired():
            raise DeadlineExceeded(
                f"Deadline of {self.deadline_seconds:.1f}s exceeded before required stage '{stage}'",
                stage=stage,
            )


def make_bounded_pool(max_workers: int = 4) -> ThreadPoolExecutor:
    """Create a bounded thread pool for independent concurrent operations."""
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-pool")


@dataclass
class RequestRun:
    """Everything scoped to one request; the engine itself holds no per-request state (D-03)."""
    request_id: str
    usage: Any
    degradations: Any
    deadline: Deadline
    trace: Any
