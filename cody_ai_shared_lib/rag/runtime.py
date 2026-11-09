"""Runtime determinism controls, Clock, Rng, and Deadlines (§13.4, RAG-EVAL-4)."""
from __future__ import annotations

import random
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence, TypeVar, runtime_checkable

from .errors import DeadlineExceeded

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


class Deadline:
    """Tracks remaining execution time budget (§13.4)."""

    def __init__(self, deadline_seconds: float, clock: Clock | None = None):
        self.deadline_seconds = deadline_seconds
        self.clock = clock or SystemClock()
        self.start_mono = self.clock.monotonic()

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
        """Raise DeadlineExceeded when no time is left for a required stage."""
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
