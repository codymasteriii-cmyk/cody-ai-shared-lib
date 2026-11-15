"""Runtime determinism controls, Clock, Rng, and Deadlines (§13.4, RAG-EVAL-4).

Everything in the engine that depends on the outside world goes through one of these small objects, so a test or a
replay can control it:

    Clock        what time is it?            SystemClock (real)      FrozenClock (fixed; advance() moves it)
    Rng          give me random bytes        SystemRng (secure)      DeterministicRng (seeded: same output every run)
    Deadline     how long may this request take, and has the caller cancelled?
    CancelToken  the caller's "stop now" flag, set from any thread
    RequestRun   the bundle of per-request state, so the engine itself stays stateless (D-03)

Nothing in the library calls `datetime.now()` or `random` directly for logic that affects results; that is what
makes "last week" and fence ids reproducible in an evaluation.
"""
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
    """Time source protocol allowing injected fixed clocks for deterministic evaluation.

    `now_utc` is calendar time (for date questions); `monotonic` is a steadily increasing counter (for measuring
    durations, unaffected by the system clock being adjusted).
    """

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
    """Fixed clock for tests and evaluation replays (AC-RAG-17, AC-RAG-35, AC-RAG-52).

    Time stands still until `advance()` is called, which moves both the calendar time and the monotonic counter.

    >>> clock = FrozenClock(datetime(2026, 1, 1, 12, 0))
    >>> clock.now_utc().isoformat()
    '2026-01-01T12:00:00+00:00'
    >>> clock.advance(90)
    >>> clock.now_utc().isoformat(), clock.monotonic()
    ('2026-01-01T12:01:30+00:00', 90.0)
    """

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
        """Move time forward, e.g. to test that a deadline expires."""
        from datetime import timedelta
        self._dt += timedelta(seconds=seconds)
        self._mono += seconds


@runtime_checkable
class Rng(Protocol):
    """Random source protocol for fence ids and sampling.

    Used mainly for the unpredictable fence ids of the prompt (§10.4): `SystemRng` must be used in production, because a
    predictable id would let a document guess its own closing tag.
    """

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
    """Seeded pseudorandom generator for deterministic evaluation.

    Not secure: for tests and replays only, never for production fence ids.

    >>> DeterministicRng(7).hex(4) == DeterministicRng(7).hex(4)
    True
    """

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

    >>> token = CancelToken()
    >>> token.is_cancelled
    False
    >>> token.cancel("user left")
    >>> token.is_cancelled, token.reason
    (True, 'user left')
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
    """Tracks remaining execution time budget (§13.4) and carries the request's CancelToken.

    Two different questions are asked of it. `check(stage)` is for a REQUIRED stage: no time left is an error
    (`DeadlineExceeded`). `fits(seconds)` is for an OPTIONAL stage: no time left means skip it and record a
    degradation (DG-10). Both honour a cancel.

    >>> clock = FrozenClock(datetime(2026, 1, 1))
    >>> deadline = Deadline(10.0, clock=clock)
    >>> clock.advance(4)
    >>> deadline.remaining_seconds()
    6.0
    >>> deadline.fits(5.0), deadline.fits(7.0)
    (True, False)
    >>> clock.advance(7)
    >>> deadline.is_expired()
    True
    """

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
        """Seconds since the request began, by the monotonic clock."""
        return self.clock.monotonic() - self.start_mono

    def remaining_seconds(self) -> float:
        """Seconds left in the budget, never negative."""
        rem = self.deadline_seconds - self.elapsed_seconds()
        return max(0.0, rem)

    def is_expired(self) -> bool:
        """True when the budget is used up."""
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
    """Create a bounded thread pool for independent concurrent operations.

    Used for the parallel legs of a search and the map step of a list answer. It is bounded so a burst of requests
    cannot start unlimited threads; code that already runs on a pool worker must not wait on the same pool (deadlock).
    """
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rag-pool")


@dataclass
class RequestRun:
    """Everything scoped to one request; the engine itself holds no per-request state (D-03).

    Created by `RAGEngine._begin()`; each field collects what happens during the request (spend, degradations, the
    deadline, the trace). Because it is a local object, two requests running at once cannot see each other's data.
    """
    request_id: str
    usage: Any
    degradations: Any
    deadline: Deadline
    trace: Any
