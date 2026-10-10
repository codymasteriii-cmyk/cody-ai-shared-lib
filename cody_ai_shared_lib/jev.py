"""Jev client — TypeSafe AI's "System One" decision model, via typesafe-sdk.

Shared across all Cody AI projects. Ported from a benchmark project's jev_client.py
(see the Jev benchmark writeup for the accuracy/latency/cost comparison against Claude
and Gemini that motivated using Jev as a cheap, fast secondary signal rather than a
primary classifier). That original client was written against an async FastAPI app
that ran Jev concurrently alongside Claude/Gemini "columns" — the asyncio wrapping
existed for that app's concurrency, not because Jev itself needs it. This port is
synchronous, matching the rest of this library's style (see llm.py) and every known
caller's own calling pattern (a sequential per-article loop), with no asyncio
dependency.

What Jev is: a non-autoregressive model that returns typed, calibrated decisions
instead of generated text. You send it `state` (the thing to be judged, e.g. an
article's title and body) and a set of typed `questions`; it answers every question in
one parallel pass, each answer governed by the question's spec:
  - Noul   — a yes/no judgment, returned as `.noul`, the probability the answer is yes.
  - Choice — one option from a fixed set you define, returned as `.choice`.
  - Score  — a rating on an ordered scale you define, returned as `.score`.
Every answer also carries `.confidence` (how sure Jev is — see the module docstring
note on calibration below).

Noul/Choice/Score are typesafe_sdk's own classes, re-exported here rather than wrapped,
since callers need to construct them to build a `questions` dict; JevClient supplies
the retry/error-handling/usage-tracking value this library adds on top.

Calibration caveat (worth knowing before gating any decision on `.confidence`): Jev is
trained with "Reinforcement Learning for Calibrated Decisions" (RLCD), so across a
GROUP of similarly-scored answers, the stated probability tends to track real accuracy.
A single answer's confidence can still be wrong, and published third-party evaluations
found Jev notably overconfident in the 0.7-0.9 confidence band specifically — do not
treat a single high-confidence answer as ground truth; use confidence to rank which
answers are worth a second look, not to silently prefer Jev's answer over a comparator.

Vercel AI Gateway routing: TypeSafe's direct signup can be waitlisted. Routing through
the Vercel AI Gateway (TYPESAFE_BASE_URL) is a working alternative if you have a Vercel
API key, but two things follow from going through a gateway instead of TypeSafe
directly:
  - The model cannot be version-pinned. A pinned id like "jev-1.13.0" is a TypeSafe
    direct-API id and 404s on the gateway; use the unversioned "typesafe-ai/jev"
    (equivalently "jev" — the gateway auto-prefixes the provider namespace).
  - Jev's decisions are stochastic run to run even at a fixed model id (confidence
    jitters in the second decimal between identical calls), and the served model can
    change without notice since there is nothing to pin against. list_models() below
    exists so a caller that cares can record what was actually served and compare runs
    — this library does not track drift itself, callers decide what "changed" means
    for their own use case.
"""
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient  # re-exported for callers

__all__ = ["Choice", "Noul", "Score", "JevClient", "JevResult", "NoulAnswer"]

logger = logging.getLogger("shared-jev")

JEV_MODEL = os.getenv("JEV_MODEL", "typesafe-ai/jev")
_DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/typesafe"


def _is_retryable(exc: Exception) -> bool:
    """Return True for rate-limit and transient server errors that warrant retry.

    Same heuristic as llm.py's _is_retryable, kept as a separate copy rather than a
    shared import — this module is deliberately standalone (Jev is not an LLM, and a
    caller should be able to use one of these two clients without the other).
    """
    msg = str(exc).lower()
    return any(t in msg for t in (
        "429", "rate limit", "quota", "resource exhausted", "too many requests",
        "503", "service unavailable", "500", "internal server error",
        "overloaded",
    ))


@dataclass
class JevResult:
    """Return value of JevClient.system_one(): every question's raw SDK answer object
    (each carries the type-specific value — .noul / .choice / .score — plus
    .confidence), alongside call metadata."""
    answers: dict[str, Any]
    tokens_in: int
    tokens_out: int
    ms: float
    model: str


@dataclass
class NoulAnswer:
    """One question's result from ask_noul_batch().

    probability is the only signal Noul provides (typesafe_sdk's NoulAnswer has no
    separate .confidence field — checked directly against the installed SDK's wire
    schema; unlike Choice/Score, a yes/no probability already encodes its own
    certainty: 0.98 or 0.02 is confident, 0.50 is not). Treat distance from 0.5 as the
    confidence signal for a Noul answer — do not expect a separate confidence value to
    ever be populated here.
    """
    probability: float          # .noul — calibrated P(yes), in [0, 1]


class JevClient:
    """Reusable Jev client. Instantiate once per project (module-level singleton,
    mirroring LLMClient in llm.py); the underlying SDK client is created lazily on
    first use.

    Reads TYPESAFE_API_KEY and TYPESAFE_BASE_URL from the environment by default;
    override per-instance via the api_key/base_url constructor arguments (e.g. a
    project that needs to call two different TypeSafe accounts/gateways).
    """

    def __init__(
        self,
        retry_max: int = 3,
        retry_base_wait_secs: float = 1.0,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.retry_max = retry_max
        self.retry_base_wait_secs = retry_base_wait_secs
        self._api_key = api_key
        self._base_url = base_url
        self._client = None

    def _get_client(self) -> TypeSafeClient:
        if self._client is None:
            api_key = self._api_key or os.getenv("TYPESAFE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "TYPESAFE_API_KEY is not set. Jev calls cannot be made without it "
                    "— there is no mock fallback in this shared client; a caller that "
                    "wants a mock/offline mode should implement that itself (see the "
                    "benchmark project's jev_client.py for a worked example of a mock "
                    "path that was useful there but is deliberately not duplicated "
                    "here, since silently fabricating decisions is the wrong default "
                    "for a shared, production-facing client)."
                )
            base_url = self._base_url or os.getenv("TYPESAFE_BASE_URL", _DEFAULT_BASE_URL)
            self._client = TypeSafeClient(api_key=api_key, base_url=base_url)
        return self._client

    @staticmethod
    def _extract_usage(resp) -> tuple[int, int]:
        """Pulls (tokens_in, tokens_out) off a system_one() response, trying both
        attribute and dict-style access — the exact shape of `.usage` is not fully
        documented, so this degrades to (0, 0) rather than raising if it's absent or
        shaped differently than expected."""
        usage = getattr(resp, "usage", None) or {}
        tokens_in = getattr(usage, "input_tokens", None)
        if tokens_in is None and isinstance(usage, dict):
            tokens_in = usage.get("input_tokens", 0)
        tokens_out = getattr(usage, "output_tokens", None)
        if tokens_out is None and isinstance(usage, dict):
            tokens_out = usage.get("output_tokens", 0)
        return tokens_in or 0, tokens_out or 0

    def system_one(self, state: str, questions: dict[str, Any], model: str = JEV_MODEL) -> JevResult:
        """One Jev call: answers every question in `questions` against `state` in a
        single parallel pass. `questions` is {key: Noul(...) | Choice(...) | Score(...)}.

        Retries on rate-limit and transient server errors with exponential backoff
        (retry_base_wait_secs * 2^attempt), matching LLMClient's retry behaviour.

        Raises:
            RuntimeError: TYPESAFE_API_KEY is unset, or all retry attempts exhausted.
        """
        client = self._get_client()
        last_exc = None

        for attempt in range(self.retry_max):
            t0 = time.perf_counter()
            try:
                resp = client.system_one(state=state, model=model, questions=questions)
                ms = (time.perf_counter() - t0) * 1000
                tokens_in, tokens_out = self._extract_usage(resp)
                answers = {key: resp.answers.get(key) for key in questions}
                logger.debug(f"[Jev] {model} succeeded (attempt {attempt + 1}), {ms:.0f}ms.")
                return JevResult(answers=answers, tokens_in=tokens_in, tokens_out=tokens_out, ms=ms, model=model)

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[Jev] {model} retryable error (attempt {attempt + 1}/{self.retry_max}): "
                        f"{exc}. Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"[Jev] {model} failed after {self.retry_max} attempts: {last_exc}")

    def ask_noul_batch(self, state: str, questions: dict[str, str], model: str = JEV_MODEL) -> dict[str, NoulAnswer]:
        """Convenience wrapper for the common case: a batch of yes/no questions, each a
        plain instruction string, all answered as Noul (probability-of-yes) in one call.

        questions: {question_key: instruction_text}. Each instruction_text can be
        reused verbatim from an existing structured-output schema's Field description
        if one already states the same yes/no judgment for an LLM — that is exactly
        how this is meant to be used as a cheap second opinion alongside an existing
        LLM classifier, not a replacement for one (see this module's calibration
        caveat, and the benchmark writeup this client was ported from, for why Jev is
        positioned as a secondary signal rather than a primary classifier here).

        Returns {question_key: NoulAnswer(probability, confidence)}.
        """
        result = self.system_one(
            state=state,
            questions={key: Noul(instructions=instruction) for key, instruction in questions.items()},
            model=model,
        )
        return {
            key: NoulAnswer(probability=getattr(answer, "noul", 0.0) if answer else 0.0)
            for key, answer in result.answers.items()
        }

    def list_models(self):
        """Thin passthrough to the SDK's model listing (client.models.list()). Exists
        for callers that want to record which model actually served a run — see this
        module's docstring on why Jev cannot be version-pinned through the gateway.
        Returns whatever the SDK returns; this method applies no interpretation."""
        return self._get_client().models.list()
