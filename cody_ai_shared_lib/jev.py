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
  - Noul   — a yes/no judgment, the probability the answer is yes.
  - Choice — one option from a fixed set you define.
  - Score  — a rating on an ordered scale you define.
The SDK returns a different object for each (a Noul answer has only `.noul`; a Choice or
Score answer has `.choice` / `.score` and a `.confidence`). JevClient.ask() wraps every one
in a `JevAnswer(kind, value, probability, raw)`, so callers read answers the same way
whatever they asked and never guess attribute names (1.2.17):

    result = client.ask(state, {"scope": Choice(criteria={...}), "rewrite": Noul(instructions="...")})
    result.answers["scope"].value, result.answers["scope"].probability      # the choice, Jev's confidence in it
    result.answers["rewrite"].value, result.answers["rewrite"].probability  # a bool, and P(yes)

Noul/Choice/Score (the QUESTION types) are typesafe_sdk's own classes, re-exported here,
since callers need to construct them to build a `questions` dict; JevClient supplies the
answer wrapping and the retry/error-handling/usage-tracking value this library adds on top.

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
from typing import Any, Literal

from typesafe_sdk import Choice, ChoiceAnswer, Noul, Score, ScoreAnswer, TypeSafeClient  # questions re-exported for callers
from typesafe_sdk import NoulAnswer as _SdkNoulAnswer

from .pricing import PRICE_TABLE_VERSION, compute_cost

__all__ = ["Choice", "Noul", "Score", "JevClient", "JevAnswer", "JevResult", "JevUsage"]

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
class JevUsage:
    """Usage accounting for a Jev call."""
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None
    price_table_version: str | None = None
    estimated: bool = False


@dataclass
class JevAnswer:
    """One question's answer, in one shape for every kind of question.

    kind:         "noul" (yes/no), "choice" (one of your options) or "score" (a rating on your scale).
    value:        the answer itself: a bool for noul (True when the probability of yes is at least 0.5), the
                  chosen option for choice, the rating for score.
    probability:  noul: the calibrated P(yes), in [0, 1]. A yes/no probability already encodes its own
                  certainty: 0.98 or 0.02 is a confident answer, 0.50 is not, so distance from 0.5 is the
                  confidence signal. choice and score: Jev's confidence in `value`, in [0, 1] (see the
                  calibration caveat in this module's docstring).
    raw:          the SDK's own answer object, for anything not covered above (for example a choice's
                  per-option probabilities).

    The meaning of `probability` differs by `kind`, so read `kind` when it matters.
    """
    kind: Literal["noul", "choice", "score"]
    value: Any
    probability: float
    raw: Any = None


def _wrap(key: str, answer: Any) -> JevAnswer:
    """Turn the SDK's answer for one question into a JevAnswer. An answer of an unknown type is an error, never a guess.

    >>> answer = _wrap("rewrite", _SdkNoulAnswer(noul=0.2))
    >>> answer.kind, answer.value, answer.probability
    ('noul', False, 0.2)
    >>> sdk = ChoiceAnswer(choice="document_overview", confidence=0.9, probabilities={"document_overview": 0.9})
    >>> _wrap("scope", sdk).value, _wrap("scope", sdk).probability
    ('document_overview', 0.9)
    """
    if isinstance(answer, _SdkNoulAnswer):
        probability = float(answer.noul)
        return JevAnswer("noul", probability >= 0.5, probability, answer)
    if isinstance(answer, ChoiceAnswer):
        return JevAnswer("choice", answer.choice, float(answer.confidence), answer)
    if isinstance(answer, ScoreAnswer):
        return JevAnswer("score", answer.score, float(answer.confidence), answer)
    raise TypeError(f"Jev answer for question {key!r} has an unknown type: {type(answer).__name__}")


@dataclass
class JevResult:
    """Return value of JevClient.ask(): every answered question as a JevAnswer, alongside call metadata
    and cost tracking. A question Jev did not answer is absent from `answers`."""
    answers: dict[str, JevAnswer]
    tokens_in: int
    tokens_out: int
    ms: float
    model: str
    cost_usd: float | None = None
    price_table_version: str | None = None
    estimated: bool = False
    usage: JevUsage | None = None


class JevClient:
    """Reusable Jev client. Instantiate once per project (module-level singleton,
    mirroring LLMClient in llm.py); the underlying SDK client is created lazily on
    first use.

    Reads TYPESAFE_API_KEY (or VERCEL_API_KEY) and TYPESAFE_BASE_URL from the
    environment by default; override per-instance via constructor arguments.
    """

    def __init__(
        self,
        retry_max: int = 3,
        retry_base_wait_secs: float = 1.0,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ):
        self.retry_max = retry_max
        self.retry_base_wait_secs = retry_base_wait_secs
        self._api_key = api_key
        self._base_url = base_url
        # Per-HTTP-request timeout in seconds handed to the SDK. None keeps the SDK's own
        # default; callers on a latency-sensitive path (e.g. a batch job) should set one.
        self._timeout = timeout
        self._client = None

    def _get_client(self) -> TypeSafeClient:
        """The SDK client, created on first use; raises RuntimeError with a clear message when no API key is set."""
        if self._client is None:
            api_key = self._api_key or os.getenv("TYPESAFE_API_KEY") or os.getenv("VERCEL_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "TYPESAFE_API_KEY (or VERCEL_API_KEY) is not set. Jev calls cannot be made without it "
                    "— there is no mock fallback in this shared client; a caller that "
                    "wants a mock/offline mode should implement that itself (see the "
                    "benchmark project's jev_client.py for a worked example of a mock "
                    "path that was useful there but is deliberately not duplicated "
                    "here, since silently fabricating decisions is the wrong default "
                    "for a shared, production-facing client)."
                )
            base_url = self._base_url or os.getenv("TYPESAFE_BASE_URL", _DEFAULT_BASE_URL)
            self._client = TypeSafeClient(api_key=api_key, base_url=base_url, timeout=self._timeout)
        return self._client

    @staticmethod
    def _extract_usage(resp) -> tuple[int, int]:
        """Pulls (tokens_in, tokens_out) off a Jev response, trying both
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

    def ask(self, state: str, questions: dict[str, Any], model: str = JEV_MODEL) -> JevResult:
        """One Jev call: answers every question in `questions` against `state` in a
        single parallel pass. `questions` is {key: Noul(...) | Choice(...) | Score(...)}.
        Returns a JevResult whose `answers` maps each answered key to a JevAnswer.

        Retries on rate-limit and transient server errors with exponential backoff
        (retry_base_wait_secs * 2^attempt), matching LLMClient's retry behaviour.

        Raises:
            RuntimeError: TYPESAFE_API_KEY is unset, or all retry attempts exhausted.
            TypeError:    Jev returned an answer of a type this client does not know.
        """
        client = self._get_client()
        last_exc = None

        for attempt in range(self.retry_max):
            t0 = time.perf_counter()
            try:
                resp = client.system_one(state=state, model=model, questions=questions)
                ms = (time.perf_counter() - t0) * 1000
                tokens_in, tokens_out = self._extract_usage(resp)
                answers = {
                    key: _wrap(key, sdk_answer)
                    for key in questions
                    if (sdk_answer := resp.answers.get(key)) is not None
                }
                logger.debug(f"[Jev] {model} succeeded (attempt {attempt + 1}), {ms:.0f}ms.")
                cost_usd = compute_cost("jev", input_tokens=tokens_in, output_tokens=tokens_out)
                usage = JevUsage(
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )
                return JevResult(
                    answers=answers,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    ms=ms,
                    model=model,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                    usage=usage,
                )

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

    def ask_noul_batch(
        self,
        state: str,
        questions: dict[str, str],
        model: str = JEV_MODEL,
        strict: bool = False,
    ) -> dict[str, JevAnswer]:
        """Convenience wrapper for the common case: a batch of yes/no questions, each a
        plain instruction string, all answered as Noul (probability-of-yes) in one call.

        questions: {question_key: instruction_text}. Each instruction_text can be
        reused verbatim from an existing structured-output schema's Field description
        if one already states the same yes/no judgment for an LLM — that is exactly
        how this is meant to be used as a cheap second opinion alongside an existing
        LLM classifier, not a replacement for one (see this module's calibration
        caveat, and the benchmark writeup this client was ported from, for why Jev is
        positioned as a secondary signal rather than a primary classifier here).

        By default a question Jev returned no answer for is reported as probability 0.0,
        which is indistinguishable from a confident "no". Pass strict=True to raise
        ValueError naming the unanswered questions instead — use it wherever a silently
        missing answer would corrupt a stored decision.

        Returns {question_key: JevAnswer}, each with kind "noul": `.probability` is P(yes) and `.value`
        is True when it is at least 0.5.
        """
        result = self.ask(
            state=state,
            questions={key: Noul(instructions=instruction) for key, instruction in questions.items()},
            model=model,
        )
        if strict:
            missing = [key for key in questions if key not in result.answers]
            if missing:
                raise ValueError(f"Jev returned no answer for: {', '.join(missing)}")
        return {key: result.answers.get(key) or JevAnswer("noul", False, 0.0) for key in questions}

    def list_models(self):
        """Thin passthrough to the SDK's model listing (client.models.list()). Exists
        for callers that want to record which model actually served a run — see this
        module's docstring on why Jev cannot be version-pinned through the gateway.
        Returns whatever the SDK returns; this method applies no interpretation."""
        return self._get_client().models.list()
