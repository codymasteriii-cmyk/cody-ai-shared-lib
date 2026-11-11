"""Model-agnostic LLM client with singleton SDK connections and cross-provider retry.

Shared across all Cody AI projects. Routes API calls by model name prefix:
  gemini-*  → Google GenAI SDK (google-genai)
  claude-*  → Anthropic Messages SDK (anthropic) — lazy import, only loaded
              if the caller actually routes to a Claude model.
  gpt-*     → OpenAI SDK (openai) — lazy import, routes to OpenAI models.

Design principles:
  - One LLMClient instance per project (singleton at module level in the
    project's own llm service layer). SDK connections are created lazily on
    first use and reused across all subsequent generate() calls.
  - Returns raw response.text only. All parsing (json.loads, fence stripping,
    Pydantic construction, field sanitisation) is the caller's responsibility.
  - Retry with exponential backoff is applied uniformly across providers for
    rate-limit (429) and transient server errors (5xx).
  - No config file dependency. All parameters are passed at instantiation time
    with sensible defaults, or overridden per-call via generate().

Usage tracking: generate_with_usage() returns real token counts and USD cost
calculated from cody_ai_shared_lib.pricing. finish_reason is normalized to
("stop", "length", "safety", "other").
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator

from .pricing import PRICE_TABLE_VERSION, compute_cost

logger = logging.getLogger("shared-llm")

__all__ = ["LLMClient", "LLMUsage", "LLMResult", "LLMStreamChunk", "ModelCallError"]


# ── Error classification & Exception Types ─────────────────────────────────────

class ModelCallError(Exception):
    """Raised when a model call fails or post-processing fails after a billed response (AC-RAG-70)."""

    def __init__(
        self,
        message: str,
        usage: LLMUsage | None = None,
        billing: str = "billed",
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.usage = usage
        self.billing = billing  # "billed" | "unknown"
        self.cause = cause


def _is_retryable(exc: Exception) -> bool:
    """Return True for rate-limit and transient server errors that warrant retry."""
    msg = str(exc).lower()
    return any(t in msg for t in (
        "429", "rate limit", "quota", "resource exhausted", "too many requests",
        "503", "service unavailable", "500", "internal server error",
        "overloaded",
    ))


# ── Usage tracking types ────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    """Token usage and cost accounting for one generate_with_usage() call.

    input_tokens:        Prompt tokens (system + user message).
    output_tokens:       Total generation spend (including thinking tokens where reported).
    cached_tokens:       Cached prompt tokens read (where reported by provider).
    cost_usd:            Calculated USD cost against pricing table, or None if unlisted.
    price_table_version: Version of pricing table used (e.g. '2026-11-08').
    estimated:           False when token counts are reported directly by provider.
    """
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float | None = None
    price_table_version: str | None = None
    estimated: bool = False


@dataclass
class LLMResult:
    """Return value of generate_with_usage(): text, token usage, cost, and finish reason."""
    text: str
    usage: LLMUsage
    finish_reason: str | None = None  # Normalized: "stop", "length", "safety", "other"


@dataclass
class LLMStreamChunk:
    """A streaming chunk from LLMClient.stream().

    delta: text generated in this chunk (empty on final metadata chunk).
    usage: filled only on the final chunk.
    finish_reason: filled only on the final chunk ("stop", "length", "safety", "other").
    """
    delta: str
    usage: LLMUsage | None = None
    finish_reason: str | None = None


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """Reusable, model-agnostic LLM client.

    Instantiate once at module/app startup. SDK connections (Gemini client,
    Anthropic client, OpenAI client) are created lazily on first use and shared
    across all subsequent generate() calls.
    """

    def __init__(
        self,
        retry_max: int = 3,
        retry_base_wait_secs: float = 2.0,
        sdk_timeout_ms: int = 60_000,
    ):
        self.retry_max = retry_max
        self.retry_base_wait_secs = retry_base_wait_secs
        self.sdk_timeout_ms = sdk_timeout_ms
        self._gemini_client = None
        self._anthropic_client = None
        self._openai_client = None

    # ── SDK singleton accessors ────────────────────────────────────────────────

    def _get_gemini_client(self):
        if self._gemini_client is None:
            from google import genai
            self._gemini_client = genai.Client(
                api_key=os.getenv("GOOGLE_API_KEY"),
                http_options={"timeout": self.sdk_timeout_ms},
            )
        return self._gemini_client

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=self.sdk_timeout_ms / 1000.0,
            )
        return self._openai_client

    # ── Finish reason normalization ────────────────────────────────────────────

    @staticmethod
    def _normalize_finish_reason(raw: Any) -> str | None:
        """Normalize provider-specific finish reason to 'stop', 'length', 'safety', or 'other'."""
        if raw is None:
            return None
        raw_str = str(raw).lower().strip()
        if "." in raw_str:
            raw_str = raw_str.split(".")[-1]

        if raw_str in ("stop", "end_turn", "stop_sequence"):
            return "stop"
        if raw_str in ("length", "max_tokens"):
            return "length"
        if raw_str in ("safety", "refusal", "content_filter", "recitation", "blocklist", "prohibited_content", "spii"):
            return "safety"
        return "other"

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None = None,
        # Gemini-specific / shared structured options
        response_mime_type: str | None = None,
        response_schema: type | None = None,
        disable_auto_func_calling: bool = False,
        thinking_level: str | None = None,
        # Claude/OpenAI-specific options
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
    ) -> str:
        """Call the LLM and return raw response text string.

        Routes by model name prefix:
          'gemini-*' → Google GenAI SDK
          'claude-*' → Anthropic Messages API
          'gpt-*'    → OpenAI API
        """
        return self.generate_with_usage(
            system_prompt, user_message, model, max_tokens,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            disable_auto_func_calling=disable_auto_func_calling,
            thinking_level=thinking_level,
            temperature=temperature,
            stop_sequences=stop_sequences,
            stream=stream,
        ).text

    def generate_with_usage(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: type | None = None,
        disable_auto_func_calling: bool = False,
        thinking_level: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        stream: bool = False,
        cache_breakpoints: bool | list[int] | None = None,
        user_blocks: list[tuple[str, bool]] | None = None,
    ) -> LLMResult:
        """Same call as generate(), but returns real token usage, cost, and finish reason.

        user_blocks: the user message as ordered (text, cacheable) blocks. Claude sends them as content
        blocks and marks the cacheable ones with cache_control; other providers use user_message.
        """
        if model.startswith("gemini"):
            return self._gemini_generate(
                system_prompt, user_message, model, max_tokens,
                response_mime_type, response_schema,
                disable_auto_func_calling, thinking_level,
                stream=stream,
                cache_breakpoints=cache_breakpoints,
            )
        if model.startswith("claude"):
            return self._claude_generate(
                system_prompt, user_message, model, max_tokens,
                temperature, stop_sequences,
                response_schema=response_schema,
                stream=stream,
                cache_breakpoints=cache_breakpoints,
                user_blocks=user_blocks,
            )
        if model.startswith("gpt"):
            return self._openai_generate(
                system_prompt, user_message, model, max_tokens,
                temperature, stop_sequences,
                response_mime_type=response_mime_type,
                response_schema=response_schema,
                stream=stream,
            )
        raise ValueError(
            f"Unknown model provider for: {model!r}. "
            "Expected prefix 'gemini-', 'claude-', or 'gpt-'."
        )

    def stream(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None = None,
        response_mime_type: str | None = None,
        thinking_level: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        cache_breakpoints: bool | list[int] | None = None,
        user_blocks: list[tuple[str, bool]] | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream chunks from the LLM.

        Yields LLMStreamChunk(delta=...). The final chunk has delta="" and
        contains usage and finish_reason.
        """
        if model.startswith("gemini"):
            yield from self._gemini_stream(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                max_tokens=max_tokens,
                response_mime_type=response_mime_type,
                thinking_level=thinking_level,
                cache_breakpoints=cache_breakpoints,
            )
            return
        if model.startswith("claude"):
            yield from self._claude_stream(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                cache_breakpoints=cache_breakpoints,
                user_blocks=user_blocks,
            )
            return
        if model.startswith("gpt"):
            yield from self._openai_stream(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                response_mime_type=response_mime_type,
            )
            return
        raise ValueError(
            f"Unknown model provider for: {model!r}. "
            "Expected prefix 'gemini-', 'claude-', or 'gpt-'."
        )

    # ── Provider implementations ───────────────────────────────────────────────

    def _gemini_generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        response_mime_type: str | None,
        response_schema: type | None,
        disable_auto_func_calling: bool,
        thinking_level: str | None = None,
        stream: bool = False,
        cache_breakpoints: bool | list[int] | None = None,
    ) -> LLMResult:
        from google.genai import types

        config_kwargs: dict = {"system_instruction": system_prompt}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
        if disable_auto_func_calling:
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(
                    disable=True, maximum_remote_calls=0
                )
            )
        if thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )

        client = self._get_gemini_client()
        last_exc = None

        for attempt in range(self.retry_max):
            try:
                if stream:
                    chunks = []
                    last_chunk = None
                    for chunk in client.models.generate_content_stream(
                        model=model,
                        contents=user_message,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ):
                        if chunk.text:
                            chunks.append(chunk.text)
                        last_chunk = chunk
                    response_text = "".join(chunks)
                    meta = last_chunk.usage_metadata if last_chunk else None
                    candidate = (
                        last_chunk.candidates[0]
                        if (last_chunk and getattr(last_chunk, "candidates", None) and len(last_chunk.candidates) > 0)
                        else None
                    )
                else:
                    response = client.models.generate_content(
                        model=model,
                        contents=user_message,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                    response_text = response.text or ""
                    meta = response.usage_metadata
                    candidate = (
                        response.candidates[0]
                        if (getattr(response, "candidates", None) and len(response.candidates) > 0)
                        else None
                    )

                logger.debug(f"[LLM] Gemini {model} succeeded (attempt {attempt + 1}).")

                visible_tokens = (meta.candidates_token_count if meta else 0) or 0
                thinking_tokens = (getattr(meta, "thoughts_token_count", 0) if meta else 0) or 0
                output_tokens = visible_tokens + thinking_tokens
                input_tokens = (meta.prompt_token_count if meta else 0) or 0
                cached_tokens = (getattr(meta, "cached_content_token_count", 0) if meta else 0) or 0

                raw_finish = getattr(candidate, "finish_reason", None) if candidate else None
                finish_reason = self._normalize_finish_reason(raw_finish)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )

                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )

                if response_schema is not None:
                    try:
                        response_schema.model_validate_json(response_text)
                    except Exception as parse_exc:
                        raise ModelCallError(
                            f"[LLM] Gemini {model} structured output parse failure: {parse_exc}",
                            usage=usage,
                            billing="billed",
                            cause=parse_exc,
                        )

                return LLMResult(text=response_text, usage=usage, finish_reason=finish_reason)

            except ModelCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] Gemini {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"[LLM] Gemini {model} failed after {self.retry_max} attempts: {last_exc}"
        )

    def _claude_generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        temperature: float | None,
        stop_sequences: list[str] | None,
        response_schema: type | None = None,
        stream: bool = False,
        cache_breakpoints: bool | list[int] | None = None,
        user_blocks: list[tuple[str, bool]] | None = None,
    ) -> LLMResult:
        client = self._get_anthropic_client()

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens if max_tokens is not None else 8192,
            "messages": [{"role": "user", "content": _claude_user_content(user_message, user_blocks)}],
        }
        if cache_breakpoints:
            create_kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            create_kwargs["system"] = system_prompt
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences

        last_exc = None

        for attempt in range(self.retry_max):
            try:
                if stream:
                    with client.messages.stream(**create_kwargs) as s:
                        response_text = s.get_final_text()
                        final_msg = s.get_final_message()
                    usage_obj = final_msg.usage
                    raw_stop = getattr(final_msg, "stop_reason", None)
                else:
                    response = client.messages.create(**create_kwargs)
                    # Support ThinkingBlock followed by TextBlock (e.g. claude-opus-5-5)
                    response_text = "".join(b.text for b in response.content if hasattr(b, "text"))
                    usage_obj = response.usage
                    raw_stop = getattr(response, "stop_reason", None)

                logger.debug(f"[LLM] Claude {model} succeeded (attempt {attempt + 1}).")

                uncached_tokens = getattr(usage_obj, "input_tokens", 0) or 0
                cached_tokens = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
                input_tokens = uncached_tokens + cached_tokens
                output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
                finish_reason = self._normalize_finish_reason(raw_stop)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )

                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )

                if response_schema is not None:
                    try:
                        response_schema.model_validate_json(response_text)
                    except Exception as parse_exc:
                        raise ModelCallError(
                            f"[LLM] Claude {model} structured output parse failure: {parse_exc}",
                            usage=usage,
                            billing="billed",
                            cause=parse_exc,
                        )

                return LLMResult(text=response_text, usage=usage, finish_reason=finish_reason)

            except ModelCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] Claude {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"[LLM] Claude {model} failed after {self.retry_max} attempts: {last_exc}"
        )

    def _openai_generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        temperature: float | None,
        stop_sequences: list[str] | None,
        response_mime_type: str | None = None,
        response_schema: type | None = None,
        stream: bool = False,
    ) -> LLMResult:
        client = self._get_openai_client()
        # Reasoning models (gpt-6 series, o1, o3) enforce max_completion_tokens and reject temperature
        is_reasoning = any(x in model.lower() for x in ("gpt-6", "o1", "o3"))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        create_kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if is_reasoning:
            if max_tokens is not None:
                create_kwargs["max_completion_tokens"] = max_tokens
        else:
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                create_kwargs["temperature"] = temperature

        if stop_sequences:
            create_kwargs["stop"] = stop_sequences

        if response_mime_type == "application/json" and response_schema is None:
            create_kwargs["response_format"] = {"type": "json_object"}

        last_exc = None

        for attempt in range(self.retry_max):
            try:
                if stream:
                    chunks = []
                    raw_reason = None
                    final_usage = None
                    response_stream = client.chat.completions.create(
                        stream=True,
                        stream_options={"include_usage": True},
                        **create_kwargs,
                    )
                    for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            chunks.append(chunk.choices[0].delta.content)
                        if chunk.choices and chunk.choices[0].finish_reason:
                            raw_reason = chunk.choices[0].finish_reason
                        if getattr(chunk, "usage", None):
                            final_usage = chunk.usage
                    response_text = "".join(chunks)
                    usage_obj = final_usage
                else:
                    response = client.chat.completions.create(**create_kwargs)
                    choice = response.choices[0]
                    response_text = choice.message.content or ""
                    raw_reason = choice.finish_reason
                    usage_obj = response.usage

                logger.debug(f"[LLM] OpenAI {model} succeeded (attempt {attempt + 1}).")

                input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
                prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                finish_reason = self._normalize_finish_reason(raw_reason)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )

                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )

                if response_schema is not None:
                    try:
                        response_schema.model_validate_json(response_text)
                    except Exception as parse_exc:
                        raise ModelCallError(
                            f"[LLM] OpenAI {model} structured output parse failure: {parse_exc}",
                            usage=usage,
                            billing="billed",
                            cause=parse_exc,
                        )

                return LLMResult(text=response_text, usage=usage, finish_reason=finish_reason)

            except ModelCallError:
                raise
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] OpenAI {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. "
                        f"Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"[LLM] OpenAI {model} failed after {self.retry_max} attempts: {last_exc}"
        )

    def _gemini_stream(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        response_mime_type: str | None = None,
        thinking_level: str | None = None,
        cache_breakpoints: bool | list[int] | None = None,
    ) -> Iterator[LLMStreamChunk]:
        from google.genai import types

        config_kwargs: dict = {"system_instruction": system_prompt}
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type
        if thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        client = self._get_gemini_client()
        last_exc = None

        for attempt in range(self.retry_max):
            try:
                stream_iter = client.models.generate_content_stream(
                    model=model,
                    contents=user_message,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                last_chunk = None
                for chunk in stream_iter:
                    if chunk.text:
                        yield LLMStreamChunk(delta=chunk.text)
                    last_chunk = chunk

                meta = last_chunk.usage_metadata if last_chunk else None
                candidate = (
                    last_chunk.candidates[0]
                    if (last_chunk and getattr(last_chunk, "candidates", None) and len(last_chunk.candidates) > 0)
                    else None
                )
                visible_tokens = (meta.candidates_token_count if meta else 0) or 0
                thinking_tokens = (getattr(meta, "thoughts_token_count", 0) if meta else 0) or 0
                output_tokens = visible_tokens + thinking_tokens
                input_tokens = (meta.prompt_token_count if meta else 0) or 0
                cached_tokens = (getattr(meta, "cached_content_token_count", 0) if meta else 0) or 0

                raw_finish = getattr(candidate, "finish_reason", None) if candidate else None
                finish_reason = self._normalize_finish_reason(raw_finish)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )
                yield LLMStreamChunk(delta="", usage=usage, finish_reason=finish_reason)
                return

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] Gemini stream {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. Retrying..."
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"[LLM] Gemini stream {model} failed: {last_exc}")

    def _claude_stream(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        cache_breakpoints: bool | list[int] | None = None,
        user_blocks: list[tuple[str, bool]] | None = None,
    ) -> Iterator[LLMStreamChunk]:
        client = self._get_anthropic_client()

        create_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens if max_tokens is not None else 8192,
            "messages": [{"role": "user", "content": _claude_user_content(user_message, user_blocks)}],
        }
        if cache_breakpoints:
            create_kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            create_kwargs["system"] = system_prompt

        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences

        last_exc = None
        for attempt in range(self.retry_max):
            try:
                with client.messages.stream(**create_kwargs) as s:
                    for text in s.text_stream:
                        yield LLMStreamChunk(delta=text)
                    final_msg = s.get_final_message()

                usage_obj = final_msg.usage
                raw_stop = getattr(final_msg, "stop_reason", None)
                uncached_tokens = getattr(usage_obj, "input_tokens", 0) or 0
                cached_tokens = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
                input_tokens = uncached_tokens + cached_tokens
                output_tokens = getattr(usage_obj, "output_tokens", 0) or 0
                finish_reason = self._normalize_finish_reason(raw_stop)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )
                yield LLMStreamChunk(delta="", usage=usage, finish_reason=finish_reason)
                return

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] Claude stream {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. Retrying..."
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"[LLM] Claude stream {model} failed: {last_exc}")

    def _openai_stream(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        response_mime_type: str | None = None,
    ) -> Iterator[LLMStreamChunk]:
        client = self._get_openai_client()
        is_reasoning = any(x in model.lower() for x in ("gpt-6", "o1", "o3"))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        create_kwargs: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if is_reasoning:
            if max_tokens is not None:
                create_kwargs["max_completion_tokens"] = max_tokens
        else:
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                create_kwargs["temperature"] = temperature

        if stop_sequences:
            create_kwargs["stop"] = stop_sequences
        if response_mime_type == "application/json":
            create_kwargs["response_format"] = {"type": "json_object"}

        last_exc = None
        for attempt in range(self.retry_max):
            try:
                response_stream = client.chat.completions.create(**create_kwargs)
                raw_reason = None
                final_usage = None
                for chunk in response_stream:
                    if getattr(chunk, "usage", None):
                        final_usage = chunk.usage
                    if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
                        c = chunk.choices[0]
                        if getattr(c, "delta", None) and getattr(c.delta, "content", None):
                            yield LLMStreamChunk(delta=c.delta.content)
                        if getattr(c, "finish_reason", None):
                            raw_reason = c.finish_reason

                input_tokens = getattr(final_usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(final_usage, "completion_tokens", 0) or 0
                prompt_details = getattr(final_usage, "prompt_tokens_details", None)
                cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                finish_reason = self._normalize_finish_reason(raw_reason)

                cost_usd = compute_cost(
                    model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
                usage = LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    cost_usd=cost_usd,
                    price_table_version=PRICE_TABLE_VERSION,
                    estimated=False,
                )
                yield LLMStreamChunk(delta="", usage=usage, finish_reason=finish_reason)
                return

            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc) and attempt < self.retry_max - 1:
                    wait = self.retry_base_wait_secs * (2 ** attempt)
                    logger.warning(
                        f"[LLM] OpenAI stream {model} retryable error "
                        f"(attempt {attempt + 1}/{self.retry_max}): {exc}. Retrying..."
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError(f"[LLM] OpenAI stream {model} failed: {last_exc}")


def _claude_user_content(user_message: str, user_blocks: list[tuple[str, bool]] | None) -> Any:
    """The user turn for Claude: plain text, or content blocks with the cacheable ones marked.

    The RAG engine puts the stable data (Tier 0 sections, admitted summaries) first, so marking that
    block lets Claude reuse it across turns instead of re-reading the whole document (§10.3, §15).
    Falls back to the plain message when the blocks do not add up to it.
    """
    if not user_blocks or "".join(text for text, _ in user_blocks) != user_message:
        return user_message
    content = []
    for text, cacheable in user_blocks:
        if not text:
            continue
        block: dict[str, Any] = {"type": "text", "text": text}
        if cacheable:
            block["cache_control"] = {"type": "ephemeral"}
        content.append(block)
    return content or user_message
