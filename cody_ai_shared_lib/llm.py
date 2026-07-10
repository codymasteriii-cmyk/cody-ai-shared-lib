"""Model-agnostic LLM client with singleton SDK connections and cross-provider retry.

Shared across all Cody AI projects. Routes API calls by model name prefix:
  gemini-*  → Google GenAI SDK (google-generativeai)
  claude-*  → Anthropic Messages SDK (anthropic) — lazy import, only loaded
              if the caller actually routes to a Claude model.

Design principles:
  - One LLMClient instance per project (singleton at module level in the
    project's own llm service layer). SDK connections are created lazily on
    first use and reused across all subsequent generate() calls.
  - Returns raw response.text only. All parsing (json.loads, fence stripping,
    Pydantic construction, field sanitisation) is the caller's responsibility.
  - Retry with exponential backoff is applied uniformly to both providers for
    rate-limit (429) and transient server errors (5xx).
  - No config file dependency. All parameters are passed at instantiation time
    with sensible defaults, or overridden per-call via generate().

Usage tracking: generate_with_usage() returns real token counts from the
provider response instead of a char-count estimate. output_tokens means the
same thing for both providers (total generation cost) — see LLMUsage
docstring for how each provider's raw response maps onto it. generate() is
unchanged (still returns a bare str) so existing callers across all projects
are unaffected.
"""
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger("shared-llm")


# ── Usage tracking types ────────────────────────────────────────────────────────

@dataclass
class LLMUsage:
    """Token usage for one generate_with_usage() call.

    input_tokens:  Prompt tokens (system + user message), both providers.
    output_tokens: Total generation spend, both providers — the number that
                    matters for cost. Claude reports this as a single combined
                    count already (thinking included when thinking was used).
                    Gemini reports visible-text and thinking tokens separately
                    (`candidates_token_count` + `thoughts_token_count`); this
                    field is their sum, folded in at the call site in
                    _gemini_generate() so callers never need to know the two
                    providers report differently.
    """
    input_tokens: int
    output_tokens: int


@dataclass
class LLMResult:
    """Return value of generate_with_usage(): text plus real token counts."""
    text: str
    usage: LLMUsage

# ── Error classification ───────────────────────────────────────────────────────

def _is_retryable(exc: Exception) -> bool:
    """Return True for rate-limit and transient server errors that warrant retry."""
    msg = str(exc).lower()
    return any(t in msg for t in (
        "429", "rate limit", "quota", "resource exhausted", "too many requests",
        "503", "service unavailable", "500", "internal server error",
        "overloaded",
    ))


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    """Reusable, model-agnostic LLM client.

    Instantiate once at module/app startup. SDK connections (Gemini client,
    Anthropic client) are created lazily on first use and shared across all
    subsequent generate() calls, avoiding repeated authentication overhead.

    Example (project-level singleton)::

        from cody_ai_shared_lib.llm import LLMClient
        _client = LLMClient()               # uses defaults
        raw = _client.generate(sys, user, "gemini-3-flash-preview")
    """

    def __init__(
        self,
        retry_max: int = 3,
        retry_base_wait_secs: float = 2.0,
        sdk_timeout_ms: int = 60_000,
    ):
        """Create an LLMClient.

        Args:
            retry_max:            Maximum attempts per generate() call on
                                  rate-limit or transient errors. Applies to
                                  both Gemini and Claude.
            retry_base_wait_secs: Exponential backoff base in seconds.
                                  Wait = base * 2^attempt → 2s, 4s, 8s, …
            sdk_timeout_ms:       HTTP timeout for the Gemini SDK (milliseconds).
                                  The Anthropic SDK manages its own timeout.
        """
        self.retry_max = retry_max
        self.retry_base_wait_secs = retry_base_wait_secs
        self.sdk_timeout_ms = sdk_timeout_ms
        self._gemini_client = None
        self._anthropic_client = None

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
            # Reads ANTHROPIC_API_KEY from environment automatically.
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        model: str,
        max_tokens: int | None = None,
        # ── Gemini-specific ───────────────────────────────────────────────────
        response_mime_type: str | None = None,
        # "application/json" → Gemini returns clean JSON with no fences, no
        # Python literals (None/True/False), no trailing commas. Eliminates
        # fence-stripping and sanitisation at the call site.
        # Ignored for claude-* models.
        response_schema: type | None = None,
        # A Pydantic model class for server-side schema enforcement.
        # Only valid alongside response_mime_type for gemini-* models.
        # Ignored for claude-* models.
        disable_auto_func_calling: bool = False,
        # Prevents Gemini from auto-invoking function tools. Set True when
        # the pipeline does not use tools, to avoid SDK deadlocks.
        # Ignored for claude-* models.
        # ── Claude-specific ───────────────────────────────────────────────────
        temperature: float | None = None,
        # Sampling temperature (0.0–1.0). Provider default if None.
        # Accepted by both Gemini and Claude; applied to whichever is routed.
        stop_sequences: list[str] | None = None,
        # Sequences that halt generation early. Supported by Claude.
        # Gemini ignores this parameter in the current implementation.
    ) -> str:
        """Call the LLM and return raw response text.

        Routes by model name prefix:
          'gemini-*' → Google GenAI SDK
          'claude-*' → Anthropic Messages API

        Retries on rate-limit and transient errors for both providers using
        exponential backoff (retry_base_wait_secs * 2^attempt).

        Args:
            max_tokens: Output token limit. None (default) means:
                        - Gemini: field omitted; model uses its own default.
                        - Claude: internal fallback of 8192 (the Anthropic API
                          requires the field; passing None would error).
                        Pass an explicit value for short structured tasks
                        (e.g. max_tokens=256 for classification) to control cost.

        Returns:
            Raw response text string. Parsing is the caller's responsibility.

        Raises:
            ValueError: Unknown model prefix.
            RuntimeError: All retry attempts exhausted.
        """
        return self.generate_with_usage(
            system_prompt, user_message, model, max_tokens,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            disable_auto_func_calling=disable_auto_func_calling,
            temperature=temperature,
            stop_sequences=stop_sequences,
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
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResult:
        """Same call as generate(), but returns real token usage alongside the text.

        See LLMUsage for what output_tokens means per provider — Gemini's
        thinking-token spend is folded in at the call site (_gemini_generate),
        so both providers' output_tokens mean the same thing: total generation
        cost, ready to multiply by a per-token price with no provider-specific
        math at the call site.

        Raises:
            ValueError: Unknown model prefix.
            RuntimeError: All retry attempts exhausted.
        """
        if model.startswith("gemini"):
            return self._gemini_generate(
                system_prompt, user_message, model, max_tokens,
                response_mime_type, response_schema,
                disable_auto_func_calling, temperature,
            )
        if model.startswith("claude"):
            return self._claude_generate(
                system_prompt, user_message, model, max_tokens,
                temperature, stop_sequences,
            )
        raise ValueError(
            f"Unknown model provider for: {model!r}. "
            "Expected prefix 'gemini-' or 'claude-'."
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
        temperature: float | None,
    ) -> LLMResult:
        from google.genai import types

        # Build config kwargs; only include optional fields when explicitly set
        # to avoid overriding Gemini model defaults unnecessarily.
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
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        client = self._get_gemini_client()
        last_exc = None

        for attempt in range(self.retry_max):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                logger.debug(f"[LLM] Gemini {model} succeeded (attempt {attempt + 1}).")
                meta = response.usage_metadata
                # Fold thinking into output_tokens here — the one place that needs
                # to know Gemini reports them separately (see LLMUsage docstring).
                visible_tokens = (meta.candidates_token_count if meta else 0) or 0
                thinking_tokens = (meta.thoughts_token_count if meta else 0) or 0
                usage = LLMUsage(
                    input_tokens=(meta.prompt_token_count if meta else 0) or 0,
                    output_tokens=visible_tokens + thinking_tokens,
                )
                return LLMResult(text=response.text, usage=usage)

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
    ) -> LLMResult:
        client = self._get_anthropic_client()

        create_kwargs: dict = {
            "model": model,
            # Claude's API requires max_tokens; use 8192 as the internal
            # fallback when the caller does not specify a limit.
            "max_tokens": max_tokens if max_tokens is not None else 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if stop_sequences:
            create_kwargs["stop_sequences"] = stop_sequences

        last_exc = None

        for attempt in range(self.retry_max):
            try:
                response = client.messages.create(**create_kwargs)
                logger.debug(f"[LLM] Claude {model} succeeded (attempt {attempt + 1}).")
                # Claude's usage.output_tokens is a single combined count — it
                # already includes thinking tokens when thinking was used, and
                # the API does not report them separately (unlike Gemini).
                usage = LLMUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
                return LLMResult(text=response.content[0].text, usage=usage)

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
