"""Tests for LLMClient.stream() across Gemini, Claude, and OpenAI providers."""
from unittest.mock import MagicMock, patch

import pytest
from cody_ai_shared_lib.llm import LLMClient, LLMStreamChunk
from cody_ai_shared_lib.pricing import PRICE_TABLE_VERSION


class MockGeminiChunk:
    def __init__(self, text, usage_metadata=None, candidates=None):
        self.text = text
        self.usage_metadata = usage_metadata
        self.candidates = candidates or []


class MockOpenAIDelta:
    def __init__(self, content):
        self.content = content


class MockOpenAIChoice:
    def __init__(self, delta_content, finish_reason=None):
        self.delta = MockOpenAIDelta(delta_content)
        self.finish_reason = finish_reason


class MockOpenAIChunk:
    def __init__(self, choices=None, usage=None):
        self.choices = choices or []
        self.usage = usage


def test_gemini_stream():
    client = LLMClient()
    mock_gemini_client = MagicMock()
    client._gemini_client = mock_gemini_client

    meta = MagicMock()
    meta.prompt_token_count = 100
    meta.candidates_token_count = 50
    meta.cached_content_token_count = 0
    meta.thoughts_token_count = 0

    cand = MagicMock()
    cand.finish_reason = "STOP"

    chunks = [
        MockGeminiChunk("Hello "),
        MockGeminiChunk("world!"),
        MockGeminiChunk("", usage_metadata=meta, candidates=[cand]),
    ]
    mock_gemini_client.models.generate_content_stream.return_value = chunks

    stream = list(client.stream(
        system_prompt="system",
        user_message="user",
        model="gemini-3.8-flash",
    ))

    deltas = [c.delta for c in stream[:-1]]
    assert "".join(deltas) == "Hello world!"

    final = stream[-1]
    assert final.delta == ""
    assert final.finish_reason == "stop"
    assert final.usage is not None
    assert final.usage.input_tokens == 100
    assert final.usage.output_tokens == 50
    assert final.usage.price_table_version == PRICE_TABLE_VERSION
    assert final.usage.cost_usd is not None


def test_claude_stream():
    client = LLMClient()
    mock_anthropic = MagicMock()
    client._anthropic_client = mock_anthropic

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__enter__.return_value = mock_stream_ctx
    mock_stream_ctx.text_stream = iter(["Part 1 ", "Part 2."])

    final_msg = MagicMock()
    final_msg.stop_reason = "end_turn"
    final_msg.usage.input_tokens = 200
    final_msg.usage.cache_read_input_tokens = 50
    final_msg.usage.output_tokens = 80
    mock_stream_ctx.get_final_message.return_value = final_msg

    mock_anthropic.messages.stream.return_value = mock_stream_ctx

    stream = list(client.stream(
        system_prompt="system",
        user_message="user",
        model="claude-sonnet-5-5",
        cache_breakpoints=True,
    ))

    # Verify cache_control was passed
    call_kwargs = mock_anthropic.messages.stream.call_args.kwargs
    assert isinstance(call_kwargs["system"], list)
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    deltas = [c.delta for c in stream[:-1]]
    assert "".join(deltas) == "Part 1 Part 2."

    final = stream[-1]
    assert final.delta == ""
    assert final.finish_reason == "stop"
    assert final.usage.input_tokens == 250
    assert final.usage.cached_tokens == 50
    assert final.usage.output_tokens == 80
    assert final.usage.cost_usd is not None


def test_openai_stream():
    client = LLMClient()
    mock_openai = MagicMock()
    client._openai_client = mock_openai

    usage = MagicMock()
    usage.prompt_tokens = 120
    usage.completion_tokens = 60
    usage.prompt_tokens_details.cached_tokens = 20

    chunks = [
        MockOpenAIChunk(choices=[MockOpenAIChoice("OpenAI ")]),
        MockOpenAIChunk(choices=[MockOpenAIChoice("stream ", finish_reason=None)]),
        MockOpenAIChunk(choices=[MockOpenAIChoice("chunk", finish_reason="stop")], usage=usage),
    ]
    mock_openai.chat.completions.create.return_value = chunks

    stream = list(client.stream(
        system_prompt="system",
        user_message="user",
        model="gpt-6.1-terra",
    ))

    deltas = [c.delta for c in stream[:-1]]
    assert "".join(deltas) == "OpenAI stream chunk"

    final = stream[-1]
    assert final.delta == ""
    assert final.finish_reason == "stop"
    assert final.usage.input_tokens == 120
    assert final.usage.output_tokens == 60
    assert final.usage.cached_tokens == 20
    assert final.usage.cost_usd is not None


# ── a retry after text was sent would show the user the start of the answer twice (1.2.17) ──────────

def _claude_context(texts, error=None):
    """A Claude stream context that yields `texts`, then raises `error` if given."""
    ctx = MagicMock()
    ctx.__enter__.return_value = ctx

    def stream():
        yield from texts
        if error is not None:
            raise error

    ctx.text_stream = stream()
    final = MagicMock()
    final.stop_reason = "end_turn"
    final.usage.input_tokens = 10
    final.usage.cache_read_input_tokens = 0
    final.usage.output_tokens = 5
    ctx.get_final_message.return_value = final
    return ctx


def _flaky_gemini(texts, error=None):
    for text in texts:
        yield MockGeminiChunk(text)
    if error is not None:
        raise error


def _flaky_openai(texts, error=None):
    for text in texts:
        yield MockOpenAIChunk(choices=[MockOpenAIChoice(text)])
    if error is not None:
        raise error


def _stream_text(client, model):
    return "".join(c.delta for c in client.stream(system_prompt="s", user_message="u", model=model))


def _provider(provider):
    """(client, the mock whose side_effect lists one stream per attempt, model name) for each provider."""
    client = LLMClient(retry_base_wait_secs=0)
    if provider == "claude":
        mock = MagicMock()
        client._anthropic_client = mock
        return client, mock.messages.stream, "claude-sonnet-5-5"
    if provider == "gemini":
        mock = MagicMock()
        client._gemini_client = mock
        return client, mock.models.generate_content_stream, "gemini-3.8-flash"
    mock = MagicMock()
    client._openai_client = mock
    return client, mock.chat.completions.create, "gpt-6.1-terra"


def _attempt(provider, texts, error=None):
    if provider == "claude":
        return _claude_context(texts, error)
    if provider == "gemini":
        return _flaky_gemini(texts, error)
    return _flaky_openai(texts, error)


@pytest.mark.parametrize("provider", ["claude", "gemini", "openai"])
def test_a_stream_is_not_retried_after_text_was_sent(provider):
    client, call, model = _provider(provider)
    call.side_effect = [
        _attempt(provider, ["The festival began on "], error=Exception("503 service unavailable")),
        _attempt(provider, ["The festival began on the first of May."]),
    ]
    received = []
    with pytest.raises(Exception, match="503"):
        for chunk in client.stream(system_prompt="s", user_message="u", model=model):
            received.append(chunk.delta)
    assert "".join(received) == "The festival began on ", "the caller must never see the start of the answer twice"
    assert call.call_count == 1


@pytest.mark.parametrize("provider", ["claude", "gemini", "openai"])
def test_a_stream_is_still_retried_before_any_text(provider):
    client, call, model = _provider(provider)
    call.side_effect = [
        _attempt(provider, [], error=Exception("429 rate limit")),
        _attempt(provider, ["Hello ", "there."]),
    ]
    assert _stream_text(client, model) == "Hello there."
    assert call.call_count == 2
