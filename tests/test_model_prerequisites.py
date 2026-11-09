"""Tests for lib-wide model prerequisites: LLMClient, embedding, jev, rerank, and pricing.

Covers:
  - AC-RAG-69: Model usage, USD cost accounting, price_table_version, estimates, and overrides.
  - AC-RAG-70: ModelCallError with usage attached on structured-output parse failure.
  - LLM finish_reason normalization ("stop", "length", "safety", "other").
  - Claude ThinkingBlock extraction.
  - OpenAI reasoning model handling (max_completion_tokens).
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from cody_ai_shared_lib import (
    PRICE_TABLE_VERSION,
    EmbeddingResult,
    EmbeddingUsage,
    JevClient,
    JevResult,
    JevUsage,
    LLMClient,
    LLMResult,
    LLMUsage,
    ModelCallError,
    RerankClient,
    RerankItem,
    RerankResult,
    RerankUsage,
    clear_price_overrides,
    generate_embedding_with_usage,
    set_price_override,
)
from cody_ai_shared_lib.pricing import TokenPricing
from cody_ai_shared_lib.rag.text.analyzer import (
    CjkAwareAnalyzer,
    TokenGroup,
    build_tsquery,
    estimate_tokens,
)


@pytest.fixture(autouse=True)
def clean_overrides():
    clear_price_overrides()
    yield
    clear_price_overrides()


class SampleSchema(BaseModel):
    name: str
    age: int


# ── LLMClient Tests ────────────────────────────────────────────────────────────

def test_llm_result_and_usage_fields():
    usage = LLMUsage(
        input_tokens=100,
        output_tokens=50,
        cached_tokens=20,
        cost_usd=0.001,
        price_table_version=PRICE_TABLE_VERSION,
        estimated=False,
    )
    result = LLMResult(text="test output", usage=usage, finish_reason="stop")
    assert result.text == "test output"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 50
    assert result.usage.cached_tokens == 20
    assert result.usage.cost_usd == 0.001
    assert result.usage.price_table_version == PRICE_TABLE_VERSION
    assert result.usage.estimated is False


def test_finish_reason_normalization():
    norm = LLMClient._normalize_finish_reason
    assert norm("STOP") == "stop"
    assert norm("end_turn") == "stop"
    assert norm("stop_sequence") == "stop"
    assert norm("MAX_TOKENS") == "length"
    assert norm("length") == "length"
    assert norm("max_tokens") == "length"
    assert norm("SAFETY") == "safety"
    assert norm("refusal") == "safety"
    assert norm("content_filter") == "safety"
    assert norm("recitation") == "safety"
    assert norm("something_unexpected") == "other"
    assert norm(None) is None


def test_gemini_cost_and_usage_accounting():
    client = LLMClient()
    mock_gemini = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"name": "Alice", "age": 30}'

    mock_meta = MagicMock()
    mock_meta.prompt_token_count = 1000
    mock_meta.candidates_token_count = 500
    mock_meta.thoughts_token_count = 100
    mock_meta.cached_content_token_count = 200
    mock_resp.usage_metadata = mock_meta

    mock_candidate = MagicMock()
    mock_candidate.finish_reason = "STOP"
    mock_resp.candidates = [mock_candidate]

    mock_gemini.models.generate_content.return_value = mock_resp
    client._gemini_client = mock_gemini

    result = client.generate_with_usage("sys", "user", "gemini-3.8-flash")
    # gemini-3.8-flash: input 1.5, output 7.5 per 1M, cached 25% = 0.375
    # uncached = (1000 - 200) = 800 -> 0.0012; cached = 200 -> 0.000075; output = 600 -> 0.0045
    # total = 0.0012 + 0.000075 + 0.0045 = 0.005775
    assert result.usage.input_tokens == 1000
    assert result.usage.output_tokens == 600
    assert result.usage.cached_tokens == 200
    assert result.usage.cost_usd == pytest.approx(0.005775, rel=1e-5)
    assert result.usage.price_table_version == PRICE_TABLE_VERSION
    assert result.finish_reason == "stop"


def test_claude_thinking_block_and_cost():
    client = LLMClient()
    mock_anthropic = MagicMock()

    # Simulate ThinkingBlock then TextBlock (e.g. claude-opus-5-5)
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    del thinking_block.text  # Anthropic ThinkingBlock has NO .text attribute

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello world from Claude"

    mock_resp = MagicMock()
    mock_resp.content = [thinking_block, text_block]
    mock_resp.stop_reason = "end_turn"

    mock_usage = MagicMock()
    mock_usage.input_tokens = 2000
    mock_usage.output_tokens = 1000
    mock_usage.cache_read_input_tokens = 500
    mock_resp.usage = mock_usage

    mock_anthropic.messages.create.return_value = mock_resp
    client._anthropic_client = mock_anthropic

    result = client.generate_with_usage("sys", "user", "claude-opus-5-5")
    assert result.text == "Hello world from Claude"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 2500
    assert result.usage.output_tokens == 1000
    assert result.usage.cached_tokens == 500
    # claude-opus-5-5: input $4/1M, output $20/1M, cached $1/1M
    # uncached 2000 * 4/1M = 0.008; 1000 * 20/1M = 0.02; 500 * 1/1M = 0.0005 -> total 0.0285
    assert result.usage.cost_usd == pytest.approx(0.0285, rel=1e-5)


def test_openai_reasoning_model_routing():
    client = LLMClient()
    mock_openai = MagicMock()

    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI response"
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 500
    mock_usage.completion_tokens = 250
    mock_details = MagicMock()
    mock_details.cached_tokens = 100
    mock_usage.prompt_tokens_details = mock_details

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = mock_usage

    mock_openai.chat.completions.create.return_value = mock_resp
    client._openai_client = mock_openai

    result = client.generate_with_usage(
        "sys", "user", "gpt-6-luna", max_tokens=1000, temperature=0.7
    )
    assert result.text == "OpenAI response"
    assert result.finish_reason == "stop"

    # Verify reasoning model passed max_completion_tokens and omitted temperature
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 1000
    assert "temperature" not in call_kwargs

    # gpt-6-luna: input $0.10/1M, output $0.50/1M, cached $0.025/1M
    # uncached 400 * 0.10/1M = 0.00004; 250 * 0.50/1M = 0.000125; 100 * 0.025/1M = 0.0000025 -> total 0.0001675 (rounded to 6 decimals: 0.000168)
    assert result.usage.cost_usd == pytest.approx(0.000168, abs=1e-6)


def test_structured_output_failure_attaches_usage_ac_rag_70():
    client = LLMClient()
    mock_gemini = MagicMock()
    mock_resp = MagicMock()
    # Invalid JSON that will fail SampleSchema parsing
    mock_resp.text = "This is not valid json {"
    mock_meta = MagicMock()
    mock_meta.prompt_token_count = 50
    mock_meta.candidates_token_count = 20
    mock_meta.thoughts_token_count = 0
    mock_meta.cached_content_token_count = 0
    mock_resp.usage_metadata = mock_meta
    mock_resp.candidates = [MagicMock(finish_reason="STOP")]

    mock_gemini.models.generate_content.return_value = mock_resp
    client._gemini_client = mock_gemini

    with pytest.raises(ModelCallError) as exc_info:
        client.generate_with_usage(
            "sys", "user", "gemini-3.8-flash", response_schema=SampleSchema
        )

    err = exc_info.value
    assert err.billing == "billed"
    assert err.usage is not None
    assert err.usage.input_tokens == 50
    assert err.usage.output_tokens == 20
    assert err.usage.cost_usd is not None


def test_unlisted_model_gives_none_cost():
    client = LLMClient()
    mock_openai = MagicMock()
    mock_choice = MagicMock(finish_reason="stop")
    mock_choice.message.content = "output"
    mock_usage = MagicMock(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None)
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice], usage=mock_usage
    )
    client._openai_client = mock_openai

    result = client.generate_with_usage("sys", "user", "gpt-unlisted-custom")
    assert result.usage.cost_usd is None
    assert result.usage.price_table_version == PRICE_TABLE_VERSION


def test_price_override_precedence():
    client = LLMClient()
    mock_gemini = MagicMock()
    mock_resp = MagicMock(text="ok")
    mock_meta = MagicMock(
        prompt_token_count=1_000_000,
        candidates_token_count=1_000_000,
        thoughts_token_count=0,
        cached_content_token_count=0,
    )
    mock_resp.usage_metadata = mock_meta
    mock_resp.candidates = [MagicMock(finish_reason="STOP")]
    mock_gemini.models.generate_content.return_value = mock_resp
    client._gemini_client = mock_gemini

    # Override gemini-3.8-flash to $10 in, $20 out
    set_price_override("gemini-3.8-flash", TokenPricing(input_cost_per_m=10.0, output_cost_per_m=20.0))

    result = client.generate_with_usage("sys", "user", "gemini-3.8-flash")
    # 1M in * 10 + 1M out * 20 = $30
    assert result.usage.cost_usd == pytest.approx(30.0, rel=1e-5)


# ── Embedding Tests ────────────────────────────────────────────────────────────

def test_generate_embedding_with_usage_empty():
    res = generate_embedding_with_usage("")
    assert res.vectors == []
    assert res.usage.input_tokens == 0
    assert res.usage.cost_usd == 0.0
    assert res.usage.estimated is True


def test_generate_embedding_with_usage_mocked():
    with patch("cody_ai_shared_lib.embedding.generate_embedding") as mock_embed:
        mock_embed.return_value = [0.1] * 768
        res = generate_embedding_with_usage("Hello world, this is a test.")
        assert len(res.vectors) == 768
        assert res.usage.input_tokens > 0
        assert res.usage.cost_usd is not None
        assert res.usage.cost_usd > 0
        assert res.usage.estimated is True
        assert res.usage.price_table_version == PRICE_TABLE_VERSION


# ── Jev Tests ──────────────────────────────────────────────────────────────────

def test_jev_cost_and_usage():
    client = JevClient(api_key="mock_key")
    mock_sdk = MagicMock()
    mock_resp = MagicMock()
    mock_resp.answers = {"q1": MagicMock(noul=0.85)}
    mock_usage = MagicMock()
    mock_usage.input_tokens = 500
    mock_usage.output_tokens = 50
    mock_resp.usage = mock_usage

    mock_sdk.system_one.return_value = mock_resp
    client._client = mock_sdk

    result = client.system_one("some state", {"q1": MagicMock()})
    assert result.tokens_in == 500
    assert result.tokens_out == 50
    assert result.usage is not None
    assert result.usage.input_tokens == 500
    assert result.usage.output_tokens == 50
    # jev is $0.04 per 1M input tokens
    assert result.cost_usd == pytest.approx(500 * 0.04 / 1_000_000, rel=1e-5)
    assert result.usage.cost_usd == result.cost_usd
    assert result.usage.estimated is False
    assert result.usage.price_table_version == PRICE_TABLE_VERSION


# ── Rerank Tests ───────────────────────────────────────────────────────────────

def test_rerank_empty_documents():
    client = RerankClient(api_key="mock_key")
    res = client.rerank("query", [])
    assert res.results == []
    assert res.usage.total_tokens == 0
    assert res.usage.cost_usd == 0.0


def test_rerank_cost_and_usage():
    client = RerankClient(api_key="mock_key")
    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "model": "jina-reranker-v2-base-multilingual",
            "usage": {"total_tokens": 10_000},
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "doc1"}},
                {"index": 1, "relevance_score": 0.42, "document": {"text": "doc2"}},
            ],
        }
        mock_post.return_value = mock_resp

        res = client.rerank("my query", ["doc1", "doc2"])
        assert len(res.results) == 2
        assert res.results[0].relevance_score == 0.95
        assert res.results[0].document == "doc1"
        assert res.usage.total_tokens == 10_000
        # jina-reranker is $0.05 per 1M tokens: 10_000 * 0.05 / 1M = 0.0005
        assert res.usage.cost_usd == pytest.approx(0.0005, rel=1e-5)
        assert res.usage.price_table_version == PRICE_TABLE_VERSION
        assert res.usage.estimated is False


# ── Text Analyzer Tests ────────────────────────────────────────────────────────

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    # 4 latin chars * 0.25 = 1.0 -> * 1.10 = 1.1 -> ceil = 2
    assert estimate_tokens("test") == 2
    # 4 CJK chars * 1.0 = 4.0 -> * 1.10 = 4.4 -> ceil = 5
    assert estimate_tokens("测试文本") == 5


def test_cjk_aware_analyzer_tokenization_and_sentences():
    analyzer = CjkAwareAnalyzer()
    # CJK bigram tokenization
    tokens = analyzer.tokenize("张三来到了北京大学")
    assert "张三" in tokens
    assert "三来" in tokens
    assert "北京" in tokens

    # Latin word tokenization
    latin_tokens = analyzer.tokenize("Hello World 2026!")
    assert latin_tokens == ["hello", "world", "2026"]

    # Sentence boundary detection
    text = "第一句话。第二句话！他说：“第三句。”3.14不是句子结尾。Last sentence."
    sents = analyzer.sentences(text)
    assert len(sents) >= 4
    last_end = analyzer.last_sentence_end(text)
    assert last_end == len(text)


def test_build_tsquery():
    groups = [
        TokenGroup(tokens=["李小", "小明"], mode="phrase", origin="exact_term"),
        TokenGroup(tokens=["search", "find"], mode="any", origin="free_text"),
    ]
    query_str = build_tsquery(groups)
    assert "(李小 <-> 小明)" in query_str
    assert "(search | find)" in query_str
