"""Unit tests for cody_ai_shared_lib.pricing module."""
from dataclasses import dataclass
import pytest
from cody_ai_shared_lib.pricing import (
    PRICE_TABLE_VERSION,
    MODEL_CATALOG,
    TokenPricing,
    ImagePricing,
    VideoPricing,
    VoicePricing,
    compute_cost,
    estimate_cost,
    get_model_pricing,
    set_price_override,
    clear_price_overrides,
    get_price_table_version,
)


@dataclass
class DummyUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    images: int = 0
    seconds: float = 0.0


def test_pricing_table_version():
    assert PRICE_TABLE_VERSION == "2026-11-08"
    assert get_price_table_version() == "2026-11-08"


def test_compute_cost_text_gemini():
    # gemini-3.8-flash: in $1.50/M, out $7.50/M
    usage = DummyUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = compute_cost("gemini-3.8-flash", usage)
    assert cost == 9.0

    # Test with dict
    cost_dict = compute_cost("gemini-3.8-flash", {"input_tokens": 100_000, "output_tokens": 50_000})
    # 0.1 * 1.50 + 0.05 * 7.50 = 0.15 + 0.375 = 0.525
    assert cost_dict == 0.525


def test_compute_cost_text_openai_gpt6():
    # gpt-6-luna: in $0.10/M, out $0.50/M
    usage = DummyUsage(input_tokens=10_000_000, output_tokens=2_000_000)
    # 10 * 0.10 + 2 * 0.50 = 1.0 + 1.0 = 2.0
    cost = compute_cost("gpt-6-luna", usage)
    assert cost == 2.0


def test_compute_cost_text_claude():
    # claude-sonnet-5-5: in $2.00/M, out $10.00/M
    usage = DummyUsage(input_tokens=500_000, output_tokens=100_000)
    # 0.5 * 2.0 + 0.1 * 10.0 = 1.0 + 1.0 = 2.0
    cost = compute_cost("claude-sonnet-5-5", usage)
    assert cost == 2.0


def test_compute_cost_embedding():
    # gemini-embedding-2: in $0.20/M, out 0
    usage = DummyUsage(input_tokens=500_000, output_tokens=0)
    cost = compute_cost("gemini-embedding-2", usage)
    assert cost == 0.1


def test_compute_cost_decision_jev():
    # jev: $0.04/M input
    usage = DummyUsage(input_tokens=1_000_000, output_tokens=50)
    cost = compute_cost("jev", usage)
    assert cost == 0.04

    cost_typesafe = compute_cost("typesafe-ai/jev", usage)
    assert cost_typesafe == 0.04


def test_compute_cost_rerank():
    # jina-reranker: $0.05/M
    usage = DummyUsage(input_tokens=200_000, output_tokens=0)
    cost = compute_cost("jina-reranker", usage)
    assert cost == 0.01

    cost_v2 = compute_cost("jina-reranker-v2-base-multilingual", usage)
    assert cost_v2 == 0.01


def test_compute_cost_image_models():
    # bytedance/seedream-5.0-lite: $0.04 per image
    usage_img = DummyUsage(images=3)
    cost = compute_cost("bytedance/seedream-5.0-lite", usage_img)
    assert cost == 0.12

    # seedream-5.0-pro: in $0.003/M, $0.04/image
    usage_pro = DummyUsage(input_tokens=1_000_000, images=1)
    cost_pro = compute_cost("bytedance/seedream-5.0-pro", usage_pro)
    assert cost_pro == 0.043


def test_compute_cost_video_models():
    # minimax/minimax-h3: $0.08 / second
    usage_video = DummyUsage(seconds=10)
    cost = compute_cost("minimax/minimax-h3", usage_video)
    assert cost == 0.8

    # minimax-h3-max: $0.05 / second
    cost_max = compute_cost("minimax/minimax-h3-max", DummyUsage(seconds=20))
    assert cost_max == 1.0


def test_compute_cost_voice_tts():
    # gemini-3.8-flash-tts: in $1.00/M, out $18.00/M
    usage_tts = DummyUsage(input_tokens=100_000, output_tokens=100_000)
    # 0.1 * 1.0 + 0.1 * 18.0 = 0.1 + 1.8 = 1.9
    cost = compute_cost("gemini-3.8-flash-tts", usage_tts)
    assert cost == 1.9


def test_compute_cost_unknown_model():
    cost = compute_cost("non-existent-model-xyz", DummyUsage(input_tokens=1000))
    assert cost is None


def test_estimate_cost():
    # Pre-call estimate for gemini-3.8-flash
    est = estimate_cost("gemini-3.8-flash", est_input_tokens=100_000, est_output_tokens=50_000)
    assert est == 0.525

    # Estimate for unknown model returns None
    assert estimate_cost("unknown-model") is None


def test_price_overrides():
    model = "custom-special-model"
    # Initially unpriced
    assert compute_cost(model, DummyUsage(input_tokens=1_000_000)) is None

    # Set override
    set_price_override(model, TokenPricing(input_cost_per_m=5.0, output_cost_per_m=15.0))
    assert compute_cost(model, DummyUsage(input_tokens=1_000_000, output_tokens=1_000_000)) == 20.0

    # Overriding an existing model takes precedence
    set_price_override("gemini-3.8-flash", TokenPricing(input_cost_per_m=0.5, output_cost_per_m=1.0))
    assert compute_cost("gemini-3.8-flash", DummyUsage(input_tokens=1_000_000, output_tokens=1_000_000)) == 1.5

    # Clear overrides
    clear_price_overrides()
    assert compute_cost("gemini-3.8-flash", DummyUsage(input_tokens=1_000_000, output_tokens=1_000_000)) == 9.0
    assert compute_cost(model, DummyUsage(input_tokens=1_000_000)) is None
