"""Central price table and cost computation for Cody AI shared models.

Centralizes per-model pricing across text, embedding, decision, rerank,
image, voice, and video modalities. Model modules (llm.py, embedding.py,
jev.py, rerank.py, image_gen.py, video_gen.py) call compute_cost() to
compute real dollar spend without hardcoding vendor prices locally.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

PRICE_TABLE_VERSION = "2026-11-08"


@dataclass(frozen=True)
class TokenPricing:
    """Pricing per 1M tokens."""
    input_cost_per_m: float
    output_cost_per_m: float = 0.0
    cached_input_cost_per_m: Optional[float] = None
    modality: str = "text"


@dataclass(frozen=True)
class ImagePricing:
    """Pricing for 2D image models (per 1M prompt tokens and/or per generated image)."""
    input_cost_per_m: float = 0.0
    cost_per_image: float = 0.0
    output_cost_per_m: float = 0.0
    modality: str = "image"


@dataclass(frozen=True)
class VideoPricing:
    """Pricing for video generation models (per prompt image, per second, or per 1M tokens)."""
    input_cost_per_image: float = 0.0
    output_cost_per_second: float = 0.0
    output_cost_per_m: float = 0.0
    input_cost_per_m: float = 0.0
    modality: str = "video"


@dataclass(frozen=True)
class VoicePricing:
    """Pricing for speech / audio synthesis models (per 1M tokens)."""
    input_cost_per_m: float
    output_cost_per_m: float
    modality: str = "voice"


ModelPricingType = Union[TokenPricing, ImagePricing, VideoPricing, VoicePricing]

# ── Central Model Catalog ────────────────────────────────────────────────────────

MODEL_CATALOG: Dict[str, ModelPricingType] = {
    # ── Text Models (Gemini direct SDK) ──
    "gemini-3.8-flash": TokenPricing(input_cost_per_m=1.50, output_cost_per_m=7.50, modality="text"),
    "gemini-3.5-flash-lite": TokenPricing(input_cost_per_m=0.30, output_cost_per_m=2.50, modality="text"),
    "gemini-4-argon": TokenPricing(input_cost_per_m=4.00, output_cost_per_m=20.00, modality="text"),

    # ── Text Models (OpenAI direct SDK) ──
    "gpt-6-luna": TokenPricing(input_cost_per_m=0.10, output_cost_per_m=0.50, modality="text"),
    "gpt-6.1-terra": TokenPricing(input_cost_per_m=1.00, output_cost_per_m=5.00, modality="text"),
    "gpt-6.1-sol": TokenPricing(input_cost_per_m=2.00, output_cost_per_m=10.00, modality="text"),
    "gpt-6-astra": TokenPricing(input_cost_per_m=10.00, output_cost_per_m=50.00, modality="text"),

    # ── Text Models (Anthropic direct SDK) ──
    "claude-fable-5-1": TokenPricing(input_cost_per_m=10.00, output_cost_per_m=50.00, modality="text"),
    "claude-opus-5-5": TokenPricing(input_cost_per_m=4.00, output_cost_per_m=20.00, modality="text"),
    "claude-sonnet-5-5": TokenPricing(input_cost_per_m=2.00, output_cost_per_m=10.00, modality="text"),
    "claude-haiku-5-5": TokenPricing(input_cost_per_m=0.10, output_cost_per_m=0.50, modality="text"),
    "claude-opus-5": TokenPricing(input_cost_per_m=5.00, output_cost_per_m=25.00, modality="text"),
    "claude-sonnet-5": TokenPricing(input_cost_per_m=2.00, output_cost_per_m=10.00, modality="text"),
    "claude-haiku-4-5-20251001": TokenPricing(input_cost_per_m=1.00, output_cost_per_m=5.00, modality="text"),

    # ── Embedding ──
    "gemini-embedding-2": TokenPricing(input_cost_per_m=0.20, output_cost_per_m=0.0, modality="embedding"),

    # ── Decision (Jev System One) ──
    "jev": TokenPricing(input_cost_per_m=0.04, output_cost_per_m=0.0, modality="decision"),
    "typesafe-ai/jev": TokenPricing(input_cost_per_m=0.04, output_cost_per_m=0.0, modality="decision"),

    # ── Reranker ──
    "jina-reranker": TokenPricing(input_cost_per_m=0.05, output_cost_per_m=0.0, modality="rerank"),
    "jina-reranker-v2-base-multilingual": TokenPricing(input_cost_per_m=0.05, output_cost_per_m=0.0, modality="rerank"),
    "jina-reranker-v3.5": TokenPricing(input_cost_per_m=0.05, output_cost_per_m=0.0, modality="rerank"),

    # ── Image Models ──
    "gemini-nano-banana-2.1": ImagePricing(input_cost_per_m=1.50, output_cost_per_m=30.00, modality="image"),
    "gemini-3-pro-image": ImagePricing(input_cost_per_m=2.00, output_cost_per_m=120.00, modality="image"),
    "moonshotai/kimi-k3.1-pro": ImagePricing(input_cost_per_m=0.29, output_cost_per_m=8.14, modality="image"),
    "moonshotai/kimi-k3.1-pro-highspeed": ImagePricing(input_cost_per_m=4.00, output_cost_per_m=20.00, modality="image"),
    "bytedance/seedream-5.0-pro": ImagePricing(input_cost_per_m=0.003, cost_per_image=0.04, modality="image"),
    "bytedance/seedream-5.0-lite": ImagePricing(cost_per_image=0.04, modality="image"),
    "bytedance/seedream-4.5": ImagePricing(cost_per_image=0.04, modality="image"),

    # ── Video Models ──
    "gemini-omni-1.2-flash": VideoPricing(input_cost_per_m=1.50, output_cost_per_m=12.00, modality="video"),
    "minimax/minimax-h3": VideoPricing(input_cost_per_image=0.04, output_cost_per_second=0.08, modality="video"),
    "minimax/minimax-h3-max": VideoPricing(output_cost_per_second=0.05, modality="video"),
    "bytedance/seedance-2.5": VideoPricing(output_cost_per_m=10.70, modality="video"),
    "bytedance/seedance-2.0-mini": VideoPricing(output_cost_per_m=3.50, modality="video"),
    "bytedance/seedance-2.0": VideoPricing(output_cost_per_m=7.00, modality="video"),
    "bytedance/seedance-2.0-fast": VideoPricing(output_cost_per_m=5.60, modality="video"),

    # ── Voice / Speech Synthesis ──
    "gemini-3.8-flash-tts": VoicePricing(input_cost_per_m=1.00, output_cost_per_m=18.00, modality="voice"),
}

# In-memory overrides (e.g. from app settings or runtime overrides)
_PRICE_OVERRIDES: Dict[str, ModelPricingType] = {}


def set_price_override(model: str, price: ModelPricingType) -> None:
    """Set an explicit price override for a model (takes precedence over MODEL_CATALOG)."""
    _PRICE_OVERRIDES[model] = price


def clear_price_overrides() -> None:
    """Clear all in-memory price overrides."""
    _PRICE_OVERRIDES.clear()


def get_price_table_version() -> str:
    """Return the active version of the price table."""
    return PRICE_TABLE_VERSION


def get_model_pricing(model: str) -> Optional[ModelPricingType]:
    """Retrieve pricing structure for a model, checking overrides first then catalog.

    Supports exact match first, then normalized model name (stripped of common provider prefixes).
    """
    if model in _PRICE_OVERRIDES:
        return _PRICE_OVERRIDES[model]
    if model in MODEL_CATALOG:
        return MODEL_CATALOG[model]

    # Try matching without prefix or version date if needed
    lower_model = model.lower()
    for cat_name, pricing in MODEL_CATALOG.items():
        if cat_name.lower() == lower_model:
            return pricing

    return None


def compute_cost(
    model: str,
    usage: Any = None,
    *,
    input_tokens: Optional[Union[int, float]] = None,
    output_tokens: Optional[Union[int, float]] = None,
    cached_tokens: Optional[Union[int, float]] = None,
    images: Optional[Union[int, float]] = None,
    seconds: Optional[Union[int, float]] = None,
    total_tokens: Optional[Union[int, float]] = None,
) -> Optional[float]:
    """Compute total dollar cost for a model call given reported token or unit usage.

    Accepts either a usage object/dict as positional second argument, or explicit keyword args.
    Returns:
        float cost in USD (rounded to 6 decimal places), or None if the model is unpriced.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    def _val(attr_name: str, kwarg_val: Optional[Union[int, float]], default: float = 0.0) -> float:
        if kwarg_val is not None:
            return float(kwarg_val)
        if usage is not None:
            if isinstance(usage, dict):
                return float(usage.get(attr_name, default) or 0.0)
            return float(getattr(usage, attr_name, default) or 0.0)
        return default

    in_tok = _val("input_tokens", input_tokens, 0.0)
    out_tok = _val("output_tokens", output_tokens, 0.0)
    cached_tok = _val("cached_tokens", cached_tokens, 0.0)
    imgs = _val("images", images, 0.0)
    secs = _val("seconds", seconds, 0.0)
    tot_tok = _val("total_tokens", total_tokens, 0.0)

    # For rerank or single-metric token models, total_tokens counts as input_tokens if in_tok == 0
    if in_tok == 0.0 and tot_tok > 0.0:
        in_tok = tot_tok

    total_cost = 0.0

    if isinstance(pricing, (TokenPricing, VoicePricing)):
        cached_rate = getattr(pricing, "cached_input_cost_per_m", None)
        if cached_rate is None:
            cached_rate = pricing.input_cost_per_m * 0.25
        effective_input = max(0.0, in_tok - cached_tok)
        total_cost += (effective_input / 1_000_000.0) * pricing.input_cost_per_m
        total_cost += (cached_tok / 1_000_000.0) * cached_rate
        total_cost += (out_tok / 1_000_000.0) * pricing.output_cost_per_m

    elif isinstance(pricing, ImagePricing):
        total_cost += (in_tok / 1_000_000.0) * pricing.input_cost_per_m
        total_cost += (out_tok / 1_000_000.0) * pricing.output_cost_per_m
        total_cost += imgs * pricing.cost_per_image

    elif isinstance(pricing, VideoPricing):
        total_cost += (in_tok / 1_000_000.0) * pricing.input_cost_per_m
        total_cost += (out_tok / 1_000_000.0) * pricing.output_cost_per_m
        total_cost += imgs * pricing.input_cost_per_image
        total_cost += secs * pricing.output_cost_per_second

    return round(total_cost, 6)


def estimate_cost(
    model: str,
    est_input_tokens: int = 0,
    est_output_tokens: int = 0,
    images: int = 0,
    seconds: float = 0.0,
) -> Optional[float]:
    """Estimate dollar cost before making a call to allow pre-call budget checks."""
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    fake_usage = {
        "input_tokens": est_input_tokens,
        "output_tokens": est_output_tokens,
        "cached_tokens": 0,
        "images": images,
        "seconds": seconds,
    }
    return compute_cost(model, fake_usage)
