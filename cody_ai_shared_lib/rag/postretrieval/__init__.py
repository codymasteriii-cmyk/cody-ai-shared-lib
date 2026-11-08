"""RAG post-retrieval pipeline stages."""
from .dedup import IdentityDeduplicator, OffsetDeduplicator
from .rerank import JinaCrossEncoderReranker
from .threshold import ThresholdFilter

__all__ = [
    "ThresholdFilter",
    "IdentityDeduplicator",
    "OffsetDeduplicator",
    "JinaCrossEncoderReranker",
]
