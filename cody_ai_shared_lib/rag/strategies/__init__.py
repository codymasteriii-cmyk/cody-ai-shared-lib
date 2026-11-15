"""RAG Retrieval strategies package.

A strategy answers one question: "given this request and these filters, which passages should the model see?"
The engine never picks one by hand; `AdaptiveRetrievalStrategy` (the default) looks at the question and at what
the store can do, and runs the first tier that fits:

    tier / name          reads                                        used when
    ───────────────────  ───────────────────────────────────────────  ───────────────────────────────────────
    0  full_context      the whole visible scope, in order, no search  the scope fits `full_context_max_tokens`
       find_then_read    search, then the text around every hit      a list question (opt-in, OI-4)
    3  hierarchical      summaries + a window + best passages          an overview question
       keyword_first     keyword search, hybrid if too few hits        the question names exact terms
    2  hybrid_rrf        vector + keyword search, fused (RRF)          the store has both
    1  dense_vector      vector search                                 the store has vectors only
       keyword_only      keyword search                                the store has keywords only

Shared steps live in `shared.py`; the opt-in ladder that loosens a too-narrow search is `relaxation.py`.
`context.py` holds the `RetrievalContext` every strategy receives and the `RetrievalStrategy` contract.
"""
from .adaptive import AdaptiveRetrievalStrategy
from .context import RetrievalContext, RetrievalStrategy
from .dense_vector import DenseVectorStrategy
from .full_context import FullContextStrategy
from .hierarchical import HierarchicalSummaryStrategy
from .hybrid_rrf import HybridRRFStrategy
from .keyword import KeywordFirstStrategy, KeywordOnlyStrategy

__all__ = [
    "RetrievalContext",
    "RetrievalStrategy",
    "FullContextStrategy",
    "DenseVectorStrategy",
    "HybridRRFStrategy",
    "HierarchicalSummaryStrategy",
    "KeywordOnlyStrategy",
    "KeywordFirstStrategy",
    "AdaptiveRetrievalStrategy",
]
