"""Relaxation counts ranked passages only (RAG_SPEC 7.6, OI-3)."""
from __future__ import annotations

from cody_ai_shared_lib.rag.engine import RAGEngine
from cody_ai_shared_lib.rag.gateway import ModelGateway
from cody_ai_shared_lib.rag.granularity import is_ranked
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import QueryIntent, RAGFilterSpec, RAGNode, RAGQueryRequest, SourceSpan
from cody_ai_shared_lib.rag.profile import EngineProfile, ModelBindings, RetrievalConfig
from tests.rag.test_several_names import DIM, _LLM, _embeddings  # noqa: F401  (autouse fixture for embeddings)

ORTHOGONAL = [0.0] * (DIM - 1) + [1.0]      # never close to the fake query vector, so only the keyword leg finds rows


def _engine(texts: list[str]) -> RAGEngine:
    store = InMemoryRAGStore()
    for i, text in enumerate(texts):
        node = RAGNode(node_id=f"n{i}", text=text, span=SourceSpan(doc_id="d", char_start=i * 40, char_end=i * 40 + len(text)))
        store.upsert_section_nodes("d", i, [node], [ORTHOGONAL])
    profile = EngineProfile(
        name="t",
        models=ModelBindings(generation=["m"], embedding="gemini-embedding-2"),
        retrieval=RetrievalConfig(
            default_strategy="hybrid_rrf",
            relaxation=["drop_required_entities"],
            min_results_before_relax=3,
            local_window_radius_chars=500,
        ),
    )
    return RAGEngine(profile, store, gateway=ModelGateway(llm_client=_LLM()))


def test_a_local_window_does_not_stop_relaxation():
    """Two ranked passages plus the reading-position window are three nodes, but only two are passages."""
    engine = _engine(["Alice one alpha.", "Alice two alpha.", "Zed three alpha.", "Zed four alpha."])
    intent = QueryIntent(search_query="alpha", entity_terms=["Alice"], entity_mode="require")
    request = RAGQueryRequest(
        question="alpha",
        cursor_offset=60,
        document_id="d",
        filters=RAGFilterSpec(doc_ids=["d"], min_similarity=0.5),
    )
    result = engine.retrieve(request, intent)
    assert any(not is_ranked(n) for n in result.nodes), "the test needs a window in the result"
    assert result.relaxations == ["drop_required_entities"]
    assert len([n for n in result.nodes if is_ranked(n)]) == 4


def test_enough_ranked_passages_still_skip_relaxation():
    engine = _engine(["Alice one alpha.", "Alice two alpha.", "Alice three alpha.", "Zed four alpha."])
    intent = QueryIntent(search_query="alpha", entity_terms=["Alice"], entity_mode="require")
    request = RAGQueryRequest(
        question="alpha", cursor_offset=60, document_id="d", filters=RAGFilterSpec(doc_ids=["d"], min_similarity=0.5)
    )
    assert engine.retrieve(request, intent).relaxations == []
