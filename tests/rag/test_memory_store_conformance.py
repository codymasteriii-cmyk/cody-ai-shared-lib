"""Conformance test for InMemoryRAGStore and negative test for BrokenStore (AC-RAG-50)."""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag.errors import ScopeViolationError
from cody_ai_shared_lib.rag.memory_store import InMemoryRAGStore
from cody_ai_shared_lib.rag.models import DefaultNodeMeta, RAGFilterSpec, RAGNode, SourceSpan
from cody_ai_shared_lib.rag.ports import OwnerScope, RAGStore, StoreCapabilities
from cody_ai_shared_lib.rag.testing.contract import BrokenStore, RAGStoreContractTests


class TestInMemoryRAGStoreConformance(RAGStoreContractTests):
    """Verifies that InMemoryRAGStore satisfies all 15 contract checks."""

    def make_store(self) -> InMemoryRAGStore[DefaultNodeMeta]:
        # Enable required_filter_keys on test instance for check 2
        caps = StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=True,
            supports_native_hybrid_rpc=True,
            returns_raw_scores=True,
            uses_analyzer_tokens=True,
            returns_snippets=True,
            supports_parent_summaries=True,
            supports_summary_levels=True,
            supports_local_window=True,
            supports_full_scope_fetch=True,
            supports_sequential_boundary=True,
            supports_fetch_nodes=True,
            supports_fingerprints=True,
            supports_delete=True,
            supports_revisions=True,
            supports_embedding_return=True,
            required_filter_keys=frozenset(),
            supported_custom_filters={"favorites_only": bool, "active": bool, "language": str},
            supported_date_fields=frozenset(["created_at", "updated_at"]),
        )
        return InMemoryRAGStore(capabilities=caps)


def _tenant_caps() -> StoreCapabilities:
    caps = InMemoryRAGStore().capabilities()
    return caps.model_copy(update={"required_filter_keys": frozenset({"tenant_or_user_id", "doc_ids"})})


class TestTenantEnforcingStoreConformance(RAGStoreContractTests):
    """The configuration a multi-user app ships: required tenant, owner on every write, scoped views (INT-05)."""

    def make_store(self) -> InMemoryRAGStore[DefaultNodeMeta]:
        return InMemoryRAGStore(capabilities=_tenant_caps())

    def make_owner(self) -> OwnerScope:
        return OwnerScope(tenant_or_user_id="kit-user")

    def make_scoped_store(self, store, owner: OwnerScope):
        return store.scoped(owner)


class ZeroStatsStore(InMemoryRAGStore):
    """Counts only indexed rows in scope_stats while fetch_full_scope serves source text (the INT-06 trap)."""

    def scope_stats(self, filters):
        return super().scope_stats(filters).model_copy(update={"char_count": 0, "est_tokens": 0})


class OneNodeStore(InMemoryRAGStore):
    """Returns the whole scope as one node (the INT-07 trap)."""

    def fetch_full_scope(self, filters):
        nodes = super().fetch_full_scope(filters)
        if not nodes:
            return nodes
        merged = nodes[0].model_copy(deep=True)
        merged.text = "".join(n.text for n in nodes)
        merged.span = merged.span.model_copy(update={"section_seq": 0, "char_start": 0, "char_end": len(merged.text)})
        return [merged]


@pytest.mark.parametrize(
    ("store_cls", "check"),
    [
        (ZeroStatsStore, "test_check_17_scope_stats_matches_full_scope"),
        (OneNodeStore, "test_check_18_full_scope_granularity"),
    ],
)
def test_kit_catches_full_scope_traps(store_cls, check):
    class Kit(RAGStoreContractTests):
        def make_store(self):
            return store_cls()

    with pytest.raises(AssertionError):
        getattr(Kit(), check)()


def test_kit_catches_a_store_that_computes_its_own_fingerprints():
    class SelfFingerprinting(InMemoryRAGStore):
        def upsert_section(self, write):          # drops the engine's fingerprints, as a legacy store would
            return RAGStore.upsert_section(self, write)

    class Kit(RAGStoreContractTests):
        def make_store(self):
            return SelfFingerprinting()

    with pytest.raises(AssertionError):
        Kit().test_check_15_fingerprints_persisted_as_given()


def test_scoped_store_refuses_lifecycle_calls_on_another_owners_document():
    store = InMemoryRAGStore(capabilities=_tenant_caps())
    kit = TestTenantEnforcingStoreConformance()
    kit.write(store, "doc_b", [RAGNode(node_id="b1", text="B text", span=SourceSpan(doc_id="doc_b"))], [[0.1] * 768],
              owner=OwnerScope(tenant_or_user_id="user-b"))
    scoped_a = store.scoped(OwnerScope(tenant_or_user_id="user-a"))

    for call in (lambda: scoped_a.get_fingerprints("doc_b"), lambda: scoped_a.delete_document("doc_b"),
                 lambda: scoped_a.delete_nodes(["b1"])):
        with pytest.raises(ScopeViolationError):
            call()
    assert "b1" in store.get_fingerprints("doc_b")


def test_broken_store_fails_checks():
    """BrokenStore must fail zero-vector and required scope checks (AC-RAG-50)."""
    broken = BrokenStore()

    # BrokenStore fails check 6: does NOT raise EmbeddingProviderError on zero vector
    res = broken.vector_search([0.0] * 768, RAGFilterSpec(), top_k=5)
    assert len(res) > 0  # It returned results instead of raising!

    # BrokenStore fails check 1 & 2: does NOT raise MissingRequiredScopeError
    res2 = broken.vector_search([0.1] * 768, RAGFilterSpec(), top_k=5)
    assert len(res2) > 0


class _AppliesPreferStore(InMemoryRAGStore):
    """Defect: excludes for a `prefer` rule, as a store that treats every rule as a filter would."""

    def vector_search(self, query_vector, filters, top_k):
        from cody_ai_shared_lib.rag.text.rules import satisfies

        rows = super().vector_search(query_vector, filters, top_k)
        return [r for r in rows if satisfies(r.text, list(filters.text_rules))]


class _FiltersFullScopeStore(InMemoryRAGStore):
    """Defect: applies rules to the Tier 0 text too."""

    def fetch_full_scope(self, filters):
        from cody_ai_shared_lib.rag.text.rules import satisfies

        return [n for n in super().fetch_full_scope(filters) if satisfies(n.text, list(filters.text_rules))]


def test_kit_catches_a_store_that_applies_prefer_rules():
    kit = type("K", (RAGStoreContractTests,), {"make_store": lambda self: _AppliesPreferStore()})()
    with pytest.raises(AssertionError, match="prefer"):
        kit.test_check_29_prefer_rule_is_never_a_filter()


def test_kit_catches_a_store_that_filters_full_scope():
    kit = type("K", (RAGStoreContractTests,), {"make_store": lambda self: _FiltersFullScopeStore()})()
    with pytest.raises(AssertionError, match="ignores text rules"):
        kit.test_check_30_rules_apply_to_ranked_searches_only()


class _IgnoresRulesStore(InMemoryRAGStore):
    """Defect: declares the kinds but applies none."""

    def _rules_ok(self, node, filters):
        return True


@pytest.mark.parametrize("check", ["test_check_26_must_mention_rule", "test_check_27_exact_rule", "test_check_28_must_not_rule"])
def test_kit_catches_a_store_that_declares_a_kind_but_ignores_it(check: str):
    kit = type("K", (RAGStoreContractTests,), {"make_store": lambda self: _IgnoresRulesStore()})()
    with pytest.raises(AssertionError, match="filter rule"):
        getattr(kit, check)()


def test_the_reference_store_declares_offset_range_scope():
    assert InMemoryRAGStore().capabilities().supports_offset_range_scope is True


class _IgnoresMinOffsetStore(InMemoryRAGStore):
    """Defect: declares offset-range scopes but returns everything from the start of the document."""

    def fetch_full_scope(self, filters):
        return super().fetch_full_scope(filters.model_copy(update={"min_char_offset": None}))


def test_kit_catches_a_store_that_ignores_the_range_start():
    kit = type("K", (RAGStoreContractTests,), {"make_store": lambda self: _IgnoresMinOffsetStore()})()
    with pytest.raises(AssertionError, match="range"):
        kit.test_check_31_offset_range_scope()
