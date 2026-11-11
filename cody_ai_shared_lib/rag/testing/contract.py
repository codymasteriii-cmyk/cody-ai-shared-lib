"""RAGStore conformance kit and BrokenStore fixture (§4.6, AC-RAG-50).

Provides:
  - RAGStoreContractTests: pytest mixin with the 24 contract checks.
  - BrokenStore: deliberately non-conforming store that must fail checks.

An app subclasses the mixin and implements `make_store()`. A store that enforces a required scope
also overrides `make_owner()`: the kit then writes every row for that owner and builds every read
filter from it (`make_filters`), so the configuration that ships is the one tested (INT-05).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
from typing import Any

import pytest

from ..errors import EmbeddingProviderError, MissingRequiredScopeError, ScopeViolationError
from ..models import (
    DefaultNodeMeta,
    PreparedNode,
    RAGFilterSpec,
    RAGNode,
    SourceSpan,
)
from ..ports import (
    EmbeddingSpace,
    Fingerprint,
    OwnerScope,
    RAGStore,
    SectionWrite,
    StoreCapabilities,
    SummaryWrite,
)
from ..text.analyzer import TokenGroup

DIM = 768


class BrokenStore(RAGStore[DefaultNodeMeta]):
    """Deliberately non-conforming store used to verify that the test kit catches defects."""

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            supports_vector_search=True,
            supports_keyword_fts=True,
            required_filter_keys=frozenset(["tenant_id"]),
            supported_custom_filters={"active": bool},
        )

    def embedding_space(self) -> EmbeddingSpace:
        return EmbeddingSpace(model="gemini-embedding-2", dims=768)

    def vector_search(self, query_vector: list[float], filters: RAGFilterSpec, top_k: int) -> list[RAGNode[DefaultNodeMeta]]:
        # Defect 1: does NOT reject zero vectors
        # Defect 2: does NOT check required_filter_keys
        # Defect 3: ignores doc_ids filter
        return [
            RAGNode(
                node_id="broken_node_1",
                text="Some text",
                span=SourceSpan(doc_id="unrelated_doc", char_start=0, char_end=10),
                score=0.5,
                score_kind="cosine",
                metadata=DefaultNodeMeta(),
            )
        ]


def _node(node_id: str, text: str, doc_id: str, start: int = 0, seq: int = 0, **meta: Any) -> RAGNode[DefaultNodeMeta]:
    return RAGNode(
        node_id=node_id,
        text=text,
        span=SourceSpan(doc_id=doc_id, section_seq=seq, char_start=start, char_end=start + len(text)),
        metadata=DefaultNodeMeta(extra=meta),
    )


class RAGStoreContractTests:
    """Pytest mixin checking the RAGStore contract (§4.6).

    Hooks an app may override:
      - make_store(): required. A clean store.
      - make_owner(): the owner every kit write is made for. Default: no owner.
      - make_filters(doc_id=None, **extra): the read filters. Default: the owner's keys plus doc_ids.
      - make_scoped_store(store, owner): a view of `store` bound to `owner` (§5.6). Default None: check 16
        is skipped. Views of one store share its data, as two users' requests share one database.
      - full_scope_filters(): filters for a multi-section scope the app seeded in make_store(), for a
        store whose full-scope text does not come from kit writes. Default None: the kit writes its own.
    """

    # ── hooks ─────────────────────────────────────────────────────────────

    def make_store(self) -> RAGStore[Any]:
        """Factory creating a clean store instance populated with test data if necessary."""
        raise NotImplementedError

    def make_owner(self) -> OwnerScope:
        return OwnerScope()

    def make_filters(self, doc_id: str | None = None, **extra: Any) -> RAGFilterSpec:
        owner = self.make_owner()
        fields: dict[str, Any] = {}
        if owner.tenant_or_user_id is not None:
            fields["tenant_or_user_id"] = owner.tenant_or_user_id
        if owner.domain_id is not None:
            fields["domain_id"] = owner.domain_id
        if owner.custom:
            fields["custom"] = dict(owner.custom)
        if doc_id is not None:
            fields["doc_ids"] = [doc_id]
        fields.update(extra)
        return RAGFilterSpec(**fields)

    def make_scoped_store(self, store: RAGStore[Any], owner: OwnerScope) -> RAGStore[Any] | None:
        return None

    def full_scope_filters(self) -> RAGFilterSpec | None:
        return None

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def fingerprint_for(store: RAGStore[Any], node: RAGNode[Any]) -> Fingerprint:
        space = store.embedding_space()
        return Fingerprint(
            content_hash=hashlib.sha256(node.text.encode("utf-8")).hexdigest(),
            embedding_signature=f"{space.model}|{space.dims}|{space.task_type_documents}|kit-1",
            fts_signature="kit_analyzer|1" if store.capabilities().uses_analyzer_tokens else None,
        )

    def write(
        self,
        store: RAGStore[Any],
        doc_id: str,
        nodes: list[RAGNode[Any]],
        vectors: list[list[float]],
        section_seq: int = 0,
        owner: OwnerScope | None = None,
    ) -> int:
        """Write nodes the way the engine does: one SectionWrite, with fingerprints and the owner."""
        passages = [
            PreparedNode[Any](node=node, vector=vector, fingerprint=self.fingerprint_for(store, node))
            for node, vector in zip(nodes, vectors)
        ]
        write = SectionWrite[Any](
            doc_id=doc_id,
            section_seq=section_seq,
            owner=owner if owner is not None else self.make_owner(),
            passages=passages,
        )
        return store.upsert_section(write)

    # ── Check 1: Filters honoured ──────────────────────────────────────────────
    def test_check_1_filters_honoured(self):
        store = self.make_store()
        vec = [0.1] * DIM
        self.write(store, "doc1", [_node("n1", "Node 1 text", "doc1")], [vec])
        self.write(store, "doc2", [_node("n2", "Node 2 text", "doc2")], [vec])

        res = store.vector_search(vec, self.make_filters("doc1"), top_k=5)
        assert [hit.node_id for hit in res] == ["n1"]

        res_bound = store.vector_search(vec, self.make_filters("doc1", max_char_offset=0), top_k=5)
        assert len(res_bound) == 0

    def test_check_1b_local_window_honours_filters_and_offsets(self):
        store = self.make_store()
        if not store.capabilities().supports_local_window:
            return
        text = "abcdefghijklmnopqrst"
        first = RAGNode(node_id="w1", text=text[:12], span=SourceSpan(doc_id="w_doc", char_start=0, char_end=12))
        second = RAGNode(node_id="w2", text=text[8:], span=SourceSpan(doc_id="w_doc", char_start=8, char_end=20))
        self.write(store, "w_doc", [first, second], [[0.1] * DIM, [0.1] * DIM])

        windows = store.fetch_local_window("w_doc", 10, 8, self.make_filters("w_doc"))
        assert len(windows) == 1
        window = windows[0]
        assert window.text == text[window.span.char_start:window.span.char_end], "overlap must not repeat text"

    def test_check_1c_rows_of_another_owner_are_invisible(self):
        store = self.make_store()
        owner = self.make_owner()
        if owner.tenant_or_user_id is None:
            return
        vec = [0.1] * DIM
        self.write(store, "own_doc", [_node("own_n", "Owned text", "own_doc")], [vec])
        other = self.make_filters("own_doc", tenant_or_user_id=f"{owner.tenant_or_user_id}-other")
        try:
            assert store.vector_search(vec, other, top_k=5) == []
        except ScopeViolationError:
            pass  # a scoped store may refuse instead of returning nothing

    # ── Check 2: Required scope enforced ───────────────────────────────────────
    def test_check_2_required_scope_enforced(self):
        store = self.make_store()
        if not store.capabilities().required_filter_keys:
            return
        with pytest.raises(MissingRequiredScopeError):
            store.vector_search([0.1] * DIM, RAGFilterSpec(), top_k=5)

    # ── Check 3: Ordering descending and ranks 1-based ─────────────────────────
    def test_check_3_ordering_and_ranks(self):
        store = self.make_store()
        nodes = [_node("n1", "Text 1", "d1"), _node("n2", "Text 2", "d1", start=10)]
        self.write(store, "d1", nodes, [[0.9] * DIM, [0.1] * DIM])
        res = store.vector_search([0.9] * DIM, self.make_filters("d1"), top_k=2)
        assert len(res) == 2
        assert res[0].score >= res[1].score
        assert res[0].vector_rank == 1
        assert res[1].vector_rank == 2

    # ── Check 4: Idempotence, deletes, fingerprints ────────────────────────────
    def test_check_4_upsert_idempotence_and_deletes(self):
        store = self.make_store()
        node = _node("n_idemp", "Original text", "d_idemp")
        self.write(store, "d_idemp", [node], [[0.1] * DIM])
        self.write(store, "d_idemp", [node], [[0.1] * DIM])

        assert "n_idemp" in store.get_fingerprints("d_idemp")
        assert store.delete_document("d_idemp") == 1
        assert store.get_fingerprints("d_idemp") == {}

    # ── Check 5: Declared capabilities match working methods ───────────────────
    def test_check_5_capabilities_match_methods(self):
        store = self.make_store()
        if store.capabilities().supports_full_scope_fetch:
            assert isinstance(store.fetch_full_scope(self.make_filters("any_doc")), list)

    # ── Check 6: Zero-vector query rejected ────────────────────────────────────
    def test_check_6_zero_vector_query_rejected(self):
        store = self.make_store()
        with pytest.raises(EmbeddingProviderError):
            store.vector_search([0.0] * DIM, self.make_filters("any_doc"), top_k=5)

    # ── Check 7: Filtered recall ───────────────────────────────────────────────
    def test_check_7_filtered_recall(self):
        store = self.make_store()
        self.write(store, "target_doc", [_node("target_n", "Target text", "target_doc")], [[0.5] * DIM])
        for i in range(10):
            self.write(store, f"dist_doc_{i}", [_node(f"dist_{i}", f"Distractor {i}", f"dist_doc_{i}")], [[0.5] * DIM])

        res = store.vector_search([0.5] * DIM, self.make_filters("target_doc"), top_k=5)
        assert [hit.node_id for hit in res] == ["target_n"]

    # ── Check 8: Native hybrid raw scores ──────────────────────────────────────
    def test_check_8_native_hybrid_raw_scores(self):
        store = self.make_store()
        if not store.capabilities().supports_native_hybrid_rpc:
            return
        self.write(store, "h_doc", [_node("h_node", "hybrid content", "h_doc")], [[0.5] * DIM])
        res = store.hybrid_search("hybrid", [0.5] * DIM, self.make_filters("h_doc"), top_k=1)
        assert len(res) == 1
        if store.capabilities().returns_raw_scores:
            assert res[0].vector_score is not None
            assert res[0].keyword_score is not None
        assert res[0].score_kind == "rrf"

    # ── Check 9: fetch_nodes honours filters ───────────────────────────────────
    def test_check_9_fetch_nodes_honours_filters(self):
        store = self.make_store()
        self.write(store, "fn_doc", [_node("fn_1", "fetch content", "fn_doc", active=True)], [[0.1] * DIM])

        assert len(store.fetch_nodes(["fn_1"], self.make_filters("fn_doc"))) == 1
        assert store.fetch_nodes(["fn_1"], self.make_filters("other_doc")) == []

    # ── Check 10: CJK keyword search finds exact name ──────────────────────────
    def test_check_10_cjk_keyword_search(self):
        store = self.make_store()
        if not store.capabilities().supports_keyword_fts:
            return
        self.write(store, "cjk_doc", [_node("cjk_n", "张三来到了北京大学", "cjk_doc")], [[0.1] * DIM])

        res = store.keyword_search("张三", self.make_filters("cjk_doc"), top_k=5)
        assert len(res) >= 1
        assert res[0].node_id == "cjk_n"

    # ── Check 11: Keyword-only hit survives cosine floor ───────────────────────
    def test_check_11_keyword_hit_survives_cosine_floor(self):
        store = self.make_store()
        if not store.capabilities().supports_native_hybrid_rpc:
            return
        # A keyword match whose vector is orthogonal-ish to the query
        self.write(store, "k_doc", [_node("kw_only", "SpecialUniqueTerm", "k_doc")], [[0.0] * (DIM - 1) + [1.0]])

        res = store.hybrid_search("SpecialUniqueTerm", [0.5] * DIM, self.make_filters("k_doc", min_similarity=0.8), top_k=5)
        assert any(hit.node_id == "kw_only" for hit in res)

    # ── Check 12: Atomic section writes ────────────────────────────────────────
    def test_check_12_atomic_section_writes(self):
        store = self.make_store()
        nodes = [_node(f"atom_{i}", f"Text {i}", "atom_doc", start=i * 10) for i in range(5)]
        vectors = [[0.1] * DIM for _ in range(5)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut = executor.submit(self.write, store, "atom_doc", nodes, vectors)
            read_res = store.vector_search([0.1] * DIM, self.make_filters("atom_doc"), top_k=10)
            fut.result()

        assert len(read_res) in (0, 5), "a reader must see none or all of a section"

    # ── Check 13: CJK phrase precision ─────────────────────────────────────────
    def test_check_13_cjk_phrase_precision(self):
        store = self.make_store()
        if not store.capabilities().uses_analyzer_tokens:
            return
        nodes = [_node("full_n", "这里有李小明同学", "d1"), _node("part_n", "这里只有小明同学", "d1", start=20)]
        self.write(store, "d1", nodes, [[0.1] * DIM, [0.1] * DIM])

        group = TokenGroup(tokens=["李小", "小明"], mode="phrase", origin="exact_term")
        res = store.keyword_search("李小明", self.make_filters("d1"), top_k=5, analyzer_tokens=[group])
        assert len(res) >= 1
        assert res[0].node_id == "full_n"
        assert all(hit.node_id != "part_n" for hit in res), "a phrase group must not match part of the name"

    # ── Check 14: Snippets arrived unchanged ───────────────────────────────────
    def test_check_14_snippets_arrive_unchanged(self):
        store = self.make_store()
        if not store.capabilities().returns_snippets:
            return
        self.write(store, "s_doc", [_node("snip_n", "Snippet test text", "s_doc")], [[0.1] * DIM])
        res = store.keyword_search("Snippet", self.make_filters("s_doc"), top_k=1)
        assert len(res) == 1
        assert res[0].snippet is not None
        assert res[0].snippet_kind == "match_excerpt"

    # ── Check 15: Fingerprints persisted as given ──────────────────────────────
    def test_check_15_fingerprints_persisted_as_given(self):
        store = self.make_store()
        if not store.capabilities().supports_fingerprints:
            return
        node = _node("fp_node", "Fingerprint text", "fp_doc")
        self.write(store, "fp_doc", [node], [[0.1] * DIM])
        stored = store.get_fingerprints("fp_doc")["fp_node"]
        assert stored == self.fingerprint_for(store, node), "a store persists the engine's fingerprint, never its own"

    # ── Check 16: Scoped store refuses another scope ───────────────────────────
    def test_check_16_scoped_store_refuses_other_scope(self):
        base_owner = self.make_owner()
        owner_a = base_owner.model_copy(update={"tenant_or_user_id": "kit-user-a"})
        owner_b = base_owner.model_copy(update={"tenant_or_user_id": "kit-user-b"})
        shared = self.make_store()
        store_a = self.make_scoped_store(shared, owner_a)
        store_b = self.make_scoped_store(shared, owner_b)
        if store_a is None or store_b is None:
            return
        vec = [0.1] * DIM
        filters_a = self.make_filters("scoped_doc", tenant_or_user_id="kit-user-a")
        filters_b = self.make_filters("scoped_doc", tenant_or_user_id="kit-user-b")
        self.write(store_a, "scoped_doc", [_node("sc_n", "Scoped text", "scoped_doc")], [vec], owner=owner_a)

        # A read or write that names another owner is refused
        with pytest.raises(ScopeViolationError):
            store_a.vector_search(vec, filters_b, top_k=5)
        with pytest.raises(ScopeViolationError):
            self.write(store_a, "scoped_doc_b", [_node("sc_b", "Other text", "scoped_doc_b")], [vec], owner=owner_b)
        # B cannot see A's rows, nor write into A's document even under its own owner
        assert store_b.vector_search(vec, filters_b, top_k=5) == []
        with pytest.raises(ScopeViolationError):
            self.write(store_b, "scoped_doc", [_node("sc_x", "Overwrite", "scoped_doc")], [vec], owner=owner_b)
        with pytest.raises(ScopeViolationError):
            store_b.delete_document("scoped_doc")
        # A still sees exactly its own row
        own = store_a.vector_search(vec, filters_a, top_k=5)
        assert [hit.node_id for hit in own] == ["sc_n"]

    # ── full-scope helpers for checks 17 and 18 ────────────────────────────────
    def _two_section_scope(self, store: RAGStore[Any]) -> tuple[RAGFilterSpec, list[tuple[int, int, int]] | None]:
        """Filters for a scope with two sections, and their (section_seq, start, end) when the kit wrote them."""
        seeded = self.full_scope_filters()
        if seeded is not None:
            return seeded, None
        first = "First section text. It has two sentences."
        second = "Second section text. It also has two."
        # Engine-shaped: the splitter's passages overlap, so section 0 arrives as two overlapping passages (K7)
        p0 = _node("fs_0", first[:30], "fs_doc", 0, 0)
        p1 = _node("fs_0b", first[20:], "fs_doc", 20, 0)
        self.write(store, "fs_doc", [p0, p1], [[0.1] * DIM, [0.1] * DIM], section_seq=0)
        self.write(store, "fs_doc", [_node("fs_1", second, "fs_doc", len(first), 1)], [[0.1] * DIM], section_seq=1)
        ranges = [(0, 0, len(first)), (1, len(first), len(first) + len(second))]
        return self.make_filters("fs_doc"), ranges

    # ── Check 17: scope_stats describes what fetch_full_scope returns ──────────
    def test_check_17_scope_stats_matches_full_scope(self):
        store = self.make_store()
        caps = store.capabilities()
        if not caps.supports_full_scope_fetch:
            return
        filters, _ = self._two_section_scope(store)
        nodes = store.fetch_full_scope(filters)
        stats = store.scope_stats(filters)
        text_chars = sum(len(node.text) for node in nodes)
        assert stats.char_count == text_chars, "scope_stats must describe the text fetch_full_scope returns (INT-06)"
        assert (stats.total_tokens > 0) == (text_chars > 0)

    # ── Check 18: full scope never crosses a section boundary ──────────────────
    def test_check_18_full_scope_granularity(self):
        store = self.make_store()
        if not store.capabilities().supports_full_scope_fetch:
            return
        filters, ranges = self._two_section_scope(store)
        nodes = store.fetch_full_scope(filters)
        assert len(nodes) >= 2, "a scope with two sections returns at least one node per section (INT-07)"

        order = [(n.span.doc_id, n.span.section_seq, n.span.char_start) for n in nodes]
        assert order == sorted(order), "nodes come in document order"
        for prev, nxt in zip(nodes, nodes[1:]):
            if prev.span.doc_id == nxt.span.doc_id:
                assert prev.span.char_end <= nxt.span.char_start, "nodes do not overlap"
        if ranges is not None:
            for node in nodes:
                seq, start, end = next(r for r in ranges if r[0] == node.span.section_seq)
                assert start <= node.span.char_start and node.span.char_end <= end, "no node crosses a section"


    # ── helpers for the 1.2.0 checks below ─────────────────────────────────────
    def write_with_summary(
        self,
        store: RAGStore[Any],
        doc_id: str,
        passages: list[RAGNode[Any]],
        summary: RAGNode[Any],
        section_seq: int = 0,
    ) -> int:
        """A section write with a section summary, as the engine sends it (§4.4.1)."""
        write = SectionWrite[Any](
            doc_id=doc_id,
            section_seq=section_seq,
            owner=self.make_owner(),
            passages=[
                PreparedNode[Any](node=n, vector=[0.1] * DIM, fingerprint=self.fingerprint_for(store, n))
                for n in passages
            ],
            section_summary=SummaryWrite[Any](
                node=summary, vector=[0.1] * DIM, fingerprint=self.fingerprint_for(store, summary)
            ),
        )
        return store.upsert_section(write)

    @staticmethod
    def _summary(node_id: str, doc_id: str, start: int, end: int, seq: int = 0) -> RAGNode[Any]:
        return RAGNode(
            node_id=node_id, text="A summary of the section.", is_summary=True, granularity="parent_summary",
            parent_kind="section", level=1,
            span=SourceSpan(doc_id=doc_id, section_seq=seq, char_start=start, char_end=end),
        )

    # ── Check 19: summaries are admitted by their end, ranked searches never return them (K1) ──
    def test_check_19_summary_boundary_and_ranked_search(self):
        store = self.make_store()
        caps = store.capabilities()
        if not caps.supports_parent_summaries:
            return
        first = _node("sb_p0", "First section text.", "sb_doc", 0, 0)
        second = _node("sb_p1", "Second section text.", "sb_doc", 100, 1)
        self.write_with_summary(store, "sb_doc", [first], self._summary("sb_doc#s0_sum", "sb_doc", 0, 100, 0), 0)
        self.write_with_summary(store, "sb_doc", [second], self._summary("sb_doc#s1_sum", "sb_doc", 100, 200, 1), 1)

        hits = store.vector_search([0.1] * DIM, self.make_filters("sb_doc"), top_k=10)
        assert hits and not any(h.is_summary for h in hits), "ranked searches never return summaries (§4.2)"

        if caps.supports_sequential_boundary:
            # The reader is inside section 1 (boundary 150): its summary ends at 200 and must not appear
            visible = store.fetch_parent_summaries(self.make_filters("sb_doc", max_char_offset=150))
            assert [s.node_id for s in visible] == ["sb_doc#s0_sum"], "summaries are admitted only if char_end <= B"

    # ── Check 20: per-document boundaries ───────────────────────────────────────
    def test_check_20_per_document_boundaries(self):
        store = self.make_store()
        if not store.capabilities().supports_sequential_boundary:
            return
        self.write(store, "pd_a", [_node("pd_a0", "A early.", "pd_a", 0), _node("pd_a1", "A late.", "pd_a", 500)],
                   [[0.1] * DIM, [0.1] * DIM])
        self.write(store, "pd_b", [_node("pd_b0", "B early.", "pd_b", 0), _node("pd_b1", "B late.", "pd_b", 500)],
                   [[0.1] * DIM, [0.1] * DIM])
        filters = self.make_filters(doc_ids=["pd_a", "pd_b"], max_char_offset_by_doc={"pd_a": 100, "pd_b": 1000})
        got = {h.node_id for h in store.vector_search([0.1] * DIM, filters, top_k=10)}
        assert got == {"pd_a0", "pd_b0", "pd_b1"}

    # ── Check 21: a partial section write keeps the section's other rows (H4) ──
    def test_check_21_partial_write_never_deletes(self):
        store = self.make_store()
        p0, p1 = _node("pw_0", "Kept passage.", "pw_doc", 0), _node("pw_1", "Changed passage.", "pw_doc", 50)
        self.write(store, "pw_doc", [p0, p1], [[0.1] * DIM, [0.1] * DIM])
        changed = _node("pw_1", "Changed passage, new text.", "pw_doc", 50)
        self.write(store, "pw_doc", [changed], [[0.1] * DIM])            # what sync sends: only the change
        got = {h.node_id for h in store.vector_search([0.1] * DIM, self.make_filters("pw_doc"), top_k=10)}
        assert got == {"pw_0", "pw_1"}, "upsert_section never deletes (§4.4.1)"

    # ── Check 22: ordering by score with distinct scores and a tie (K2) ─────────
    def test_check_22_ordering_with_a_tie(self):
        store = self.make_store()
        query = [1.0] + [0.0] * (DIM - 1)
        near = [1.0, 0.2] + [0.0] * (DIM - 2)
        far = [1.0, 1.0] + [0.0] * (DIM - 2)
        nodes = [_node("or_far", "far", "or_doc", 0), _node("or_t1", "tie one", "or_doc", 10),
                 _node("or_t2", "tie two", "or_doc", 20)]
        self.write(store, "or_doc", nodes, [far, near, near])
        first = store.vector_search(query, self.make_filters("or_doc"), top_k=3)
        second = store.vector_search(query, self.make_filters("or_doc"), top_k=3)
        assert [h.node_id for h in first][-1] == "or_far"
        assert all(a.score >= b.score for a, b in zip(first, first[1:]))
        assert [h.node_id for h in first] == [h.node_id for h in second], "ties are stable"
        assert [h.vector_rank for h in first] == [1, 2, 3]

    # ── Check 23: deletes and revisions behave as declared (K3, K4) ─────────────
    def test_check_23_deletes_and_revisions(self):
        store = self.make_store()
        caps = store.capabilities()
        nodes = [_node(f"dl_{i}", f"Text {i}", "dl_doc", i * 10, seq=i) for i in range(3)]
        for i, node in enumerate(nodes):
            node.revision = "r1"
            self.write(store, "dl_doc", [node], [[0.1] * DIM], section_seq=i)
        if caps.supports_revisions:
            assert store.current_revisions(["dl_0"]) == {"dl_0": "r1"}
        if caps.supports_delete:
            assert store.delete_nodes(["dl_0"]) == 1
            assert store.delete_sections("dl_doc", 2) == 1
            assert set(store.get_fingerprints("dl_doc")) == {"dl_1"}
        if caps.supports_fetch_nodes:
            assert [n.node_id for n in store.fetch_nodes(["dl_1"], self.make_filters("dl_doc"))] == ["dl_1"]

    # ── Check 24: a reader sees a section's passages and summary together (K6) ──
    def test_check_24_atomic_passages_and_summary(self):
        store = self.make_store()
        if not store.capabilities().supports_parent_summaries:
            return
        passages = [_node(f"at_{i}", f"Atomic text {i}", "at_doc", i * 10) for i in range(4)]
        summary = self._summary("at_doc#s0_sum", "at_doc", 0, 40)
        stop = []
        seen: list[tuple[int, int]] = []

        def reader():
            while not stop:
                hits = store.vector_search([0.1] * DIM, self.make_filters("at_doc"), top_k=10)
                sums = store.fetch_parent_summaries(self.make_filters("at_doc"))
                seen.append((len(hits), len(sums)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(reader)
            self.write_with_summary(store, "at_doc", passages, summary)
            stop.append(True)
            future.result()
        # Two calls are not one snapshot, but a reader must never see a partial set of passages
        assert all(n in (0, 4) for n, _ in seen)
