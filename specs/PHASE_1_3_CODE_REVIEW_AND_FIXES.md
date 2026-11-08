# Phase P1–P3 Deep Code Review & Fix Explanation Note

> **Notice**: *This is a temporary code review and fix explanation document on branch `feat/rag`. It documents every finding from the deep code review of the Phase P1–P3 RAG Engine implementation against `specs/RAG_SPEC.md`, explains why each finding was an issue, details the exact code and spec fixes applied, and provides a verification walkthrough for the next reviewing agent.*

---

## 1. Executive Summary

Following the initial Phase P1–P3 implementation of `cody_ai_shared_lib.rag` (documented in [`specs/PHASE_1_3_IMPLEMENTATION_REVIEW.md`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/specs/PHASE_1_3_IMPLEMENTATION_REVIEW.md)), a comprehensive architectural and line-by-line code review was conducted against [`specs/RAG_SPEC.md`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/specs/RAG_SPEC.md).

While the initial implementation had 160 passing tests and a solid high-level pipeline structure, the deep review uncovered **30+ concrete issues** across 8 categories:
1. **4 Critical Runtime Bugs & Crashes** (latent `AttributeError` crashes on multi-turn conversations, 0-token scope estimation bypassing Tier 0, ignored `RAGNode` span aliases, and broken `NodeRenderer.timestamp` calls).
2. **7 Security, Guard & Grounding Defects** (unparameterized built-in guards silently becoming no-ops unless manually instantiated by callers, missing pre-store filter validation, prompt-fence zero-width/Unicode bypasses, unverified citation markers left in output text, fabricated quotes passing `QuoteVerifier`, and unstripped external markdown links).
3. **7 Missing Phase 1–3 Spec Deliverables** (`generate_from_nodes`, `verify_answer`, `with_profile`, `InMemoryRAGStore.to_jsonl`/`from_jsonl`, `InMemoryLRUCache` + query embedding cache, `diff_retrievals`, and multi-zone `TokenBudgeter` + 3-block cache layout).
4. **11 Algorithmic & Logic Discrepancies vs. Spec** (partial relaxation ladder, static RRF weights, wrong `score_kind`, unshared Tier 0–3 routing logic in `engine.plan()`, hardcoded `BudgetGuard` token estimate and missing embed/rerank guard checks, missing `DG-07` zero-vector validation, un-narrowed `RAGFilterSpec.patch()` dates, and missing `tokenize_query_groups`).
5. **4 OOP, Module Segregation & Import Hierarchy Issues** (top-level `embedding.py` importing downward into `rag.text.analyzer`, flat placement of ingestion modules instead of `rag/ingestion/`, missing `rag/eval/` package, and per-call `ThreadPoolExecutor` creation).
6. **4 Dead Code & Unused Parameter Issues** (dead regexes, unused parameters, and missing `[RAG] id=...` summary log).
7. **Documentation Inaccuracies** in [`specs/PHASE_1_3_IMPLEMENTATION_REVIEW.md`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/specs/PHASE_1_3_IMPLEMENTATION_REVIEW.md) (broken links to non-existent file names).
8. **3 Minor Spec Contradictions** in [`specs/RAG_SPEC.md`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/specs/RAG_SPEC.md) (updated in version `1.1.12-draft`).

All findings have been resolved, 11 new multi-assertion verification tests were added in [`tests/rag/test_code_review_fixes.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_code_review_fixes.py), and all **171 tests across 17 test modules pass in ~4.3s with 0 warnings**.

---

## 2. Category 1: Critical Runtime Bugs & Crashes

### 1.1 `AttributeError: 'ChatTurn' object has no attribute 'text'` on Multi-Turn Conversations
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/context/assembler.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py)
  - [`cody_ai_shared_lib/rag/intent/router.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/intent/router.py)
  - [`cody_ai_shared_lib/rag/models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py)
- **Why It Was an Issue**:
  - [`ChatTurn`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L369-L374) defines `role: Literal["user", "assistant"]` and `content: str` (per §3.8).
  - However, `ContextAssembler.assemble()` accessed `t.text` (`f"{t.role}: {t.text}"`) and `AutoRouter.route()` accessed `getattr(t, 'text', str(t))` (which fell back to `repr(ChatTurn)` instead of `t.content`).
  - Furthermore, `RAGQueryRequest(query=..., conversation=conv)` only mapped `query -> question` in `_populate_question`, silently dropping `conversation` because the field was named `state: ConversationState | None`, and `ConversationState` lacked `turns: list[ChatTurn]`. Existing tests never passed `history` or `conversation` with `ChatTurn` objects, hiding the crash.
- **How It Was Fixed**:
  1. Added `turns: list[ChatTurn] = Field(default_factory=list)` to [`ConversationState`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L387-L398).
  2. Updated [`RAGQueryRequest._populate_question`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L421-L438) to map `conversation -> state` and automatically populate `history` from `state.turns` when `history` is not explicitly provided.
  3. Updated [`ContextAssembler`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py#L176-L185) and [`AutoRouter`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/intent/router.py#L64-L78) to read `getattr(t, "content", None) or getattr(t, "text", str(t))`.
  4. Updated [`ConversationManager.update`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/conversation.py#L14-L98) to append the current user/assistant `ChatTurn` pair to `state.turns` and populate full provenance on `NodeRef` (`doc_id`, `section_seq`, `char_start`, `char_end`, `score`).

---

### 1.2 `InMemoryRAGStore.scope_stats()` Returned `est_tokens = 0` When `node.token_estimate` Was Unset
- **Affected File**: [`cody_ai_shared_lib/rag/memory_store.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py#L406-L424)
- **Why It Was an Issue**:
  - `RAGNode.token_estimate` defaults to `0`. When nodes were constructed directly (without `prepare_nodes()`), `scope_stats()` summed `n.token_estimate` (`0`) and returned `ScopeStats(..., est_tokens=0, has_summaries=False)`.
  - Because `ScopeStats.total_tokens` checks `self.est_tokens if self.est_tokens is not None else self.char_count`, `total_tokens` evaluated to `0` (`0 is not None`), causing the Tier 0 token gate (`0 <= full_context_max_tokens`) to erroneously trigger even on massive documents, and never setting `has_summaries=True`.
- **How It Was Fixed**:
  - Updated [`InMemoryRAGStore.scope_stats`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py#L406-L424) to compute `n.token_estimate if n.token_estimate > 0 else estimate_tokens(n.text)` for each matching node and set `has_summaries = any(n.granularity in ("section_summary", "unit_summary") for n in matched)`.

---

### 1.3 `RAGNode._populate_aliases` Ignored Top-Level `doc_id` / `section_id` When `span` Was Provided
- **Affected File**: [`cody_ai_shared_lib/rag/models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L221-L263)
- **Why It Was an Issue**:
  - When a caller constructed `RAGNode(node_id="n1", doc_id="d1", text="...", span=SourceSpan(char_start=0, char_end=10))`, `_populate_aliases` had `if "span" not in data:` and skipped copying `doc_id`, `section_id`, `section_seq`, and `position_ratio` into `span`. As a result, `node.span.doc_id` remained `""`, breaking `ScopeGuard`, `SequentialBoundaryGuard`, and `InMemoryRAGStore` document filters.
- **How It Was Fixed**:
  - Updated [`RAGNode._populate_aliases`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L221-L263) to merge top-level `doc_id`/`document_id`, `section_id`, `section_seq`, and `position_ratio` into `span` (whether `span` was passed as a `SourceSpan` instance or a `dict`) whenever the corresponding field on `span` is unset/default.

---

### 1.4 `RecencyBooster._extract_timestamp` Passed `RAGNode` Instead of `node.metadata` to `NodeRenderer.timestamp()`
- **Affected File**: [`cody_ai_shared_lib/rag/postretrieval/boosters.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/boosters.py#L16-L31)
- **Why It Was an Issue**:
  - Per §3.4, `NodeRenderer[MetaT].timestamp(self, meta: MetaT) -> datetime | None` takes `node.metadata`, not `RAGNode`.
  - `RecencyBooster` called `renderer.timestamp(node)`, which failed whenever a custom `NodeRenderer` accessed metadata attributes or dict keys on its argument.
- **How It Was Fixed**:
  - Added [`extract_node_timestamp(node, renderer)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/boosters.py#L16-L51) which calls `renderer.timestamp(node.metadata)` first (with fallback to `renderer.timestamp(node)` and standard metadata fields).

---

## 3. Category 2: Security, Guard & Grounding Defects

### 2.1 Built-in Guards Were Instantiated With Default `None` Parameters and Ignored `request.filters`
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/guards.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/guards.py)
  - [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py)
- **Why It Was an Issue**:
  - `RAGEngine.__init__` created `self._builtin_guards` once at engine initialization with empty defaults (`SequentialBoundaryGuard(max_char_offset=None)`, `TenantGuard(tenant_id=None)`, `ScopeGuard(required_doc_ids=None)`, `TemporalGuard(date_range_utc=None)`).
  - When a caller followed the §16.1 integration contract—setting `profile = long_form_reader(...)` (which sets `guards=("sequential_boundary", "scope")`) and passing `RAGQueryRequest(query=..., filters=RAGFilterSpec(doc_ids=["book1"], max_char_offset=25000))` without manually constructing `SequentialBoundaryGuard` in `request.guards`—the engine's built-in `SequentialBoundaryGuard` had `self.max_char_offset = None` and silently became a no-op!
  - Additionally, `SequentialBoundaryGuard.prompt_rules()` returned the literal unformatted string `"...beyond offset {B}."` instead of substituting the actual offset, and `ContextAssembler` never called `guard.prompt_rules()`.
- **How It Was Fixed**:
  1. Implemented [`build_turn_guards(guard_names, filters, required_filter_keys)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/guards.py#L186-L240) to construct per-turn guard instances parameterized from the turn's effective `RAGFilterSpec` (`max_char_offset`, `max_char_offset_by_doc`, `tenant_id`, `doc_ids`, `date_range_utc`) and `profile.required_filter_keys`.
  2. Wired [`RAGEngine._collect_active_guards`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L285-L299) into `retrieve()`, `query()`, `stream_query()`, and `generate_from_nodes()`.
  3. Formatted `{B}` in [`SequentialBoundaryGuard.prompt_rules()`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/guards.py#L81-L86) and injected guard prompt rules into the system prompt in [`ContextAssembler`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py#L118-L125).

---

### 2.2 `required_filter_keys` and `supported_custom_filters` Were Never Validated Pre-Store
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/profile.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/profile.py)
  - [`cody_ai_shared_lib/rag/ports.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ports.py)
  - [`cody_ai_shared_lib/rag/errors.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/errors.py)
  - [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py)
- **Why It Was an Issue**:
  - Per §3.6, §5.6, and §13.3, the engine must validate `profile.required_filter_keys` (raising `MissingRequiredScopeError`) and `store.capabilities.supported_custom_filters` (raising `UnsupportedFilterError`) before invoking any store method.
  - Previously, `EngineProfile` did not declare `required_filter_keys`, and `RAGEngine` never raised `MissingRequiredScopeError` or `UnsupportedFilterError`.
- **How It Was Fixed**:
  1. Added `required_filter_keys: frozenset[str] = Field(default_factory=frozenset)` to [`EngineProfile`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/profile.py#L156).
  2. Added [`RAGEngine._validate_filters(filters)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L301-L343) and called it in `retrieve()` and `plan()` before any store interaction.

---

### 2.3 Prompt Fence Bypass & Missing XML Escaping (`ContextFencer` / `ContextAssembler`)
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/context/fence.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/fence.py)
  - [`cody_ai_shared_lib/rag/context/assembler.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py)
- **Why It Was an Issue**:
  - Per §10.4, untrusted text must have zero-width characters (`U+200B`–`U+200D`, `U+2060`, `U+FEFF`) stripped **before** regex tag scrubbing (so `</d\u200bata-...>` cannot bypass detection), neutralize `<` and `‹` (`U+2039`) before `data`, `history`, `question`, or `node`, XML-escape `<node>` attribute values, wrap history in `<history-ID>` and question in `<question-ID>`, and rotate `fence_id` on collision.
- **How It Was Fixed**:
  1. Updated [`ContextFencer.sanitize_content`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/fence.py#L42-L53) to strip zero-width characters first, then neutralize `<data-` / `‹data-` and any `<` or `‹` preceding `data`, `history`, `question`, or `node` with `<sanitized-...`.
  2. Added `escape_xml_attr` in [`fence.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/fence.py#L27-L29) and applied it to all `<node ...>` attributes in [`ContextAssembler`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py#L140-L157).
  3. Wrapped retrieved context in `<data-ID>`, conversation history in `<history-ID>`, and the user question in `<question-ID>`, with collision rotation across all untrusted texts.

---

### 2.4 `CitationExtractor` Left Out-of-Range Markers (`[99]`) in Cleaned Answer Text
- **Affected File**: [`cody_ai_shared_lib/rag/postgen/citations.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/citations.py#L22-L82)
- **Why It Was an Issue**:
  - Per §11.2, when an LLM emits an out-of-range citation marker like `[99]` (when only 5 nodes were in the prompt), `on_unverified="filter"` must strip `[99]` from the cleaned answer text while recording `RAGCitation(index=99, verified=False)` in `response.citations` for audit; `"flag"` must replace `[99]` with `[?]`; `"strict"` must abstain.
  - Previously, `CitationExtractor` returned `answer_text` completely unmodified under `"filter"`.
- **How It Was Fixed**:
  - Updated [`CitationExtractor.extract`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/citations.py#L22-L82) to strip unverified `[N]` markers (and collapse double spaces before punctuation) when `on_unverified == "filter"`, replace with `[?]` when `on_unverified == "flag"`, and record `RAGCitation(index=idx, node_id="", verified=False, verification_score=0.0)` in the returned citation list.

---

### 2.5 `QuoteVerifier` Always Returned `1.0` Because `CitationExtractor` Pre-Populated `cite.quote = node.text[:200]`
- **Affected File**: [`cody_ai_shared_lib/rag/postgen/quotes.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/quotes.py#L22-L123)
- **Why It Was an Issue**:
  - `CitationExtractor` sets `cite.quote = node.text[:200]` as a default preview excerpt. `QuoteVerifier.verify()` then compared `cite.quote` against `node.text` without inspecting the actual quoted spans in `answer_text`! Even if the LLM fabricated a quote in `answer_text` (`'The document states "fabricated claim" [1].'`), `QuoteVerifier` compared `node.text[:200]` against `node.text` and scored it `1.0` (`verified = True`).
- **How It Was Fixed**:
  - Updated [`QuoteVerifier.verify`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/quotes.py#L34-L93) to extract inline quoted spans (`"..." [N]`, `“...” [N]`, `「...」[N]`, `『...』[N]`) from `answer_text` associated with citation `[N]` and verify those actual quoted spans against `node.text` using normalized substring and `difflib.SequenceMatcher` window matching against `min_match_ratio`.

---

### 2.6 `OutputSanitizer` Never Stripped External Markdown Links
- **Affected File**: [`cody_ai_shared_lib/rag/postgen/sanitizer.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/sanitizer.py#L9-L76)
- **Why It Was an Issue**:
  - Per §11.6 and AC-RAG-43, `OutputSanitizer` must strip or neutralize markdown links `[label](https://...)` whose host is not present in retrieved node metadata URLs or `profile.allowed_link_hosts`, in addition to stripping markdown images `![alt](url)`, fence tags, and `[[answer_found=...]]` flags.
- **How It Was Fixed**:
  - Updated [`OutputSanitizer.sanitize`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/sanitizer.py#L31-L76) to collect hostnames from `nodes` metadata URLs and `allowed_link_hosts` (supporting exact host and subdomain matching) and replace unauthorized `[label](url)` links with plain `label`.

---

### 2.7 `RAGResponse.sources` Was Typed as `list[RAGNode]` Instead of `list[RAGCitation]`
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L455-L474)
  - [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py)
- **Why It Was an Issue**:
  - Per §3.11 and §11.1 (`RAG-CIT-7`), `RAGResponse.sources` is a `list[RAGCitation]` containing one entry per retrieved node included in the prompt (`index = 1..M`, `verified = False`), whereas `citations` contains only the markers actually cited in the answer.
- **How It Was Fixed**:
  - Changed `RAGResponse.sources` to `list[RAGCitation]` in [`models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L461) and constructed `RAGCitation` entries for all prompt nodes in [`RAGEngine._build_sources_from_nodes`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L389-L413).

---

## 4. Category 3: Missing Phase 1–3 Spec Deliverables

### 3.1 `RAGEngine.generate_from_nodes()` and `RAGEngine.verify_answer()` (§3.1, §14.1, §16)
- **Affected File**: [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L878-L982)
- **Why It Was an Issue**:
  - §3.1, §14.1 (`RAG-EVAL-2`), and §16 require `engine.generate_from_nodes(request, nodes)` (runs guard verification, context assembly, generation, and post-generation verification without store retrieval) and `engine.verify_answer(answer_text, response)` (runs deterministic citation/quote verification without an LLM call).
- **How It Was Fixed**:
  - Implemented [`RAGEngine.generate_from_nodes`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L878-L962) and [`RAGEngine.verify_answer`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L964-L982).

---

### 3.2 `RAGEngine.with_profile(profile)` (§3.1, §12.2, §16.2)
- **Affected File**: [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L261-L269)
- **Why It Was an Issue**:
  - Apps switching between profile variants (e.g., `news_chat` and `news_search` in §16.2) call `engine.with_profile(profile_variant)` to get a lightweight engine wrapper sharing the underlying store, gateway, and thread pool.
- **How It Was Fixed**:
  - Implemented [`RAGEngine.with_profile(profile)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L261-L269) which validates the new profile against the store capabilities and shares `store`, `gateway`, `renderer`, `clock`, `rng`, and `pool`.

---

### 3.3 `InMemoryRAGStore.to_jsonl(path)` and `from_jsonl(path)` (§14.1 `RAG-EVAL-10`)
- **Affected File**: [`cody_ai_shared_lib/rag/memory_store.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py#L483-L518)
- **Why It Was an Issue**:
  - §14.1 (`RAG-EVAL-10`) and the Phase P1 build list require JSONL snapshot export/import on `InMemoryRAGStore` for deterministic offline evaluation fixtures.
- **How It Was Fixed**:
  - Implemented [`to_jsonl(path)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py#L483-L497) and [`from_jsonl(path)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py#L499-L518) preserving nodes, document revisions, and embedding signatures.

---

### 3.4 `Cache` Protocol, `InMemoryLRUCache`, and Query Embedding Caching (§13.5)
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/cache.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/cache.py)
  - [`cody_ai_shared_lib/rag/gateway.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/gateway.py#L243-L390)
- **Why It Was an Issue**:
  - §13.5 and the Phase P2 build list require a `Cache` protocol (`get`, `set`, `delete`), a thread-safe byte-bounded `InMemoryLRUCache(max_bytes=32 MiB)`, and `RETRIEVAL_QUERY` embedding caching keyed by `rag:qemb:v1:{model}:{task_type}:{sha256(norm_query)}` with `DG-12` degradation on cache errors.
- **How It Was Fixed**:
  - Created [`cody_ai_shared_lib/rag/cache.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/cache.py) and wired query embedding cache lookup/population and `DG-12` fallback handling into [`ModelGateway.embed`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/gateway.py#L243-L390).

---

### 3.5 `diff_retrievals(result_a, result_b, k=10)` (§14.2)
- **Affected Files**:
  - [`cody_ai_shared_lib/rag/testing/diff.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/testing/diff.py)
  - [`cody_ai_shared_lib/rag/models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L555-L566)
- **Why It Was an Issue**:
  - §14.2 and the Phase P3 build list require `diff_retrievals(result_a, result_b, k=10) -> RetrievalDiff` computing `overlap_at_k`, Spearman `rank_correlation`, `only_in_a`, `only_in_b`, and `rank_deltas`.
- **How It Was Fixed**:
  - Created [`cody_ai_shared_lib/rag/testing/diff.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/testing/diff.py) and exported `diff_retrievals` from `cody_ai_shared_lib.rag.testing` and `cody_ai_shared_lib.rag`.

---

### 3.6 Multi-Zone `TokenBudgeter`, Turn-Distance Decay & 3-Block Cache Layout (§10.2, §10.3, §10.7)
- **Affected File**: [`cody_ai_shared_lib/rag/context/assembler.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py)
- **Why It Was an Issue**:
  - §10.2 specifies 4 token zones (`system_and_rules`, `conversation_state`, `parent_summaries`, `retrieved_passages`) where unused tokens from earlier zones roll into `retrieved_passages`.
  - §10.3 specifies turn-distance decay ($\text{effective\_score} = \text{base\_score} \times 0.85^{\text{current\_turn} - \text{last\_used\_turn}}$) for reused nodes.
  - §10.7 specifies a 3-block prompt cache layout (`static_system_prefix`, `semi_stable_context`, `per_turn_suffix`) with `cache_breakpoints`.
- **How It Was Fixed**:
  - Implemented [`TokenBudgeter`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py#L21-L68) and updated [`ContextAssembler`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/assembler.py#L71-L246) with zone rollover, turn-distance decay, and `assemble_with_breakpoints()`.

---

## 5. Category 4: Algorithmic & Logic Fixes vs. Spec

### 4.1 Complete 4-Step Relaxation Ladder (§7.6)
- **Fix**: Updated [`RAGEngine.retrieve`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L510-L579) to trigger relaxation when `len(nodes) < profile.retrieval.min_results_before_relax` across all 4 steps in order:
  1. `widen_time_window(x2)` (soft date range only)
  2. `drop_time_window` (soft date range only)
  3. `drop_entity_terms` (soft `intent.entity_terms` only)
  4. `lower_min_similarity:{old:.2f}->{new:.2f}` (lowers profile threshold by `0.10`, floor `0.30`; never lowers caller hard `request.filters.min_similarity`).

### 4.2 Intent-Driven RRF Weights & `score_kind = "rrf"` (§8.4)
- **Fix**: Updated [`HybridRRFStrategy`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hybrid_rrf.py#L31-L157) to select fusion weights from `profile.retrieval.fusion_weights` based on query intent (`exact_terms` $\rightarrow$ `(0.35, 0.65)`, `entity_terms` $\rightarrow$ `(0.45, 0.55)`, conceptual $\rightarrow$ `(0.70, 0.30)`) and set `node.score_kind = "rrf"` on fused nodes.

### 4.3 `JinaCrossEncoderReranker` `TRAIT_RERANKABLE` & `score_kind = "rerank"` (§9.3)
- **Fix**: Updated [`JinaCrossEncoderReranker`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/rerank.py#L23-L93) to only send nodes with `TRAIT_RERANKABLE` to the cross-encoder, set `node.score_kind = "rerank"` on reranked nodes, and re-merge non-rerankable summary nodes cleanly.

### 4.4 Shared Tier Decision Logic Between `AdaptiveStrategy` and `RAGEngine.plan()` (§8.2, §8.12)
- **Fix**: Extracted [`AdaptiveStrategy.decide_tier()`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/adaptive.py#L43-L119) implementing Rules 1–7 from §8.2 (`full_context` $\rightarrow$ `hierarchical` $\rightarrow$ `local_window` $\rightarrow$ `hybrid_rrf` $\rightarrow$ `dense_vector`) and reused it directly in [`RAGEngine.plan()`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L271-L283).

### 4.5 `BudgetGuard` Token Estimation, Dual Signature Support & Embed/Rerank Wiring (§13.1)
- **Fix**:
  - Added [`invoke_budget_guard`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/usage.py#L97-L116) supporting both 5-argument `before_call(stage, model, est_input_tokens, optional, ctx)` and 1-argument `before_call(ctx)` guards.
  - Replaced hardcoded `est_input_tokens=1000` in [`ModelGateway`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/gateway.py) with `estimate_tokens(...)` and wired `budget_guard` into `generate()`, `stream()`, `embed()`, and `rerank()`.

### 4.6 Zero Query Vector Handling (`DG-07`, §8.3)
- **Fix**: Added zero-vector detection in [`DenseVectorStrategy`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/dense_vector.py#L45-L62) and [`HybridRRFStrategy`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hybrid_rrf.py#L64-L85): if the query embedding is empty or all-zeros, records `DG-07` and degrades to keyword search if `supports_keyword_fts` is available, or raises `EmbeddingProviderError`.

### 4.7 `RAGFilterSpec.patch()` Date Range Narrowing (§4.3)
- **Fix**: Updated [`RAGFilterSpec.patch`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py#L145-L185) so `date_range_utc` narrows to `(max(start_a, start_b), min(end_a, end_b))` instead of overwriting.

### 4.8 `tokenize_query_groups` (§6.1)
- **Fix**: Implemented [`tokenize_query_groups(analyzer, query, exact_terms, entity_terms)`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/text/analyzer.py#L237-L263) in `cody_ai_shared_lib/rag/text/analyzer.py` and wired it into `KeywordStrategy` and `HybridRRFStrategy`.

---

## 6. Category 5 & 6: Module Segregation, Import Hierarchy & Dead Code Cleanup

1. **Fixed Inverted Dependency (`embedding.py` $\rightarrow$ `rag.text.analyzer`)**:
   - Moved `is_cjk_char` and `estimate_tokens` into [`cody_ai_shared_lib/pricing.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/pricing.py#L121-L150) so [`cody_ai_shared_lib/embedding.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/embedding.py#L14) imports from `.pricing` rather than reaching down into `.rag.text.analyzer`. `rag/text/analyzer.py` re-exports both functions for backward compatibility.
2. **Created `cody_ai_shared_lib/rag/ingestion/` and `cody_ai_shared_lib/rag/eval/` Subpackages**:
   - Created [`cody_ai_shared_lib/rag/ingestion/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/__init__.py) (`splitter.py`, `embedder.py`, `summarizer.py`) per §2.2 while maintaining backward-compatible re-exports in `rag/text/splitter.py`, `rag/embedder.py`, and `rag/summarizer.py`.
   - Created reserved [`cody_ai_shared_lib/rag/eval/__init__.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/eval/__init__.py) per §14.2 (`AC-RAG-53`).
3. **Engine-Level Bounded Thread Pool & Single-Line `[RAG] id=...` Summary Log (§13.2, §13.6)**:
   - `RAGEngine.__init__` now initializes a shared bounded thread pool (`self.pool = pool or make_bounded_pool()`) and passes it into `RetrievalContext`.
   - Added [`TraceCollector.format_summary_line`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/trace.py#L64-L88) and emitted the single-line `[RAG] id=... profile=... route=... strategy=...` structured log at the end of every `query()` and `stream_query()`.

---

## 7. Category 8: Spec Updates Applied to `specs/RAG_SPEC.md` (`1.1.12-draft`)

Three minor internal spec inconsistencies were harmonized in [`specs/RAG_SPEC.md`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/specs/RAG_SPEC.md) (bumped to `1.1.12-draft` with a new Appendix B revision entry):
1. **§3.11 `RAGStreamEvent.event`**: Added `"degradation"` to the `Literal` union in §3.11 to match the §10.8 event table and `AC-RAG-23`.
2. **§6.6 `ReindexStatus`**: Added `doc_id: str | None = None` to `ReindexStatus` so per-document staleness checks carry their document identifier.
3. **§13.1 `BudgetGuard.before_call`**: Documented support for both the 5-argument `(stage, model, est_input_tokens, optional, ctx)` signature and the 1-argument `(ctx: CallContext)` signature.

---

## 8. Verification Guide for the Next Reviewing Agent

To verify all fixes and run the full test suite:

```powershell
pytest -q
```

- **Test Suite Summary**: **171 passed in ~4.3s** across 17 test files.
- **Dedicated Code Review Fix Tests**: Inspect [`tests/rag/test_code_review_fixes.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_code_review_fixes.py), which directly exercises:
  1. `test_context_assembler_and_auto_router_with_conversation_history`
  2. `test_in_memory_store_scope_stats_estimates_tokens_when_unset`
  3. `test_parameterized_builtin_guards_from_request_filters`
  4. `test_pre_store_filter_validation_missing_scope_and_unsupported_custom`
  5. `test_fencer_zero_width_and_u2039_scrubbing`
  6. `test_citation_extractor_unverified_and_quote_verifier_fabricated`
  7. `test_output_sanitizer_allowed_link_hosts`
  8. `test_generate_from_nodes_verify_answer_and_with_profile`
  9. `test_memory_store_jsonl_roundtrip_and_diff_retrievals`
  10. `test_lru_cache_and_query_embedding_cache`
  11. `test_tokenize_query_groups_and_recency_booster_renderer_metadata`
