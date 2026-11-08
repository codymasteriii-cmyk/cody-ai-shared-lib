# RAG Engine Implementation Explanation (Phases P1 – P3)
> **Notice**: *This is a temporary implementation explanation document prepared for senior code review on branch `feat/rag`. It details the architecture, code modifications, algorithms, and test coverage across Phases P1, P2, and P3 of `specs/RAG_SPEC.md`. This file will be deleted once the code review is approved.*

---

## 1. Executive Summary & Architecture Overview

The `cody_ai_shared_lib.rag` package provides a unified, object-oriented, production-hardened Retrieval-Augmented Generation (RAG) framework shared across all Cody AI applications:
- **AudioReading** (`audiolib.codymasterai.com`): Long-form audiobooks, verbatim chapter-level context, sequential spoiler guards, and streaming audio QA.
- **NewsIntelligence / Pulse Nexus** (`ai.codymasterai.com`): Dated multi-topic corpora, hybrid search, recency scoring, comparison fan-out, temporal query resolution, and high-throughput feed search.
- **KnowledgeBase**: Fail-loud ADRs, wiki entries, and markdown structural analysis.

### Core Architectural Principles & Invariants
1. **DB-Free Core with Inversion of Control (D-01, D-02)**: The engine contains zero database drivers, SQL connections, or migrations. The data access contract is defined by the `RAGStore` protocol. Apps implement adapters (or use the provided reference PostgreSQL/pgvector SQL templates in `cody_ai_shared_lib/rag/sql/`), verified against the built-in conformance suite (`InMemoryRAGStore`).
2. **Stateless & Reentrant Facade (D-03)**: `RAGEngine` instances maintain no per-request state or cross-request session state. Turn state is encapsulated in `ConversationState`, safely passed to and from callers with size bounds and fence-checked cryptographic verification.
3. **Hard vs. Soft Constraint Separation (D-04)**: Security boundaries (guards, tenant isolation, required scope keys, caller-specified filters) are strictly **hard** and can never be relaxed or bypassed. Inferred or intent-derived constraints (temporal windows, entity requirements, score thresholds) are **soft** and subject to the relaxation ladder.
4. **Closed Degradation Model (DG-01 – DG-14, §13.3)**: Instead of silent degradation or unstructured exceptions, all edge-case fallbacks are governed by an agreed closed table of degradations recorded in `DegradationTracker` and surfaced in telemetry.
5. **Injected Runtime Determinism (RAG-EVAL-4)**: Clocks (`Clock`, `FrozenClock`), random number generators (`Rng`, `DeterministicRng`), and thread pools (`make_bounded_pool`) are strictly injected. The core never reads system time or global random state directly.
6. **Pydantic v2 Modernization**: Fully native Pydantic v2 models with typed generics (`RAGNode[MetaT]`, `QueryIntent[ExtT]`), field validators, and explicit package discovery in `pyproject.toml`.

---

## 2. Phase P1: Shared Foundation

Phase P1 established the shared library-wide model infrastructure, core contracts, data models, retrieval primitives, prompt assembly, and the primary execution pipeline.

### 2.1 Model Modules & Centralized Pricing Catalog (`cody_ai_shared_lib/`)
- **Central Pricing Catalog (`cody_ai_shared_lib/pricing.py`)**:
  - Implemented exact per-token and per-unit billing across providers: Gemini (`gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-4-argon`, `gemini-embedding-2`), OpenAI (`gpt-6-luna`, `gpt-6.1-terra`, `gpt-6.1-sol`, `gpt-6-astra`), Claude (`claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`), Jev Decision Router (`jev`), and cross-encoder rerankers (`jina-reranker`).
  - Added support for multimodal input/output units: images (`/img`) and video/voice seconds (`/sec`).
  - Implemented `calculate_cost()` with estimated flags and price-table versioning.
- **Provider Clients**:
  - [`LLMClient`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/llm.py): Unified text generation client wrapping Google GenAI SDK, Anthropic API, and OpenAI API, returning structured usage (`UsageTotal`), cost in USD, and provider-normalized `finish_reason` (`"stop"`, `"length"`, `"safety"`, `"other"`).
  - [`JevClient`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/jev.py): Fast-path decision router with structured JSON extraction and latency tracking.
  - [`RerankClient`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rerank.py): Isolated cross-encoder reranker calling Jina Rerank API with DG-06 degradation on network failure.

### 2.2 RAG Core Contracts & Data Models (`cody_ai_shared_lib/rag/`)
- **Node & Metadata Models ([`models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py))**:
  - `RAGNode[MetaT]`: Core unit of retrieval with character spans (`SourceSpan`), granularity, content hashes, embeddings, vector/keyword/rerank scores, and extensible metadata (`MetaT = TypeVar("MetaT", bound=BaseModel, default=DefaultNodeMeta)`).
  - `SourceSpan`: Exact document coordinates with character offsets (`char_start`, `char_end`), chapter/section numbers, and breadcrumbs.
  - `RAGFilterSpec`: Multi-tenant filters supporting `tenant_or_user_id`, `document_ids`, `date_range_utc`, boundary offsets (`max_char_offset`), and typed `custom` filters.
  - `QueryIntent[ExtT]`: Structured intent parsed from user queries including routing route (`"rag"`, `"no_retrieval"`), retrieval scope (`"local_lookup"`, `"document_overview"`), comparison flags, temporal ranges, entity terms, and suggested `top_k`.
- **Ports & Protocols ([`ports.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ports.py))**:
  - `RAGStore`: Standardized port for vector search, keyword search, hybrid search, node hydration, scope statistics (`scope_stats`), local window fetching, and atomic section ingestion.
  - `StoreCapabilities`: Detailed capability declaration (`supports_vector_search`, `supports_keyword_fts`, `supports_native_hybrid_rpc`, `supports_sequential_boundary`, `supports_hydration`, etc.).
  - `Guard`: Protocol for pre-retrieval filter patches (`to_filter()`) and post-retrieval candidate verification (`verify()`).
- **In-Memory Reference Store ([`memory_store.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/memory_store.py))**:
  - High-performance, thread-safe reference adapter implementing cosine vector similarity, BM25 term-overlap text search, reciprocal rank fusion (RRF), local window extraction, and scope stats for unit testing and offline development.

### 2.3 Guards & Visibility Control ([`guards.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/guards.py))
- `SequentialBoundaryGuard`: Prevents spoiler leakage in audiobooks and novels. Enforces that verbatim passages start strictly before `max_char_offset`, and summary nodes end before the boundary. Injects prompt rules informing the LLM never to mention future events.
- `TemporalGuard`: Drops nodes whose timestamps lie outside caller-specified date ranges.
- `TenantGuard`: Enforces multi-tenant isolation by dropping nodes whose tenant ID differs from the request.
- `ScopeGuard`: Requires mandatory scope keys (e.g. `doc_id` or `tenant_id`) preventing unrestricted global table scans.

### 2.4 Intent Routing Pipeline ([`intent/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/intent/))
- **`HeuristicRouter`**: 0ms regex-based classifier detecting greetings, document overviews (English & Chinese), continuation/follow-up prompts, and quoted exact terms.
- **`JevGateRouter`**: Fast System One classifier using TypeSafe Jev routing with strict timeout budgets.
- **`LLMIntentRouter`**: Structured LLM question rewriter producing clamped `QueryIntent` instances with DG-04 retry on malformed JSON and DG-05 value clamping.
- **`AutoRouter`**: Tiered orchestrator coordinating Heuristic -> Jev Gate -> LLM Rewriter -> Passthrough fallback, enriched with temporal extraction.

### 2.5 Retrieval Strategies Tier 1 & Tier 2 ([`strategies/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/))
- **Tier 1 Dense Vector ([`dense_vector.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/dense_vector.py))**: Embeds the query vector using the configured embedding model and queries `store.vector_search()`.
- **Tier 2 Hybrid RRF ([`hybrid_rrf.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hybrid_rrf.py))**: Dispatches native store hybrid RPC if supported (`supports_native_hybrid_rpc`); otherwise runs client-side vector search and keyword full-text search, combining rankings via Reciprocal Rank Fusion:
  $$\text{RRF Score} = \sum_{m \in \{\text{vec}, \text{kw}\}} \frac{w_m}{\text{rrf\_k} + \text{rank}_m}$$

### 2.6 Post-Retrieval Pipeline & Context Assembly
- **Threshold Filtering ([`threshold.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/threshold.py))**: `ThresholdFilter(min_similarity, kind="cosine")` strictly evaluates vector similarity on vector hits while preserving non-vector keyword hits in hybrid results. Raises `ProfileConfigError` if configured with impossible score kinds (AC-RAG-31).
- **Deduplication ([`dedup.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/dedup.py))**: `OffsetDeduplicator` eliminates chunks with >= 80% span overlap; `IdentityDeduplicator` merges identical document copies.
- **Context Assembly & Prompt Fencing ([`context/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/context/))**:
  - `ContextFencer`: Wraps retrieved context in cryptographic XML delimiters (`<context fence_id="...">`) to defeat prompt-injection and context boundary escape attacks.
  - `ContextAssembler`: Allocates context token budgets across system instructions, conversation history, verbatim passages, and summaries.
  - `ConversationManager`: Prunes older turns and rolling summaries within hard token caps.

### 2.7 Generation & Post-Generation Verification
- **Generation ([`generation/stuff.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/generation/stuff.py))**: Assembles fenced prompts with grounding instructions under `"strict"`, `"labelled_fallback"`, or `"unconstrained"` modes.
- **Citation Extraction & Verification ([`postgen/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postgen/))**:
  - `CitationExtractor`: Extracts bracketed citation markers `[1]`, `[2]`.
  - `QuoteVerifier`: Verifies direct quotes against source nodes with fuzzy whitespace normalization, flagging unverified claims.
  - `OutputSanitizer`: Strips leaked internal prompt instructions and fence markers before returning to callers.

---

## 3. Phase P2: AudioReading Adoption

Phase P2 focused on long-form sequential content, audiobook chapter ingestion, anti-spoiler guarantees, token-budgeted adaptive retrieval, and streaming SSE pipelines.

### 3.1 Streaming Engine & Cache Breakpoint Pass-Through (§15)
- **`LLMClient.stream()`**: Added native chunk streaming returning an iterator of token deltas, usage totals, and finish reason.
- **Cache Breakpoints**: Enabled Anthropic prompt caching headers and Google GenAI cache breakpoints across long-form book system prompts.

### 3.2 Text Splitters & Ingestion Lifecycle ([`ingestion/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/), [`ingestion/splitter.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/splitter.py), [`ingestion/embedder.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/embedder.py), [`ingestion/summarizer.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/summarizer.py))
- **`SentenceSplitter`**:
  - Sentence-aligned text splitter with target chunk sizes (default 1,200 chars) and overlap (default 150 chars).
  - Preserves exact character offsets such that `original_text[chunk.local_start:chunk.local_end] == chunk.text`.
  - Full support for Latin sentence boundaries (`.`, `!`, `?`) and CJK sentence terminators (`。`, `！`, `？`, `\n\n`). Subdivides oversized clauses at comma/semicolon boundaries (`，`, `；`, `,`, `;`).
- **`BatchEmbedder`**:
  - Batching embedder with maximum batch sizing (default 100 items), concurrent worker pools, and automatic exponential backoff retry on transient 429/503 errors.
- **`SectionSummarizer`**:
  - Generates hierarchical chapter and section summaries using dedicated summary LLM models, storing span coordinates and token estimates.
- **Ingestion Lifecycle & Synchronization ([`engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L980-L1080))**:
  - `sync_document()`: Fingerprint-driven synchronization checking sha256 content hashes, embedding model signatures, and revision IDs. Detects whether a document is current, needs partial section reindexing, or requires full reindexing.
  - `prepare_nodes()`: Computes node spans, tokens, fingerprints, and search tokens before store writes, ensuring deterministic store upserts.
  - Atomic section upsert writes via `store.upsert_section()` ensuring multi-statement consistency.

### 3.3 Adaptive Retrieval: Tier 0 Gate & Tier 3 Hierarchical
- **Tier 0 Token Gate (`full_context`, §8.2)**:
  - When the entire active document or scope has total tokens $\le \text{full\_context\_max\_tokens}$ (e.g. 120,000 tokens for Gemini Flash), the engine bypasses vector/keyword search entirely and loads the verbatim text directly into the prompt with zero retrieval distortion.
- **Tier 3 Hierarchical Strategy ([`hierarchical.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hierarchical.py))**:
  - For broad or structural questions, fetches parent chapter summaries first (`fetch_parent_summaries`), identifies matching regions, and expands local verbatim windows (`fetch_local_window`) around the match point up to `local_window_radius_chars` (default 6,000 chars).
- **Execution Plan Inspection ([`engine.plan()`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L257))**:
  - Evaluates adaptive strategy rules without incurring embedding, retrieval, or LLM generation costs, returning the predicted tier, strategy name, reason, and scope statistics.

### 3.4 Conversation State & Node Reuse (§7.7, AC-RAG-29, 30)
- Bounded `ConversationState` holding `prior_nodes` (capped by `max_reused_nodes = 24`), `rolling_summary`, and `profile_fingerprint`.
- On follow-up questions (`intent.is_followup = True`), re-fetches prior turn nodes via `store.fetch_nodes()` and passes them through all security guards under the current turn's boundaries before new retrieval.

### 3.5 Native Streaming & Flag Detection Buffer (§10.5, §10.9)
- **`RAGEngine.stream_query()` & `RAGStream`**:
  - Generates an event generator yielding tokens, citation updates, degradation notices, usage metrics, and completion events.
- **Flag Detection Buffer**:
  - Monitors streaming tokens for internal model control markers (e.g. `[[answer_found=yes]]`, `[[answer_found=no]]`).
  - Buffers characters matching the prefix `[[answer_found=` and releases normal text immediately upon disqualifying characters, preventing raw control tags from leaking to user UIs while avoiding streaming stalls.
- **Server-Sent Events (`format_event`)**:
  - Standard SSE formatting helper serializing typed events (`event: delta\ndata: ...\n\n`) compatible with AudioReading's streaming listeners.

### 3.6 Security Canary Anti-Spoiler Verification (§5.2, AC-RAG-10, 45)
- End-to-end security canary test (`test_canary_sequential_boundary_end_to_end`) verifying that with `max_char_offset = 25_000`, a canary token embedded at character offset 35,000 never appears in candidate retrieved nodes, prompt context, or generated LLM answers.

### 3.7 Reference SQL Templates (`cody_ai_shared_lib/rag/sql/`, §4.7)
Created and verified 7 production PostgreSQL/pgvector reference SQL templates:
1. [`hybrid_search.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/hybrid_search.sql): Dual-branch vector and FTS search with per-branch caps and RRF scoring.
2. [`vector_search.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/vector_search.sql): Cosine vector search with iterative index scan and filtered recall protection.
3. [`sequential_boundary.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/sequential_boundary.sql): Boundary offset predicates with per-document array arguments.
4. [`fingerprints.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/fingerprints.sql): Fingerprint inspection queries for content hashes and embedding model versions.
5. [`upsert_section.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/upsert_section.sql): Single-RPC atomic section transaction upserting passages and summaries.
6. [`indexes.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/indexes.sql): HNSW vector indexes, GIN full-text indexes, and B-tree compound indexes.
7. [`cjk_fts.sql`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/cjk_fts.sql): Simple-config `tsvector` over analyzer tokens and PGroonga CJK full-text search.

---

## 4. Phase P3: NewsIntelligence Migration

Phase P3 migrated the high-throughput, dated news corpus capabilities: temporal intent resolution, soft-constraint relaxation ladders, comparative query fan-out, recency score boosting, multi-signal score fusion, snippet extraction, and 2,000-result feed search sizing.

### 4.1 Pass-Through Splitter ([`ingestion/splitter.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/splitter.py), AC-RAG-03)
- `PassThroughSplitter`: Ingests pre-chunked news articles or standalone paragraphs where each input unit corresponds to exactly one retrievable node, preserving exact offsets `[0, len(text)]`.

### 4.2 Temporal Query Extraction & Resolution ([`temporal.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/intent/temporal.py), AC-RAG-17)
- **`TemporalExtractor`**: Resolves calendar expressions relative to an injected `Clock`:
  - `"last week"` / `"previous week"`: Resolves to Monday 00:00:00 UTC through Sunday 23:59:59 UTC of the preceding week.
  - `"this week"`: Monday 00:00:00 UTC through current time.
  - `"yesterday"` / `"today"`, `"past N days"`, and year/quarter patterns (`"2026-Q1"`, `"in 2025"`).
- **`resolve_range()`**: Automatically repairs reversed date bounds (if $\text{start} > \text{end}$, swaps them).
- **`applied_filters` Echo**: Returns the resolved date window in `RAGRetrievalResult.applied_filters.date_range_utc`, allowing client UIs to display "Searched Nov 2 – Nov 8".

### 4.3 Soft vs. Hard Constraints & Relaxation Ladder (§7.6, AC-RAG-18)
- **Constraint Distinction (D-04)**:
  - **Hard Constraints**: Explicitly supplied by the caller in `request.filters.date_range_utc` or guards. These are never relaxed under any circumstances.
  - **Soft Constraints**: Derived from intent temporal expressions (`intent.has_time_expression`).
- **Ladder Execution**: When retrieval against a soft time window returns zero candidate nodes, the engine executes the relaxation ladder:
  1. `widen_time_window(x2)`: Widens the date window by doubling the time delta backwards from the end date.
  2. `drop_time_window`: If widened search still yields zero results, drops the temporal constraint entirely.
  3. Records all triggered relaxation steps in `retrieval.relaxations` (`["widen_time_window(x2)"]` or `["drop_time_window"]`).

### 4.4 Secondary Query Fan-Out & Fair Interleaving ([`strategies/transforms.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/transforms.py), AC-RAG-19)
- For comparison queries (e.g. *"Compare Apple and Microsoft financials"*), `SecondaryQueryFanOut` extracts secondary query variants:
  - Variant 1: `Apple financial results` (weight 1.0)
  - Variant 2: `Microsoft financial results` (weight 1.0)
- Executes retrieval for each variant independently and merges candidate pools using `fair_interleave()`, alternating rank positions ($A_1, B_1, A_2, B_2, \dots$) so neither subject crowds out the other.

### 4.5 Post-Retrieval Boosters & Multi-Signal Fusion ([`boosters.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/boosters.py), AC-RAG-35)
- **`RecencyBooster`**:
  - Applies deterministic exponential time decay relative to the injected clock:
    $$\text{Score}_{\text{new}} = \text{Score}_{\text{old}} \cdot \left(1 + \text{weight} \cdot 2^{-\frac{\text{age\_days}}{\text{half\_life\_days}}}\right)$$
  - Reads timestamps from `renderer.timestamp(node)`, `node.metadata.created_at`, or `node.metadata.__pydantic_extra__`.
  - Stably re-sorts candidate nodes descending by boosted score.
- **`ScoreFusionStage`**:
  - Performs min-max normalized weighted fusion across multiple distinct retrieval signals:
    $$\text{Score}_{\text{fused}} = w_{\text{vec}} \tilde{S}_{\text{vec}} + w_{\text{kw}} \tilde{S}_{\text{kw}} + w_{\text{rec}} \tilde{S}_{\text{rec}} + w_{\text{feat}} \tilde{S}_{\text{feat}}$$
  - Sets `node.score_kind = "fused"` and stores raw component scores.

### 4.6 Snippet Handling & Prompt Exclusion (AC-RAG-63)
- When a store adapter declares `returns_snippets=True`, keyword search populates `node.snippet` (e.g. `match_excerpt` or summary).
- `ContextAssembler` preserves the snippet on the returned node for UI display while strictly excluding snippets from the generated prompt sent to LLMs.

### 4.7 Feed Search Sizing (2,000 Results) & Truncation Flag (§8.10)
- **High-Throughput Retrieval**: Configured `news_search` variant with `retrieve_max_results = 2000`, `top_k = 2000`, and `hydration_projection = "card"`.
- **Look-Ahead Truncation Detection**: Strategies query the store for `target_limit + 1` rows. If $N > \text{target\_limit}$, the result is sliced to exactly `target_limit` and `RAGRetrievalResult.truncated` is set to `True`.
- **Card Hydration**: Hydrates lightweight metadata (`"card"` projection) without loading heavy passage bodies.

### 4.8 Profile Presets & Variant Factory ([`profile.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/profile.py))
- Added `news_intelligence()` preset with named variants:
  - `news_chat()`: Interactive conversational news assistant with hybrid RRF, reranking, and recency boosting (`top_k = 12`).
  - `news_search()`: Feed search variant with passthrough router, 2,000-result sizing, card hydration, and no reranker.
- Implemented `EngineProfile.variant(name, **overrides)` and `EngineProfile.with_overrides()` supporting deep dictionary merges.

---

## 5. Build, Packaging & Flat-Layout Fix

### 5.1 The Flat-Layout Bug & Fix
- **Root Cause**: `pyproject.toml` previously had no `[tool.setuptools.packages.find]` directive. When `specs/` was added to the repository root, setuptools automatic package discovery detected multiple top-level directories (`['specs', 'cody_ai_shared_lib']`) and failed the build with `error: Multiple top-level packages discovered in a flat-layout`.
- **Fix**:
  1. Configured explicit package discovery in `pyproject.toml`:
     ```toml
     [tool.setuptools.packages.find]
     include = ["cody_ai_shared_lib*"]
     ```
  2. Moved reference SQL templates from `specs/sql/` to package data at `cody_ai_shared_lib/rag/sql/`.
  3. Registered SQL templates in `pyproject.toml`:
     ```toml
     [tool.setuptools.package-data]
     "cody_ai_shared_lib.rag" = ["sql/*.sql"]
     ```
  4. Added superseded warning banner to `specs/archive/RAG_SPEC_1.0.0-draft.md`.

### 5.2 Clean Export Build Verification
Verified build integrity in an isolated temp directory using `git archive`:
1. Ran `python -m pip wheel --no-deps -w <out> .` from clean export.
2. Successfully built `cody_ai_shared_lib-0.8.0-py3-none-any.whl`.
3. Verified wheel zip contents:
   - Exactly `cody_ai_shared_lib/` and `cody_ai_shared_lib-0.8.0.dist-info/` at top level.
   - Zero files from `specs/` or `tests/`.
   - All 7 reference SQL templates packaged in `cody_ai_shared_lib/rag/sql/*.sql`.

---

## 6. Complete Verification Matrix & Test Mapping

The test suite contains **171 tests** across 17 test modules. All pass with **100% pass rate and 0 warnings** in ~4.5 seconds.

| Test Module | Tests | Scope & Acceptance Criteria Covered |
| :--- | :---: | :--- |
| [`test_code_review_fixes.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_code_review_fixes.py) | 11 | Multi-turn conversation assembly & routing, scope_stats token estimation, parameterized built-in guards, pre-store filter validation, fencer zero-width/U+2039 scrubbing, unverified citation filtering/flagging, QuoteVerifier fabricated quote detection, OutputSanitizer host allowlist, `generate_from_nodes`/`verify_answer`/`with_profile`, JSONL snapshots, `diff_retrievals`, `InMemoryLRUCache` & query embedding cache, `tokenize_query_groups`, and `RecencyBooster` renderer metadata contract. |
| [`test_contracts_and_models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_contracts_and_models.py) | 10 | AC-RAG-40, 41, 46, 55, 56. Immutability, Pydantic v2 generics, Granularity registry. |
| [`test_engine_pipeline.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_engine_pipeline.py) | 17 | AC-RAG-20, 25, 27, 28, 31, 32, 33, 34, 36, 37, 39, 43, 44, 47, 51. End-to-end query, dedup, rerank, hydration, grounding modes. |
| [`test_gateway_profile_runtime.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_gateway_profile_runtime.py) | 16 | AC-RAG-26, 49, 50, 53, 57, 61, 62, 68, 69. Profile validation, budget guards, deadline enforcement, traces, clock/RNG determinism. |
| [`test_guards.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_guards.py) | 8 | AC-RAG-09, 10, 12, 15, 60. Sequential boundary guard, temporal guard, tenant isolation, scope guard. |
| [`test_ingestion_lifecycle.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_ingestion_lifecycle.py) | 5 | AC-RAG-05, 06, 07, 65, 71. `sync_document`, fingerprint cache, signature mismatches, `prepare_nodes`. |
| [`test_intent_and_temporal.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_intent_and_temporal.py) | 8 | AC-RAG-13, 14, 16. Heuristic regex classifier, Jev router fast-path, LLM rewriting, clamping. |
| [`test_memory_store_conformance.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_memory_store_conformance.py) | 16 | AC-RAG-46, 51. Conformance kit checks 1–15 for store adapters. |
| [`test_news_migration.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_news_migration.py) | 7 | **Phase P3**: AC-RAG-03 (`PassThroughSplitter`), AC-RAG-17 (`TemporalExtractor` + fixed clock), AC-RAG-18 (Relaxation ladder), AC-RAG-19 (`SecondaryQueryFanOut` + fair interleave), AC-RAG-35 (`RecencyBooster` + `ScoreFusionStage`), AC-RAG-63 (Snippet prompt exclusion), §8.10 (2,000 feed search sizing + card projection). |
| [`test_splitters_and_embedder.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_splitters_and_embedder.py) | 5 | **Phase P2**: AC-RAG-01 (`SentenceSplitter` offsets), AC-RAG-04 (`BatchEmbedder` 429 retry), AC-RAG-08 (`SectionSummarizer`), AC-RAG-59 (Chinese CJK punctuation splitting). |
| [`test_streaming_and_canary.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_streaming_and_canary.py) | 6 | **Phase P2**: AC-RAG-23 (`stream_query`), AC-RAG-45 / 72 / 73 (Anti-spoiler canary test), AC-RAG-54 (Flag detection buffer), AC-RAG-67 (Reference SQL package data in `sql/`), AC-RAG-70 (Usage on failure). |
| [`test_tier0_tier3_plan_conversation.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_tier0_tier3_plan_conversation.py) | 6 | **Phase P2**: AC-RAG-22 (Tier 0 token gate), AC-RAG-29 / 30 (Conversation state & prior node reuse), AC-RAG-38 (Tier 3 local window), AC-RAG-48 / 58 (`engine.plan()` dry-run). |
| [`test_fetcher_pdf.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/test_fetcher_pdf.py) | 11 | Existing pre-RAG PDF fetcher tests. |
| [`test_llm_stream.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/test_llm_stream.py) | 3 | Existing `LLMClient.stream()` delta and error tests. |
| [`test_model_prerequisites.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/test_model_prerequisites.py) | 16 | Existing model prerequisites and API key validation tests. |
| [`test_pdf_text.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/test_pdf_text.py) | 13 | Existing PDF text extraction tests. |
| [`test_pricing.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/test_pricing.py) | 13 | Central pricing catalog calculations, token counts, multimodal units, and overrides. |
| **Total** | **171** | **All passing (100% pass rate, 0 failures, 0 warnings)** |

---

## 7. Reviewer Guide: Key Files & Commits

### Git Branch & Commit Log (`feat/rag`)
- `0f17584`: `feat(rag): move reference SQL to package data, update spec references, and mark archive superseded`
- `f0a716f`: `Merge branch 'main' into feat/rag` (bringing in setuptools flat-layout fix `ba6a4b1` / `3d2cdea`)
- `803f833`: `feat(rag): implement NewsIntelligence migration components (AC-RAG-03, 17-19, 35, 63, §8.10)`
- `83f6293`: `feat(rag): implement native streaming, flag detection buffer, SSE helper, canary test, and reference SQL (AC-RAG-23, AC-RAG-45, AC-RAG-54, AC-RAG-67, AC-RAG-70, AC-RAG-72, AC-RAG-73)`
- `afb00b6`: `feat(rag): implement Tier 0 gate, Tier 3 hierarchical strategy, engine.plan(), and conversation reuse (AC-RAG-10, AC-RAG-22, AC-RAG-29, AC-RAG-30, AC-RAG-38, AC-RAG-48, AC-RAG-58)`
- `0f13ec6`: `feat(rag): implement ingestion lifecycle, sync_document, and prepare_nodes (AC-RAG-05, AC-RAG-06, AC-RAG-07, AC-RAG-65, AC-RAG-71)`
- `408143d`: `feat(rag): add text splitters, BatchEmbedder, and SectionSummarizer (AC-RAG-01, AC-RAG-03, AC-RAG-04, AC-RAG-08, AC-RAG-59)`
- `40b80d5`: `feat(llm): implement LLMClient.stream() and cache-breakpoint pass-through (§15)`
- `5d9b9a7`: `feat(rag): implement complete query pipeline, retrieval strategies, context assembly, generation, and RAGEngine facade (Phase P1)`

### Recommended Code Review Walkthrough Order
1. **Core Data Contracts**: [`cody_ai_shared_lib/rag/models.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/models.py) and [`ports.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ports.py) — inspect generic `MetaT` defaults, `RAGNode`, `SourceSpan`, and store capabilities.
2. **Main Engine Facade**: [`cody_ai_shared_lib/rag/engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py) — review `query()`, `retrieve()`, `stream_query()`, `sync_document()`, `prepare_nodes()`, and `plan()`.
3. **Retrieval Strategies**:
   - Tier 0 & Tier 1: [`strategies/dense_vector.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/dense_vector.py) and [`strategies/full_context.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/full_context.py).
   - Tier 2 & Tier 3: [`strategies/hybrid_rrf.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hybrid_rrf.py) and [`strategies/hierarchical.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/hierarchical.py).
   - Fan-Out & Comparison: [`strategies/transforms.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/strategies/transforms.py).
4. **Post-Retrieval Boosters & Fusion**: [`postretrieval/boosters.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/boosters.py) and [`postretrieval/threshold.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/postretrieval/threshold.py).
5. **Ingestion & Text Processing**: [`ingestion/splitter.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/splitter.py), [`ingestion/embedder.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/embedder.py), and [`ingestion/summarizer.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/ingestion/summarizer.py).
6. **Streaming & SSE**: [`engine.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/engine.py#L650) (`stream_query`), flag detection buffer, and [`integrations/sse.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/integrations/sse.py).
7. **Reference SQL**: [`cody_ai_shared_lib/rag/sql/`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/cody_ai_shared_lib/rag/sql/).
8. **Phase Tests**: [`tests/rag/test_news_migration.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_news_migration.py), [`test_streaming_and_canary.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_streaming_and_canary.py), and [`test_tier0_tier3_plan_conversation.py`](file:///c:/Users/minfa/AI%20Test/cody-ai-shared-lib/tests/rag/test_tier0_tier3_plan_conversation.py).
