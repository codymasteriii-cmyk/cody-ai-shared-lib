# RAG Engine Review: Code vs. Spec (1.2.0-draft), Findings and Fix Suggestions

| | |
| :--- | :--- |
| **Date** | 2026-11-11 |
| **Branch / state reviewed** | `feat/rag-1.2.0`, working tree at session start (uncommitted changes to `rag/*`, `specs/RAG_SPEC.md`, tests; untracked `rag/ingestion/ids.py`, `tests/rag/test_integration_round2.py`). HEAD `37d76fd`. |
| **Spec reviewed** | `specs/RAG_SPEC.md`, Document Version 1.2.0-draft (read in full except the tail of §17 and Appendices A–C). |
| **Code reviewed** | `cody_ai_shared_lib/rag/**` (see §1.3 for exactly which files were read), `cody_ai_shared_lib/llm.py` (cache-breakpoint paths only), `rag/sql/*.sql`. |
| **Reviewer** | Claude (Sonnet 5.5), read-only review. **No source, test or spec file was modified.** This file is the only artifact written. |
| **Audience** | A second agent who will review these findings. Every finding carries evidence you can re-check, a suggested fix, and a suggested test. |
| **Test baseline** | `pytest tests/rag` → **359 passed** (Python 3.13.5). All findings below are things the current tests do *not* catch. |

---

## 0. How to use this document

- Findings are grouped: **Blocker (B)**, **High (H)**, **Medium (M)**, **Conformance-kit weaknesses (K)**, **Spec-vs-code gaps (G)**, **Spec-text defects (S)**.
- Each finding has: **Where** (file:line or function), **Evidence**, **Impact**, **Suggested fix**, **Suggested test**.
- **Status tags**
  - **[PROBED]**: I reproduced it by running code (probe scripts and output in Appendix A).
  - **[READ]**: derived from reading code and spec; not executed. A reviewer should treat these as "likely" until reproduced.
- Line numbers refer to the working tree at review time and may drift after edits. Function/class names are given as a second anchor.
- §7 lists things I checked and found **correct**, so you do not need to re-litigate them.
- §8 lists what I did **not** read, so you know the coverage limits.

---

## 1. Scope, method, limits

### 1.1 Method
1. Read the whole spec, then the code in dependency order: engine facade, ports, models, ingestion (nodes, ids, sync, summarizer, embedder, splitter), memory store, retrieval pipeline, adaptive strategy and tiers, guards, assembler/fence/budgeter, answering/citations/quotes/sanitizer, gateway/usage/degradation/runtime, profile/registry, router/LLM rewriter/temporal, post-retrieval stages, conformance kit, SQL templates.
2. Formed hypotheses from spec-vs-code comparison, then wrote throwaway probe scripts (scratchpad only, outside the repo) against the real classes, using the same fake-embedding technique as `tests/rag/test_blocker_regressions.py`.
3. Ran `pytest tests/rag -q -p no:cacheprovider` with `PYTHONDONTWRITEBYTECODE=1` to avoid touching the tree. `git status` after the review was identical to the start.

### 1.2 One hypothesis that was **wrong** (dropped)
I suspected that `strategy_override` naming a strategy the store can't run would surface as `RetrievalError(NotImplementedError)`. Probe P12 showed it raises `StoreCapabilityError` (each strategy checks `is_runnable` in `retrieve`). Not a finding.

### 1.3 Files read (fully unless noted)
`rag/engine.py`, `ports.py`, `models.py`, `profile.py`, `errors.py`, `gateway.py`, `degradation.py`, `usage.py`, `runtime.py`, `granularity.py`, `guards.py`, `memory_store.py`, `retrieval_pipeline.py`, `answering.py`, `registry.py` (first ~60 lines + `register_granularity`), `ingestion/{nodes,ids,sync,summarizer,embedder,splitter}.py`, `strategies/{adaptive,shared,hybrid_rrf,hierarchical,keyword,dense_vector,full_context,context,relaxation}.py`, `context/{assembler,fence,budgeter,conversation}.py`, `generation/{flag_buffer,stuff}.py`, `postgen/{citations,quotes,sanitizer}.py`, `postretrieval/{dedup,rerank,threshold,boosters,stages}.py`, `transforms/fusion.py`, `intent/{router,llm_rewriter}.py`, `intent/temporal.py` (first 70 lines and the tail), `text/analyzer.py`, `testing/contract.py`, `integrations/sse.py`, `sql/{hybrid_search,vector_search,upsert_section,indexes}.sql` (parts), `sql/sequential_boundary.sql` (grep), `llm.py` (cache-breakpoint sections only), `tests/rag/test_integration_round2.py` (first 120 lines), `tests/rag/test_blocker_regressions.py` (first 140 lines).

---

## 2. Summary table

| ID | Sev | Area | One-line | Status |
| :-- | :-- | :-- | :-- | :-- |
| **B1** | Blocker | Packaging | `TypeVar(default=...)` needs Python 3.13; project claims 3.11+ | READ |
| **H1** | High | Sync | Summaries never re-embedded after signature change; `needs_reindex` stays stale forever | PROBED |
| **H2** | High | Fingerprints | `EmbeddingSpace.splitter_version` defaults to `"1.0.0"`, so splitter bumps are invisible | PROBED |
| **H3** | High | Scoping | Reference scoped store lets owner B overwrite owner A's document | PROBED |
| **H4** | High | SQL template | `upsert_section_nodes` RPC deletes the whole section first; breaks sync and §4.4.1 | READ |
| **H5** | High | SQL template | Hybrid/vector templates admit summaries by start offset; text-by-ref slices with wrong offset base; no tenant scope; unpinned `search_path` | READ |
| **H6** | High | Citations | `[n]` marker directly after a CJK character is not parsed | PROBED |
| **H7** | High | Caching | Anthropic cache breakpoint covers only the system prompt; stable data block never cached | READ |
| **H8** | High | Filters | Hard-filter conflict raises `MissingRequiredScopeError` instead of empty-scope answer | PROBED |
| **H9** | High | Temporal | Caller date range cancels (not intersects) the intent date window | READ |
| **H10** | High | Temporal | `date_window_days` is never converted into a date range | READ |
| **H11** | High | Retrieval | Unit summaries appear as ranked hits in Tier 1/2 and bypass `top_k` | PROBED |
| **H12** | High | Tier 3 | Section summary and its unit summaries are both returned | PROBED |
| **H13** | High | Tier 0 | Reference `fetch_full_scope` returns overlapping passages | PROBED |
| **M1–M31** | Medium | various | See §4 | mixed |
| **K1–K8** | Kit | Conformance | Kit checks are far weaker than §4.6 describes | READ |
| **G1–G12** | Gap | Spec vs code | Spec-described features/params missing or differing | READ/grep |
| **S1–S5** | Doc | Spec text | Internal spec inconsistencies | READ |

---

## 3. Blocker and High findings

### B1. Library cannot be imported on Python 3.11 / 3.12

- **Where:** `rag/models.py:30–31`, `MetaT = TypeVar("MetaT", bound=BaseModel, default=DefaultNodeMeta)` and `ExtT` likewise; `TypeVar` is imported from `typing` (line 9).
- **Evidence:** `pyproject.toml` → `requires-python = ">=3.11"`; spec header → "Python 3.11+". `default=` on `typing.TypeVar` is PEP 696, added in **3.13**. No `typing_extensions` import anywhere under `cody_ai_shared_lib/` (grep). The suite passes only because the dev machine runs 3.13.5. On 3.11/3.12 the module raises `TypeError: TypeVar.__init__() got an unexpected keyword argument 'default'` at import. (I could not run 3.11/3.12 here: `py` launcher absent. Based on the language change, not executed.)
- **Impact:** The 512 MB hosts (News, Audio) cannot import `cody_ai_shared_lib.rag` unless they run 3.13.
- **Fix (pick one):**
  1. `from typing_extensions import TypeVar` for `MetaT`/`ExtT`; add `typing_extensions>=4.4` to `dependencies` (already transitively present via pydantic; make it explicit).
  2. Or raise `requires-python` to `>=3.13` and update the spec header.
- **Test:** CI matrix on 3.11, 3.12, 3.13 running `python -c "import cody_ai_shared_lib.rag"` and the suite.

---

### H1. `sync_document` cannot refresh stale summaries; `needs_reindex` never clears  **[PROBED]**

- **Where:** `rag/ingestion/sync.py`: `_is_stale` (≈L173) and the loop at ≈L95–146; `engine.needs_reindex` (engine.py ≈L967).
- **Evidence (probe P5):** ingest a section with unit summaries, change the store's embedding signature (simulating a model/splitter upgrade), call `needs_reindex` → `embeddings_stale=True`; call `sync_document` → `needs_reindex` still `embeddings_stale=True`. Rows still on the old signature after sync: `['b#s0_u0_sum','b#s0_u1_sum','b#s0_u2_sum','b#s0_sum']`, i.e. every summary row.
- **Cause:** `to_embed` is computed only from `section_nodes[seq]` (passages). Summaries are rewritten only when `seq in changed_sections` (text changed). The class docstring states the intent ("a model upgrade re-embeds passages without paying to summarize again") but never re-embeds the *kept summaries*, which remain in the old vector space (RAG-ING-7 / NF-28: spaces must never be mixed). Separately, `_is_stale` compares only `content_hash` and `embedding_signature`, never `fts_signature`, so an analyzer-version change is detected by `needs_reindex` (`fts_stale`) but nothing can fix it (`reconcile()` does not exist; see G1).
- **Impact:** Permanent "stale" status; `reconcile` loops (once built); summary vectors from a different model than passage/query vectors if summaries are ever searched by vector.
- **Suggested fix:**
  - In `sync`, after computing `signature`, also collect stored summary ids (`parse_node_id(...).is_summary`) of *unchanged* sections whose `fingerprint.embedding_signature != signature`. For those sections:
    - if `caps.supports_fetch_nodes`, `fetch_nodes(ids, filters_for_owner)` to get the summary text, re-embed it with `embedder.embed_nodes(..., "summary_embedding")`, and include `SummaryWrite(node, vector, fingerprint=signatures.fingerprint(node), unit_seq=...)` in the section write;
    - else regenerate (documented cost) or raise a typed error telling the app that summary re-embedding needs `supports_fetch_nodes`.
  - Add `fts_signature` to staleness: when `caps.uses_analyzer_tokens` and a stored fingerprint's `fts_signature != signatures.fts`, rewrite `search_tokens` for that node. `SectionWrite` currently requires a vector per passage, so either (a) re-use the engine's embedding path (cost) or (b) add an optional `RAGStore.update_search_tokens(items)` method and have `reconcile`/`sync` call it. Decision belongs to the spec owner; document it in §6.6.
  - Implement `reconcile` (G1) so `needs_reindex` has a consumer.
- **Suggested test:** extend `test_ingestion_lifecycle.py`: ingest → bump `splitter_version` (or embedding model) → `sync_document` → assert `needs_reindex(...).embeddings_stale is False` and that every fingerprint (including summaries) carries the new signature, and that no LLM summarize call was made.

---

### H2. Splitter version bumps are never detected  **[PROBED]**

- **Where:** `rag/ports.py:33` `EmbeddingSpace.splitter_version: str | None = "1.0.0"`; `rag/memory_store.py:81–86` also passes `"1.0.0"`; consumer: `rag/ingestion/nodes.py:65` `space.splitter_version or getattr(splitter, "version", "1.0.0")`.
- **Evidence (probe P4):** default store reports `splitter_version = 1.0.0`; after `eng.splitter.version = "9.9.9"`, `needs_reindex("b")` → `embeddings_stale=False`.
- **Cause:** spec §4.1 declares the default as `None` ("or the splitter's own version when the store declares none", §6.6). The non-`None` default means *every* store that doesn't override it "declares" 1.0.0, so the fallback is dead code.
- **Fix:** default `splitter_version: str | None = None` in `EmbeddingSpace`; in `InMemoryRAGStore.__init__` pass `None` (or have its legacy `_fingerprint()` use `space.splitter_version or "unspecified"`). Keep the engine fallback.
- **Test:** AC-RAG-80 already says "Changing the splitter version or the analyzer version makes exactly the matching flag true". Add the splitter case against a default `InMemoryRAGStore()` (not one that sets the space explicitly).

---

### H3. Reference scoped store allows cross-owner overwrite  **[PROBED]**

- **Where:** `rag/memory_store.py:845–847` `ScopedInMemoryRAGStore.upsert_section`; `InMemoryRAGStore.upsert_section` (≈L542–555).
- **Evidence (probe P6):** owner A ingests doc `b`; a scoped store for owner B then ingests a different text under the same `doc_id` `b` with `owner=B`. Result: "B overwrote A's doc; doc_owner now: tenant_or_user_id='B'", and the stored node text became "Overwritten by B…".
- **Cause:** `upsert_section` only calls `_check_owner(write.owner)` (does the write's owner equal the store's owner). It never calls `_check_doc(write.doc_id)` (does the *existing* doc belong to this owner), unlike `get_fingerprints`/`delete_*`. Node ids are `{doc}#s{seq}p{i}` so same-doc writes collide by design.
- **Impact:** In a multi-user deployment where `doc_id` is attacker-influenced (KB page ids, user-chosen ids), a user can overwrite another user's content. Spec §5.6 says a scoped store must refuse "a read, write, fingerprint lookup or delete that names another scope". This is the reference implementation apps are told to copy.
- **Fix:** in `ScopedInMemoryRAGStore.upsert_section` call `self._check_doc(write.doc_id)` before delegating. Also make `InMemoryRAGStore.upsert_section` refuse (or at least log) when an *existing* doc owner differs from `write.owner` and both are non-empty. Document in §5.6 that adapters must enforce ownership of the target document on writes, not only the owner argument.
- **Test:** extend kit check 16 (`make_scoped_store`): after A writes `scoped_doc`, a store scoped to B writing `scoped_doc` with `owner=B` must raise `ScopeViolationError` (see K-list).

---

### H4. SQL upsert template deletes the section before writing  **[READ]**

- **Where:** `rag/sql/upsert_section.sql` ≈L18–23: `DELETE FROM passages WHERE doc_id = p_doc_id AND section_seq = p_section_seq;` followed by `INSERT`.
- **Spec conflict:** §4.4.1: "Idempotent on `node_id`, **never deletes**. During sync a write may hold only the changed nodes of a section, or only its refreshed summaries." The engine's `sync` does exactly that (`SectionWrite.passages` = only `to_embed` nodes).
- **Impact:** an adapter that uses this template as written loses all *unchanged* passages and all unit summaries of a section the first time `sync_document` re-embeds one passage. AudioReading is named as the template's first real user (§4.7).
- **Also stale vs 1.2.0:** the RPC signature is `(p_doc_id, p_section_seq, p_nodes jsonb, p_summary jsonb)`: no unit summaries (`unit_seq`), no per-item fingerprint fields beyond a few columns, no owner/tenant, summary insert omits `search_tokens`/`fts_search_tokens`/`is_active`, and `SECURITY DEFINER` is unpinned (H5).
- **Fix:** rewrite as `INSERT ... ON CONFLICT (id) DO UPDATE` with no delete; accept `p_passages`, `p_unit_summaries`, `p_section_summary` (jsonb arrays, each item carrying `fingerprint` fields and `unit_seq`), plus `p_owner jsonb`; write all in one function body (single transaction). Keep deletion in a separate `delete_nodes` RPC. Add a template test or at least a documented "contract: never deletes" header.
- **Test:** none possible inside the lib (no DB). Add to the conformance kit a check that a partial write (one passage of a two-passage section) leaves the other passage readable (K-list), which any adapter built from this template would fail.

---

### H5. Hybrid/vector SQL templates: boundary, offsets, tenant, hardening  **[READ]**

- **Where:** `rag/sql/hybrid_search.sql` ≈L61–66 & L84–89, `vector_search.sql` ≈L41–47, compared with `sequential_boundary.sql` ≈L49–59 and `indexes.sql`.
- **H5a. Summary admitted by start offset.** `sequential_boundary.sql` keeps summaries in the same `passages` table (`granularity = 'parent_summary'`) and admits them only if `char_end <= boundary`. `hybrid_search.sql` and `vector_search.sql` apply `p.char_start < p_boundary` to *all* rows. If summary rows are embedded (`embed_summaries=True` default) they are candidates in both branches, so a summary whose section starts before the boundary passes: the exact 1.0.0 spoiler leak (§0 item 1, C-01). The same SQL also doesn't exclude summary rows from passage ranking (cf. H11).
- **H5b. Wrong base for text-by-reference.** All three use `SUBSTRING(c.text FROM (p.char_start + 1) FOR (p.char_end - p.char_start))`. Spec §3.2: for AudioReading, `char_start/char_end` are **global article offsets** and the source is a *chunk* table. Slicing a chunk's text with a global offset is only correct for a chunk starting at 0. Needs `p.char_start - c.char_start`.
- **H5c. No tenant/owner scope anywhere.** `grep -i "tenant|owner|user_id" *.sql` matches only a comment in `indexes.sql`. §5.6/§6.4.2 require `tenant_or_user_id` for Audio; the template RPCs are `SECURITY DEFINER`, called with the service role, so the RPC is the only isolation layer. `indexes.sql` item 4 says "B-Tree on tenant or domain scope column" but creates `(doc_id, is_active)`.
- **H5d. `SECURITY DEFINER` without `SET search_path`** on all five functions (Supabase linter flags this; REVOKE mitigates but does not replace pinning).
- **H5e. Filtered-recall comments, no mechanism.** `vector_search.sql` says "exact table scan / indexed btree … is preferred" but the query is a plain `ORDER BY embedding <=> q LIMIT n` with a `WHERE doc_id = …`; whether HNSW post-filters is up to the planner. No `SET LOCAL hnsw.iterative_scan` (pgvector ≥ 0.8) and no forced exact path.
- **Fix:** boundary predicate `CASE WHEN p.granularity = 'parent_summary' THEN p.char_end <= b ELSE p.char_start < b END` in every branch (or exclude summaries from passage search entirely); fix offset base; add `p_tenant_or_user_id`/`p_domain_id` params with a mandatory predicate and the matching index; `SET search_path = public, pg_temp` on every function; add `SET LOCAL hnsw.iterative_scan = 'relaxed_order'` and/or an exact-scan branch when `p_doc_id IS NOT NULL`.
- **Test:** the kit's check 1 should include a summary-near-boundary case (K1), so any adapter built from these templates is caught.

---

### H6. Citation marker glued to a CJK character is not recognised  **[PROBED]**

- **Where:** `rag/postgen/citations.py:61` `RefCitationFormatter.pattern = r"(?<![\w)])\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]"`.
- **Evidence (probe P-cit, output):**
  ```
  '张三是主角[1]。'            -> []
  '张三是主角。[1]'            -> [('[1]', 6, 9)]
  'He fled the city [1].'     -> [('[1]', 17, 20)]
  'He fled the city[1].'      -> []
  'The result (see table)[2].' -> []
  ```
- **Cause:** in Python 3 regex, `\w` matches CJK characters, so the lookbehind rejects a marker immediately following any CJK character. The intent (avoid `arr[1]`, `[2024]`) is reasonable for ASCII.
- **Impact:** For Chinese answers (the primary AudioReading language) a very common style, "…主角[1]。", yields no citation, no verification, the literal `[1]` left in `answer_final`, and zero citation coverage.
- **Fix:** replace the lookbehind with an ASCII-only exclusion, e.g. `(?<![A-Za-z0-9_\]])`. Better: parse permissively and accept a marker only if it **resolves in `turn_map`** (already done for multi-marker lists); keep the existing code-span exclusion. Decide on `)` explicitly (spec silent).
- **Test:** parametrised test with the five strings above plus `"见第[3]章"` where `[3]` is not in the turn map (should stay text).

---

### H7. Anthropic prompt caching cannot cover the stable data block  **[READ]**

- **Where:** `rag/context/assembler.py:226–236` (`cache_breakpoints = [len(system_prompt), len(f"Retrieved Context:\n<data-…>\n{data_blocks[0]}")]`), `llm.py:463–468` and `:769–772`.
- **Evidence:** In `llm.py` the value of `cache_breakpoints` is used only as a boolean: `if cache_breakpoints: system=[{"type":"text","text":system_prompt,"cache_control":{"type":"ephemeral"}}]`. The user message is a single string (`"messages":[{"role":"user","content":user_message}]`), so no `cache_control` can be placed inside it. The second breakpoint is an offset into the **user message**, the first into the **system prompt**: two coordinate systems in one list.
- **Impact:** RAG-CTX-1's whole rationale ("Tier 0 cost is dominated by re-sending the same document … keep that prefix byte-stable") is not realised for Claude models: only the system prompt (usually far below Anthropic's minimum cacheable prefix, ~1024 tokens) is marked; the 100k-token stable block is not. Gemini implicit caching may still work if the prefix is stable, but nothing in the code verifies or reports it for Claude.
- **Fix:** make the assembler return structured blocks, e.g. `AssembledPrompt.user_blocks: list[PromptBlock(text:str, cache:bool)]` (stable data block `cache=True`, volatile `False`), keep `user_message` as the concatenation for providers that take a string. Teach `LLMClient` (Anthropic branch) to send `content=[{"type":"text","text":…, "cache_control":{"type":"ephemeral"}}, {"type":"text","text":…}]`. Define `cache_breakpoints` semantics in §10.3/§15.1 (block indices rather than char offsets) and keep it additive.
- **Test:** unit test on `LLMClient` with a fake Anthropic client asserting the stable block carries `cache_control`; test that `cached_tokens` is reported when the fake returns `cache_read_input_tokens`; assembler test that block texts concatenate to `user_message`.

---

### H8. Hard-filter conflict raises the wrong error for stores requiring `doc_ids`  **[PROBED]**

- **Where:** `rag/retrieval_pipeline.py:98–110` (`scoped_filters` calls `validate_scope` **before** `run()` checks `filters.is_empty_scope` at L131); `rag/ports.py:186–196` (`validate_scope` treats `[]` as missing); `rag/models.py` `_narrow` sets `doc_ids = []` as the "empty scope" marker.
- **Evidence (probe P1):** store with `required_filter_keys={"tenant_or_user_id","doc_ids"}`; request with `filters.doc_ids=["a"]`, `document_id="b"`, tenant `u1`. Result: `RAISED MissingRequiredScopeError: Required filter keys are missing or empty: ['doc_ids']`.
- **Spec:** §5.5 "A conflict yields an **empty scope**, and the engine returns the empty-retrieval answer (RAG-RES-4)"; AC-RAG-28.
- **Impact:** fail-closed, so not a leak, but the caller gets a misleading 5xx-style error naming a missing key instead of the "not found" answer; `plan()` fails the same way.
- **Fix (minimal):** in `scoped_filters`, if the narrowed filters are an empty scope, skip the required-key check for keys satisfied only by the empty marker (no store call will happen). **Fix (cleaner):** stop overloading `doc_ids=[]`: add an explicit `RAGFilterSpec.empty_scope: bool = False` (additive; the model is `extra="forbid"` so it must be declared), set by `_narrow` on conflict, honoured by `is_empty_scope`, and make stores treat `doc_ids=[]` as "match nothing" independent of it.
- **Test:** the P1 scenario for `retrieve`, `query`, `plan`: expect an empty retrieval result / `answer_found="no"`, no store call.

---

### H9. Caller date range cancels the intent's date window instead of intersecting  **[READ]**

- **Where:** `rag/retrieval_pipeline.py:79–86` `has_soft_date_window`: `... and request.filters.date_range_utc is None and ...`; `effective_filters` L75–76.
- **Spec:** §5.5: "Effective filters = intersection of all hard patches, then intersected with soft constraints … date ranges intersect."
- **Impact:** caller bound "last 6 months" + question "last week" → the "last week" window is dropped; the whole 6 months is searched. Results are within the hard bound (no isolation issue) but ignore user intent.
- **Fix:** keep the caller range as the guard input, then intersect the intent window onto the effective filters (`filters = filters.patch({"date_range_utc": window})`), recording which part is soft so relaxation (`widen/drop_time_window`) can restore only the soft part (it must never go outside the caller range; `RelaxationState` needs the caller range as its hard limit, as it already does for `min_similarity`).
- **Test:** hard range [Jan, Jun] + intent window [Jun 1–7] → applied filter [Jun 1–7]; relaxation `drop_time_window` → [Jan, Jun], never wider.

---

### H10. `date_window_days` is never resolved into a range  **[READ]**

- **Where:** `rag/retrieval_pipeline.py:85` requires `intent.range_start_utc or intent.range_end_utc`; `rag/intent/temporal.py:161–167` `resolve_range` only repairs reversed bounds and is referenced only by `tests/rag/test_news_migration.py:113` (grep). `rag/intent/llm_rewriter.py:268–271` only fills ranges from the regex extractor.
- **Spec:** §7.6 "`TemporalResolver(clock)` converts the intent's temporal fields into a soft date window … `date_window_days` counts back from `clock.now()`" and "Reversed ranges are repaired."
- **Impact:** an LLM intent such as `has_time_expression=true, temporal_scope="recent", date_window_days=3` (phrasing the regex doesn't know, or a non-English phrase) yields **no time filter**. A caller-supplied intent (`engine.query(req, intent=…)`) never goes through the resolver at all, and a reversed LLM-provided range is never repaired.
- **Fix:** add `TemporalResolver.resolve(intent) -> QueryIntent` (fills `range_*` from `date_window_days` via the injected clock; repairs reversed ranges; end-of-day inclusive) and call it from `RAGEngine._resolve_intent` for **both** routed and supplied intents, before `has_soft_date_window`.
- **Test:** FrozenClock; intent with `date_window_days=3` and no ranges → window `[now-3d, now]`; reversed LLM range repaired; supplied intent path covered.

---

### H11. Unit summaries are returned as ranked hits and bypass `top_k`  **[PROBED]**

- **Where:** `rag/memory_store.py:281–289` (`vector_search` iterates `self._nodes`, which includes unit summaries because `_write_section` stores unit-summary entries in `_nodes`; only the *section* summary goes to `_section_summaries`); same for `keyword_search` (L318–323). SQL templates behave the same (H5).
- **Evidence (probe P2):** profile `dense_vector`, `top_k=3`; retrieve → `[('b#s0_u0_sum','parent_summary'), ('b#s0_u1_sum','parent_summary'), ('b#s0_u2_sum','parent_summary'), ('b#s0p0','passage')]`.
- **Cause chain:** summary nodes have traits `{summary, ordered, atomic}` → `is_ranked()` is False → `limit_ranked` exempts them from `top_k`; the assembler then routes them to the summary budget zone. Tier 1/2 results thus contain summary layers the strategy did not ask for, and they are not counted against `top_k` (AC-RAG-74 wording: "keeps every summary … and cuts only passages" applies to Tier 3, not Tier 1/2).
- **Fix:** ranked search methods must return only non-summary nodes: in the reference store skip `is_summary or has_trait(..., summary)` in `vector_search`/`keyword_search`/`hybrid_search`; add a kit check ("a search never returns a summary node; `fetch_parent_summaries` does"). Defence in depth: `strategies/shared.py` helper that drops `is_summary` nodes from `vector_search`/`keyword_search` results.
- **Test:** the P2 scenario must return only passages.

---

### H12. Tier 3 returns section summary **and** its unit summaries  **[PROBED]**

- **Where:** `rag/strategies/hierarchical.py:29` `ctx.store.fetch_parent_summaries(ctx.filters)` (no level, no selection); grep shows nothing in strategies/assembler uses `parent_kind == "section_unit"` or `level`.
- **Evidence (probe P3):** `[('b#s0_sum','section'), ('b#s0_u0_sum','section_unit'), ('b#s0_u1_sum','section_unit'), ('b#s0_u2_sum','section_unit')]` for one section.
- **Spec:** §6.3: "Tier 3 uses unit summaries only for the section that contains the boundary, or when the section summary is not admitted; otherwise it uses the section summary." §8.6: "picks the deepest level whose summaries fit the summary sub-budget".
- **Impact:** every earlier section is represented twice (once whole, once in units), wasting the 25 % summary zone; because `fit()` in the assembler `break`s at the first node that doesn't fit (summaries in document order), the **summaries nearest the reader's position are the first to be dropped**.
- **Fix:** in `HierarchicalSummaryStrategy`, group returned summaries by `span.section_seq`; per section keep the section summary if present (it passed the guard, i.e. ends before the boundary); otherwise keep that section's units. Then, if `supports_summary_levels`, apply the budget-fit rule from §8.6. Also change `fit()` to prefer dropping the *oldest* summaries when over budget (see M-assembler).
- **Test:** AC-RAG-59 extended: three sections, boundary in the third → section summaries for 1–2, **units only** for 3 (those ending before B), no duplicates.

---

### H13. Reference `fetch_full_scope` returns overlapping passages  **[PROBED]**

- **Where:** `rag/memory_store.py:504–512` (returns every non-summary node sorted by position); `scope_stats` L479–502 sums `len(n.text)` of the same nodes.
- **Evidence (probe P9):** engine-ingested doc with the default splitter (1200/150) → `nodes 4 overlapping neighbours 3`.
- **Spec:** §4.2: "Document order, real spans, **no overlap**, and no node crosses a section boundary". Note the same module's `fetch_local_window` deliberately de-overlaps ("Passages overlap, so each one contributes only the text after the previous one ended").
- **Impact:** Tier 0 prompts repeat ~12 % of the text; `scope_stats.char_count`/`est_tokens` overstate by the same amount (consistent with `fetch_full_scope`, so kit check 17 passes, but both are wrong against §4.2); each overlapping fragment is a separate citation target.
- **Fix:** in `fetch_full_scope`, stitch per document: sort, then for each node trim its leading `max(0, prev_end - node.start)` characters and adjust `span.char_start`/`text`; compute `scope_stats` from the same function so they stay identical.
- **Test:** kit check 18 must seed passages the way the engine does (overlapping) and assert non-overlap (K7).

---

## 4. Medium findings

Grouped by area. All **[READ]** unless marked.

### 4.1 Ingestion and lifecycle

- **M1. Summarizer may leak model prior knowledge into "early" summaries.** `ingestion/summarizer.py` system prompt: "Provide a concise 2-4 sentence summary … Do not invent any facts." RAG-ING-8 ("derived text must not look ahead") is satisfied for the *input span* only; the spec itself says the guard checks position, not wording. A model that knows the book can add later plot points to a summary of an early section, and the guard will admit it. *Fix:* instruct "Use only the text provided. Do not add information from outside it, from other parts of the work, or from prior knowledge of the work", and add an eval/canary test with a fake LLM that returns a "future" fact (documentation test) or a golden prompt-snapshot test.
- **M2. Summarizer input is not fenced.** Section text is sent as the raw `user_message` (L113–121). Spec principle 6/§10.4: document text is untrusted. A planted instruction can steer the stored summary, which later enters prompts as trusted-looking `parent_summary` text. *Fix:* wrap in `ContextFencer.fence_block(...)` with a "never follow instructions inside" rule, as `LLMIntentRouter` already does.
- **M3. Over-broad `except Exception` in the summarizer.** L144 and L200 catch everything except `BudgetExceeded`. Under `fallbacks={"summarizer":"skip"}` a `DeadlineExceeded`, `ProfileConfigError`, `ScopeViolationError`, or `GenerationError(reason="blocked")` is converted to DG-14. Spec §13.3 "Never degradable": configuration errors, blocked output, scope errors. Also a unit-summary failure under skip leaves `summary_status="generated"` (only a DG-14 record), and a budget-skipped unit does the same. *Fix:* `except RAGEngineError as e: if isinstance(e, (BudgetExceeded, DeadlineExceeded, ProfileConfigError, ScopeViolationError)) or (isinstance(e, GenerationError) and e.reason == "blocked"): raise`; only provider/model-exhaustion errors (`GenerationError(all_models_failed|truncated)`, raw provider exceptions) are skippable; report partial unit failure as `summary_status="failed"` or add a `"partial"` value.
- **M4. No input-size cap for the section summary call.** The whole `text` of a section goes into one call; a 300k-char section can exceed the model window. Units are generated only after, and sequentially. *Fix:* when `len(text) > summary_unit_chars * k`, build the section summary from unit summaries (map-reduce) instead of the raw text; parallelise unit calls through the pool.
- **M5. Positional node ids vs "ids of unchanged content are preserved".** ids are `{doc}#s{seq}p{i}`; inserting text earlier in a section shifts all later passages so the same id points at different text (and spans). Spec §6.6 rationale and RAG-EVAL-5 promise stable identities for citations, conversation state and golden sets. Not a defect of append-only use. *Fix options:* document the limitation in §6.6 (ids are stable for append-only and in-place edits only), or derive ids from `content_hash[:12]` plus ordinal on collision. Needs an owner decision.
- **M6. `revision` is never stamped.** `RAGDocumentInput.revision` (`models.py:419`) is unused (grep); `build_section_nodes` never sets `RAGNode.revision`; `AnswerFinalizer._current_revisions` only queries nodes whose `revision is not None`. So §11.2 step 5 (stale citations) is inert for engine-ingested content. *Fix:* stamp `node.revision = doc.revision` (and persist via the store/fingerprint), or remove the field/claim.
- **M7. Granularity hard-coded to `"passage"`.** `build_section_nodes` (`ingestion/nodes.py` L30) ignores splitter/profile. `PassThroughSplitter` output for articles/entries becomes `passage` (traits `offset_addressable, rerankable`, **not atomic**), so a budget can truncate an article mid-text; §3.3 says `article`/`entry` are atomic. *Fix:* add `granularity: str = "passage"` to `SplitChunk` or `IngestionConfig`, defaulting to `passage`, and let `pass_through` set the profile's granularity.
- **M8. `prepare_nodes` failure reason is constant.** engine.py ≈L962 always `"empty or zero embedding"`. *Fix:* carry the cause (zero vector vs missing vector vs provider error) into `FailedNode.reason`.
- **M9. Embedder task type is hard-coded.** `ingestion/embedder.py` uses `task_type="RETRIEVAL_DOCUMENT"` regardless of `EmbeddingSpace.task_type_documents`, while the fingerprint uses the latter. A store declaring another task type would sign vectors it did not produce. *Fix:* pass `space.task_type_documents` to `BatchEmbedder`.
- **M10. `sync_document` deletes everything not in the supplied document.** `to_delete` = stored ids − current − regenerated − kept summaries (sync.py ≈L148–158). Passing a *partial* document (e.g. only later sections of a book, the append-only pattern of `ingest_document`) deletes all earlier sections' passages. By spec this is "diff", but it is a footgun with no guard. *Fix:* add `sync_document(..., scope="document"|"sections")` or refuse when `doc.total_char_length` indicates incomplete input; at minimum document it in §6.6.
- **M11. Engine ingestion embeds sections sequentially and unit summaries sequentially.** Performance only (pool unused for ingestion).

### 4.2 Retrieval and context

- **M12. Reranker ignores fan-out variants.** `postretrieval/stages.py` `RerankStage.process` sends `ctx.intent.search_query` only. Spec §9.3: "For comparisons, it reranks per variant and merges with `fair_interleave`, so the second subject is not crowded out." With rerank active, candidates for the second subject are scored against the first subject's query and typically cut by `top_k`, defeating AC-RAG-19 end to end. *Fix:* when `ctx.variants` > 1, rerank per variant (parallel on the pool, `ctx.pool` is set to `None` only inside variant workers) and merge with `fair_interleave`.
- **M13. Boosters/fusion do not exempt `ordered` nodes.** `boosters.py:57` `RecencyBooster.boost` sorts the entire list by `(-score, node_id)` and `ScoreFusionStage.fuse` (L88–153) overwrites `score`/`score_kind` and includes ordered nodes in min-max normalisation. §3.3: `ordered` nodes are "never reranked or reordered by score". With the `knowledge_base()` preset (`score_fusion(confidence)` and a 100k Tier 0 gate) Tier 0 `full_doc` text passes through `score_fusion`. The assembler later re-sorts stable blocks by document order so the prompt is mostly unaffected, but marker indexes, `retrieved.nodes` order, and normalisation of ranked nodes are. *Fix:* partition `is_ranked` nodes before boosting/fusing and re-append ordered nodes unchanged.
- **M14. `ThresholdFilter` does not validate against the pipeline.** `threshold.py` only validates the kind *name*. §9.1/AC-RAG-31: "A threshold configured for a kind the pipeline cannot produce raises `ProfileConfigError` at construction." `threshold(rerank)` without a rerank stage, or `bm25` on a dense-only pipeline, silently passes all nodes (`val is None` → keep). `accepts_score_kinds` is `None` on every stage. *Fix:* give stages `accepts_score_kinds`, and have `EngineProfile`/pipeline validation compute the producible kinds from strategy + stage order.
- **M15. Tier 0 relabels all nodes `full_doc`.** `strategies/full_context.py:25–26` sets `granularity = "full_doc"` (spec §8.2 says so) for every returned node, including KB `entry` nodes, dropping `atomic` and making them `offset_addressable`. The budgeter can truncate an entry; offset-citation/guard logic then applies to nodes with no meaningful offsets (KB-02). Also the same content has different granularity in Tier 0 vs Tier 2. *Fix (spec + code):* relabel only nodes whose granularity is `passage`; keep `atomic` kinds as they are, or introduce a trait-preserving `layer="full_scope"` marker instead of mutating `granularity`.
- **M16. Fan-out repeats full-scope work.** `retrieval_pipeline._fetch` (≈L308–335) runs the whole strategy per variant; for Tier 0 that is N × (`scope_stats` + `fetch_full_scope`), merged afterwards by `ordered.setdefault`. *Fix:* decide the tier once (before fan-out); skip fan-out for strategies whose result doesn't depend on the query (Tier 0, Tier 3's summaries and window).
- **M17. `top_k`/`retrieve_max_results` not checked against `max_top_k`.** `strategies/context.py:60–66` uses `max_top_k` only for the rerank pool. `news_search` sets `retrieve_max_results=2000` while the default `StoreCapabilities.max_top_k` is 200; the engine passes `top_k=2001` straight to the store, and `truncated` is then computed against the store's silently-capped result. *Fix:* validate at startup (`retrieve_max_results <= max_top_k` or raise/clamp with a recorded note), and clamp `top_k_override` too (spec says retrieve size is bounded by `retrieve_max_results`; the code lets `top_k_override` exceed it, `context.py:47–52`).
- **M18. Fence secret is per `ContextFencer` instance.** `engine.py` builds a new `ContextFencer` in every `RAGEngine.__init__`, including `with_store()` siblings (L342–364). `stable_id()` hashes a per-instance `_secret`, so the stateless fence id changes for every per-request scoped engine, defeating D-06 ("calls on the same scope share a prefix"). Stateful conversations are unaffected (id lives in `ConversationState`). With a shared `DeterministicRng`, creating siblings also consumes RNG state, making ids depend on construction order. *Fix:* pass the parent's `ContextFencer` into siblings (or derive `_secret` from a constructor argument).
- **M19. `KeywordFirst` records no fallback reason.** `strategies/keyword.py:41–49`; AC-RAG-21 requires "falls back with a recorded reason"; `min_hits` is not profile-configurable and `fallback=` parameter from §8.3 is absent.
- **M20. Dead/misleading budget config.** `ContextConfig.retrieved_ratio` is validated in the ratio sum but never used (`budgeter.py:68` takes the remainder); `local_window_ratio` default 0.20 vs spec 25 %; the ratio-sum check mixes whole-context fractions (`history_ratio`) with data-zone fractions. Also Tier-0 validation omits `question_reserve_tokens` (spec §12.1 lists instruction, history **and question**).
- **M21. Assembler `fit()` drops by position.** `assembler.py:178–198`: when a node doesn't fit and something is already accepted it `break`s, discarding all later nodes. For summaries in document order that drops the most recent (nearest the reader). *Fix:* for ordered summary layers iterate nearest-first or drop oldest first.
- **M22. History is not boundary-aware.** `ConversationState`/`history` can carry assistant answers produced when the reader was further ahead; after a backward seek, those answers (and `rolling_summary`) re-enter the prompt unguarded. D-08 only re-guards reused *nodes*. Violates the spirit of principle 5 ("visibility enforced in code, not in prompts"). *Fix:* store `boundary_seen` per turn in `ConversationState` (and for `rolling_summary`), drop/rebuild turns whose boundary exceeds the current one; or document that the app must clear history on backward seeks.

### 4.3 Guards and safety

- **M23. Sequential guard is fail-open for docs missing from `max_char_offset_by_doc`**, and an inactive guard is silently built for a profile with `guards=["sequential_boundary"]` and a request without a position (`guards.py:48–53`, `build_turn_guards` L292–297). *Fix:* raise `ProfileConfigError`/`UnsupportedFilterError` when the profile requires the guard but the request carries no boundary, and treat a doc absent from the per-doc map as boundary 0 or reject (spec decision).
- **M24. Built-in guards read metadata attributes, not the renderer.** `TemporalGuard.verify` reads `metadata.<date_field>`; spec §5.1: "Drops nodes whose `renderer.timestamp` is outside the range; a node without a timestamp is kept only if the store applied the filter." Code always keeps timestamp-less nodes and ignores the renderer. `TenantGuard`/`ScopeGuard` read `tenant_id`/`domain_id`/`category`, so an app whose metadata uses other names gets a no-op "defence in depth". *Fix:* give guards the renderer (constructor arg from `build_turn_guards`) and use `renderer.timestamp(meta)`; document the field-name contract for tenant/domain/category or add `NodeRenderer.scope_value(meta, key)`.
- **M25. Reference store `_passes_filters` is fail-open for missing attributes** (`memory_store.py:117–216`): nodes with no tenant, domain, category, date or custom value pass any filter. A store claiming to be the §5.6 reference should fail closed for tenant/domain on rows without an owner when the filter is set.
- **M26. Output sanitizer misses reference-style Markdown and autolinks.** `postgen/sanitizer.py:50–59` only matches inline `![..](..)` and `[..](..)`. `![x][1]` with a `[1]: https://evil/?q=…` definition, `<https://evil/…>` autolinks and bare URLs are untouched. §11.6/RAG-SEC-6 target Markdown image exfiltration generally. *Fix:* also neutralise reference definitions (`^\s*\[[^\]]+\]:\s*\S+`) and `<scheme://…>` autolinks with untrusted hosts.
- **M27. Quote validation is trivially satisfiable.** `citations.py:236` pre-fills `citation.quote = node.text[:200]`; `quotes.py:100` falls back to `[citation.quote]` when no quote is found next to the marker, and `quote_match_ratio(source[:200], source)` is 1.0. With `verify_quotes=True`, any citation without a quote in the exact `"…" [n]` form is "verified, score 1.0". Also `_QUOTED_CITE_RE` keys on the raw marker string, so `[1, 2]` never matches the expanded `[1]`/`[2]` keys; quotes shorter than 4 characters are skipped (Chinese quotes of 2–3 characters). *Fix:* do not pre-fill `quote`; treat "no quote found" as *unchecked* (`verification_score=None`/neutral), not 1.0; match by expanded markers.
- **M28. Offset formatter verifies by `section_seq` only.** `OffsetCitationFormatter.locate` (L78–89) ignores `doc_id`; for a multi-doc scope a fabricated `[c3:10-20]` matches the first node of *any* doc with section 3 and is marked verified. *Fix:* include `doc_id` in the marker or match on `(doc_id, seq)` when the scope has several docs.

### 4.4 Config, errors, operations

- **M29. Silent router degradation.** `RouterConfig.name` defaults to `"auto"`; `HybridIntentRouter.route` (router.py L161–171) uses the LLM only if `models.intent` is set, otherwise silently returns the heuristic/passthrough intent. `EngineProfile.check` requires `models.intent` only for router `"llm"`. §13.3: "nothing knowable in advance becomes a per-request fallback." *Fix:* require `models.intent` for `auto` unless the profile explicitly opts into heuristic-only (e.g. `router.name="heuristic"`), or record a startup warning.
- **M30. Intent LLM error taxonomy.** `llm_rewriter.py:286–297`: a model-call failure is recorded as **DG-04** (parse failure) rather than DG-02, wrapped `GenerationError`/`DeadlineExceeded`/`ProfileConfigError` are swallowed by `except Exception` (only `BudgetExceeded` re-raised), and the final error is always `IntentParseError("All intent models failed to produce valid JSON…")` even for a provider outage. Also `reason=f"…: {exc}"` is unbounded (M31).
- **M31. Provider error text is copied unbounded into `Degradation.reason`/logs.** `gateway.py` (DG-02 L254, DG-12 L339/L411, DG-06 L497), `llm_rewriter.py:292`, `summarizer.py:146–150`, `gateway.generate`'s `GenerationError("All models failed … {last_exc}")`. These reach `RAGResponse.degradations`, stream `degradation` events (sent to browsers), traces and logs. Spec §13.8: "Error messages from providers are truncated to 200 characters and must not include prompt text"; §3.9: `message` "no document or question text"; RAG-SEC-9. *Fix:* central `safe_message(exc)` (type name + first 200 chars, prompt-free) used by `DegradationTracker.record` for every `reason`/`message`.

Additional lower-severity items (each **[READ]**):
- `CitationConfig.policy: str` accepts any value; `"repair"`/`"raise"` (spec §11.4) are accepted and ignored, `min_citation_coverage` is never read (grep: only its definition). `fallbacks` values are not validated (only keys). `profile.check` matches `"rerank" in post_stages` exactly, so `rerank(x)` bypasses the binding check.
- `ComponentRegistry.register_granularity` (`registry.py:72–75`) writes the **process-global** `granularity._DEFAULT_REGISTRY`; engines with different registries share traits, contradicting D-18 ("two engines in one process can use different component sets") and bypassing the duplicate-name rule.
- `InMemoryRAGStore`: `to_jsonl` recomputes summary fingerprints (L707) instead of saving the stored ones; `from_jsonl` drops owners (`_owners`/`_doc_owners`) so isolation info is lost on round trip; `delete_nodes`/`delete_document` do not clean `_section_nodes`/`_owners`/`_doc_owners`; `delete_document` count excludes section summaries while `delete_nodes` includes them; `vector_search` deep-copies every candidate before sorting (O(N) copies per query).
- `EngineProfile.fingerprint()` hashes `model_dump_json()` without key canonicalisation; logically equal profiles built with different dict insertion order (e.g. `variants`, `fusion_weights`, `stage_budgets`) hash differently. `allowed_link_hosts` is a `list` (spec: `frozenset`).
- Stream payloads: `citations` event sends `citation_coverage` as a bare float (`engine.py ≈L705–715`; `else 1.0` branch is dead because `finish()` always returns an object) while §3.10/`RAGResponse` use the `CitationCoverage` object; `retrieval` event lacks labels (spec: "node refs, labels"); a cancelled stream never calls `on_complete` (RAG-EVAL-8 says every request); `generate_from_nodes` ignores `request.variant` while every other public method delegates to the variant engine.
- `RAGEngine._engine_problems` compares only `space.model`, not `dims`, with the engine's embedder (RAG-ING-7 says model and dims). `RouterConfig.jev_timeout` is 2.0 (spec: `jev_timeout_seconds` 2.5). The Jev key check also accepts `VERCEL_API_KEY` (spec: `TYPESAFE_API_KEY`).
- `gateway.generate` estimates cost with `output_tokens=max_tokens or 500`, so a dollar-limit `BudgetGuard` underestimates unbounded generations (spec says `pricing.estimate_cost`).
- `SentenceSplitter`/analyzer: `\n\n` is a paragraph terminator in `sentences()` but `last_sentence_end` treats a paragraph that lacks terminal punctuation as an "unclosed tail" (returns the previous sentence end or `None`), so the §5.2 cut can be earlier than necessary (safe: never past B). `\r\n\r\n` is not recognised as a paragraph break.
- `rag/sql/*`: see H4/H5.

---

## 5. Conformance-kit weaknesses (§4.6 vs `rag/testing/contract.py`)

The spec presents the kit as "what makes a thin adapter per app safe". The implementation is materially weaker. All **[READ]**.

| ID | Spec says | Kit does | Consequence |
| :-- | :-- | :-- | :-- |
| **K1** | Check 1: every `RAGFilterSpec` field honoured: tenant, domain, docs, categories, boundary (single and per-doc), date bounds (inclusive, `date_field`), `custom` keys | `test_check_1` tests `doc_ids` and `max_char_offset=0` only | Domain, categories, per-doc boundary, `min_char_offset`, dates, custom, `min_similarity` are untested. **Summary admission (`char_end <= B`) is untested**, so an adapter with the 1.0.0 spoiler bug passes. |
| **K2** | Check 3: descending by declared score, ranks from 1, **ties stable** | Query `[0.9]*768`; docs `[0.9]*768` and `[0.1]*768` | Both vectors are parallel to the query → cosine 1.0 for both. `res[0].score >= res[1].score` holds for any order. No keyword ordering, no tie-stability. |
| **K3** | Check 4: idempotence, deletes, fingerprints, **revisions** | Writes same node twice; checks fingerprint exists; `delete_document` returns 1 | `delete_nodes`, `delete_sections`, `current_revisions` untested. |
| **K4** | Check 5: "every declared capability has a working method; every undeclared one is never called" | Calls `fetch_full_scope` only | Summaries, fetch_nodes (partly check 9), delete, revisions untested; "never called" is not checked at all. |
| **K5** | Check 7: filtered recall among many distractors | 10 distractor docs, identical vectors, `top_k=5` | Weak signal for the HNSW post-filter trap (spec says only meaningful on real Postgres; kit gives no way to scale distractors). |
| **K6** | Check 12: while one thread writes a section **with passages and summaries**, a reader sees none or all | One reader call racing a 5-node write; passages only | Racy (can pass by reading before the writer starts) and doesn't test the passages-before-summaries bug AC-RAG-66 names. |
| **K7** | Check 18: no overlap; check 17: stats match | Seeds non-overlapping nodes | Passes on stores that violate §4.2 for engine-style overlapping passages (H13). |
| **K8** | Check 16: scoped store refuses other scope | Tests read naming B and write with owner B | Does not test writing to a *document owned by A* from B's scoped store (H3), nor lifecycle calls. |

**Suggested additions (new checks):** summary boundary admission and exclusion from ranked search; per-doc boundary map; date inclusivity + `date_field`; `custom` filters; partial `SectionWrite` leaves other rows intact (H4); a deterministic atomicity test (hold a barrier between passage and summary commit via a store hook, or hammer reads in a loop while writing); engine-shaped overlapping passages for checks 17/18; cross-owner overwrite for check 16; ordering check with non-parallel vectors and an explicit tie.

---

## 6. Spec-vs-code gaps and spec-text defects

### 6.1 Gaps (G)

| ID | Spec item | Code | Phase per spec |
| :-- | :-- | :-- | :-- |
| **G1** | `engine.reconcile(limit, dry_run)` (§2.2, §6.6); `needs_reindex(doc_id: str \| None = None)` | No `reconcile` anywhere (grep); `needs_reindex` requires `doc_id` | P4 |
| **G2** | `degradation_disclosure` (§10.10, AC-RAG-57), listed under P1 tests | No field in `GenerationConfig`, not in assembler | P1 |
| **G3** | `profile.injection_scanner`, `suspect_injection` node attribute (§10.4) | Absent | optional hook |
| **G4** | `FilterInferencer` (§5.5, AC-RAG-28 in P1 list) | Absent (grep) | P1 |
| **G5** | `CitationPolicy`, `RepairPolicy`, `postgen/policy.py`, `repair.py` (§11.4–11.5) | Absent; config accepted but ignored (M-list) | P4 |
| **G6** | `RecordingGateway`/`ReplayGateway` (§2.3, RAG-EVAL-7, AC-RAG-52) | Absent | P4 |
| **G7** | `LexiconExpander`, KB preset `transforms: secondary_fanout, lexicon` | Absent; preset has default transforms only | P3/P4 |
| **G8** | `MarkdownStructuralSplitter`; KB preset splitter | Only `sentence` and `pass_through`; `knowledge_base()` leaves splitter at `sentence` | P4 |
| **G9** | `trace_include_text` opt-in | Absent | verbose |
| **G10** | Extras `[rag-cjk]`, `[rag-eval]` in `pyproject.toml` (§2.5) | Not declared | P1 |
| **G11** | Context compressor (`context/compressor.py`), rolling-summary generation (§10.2 "later") | Absent (spec says later) | later |
| **G12** | AC-RAG ids with no reference in tests/code (grep of `AC-RAG-n`): 02, 09, 26, 27, 28, 36, 56, 57, 61, 68, 69, 74–87 | Some are likely tested without the tag (e.g. 74–87 in `test_integration_round2.py` under `INT-` names). *Not* proof of missing tests | n/a |

Items below the line in P4 may be intentionally pending; G2, G4, G10 are P1-tagged and therefore overdue by the spec's own build order.

### 6.2 Spec-text defects (S)

- **S1.** Header `Last Updated: 2026-11-08` but §6.4.2 rationale cites a 2026-11-11 decision and HEAD commit is 2026-11-11.
- **S2.** §4.4 code block shows `upsert_section_nodes`, `get_fingerprints`, … as methods **inside `class Fingerprint`** (a code-fence/indent error; they belong to `RAGStore`).
- **S3.** §2.5 package layout omits files that exist (`answering.py`, `retrieval_pipeline.py`, `vectors.py`, `postretrieval/dedup.py` is listed but `registry`-level helpers are not) and lists files that don't exist (`context/compressor.py`, `postgen/policy.py`, `postgen/repair.py`, `transforms/` `LexiconExpander`).
- **S4.** AC-RAG-74: "Tier 0 returns the whole scope under any strategy override or secondary query" conflicts with §8.9 rule 1 (`has_strategy_override` precedes `fits_full_context`). Clarify whether an override of `full_context` is meant.
- **S5.** §4.2/§4.4 contract vs reference-store reality (H11, H13): the spec never says ranked searches must exclude summary nodes, and never says the `fetch_full_scope` de-overlap rule applies to engine-produced overlapping passages. Add both as normative text and as kit checks.

Spec **design gaps** worth an owner decision (code can't fix these alone): M5 (positional ids vs stable identities), M22 (history vs spoiler guard), M10 (`sync_document` with partial documents), H1's fts refresh path without `reconcile`.

---

## 7. Checked and found correct (do not re-litigate)

- **Spoiler cut never exceeds B.** `SequentialBoundaryGuard.verify` examines only `text[:limit]`; passage and local window at the same B get the same cut; re-application is idempotent; hard cut at B when no sentence end or when snap-back > 300 (read; also covered by AC-RAG-73 tests). Probe: `last_sentence_end` on `3.14` → `None`; `"他说：“走吧。”然后他"` → 8; `"Wait... what happened"` → 7.
- **Strategy override against an unsupported store** → `StoreCapabilityError` (probe P12).
- **Flag stream buffer** (`generation/flag_buffer.py`) matches §10.5 rules 1–5 on read, including the 32-character cap and end-of-stream release.
- **Fence scrubbing regex** neutralises `<data-…>`, `</ data>`, case variants and zero-width splits; fence id collision rotation exists; no HTML escaping of content.
- **Relaxation** touches only soft constraints: `WidenTimeWindow`/`DropTimeWindow` require `soft_date_window`; `LowerThreshold` clamps to the caller's `min_similarity`.
- **Variant pool deadlock**: variant workers get `pool=None`, so strategies don't wait on the pool they run in.
- **`UsageTracker`** is lock-protected; cost-completeness and `billing="unknown"` handling match §13.1.
- **`ScopeViolationError`** passes through `call_store` unchanged.
- **Owner validation** (`validate_owner`) runs before any model or store call in `ingest_document`/`sync_document`/`prepare_nodes`.
- **Truncation retry / safety block** in `ModelGateway.generate` matches §10.7 (safety not retried on another model; truncation retried once).
- **Tier 0 gate/hint logic** (`adaptive.fits_full_context`): hint can only rule Tier 0 out; the diagnostic WARNING for zero stats exists.
- **Tests:** `pytest tests/rag` → 359 passed.

---

## 8. Coverage limits (what I did not read or verify)

Not read: `rag/intent/jev_gate.py`, `intent/heuristic.py`, most of `intent/temporal.py` (middle regex branches), `rag/trace.py` (verbose snapshots), `rag/cache.py`, `rag/render.py`, `rag/__init__.py` export list, `rag/transforms/{base,fanout}.py`, `rag/testing/{canary,diff}.py`, `rag/eval`, `cody_ai_shared_lib/{pricing,embedding,jev,rerank,tokens,llm (other than cache paths)}.py`, `sql/{fingerprints,cjk_fts}.sql` (grep only), `sequential_boundary.sql` (grep only), and most test files beyond the first lines of two. Spec: §17 D-09…D-25 text and Appendices A–C were not read in detail. Python 3.11/3.12 were not available to run B1. No real database was available for the SQL findings (H4, H5), which are from reading the templates.

---

## 9. Suggested fix order and test plan

**Phase 1: small, high value (≈1 day)**
1. B1 (typing_extensions or floor 3.13) + CI matrix.
2. H2 (default `None`), H3 (`_check_doc` on write) + kit check 16 extension.
3. H6 (citation lookbehind) + parametrised test.
4. H8 (empty-scope handling before `validate_scope`) + 3 tests.
5. M31 (central safe message) – mechanical, wide.

**Phase 2: correctness of the lifecycle and retrieval layers**
6. H1 (summary re-embed + fts staleness; decide `reconcile` shape first, G1).
7. H10 then H9 (temporal resolution, intersection).
8. H11, H12, H13 (summary exclusion from ranked search; Tier 3 selection; full-scope de-overlap) + new kit checks K1, K6, K7.
9. M12 (rerank per variant), M13 (ordered nodes exempt), M16 (tier decided once), M17 (`max_top_k`).

**Phase 3: before AudioReading builds on the SQL templates**
10. H4, H5 (rewrite templates; add tenant, search_path, summary-aware boundary, offset base, iterative scan). Have AudioReading's agent review these first.
11. H7 (structured prompt blocks + Anthropic cache control).

**Phase 4: hardening and spec alignment**
12. M1–M4 (summarizer prompt, fencing, error handling, size cap), M23–M28 (guards/sanitizer/quotes), M29 (router validation).
13. Close G2, G4, G10 (P1-tagged); schedule G1, G5–G9.
14. Spec edits S1–S5 and owner decisions on M5, M10, M22.

**Cross-cutting test additions**
- Python-version matrix (B1).
- A "spec-conformance" test module that asserts the *defaults* the spec states (`EmbeddingSpace.splitter_version is None`, `jev_timeout`, preset values in §12.2 table) so drift is caught.
- Engine-shaped fixtures for the kit (overlapping passages, unit + section summaries, two owners sharing a doc id).

---

## Appendix A. Probe scripts and output

Run from the repo root with `PYTHONDONTWRITEBYTECODE=1 python -I -X utf8 <script>`. They import test helpers (`tests/rag/test_blocker_regressions.py`, `test_integration_round2.py`) and monkeypatch `gateway.generate_embedding_with_usage` with a constant vector, as the tests do. Originals lived in the session scratchpad (not in the repo); the essential logic is reproduced here so they can be re-created.

### A.1 Setup (common)
```python
import sys; sys.dont_write_bytecode = True
sys.path.insert(0, r"<repo root>")
from cody_ai_shared_lib.embedding import EmbeddingResult, EmbeddingUsage
from cody_ai_shared_lib.rag import gateway as gm
DIM = 768
def fake(text, task_type="RETRIEVAL_DOCUMENT"):
    v = [0.1]*DIM if isinstance(text, str) else [[0.1]*DIM for _ in text]
    return EmbeddingResult(vectors=v, usage=EmbeddingUsage(input_tokens=1, cost_usd=0.0))
gm.generate_embedding_with_usage = fake
from tests.rag.test_blocker_regressions import FakeLLM, engine, profile, document, seeded_store
from tests.rag.test_integration_round2 import unit_profile, UNIT_TEXT, tenant_store
```

### A.2 P1, empty scope on a store requiring `doc_ids` (H8)
```python
store = tenant_store("tenant_or_user_id", "doc_ids")
eng = engine(profile(), store)
req = RAGQueryRequest(question="q", filters=RAGFilterSpec(tenant_or_user_id="u1", doc_ids=["a"]), document_id="b")
eng.retrieve(req)
```
Output: `RAISED MissingRequiredScopeError Required filter keys are missing or empty: ['doc_ids']`

### A.3 P2, unit summaries in Tier 1 (H11)
```python
store = InMemoryRAGStore()
eng = engine(unit_profile(retrieval=RetrievalConfig(default_strategy="dense_vector", top_k=3)), store, llm=FakeLLM("A short summary."))
eng.ingest_document(document(UNIT_TEXT))
[(n.node_id, n.granularity) for n in eng.retrieve(RAGQueryRequest(question="sentence")).nodes]
```
Output: `[('b#s0_u0_sum','parent_summary'), ('b#s0_u1_sum','parent_summary'), ('b#s0_u2_sum','parent_summary'), ('b#s0p0','passage')]`

### A.4 P3, Tier 3 summaries (H12)
Same store, strategy `hierarchical`. Output of `[(n.node_id, n.parent_kind) for n in nodes if n.is_summary]`:
`[('b#s0_sum','section'), ('b#s0_u0_sum','section_unit'), ('b#s0_u1_sum','section_unit'), ('b#s0_u2_sum','section_unit')]`

### A.5 P4, splitter version change (H2)
```python
store = InMemoryRAGStore(); eng = engine(profile(), store)
eng.ingest_document(document("Hello world. " * 40))
store.embedding_space().splitter_version   # '1.0.0'
eng.splitter.version = "9.9.9"; eng.needs_reindex("b")
```
Output: `space.splitter_version = 1.0.0` / `after splitter bump: doc_id='b' embeddings_stale=False fts_stale=False`

### A.6 P5, sync after signature change (H1)
```python
eng = engine(unit_profile(), store, llm=FakeLLM("A short summary.")); doc = document(UNIT_TEXT); eng.ingest_document(doc)
store._embedding_space = store._embedding_space.model_copy(update={"splitter_version": "2.0.0"})
eng.needs_reindex("b"); eng.sync_document(doc); eng.needs_reindex("b")
[nid for nid, fp in store.get_fingerprints("b").items() if "2.0.0" not in fp.embedding_signature]
```
Output: before: `embeddings_stale=True`; after sync: `embeddings_stale=True`; stale rows `['b#s0_u0_sum','b#s0_u1_sum','b#s0_u2_sum','b#s0_sum']`.

### A.7 P6, scoped-store cross-owner overwrite (H3)
```python
store = InMemoryRAGStore(); a = OwnerScope(tenant_or_user_id="A"); b = OwnerScope(tenant_or_user_id="B")
store._capabilities = store.capabilities().model_copy(update={"required_filter_keys": frozenset({"tenant_or_user_id"})})
engine(profile(), store.scoped(a)).ingest_document(document("Secret of A. " * 30), owner=a)
engine(profile(), store.scoped(b)).ingest_document(document("Overwritten by B. " * 30), owner=b)
store.doc_owner("b"); [n.text[:20] for n in store._nodes.values()][:2]
```
Output: `B overwrote A's doc; doc_owner now: tenant_or_user_id='B' …` / `['Overwritten by B. Ov']`

### A.8 P9, full-scope overlap (H13)
```python
eng.ingest_document(document("Alpha beta gamma. " * 200))
nodes = store.fetch_full_scope(RAGFilterSpec(doc_ids=["b"]))
sum(1 for x, y in zip(nodes, nodes[1:]) if y.span.char_start < x.span.char_end)
```
Output: `nodes 4 overlapping neighbours 3`

### A.9 P12, strategy override on unsupported store (not a finding)
Output: `StoreCapabilityError HierarchicalSummaryStrategy requires store capability 'supports_parent_summaries'`

### A.10 Citation parse and sentence probes (H6, §7)
```python
RefCitationFormatter().parse(t) for t in [...]   # see H6 evidence block for the five outputs
CjkAwareAnalyzer().last_sentence_end("他说：“走吧。”然后他")            # 8
CjkAwareAnalyzer().last_sentence_end("Mr. Smith went to Washington. He said hi and then")  # 29
CjkAwareAnalyzer().last_sentence_end("The price was 3.14 dollars and")  # None
CjkAwareAnalyzer().last_sentence_end("Wait... what happened to him")    # 7
```

## Appendix B. Reproducing the baseline
```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/rag -q -p no:cacheprovider
# 359 passed
```
