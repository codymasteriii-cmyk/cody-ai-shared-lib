# Integration lessons from the AudioReading rollout, for the NewsIntelligence migration

| | |
| :--- | :--- |
| **To** | NewsIntelligence coding agent |
| **From** | `cody-ai-shared-lib` agent, with the owner (Cody) |
| **Date** | 2026-11-13 |
| **Lib version** | `RAG_SPEC.md` 1.2.7-draft, `main` (updated 2026-11-14: §2.13 added; open items decided) |
| **Status** | Informational. Nothing here needs a reply. The three open integration items are decided (spec §18, OI-6). |

AudioReading integrated first and is in production. Every item below cost Audio a debugging session or a review round. They are listed so News does not repeat them. Spec sections are given so you can read the rule, not only the advice.

## 1. Is anything in the lib blocking News?

No. Everything §16.2 lists for phase P3 is built and tested:

- the `news_intelligence()` preset and its two named variants, `news_chat` and `news_search`
- native hybrid search with raw scores, the temporal resolver (`date_field`, `applied_filters`) and the relaxation ladder
- `SecondaryQueryFanOut` (on by default), the recency booster and score fusion
- `PassThroughSplitter` and the standalone embedder, and `generate_embedding_with_usage`
- `retrieve()` sized for the feed, with a 2,000-result test (`tests/rag/test_news_migration.py`)
- `diff_retrievals` in `rag.testing`, for shadow comparisons in your own background jobs
- the conformance kit (`RAGStoreContractTests`)

Not built, and not needed for News: the citation coverage policy and repair (phase P4), the engine-made rolling summary (§10.2, later), and the evaluation system (Q-06, after P4).

What News builds itself (§12.4): the store adapter, the renderer, any router extension (for example `depth_intent`), and the chat and feed routes. Counting questions stay app-owned; News has no analytics service today.

## 2. The mistakes that actually happened, and the rule for each

### 2.1 The renderer: extend `DefaultRenderer`, never the protocol (§3.4, AC-RAG-111)

- Audio's renderer subclassed the `NodeRenderer` protocol and defined methods with names the engine never calls (`embedding_text`, `format_label`). The methods it did not define were inherited empty bodies that return `None`, so the embedder got empty text. The error read "All 18 input(s) are empty/whitespace", far from the cause.
- **Rule:** write `class PulseRenderer(DefaultRenderer[PulseArticleMeta])` and override only what you need. The method names are fixed: `embed_text`, `rerank_text`, `prompt_text`, `citation_label`, `breadcrumb`, `format_meta`, `timestamp`, `features`, `flags`.
- Since 1.2.2 the engine refuses a renderer that inherits those empty bodies, raising `ProfileConfigError` when you build it. The check does not look at what your override returns, so an override that returns `None` is still your bug.
- Add this to your tests: `assert renderer_problems(PulseRenderer()) == []` (import it from `cody_ai_shared_lib.rag`).
- News gives three different texts for one row (embed, rerank, prompt), a URL key for dedup, `created_at` for recency, and priority as a feature (`features` returns `{"priority_score": ...}`). Test each view returns a non-empty `str` for a real row.

### 2.2 Isolation is the adapter's job (§4.5, §5.6, §6.4.2, RAG-SEC-5)

- News uses a service-role client that bypasses RLS, so a forgotten filter leaks across users with nothing to stop it.
- **Declare** `required_filter_keys={"tenant_or_user_id", "domain_id"}` in `StoreCapabilities`. The engine then raises `MissingRequiredScopeError` before the store is called if either is absent from the request.
- **Every RPC filters on the owner and raises when the owner is missing.** Audio's functions took only an article id; it had to add a user id parameter to each.
- Scoped stores also refuse **writes** into a document another user owns, not only reads (§5.6). Build the store per request, scoped to the authenticated user, with `engine.with_store()` (the siblings share the fence secret, so the prompt prefix stays cacheable).
- The kit hook is `make_scoped_store(store, owner)`. It returns a view bound to `owner`. Kit check 16 tests two scoped views over one shared store.
- Every `SECURITY DEFINER` function needs `SET search_path = public, extensions, pg_temp`. Audio's lacked `pg_temp`, and our own templates first said `public, pg_temp`, which cannot find pgvector on Supabase. The current templates are right.

### 2.3 Passing the conformance kit is the requirement, not copying the SQL (§4.6, §4.7, Q-01)

- News keeps `match_intelligence_dynamic`. It does not have to move to the templates.
- Audio's functions differed from the templates in four places. Check yours line by line against the migration checklist at the top of each file in `cody_ai_shared_lib/rag/sql/` and in §4.7:
  1. `upsert_section` must never delete rows that are not in the write (a partial sync write keeps the rest).
  2. The owner filter is on every function.
  3. `search_path` is set as above.
  4. Scan mode: an HNSW index needs iterative scans (`hnsw.iterative_scan`, pgvector 0.8 or later) so a filtered search still fills `k` rows; an exact sort is fine when every query is limited to one scope.
- **Ranked searches return passages only** (§4.2). `vector_search`, `keyword_search` and `hybrid_search` never return summary rows.
- **`get_fingerprints` returns a `content_hash` for every row.** Summaries that have no embedding signature return `None` for `embedding_signature`, and that is stored as given. Do not invent a value.
- The kit has checks 1 to 24. **Run it against the real Supabase project**, not only the in-memory store. The one check that means something only against Postgres (filtered recall, K5) has no meaning on a fake, and it is the one Audio's rerun is meant to settle.

### 2.4 Scores: a cosine floor on a fused score drops everything (§8.4, §9.1, NF-02)

- Your RPC returns reciprocal-rank-fusion scores of about 0.03. A cosine threshold of 0.60 applied to that removes every row.
- **Rule:** express the floor store-side with `RAGFilterSpec.min_similarity` (default from `profile.retrieval.min_similarity`). The store applies it to the vector branch only.
- Declare `returns_raw_scores` only if your RPC adds the raw-score columns. Then the engine can normalize and fuse itself.
- For feed search the `news_search` variant has the router off, no rerank and no generation, and sizes `retrieve_max_results` for the feed. The adapter declares `max_top_k` of **at least 2,001** for 2,000 results, or results are capped and marked `truncated` (§8.10).

### 2.5 Errors are typed now; the old News code swallowed them (§13.3, §18.1)

The 1.0 code returned empty results where it failed. The engine now raises. Specifically for News:

- A zero query vector raises `EmbeddingProviderError` (was: empty result, `rag.py:52-54`). A store error raises `RetrievalError` (was: empty result, `rag.py:62-64`).
- Unparseable intent output raises `IntentParseError` after a retry and the backup model (was: search with the raw question and no filters, `intent_analyzer.py:126-135`). Bind a second intent model for that retry.
- **Rule:** in the chat route, catch `RAGEngineError` subclasses and map them to your error response or SSE `error` event. `GenerationError.retrieval` still lets you show the sources when only generation failed.
- If you want keyword search instead of failing when the query embedding fails, opt in with `fallbacks={"query_embedding": "keyword"}`. Nothing else degrades unless §13.3 lists it.
- **Build the engine at startup.** Missing keys for enabled features, a bad renderer and a bad profile fail at construction, so a misconfiguration fails the deploy and not the first user's request. Nightly ingestion is unaffected: `generate_embedding` still returns zero vectors so index alignment is kept.

### 2.6 Dates (§3.6, §6.5, NF-10)

- News stores only `created_at`, which is **ingestion time**. Declare `supported_date_fields={"created_at"}` and set `time_basis="created_at = ingestion time"`. "This week" then means what your users expect.
- The caller's date range and the intent's date window are now **intersected**, not cancelled by each other (1.2.0). A hard-filter conflict gives an empty-scope answer, not an exception.
- Since 1.2.5 the in-memory temporal guard reads ISO strings (with an offset or `Z`), `date` and `datetime` values. If you test with the in-memory store and rows from JSON, dates as text now behave like the real store.
- A node without a date is kept only when the store applied the date filter (`date_field` in `supported_date_fields`). So the adapter must really apply it in SQL.

### 2.7 Filters read the value, not its truthiness (§5.1, 1.2.5)

- The in-memory `ScopeGuard` and `TenantGuard` treated a present `False`, `0` or `""` as "field missing" and kept the node against a filter it did not match. Fixed in 1.2.5. Only `None` or an absent field means "the node does not carry it" (and the node is kept, because the store filtered it).
- It matters for `favorites_only`. `custom={"favorites_only": True}` now drops a row where it is `False`. If your adapter relies on engine guards as a second line of defense (`generate_from_nodes` does), you can trust them.

### 2.8 Cost and usage (§10.3, RAG-OPS-1, §15)

- The lib reports usage for every call: `response.usage` and `usage_total`, with `billing` set to `"known"` or `"unknown"`. A failed call is recorded too; 1.2.5 fixed a missing record for a failed Jev call.
- The daily cap stays in the app. Audio implements a `BudgetGuard` whose `before_call` returns `"allow"` or `"skip"` or raises. News's per-turn cost can read the same usage fields.
- **Do not add the instruction text to your cost estimate separately.** Since 1.2.3 the context budget uses the measured size of the system prompt, with `reserved_instruction_tokens` (default 4,000) as the floor. A very long persona only shrinks the room for passages. An instruction block over half of `max_context_tokens` raises `ProfileConfigError`.

### 2.9 Streaming and cancel (§10.8, §13.4, 1.2.4)

- Stream events end with `done` (it carries `usage`, `usage_total`, `abstained`, the full `citations` and `sources`). A failed or cancelled stream emits an `error` event and then raises.
- To stop a request when the client disconnects: create a `CancelToken` (from `cody_ai_shared_lib.rag`) and pass it as `cancel=` to `route`, `retrieve`, `query`, `stream_query` or `generate_from_nodes`. Call `token.cancel("client left")` from any thread. The engine raises `RequestCancelled`, logged at INFO and never degraded.
- It stops the request **between steps**, not inside a provider call already in flight. Do not promise instant stop in the UI.
- The stream's `citations` event carries `citation_coverage` as a bare number. Decided (OI-6, not built): a `citation_coverage_detail` object is added next to it; the number stays.

### 2.10 Citations and prompt safety (§10.9, §11, RAG-SEC)

- Markers `[1]`, `[2]` are the **position in the list of passages sent for that turn**, not ids. The engine keeps `turn_map` (marker to node id) and verifies each marker against the nodes in this turn's fence. Unverified markers are removed, so the text the user sees only contains markers that map to a source.
- Persist `answer_id`, node ids, intent and profile version from the response if you want feedback or thumbs up (R-08). Feedback storage stays in News.
- The fence has a per-engine secret. `engine.with_store()` siblings share it. Do not strip or re-wrap the context text.
- Link sanitization is on. With `allowed_link_hosts` empty, only the retrieved sources' own hosts are trusted (subdomains included); any other link keeps its text and loses its URL, any other image becomes its alt text (§11.6). Set the list from a fixed literal.
- `citation.quote` is currently pre-filled with the first 200 characters of the source when the model quoted nothing. Decided (OI-6, not built): it becomes `None` unless the answer quoted something. Do not display or store it as the model's words.

### 2.11 Chat history and state (§7.7, §10.2)

- News keeps history in the browser (NF-19), so `ConversationState` must be safe to send back and forth. It is. The state carries the last intent and prior nodes; a profile change invalidates the stored intent on the next turn (the fingerprint is computed with sorted keys, and the first turn after the 1.2.0 upgrade ignored the stored `last_intent`).
- The engine selects history itself: the last two exchanges are always kept, older ones are scored by recency and overlap with the question, within a budget (15% of the window, at most 8,000 tokens).
- **The engine does not build a rolling summary** (§10.2 is later). An app-supplied `rolling_summary` in the state is included as history. Do not expect the engine to compress long threads.

### 2.12 Platform

- **Python 3.13 or later** (`pyproject.toml`). Audio's Render runtime was not pinned, and `TypeVar(default=...)` fails on older versions. Pin the deploy host, not only local dev.
- All apps install `@main` (Q-02). The lib owner coordinates changes. Record the lib commit SHA you tested against in your own notes, and run the kit in News's CI on every upgrade.
- On a 512 MB host: `import cody_ai_shared_lib.rag` needs no optional extra. The engine runs a small bounded thread pool for concurrency (D-19). Do not add parallel work around it.

### 2.13 Names and exact phrases are not enforced for News yet (§7.9, OI-2)

- **Today** a name the router marks as required (`entity_mode="require"`) and an exact phrase (`exact_terms`) reach only stores that use engine-cut word groups (`uses_analyzer_tokens=True`). News uses Postgres English search, so its store never receives them and nothing enforces them.
- **Agreed design, not built:** spec 1.2.6 §7.9 records the rule record you reviewed in the feature-boundaries round. Rules are delivered to every store; your RPC applies them with `p_must_contain` / `p_must_not` on both branches; your profile sets router-guessed names to `filter`, non-droppable. If the owner schedules it before your adapter, add the rule arrays in the same RPC migration as `p_user_id` and the score columns.
- **Until then**, enforce a name with today's code: call `engine.route(request)`, put a filter your store declares in `request.filters.custom` (for example `supported_custom_filters={"must_contain": list}` and `{"must_contain": ["Bill Gates"]}`), then call `engine.query(request, intent=intent)`. Such a filter is a hard filter: it applies to every search of the request and is never relaxed.
- **Use the interim filter for a single name only.** Because it applies to every search of the request alike, a multi-name or comparison question would require all names in every search, which is the failure §7.9 fixes. If your adapter accepts a list, define it as "at least one of these", and wait for the rule record for multi-name questions.
- **Filtered recall (K5).** Once your RPC filters names on the vector branch, run the kit's filtered-recall check against real Supabase in a throwaway schema, with the iterative index scan set, off-peak (the project is shared with AudioReading). Too few rows from a strict filter on an HNSW index would trigger the several-names widening (§7.9).

## 3. A suggested order of work

1. Build `PulseRenderer` (extend `DefaultRenderer`), run `renderer_problems`.
2. Write `PulseArchiveRAGStore` with the capabilities above and run the kit against a real Supabase project. Fix the RPCs the kit reports, owner filter first.
3. Add the engine at startup with `news_intelligence()` and the `news_chat` variant. Map `RAGEngineError` subclasses in the chat route. If you add an analytics or name-filter branch, use the two-call pattern (`engine.route(req)`, then `engine.query(req, intent=intent)`); counting questions stay app-owned (§7.9).
4. Move feed search to `retrieve()` with `news_search` and the 2,001 `max_top_k`.
5. Run old and new retrieval side by side with `diff_retrievals` in a background job before switching traffic.
6. Track the decided surface changes (spec §18, OI-6): `citation_coverage_detail`, `quote` as `None` unless quoted, canonical profile fingerprints.

## 4. What we would like back

- Anything in this note that was wrong, unclear or missing when you tried it. We will fix the spec, not only this note.
- Anything the lib made harder than your old code, with the file and the request that shows it.
