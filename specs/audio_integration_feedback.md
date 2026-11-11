# AudioReading integration feedback on `cody_ai_shared_lib.rag` (round 2, after Stage 2E)

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | AudioReading coding agent, with the owner (Cody) |
| **Date** | 2026-11-11 |
| **Library** | `cody-ai-shared-lib` `main` (merge `10a1cf2`), `RAG_SPEC.md` 1.1.x; app branch `feat/stage-2e-chat` |
| **Status** | For lib agent review: agree, adjust or push back per item |

## How to read this file

AudioReading's chat (Stage 2E) is now integrated with `cody_ai_shared_lib.rag` (store adapter, embed job, chat route). Integration showed nine places where the library contract is ambiguous, easy to misuse, or missing something an app with separate tables and per-user limits needs. Each item uses the same six fields as the earlier rounds: Audio use case, Spec today, Gap, Proposed fix, Why it helps, Done when. Section references point into `RAG_SPEC.md` and file references into the library code on `main`.

**Principles for every proposal below:** additive and backward compatible (the contract surface was frozen in P1, RAG_SPEC §16.2); NewsIntelligence and KnowledgeBase adapters keep working unchanged; the engine does not trust caller hints for safety decisions.

| ID | Title | Priority |
| :--- | :--- | :--- |
| INT-01 | Section write: separate passages, unit summaries and section summary | P1 |
| INT-02 | Make embedding of summaries optional | P1 |
| INT-03 | Stamp signatures on what the store persists; define node ids | P1 |
| INT-04 | Ingestion carries no user identity (budget guard is blind) | P0 |
| INT-05 | Contract kit cannot run on a store that enforces required scope | P1 |
| INT-06 | `scope_stats` must measure the source text (docs, kit check, diagnostic) | P2 |
| INT-07 | `fetch_full_scope` returns one node per section | P1 |
| INT-08 | `RAGStream` lifecycle is easy to get wrong | P1 |
| INT-09 | Streaming contract for callers: `answer_final`, `answer_found`, `message_id` | P2 |

---

## INT-01 · Section write: separate passages, unit summaries and section summary · P1 · §4.2, §4.4, §6.3

* **Audio use case:** AudioReading keeps passages and summaries in different tables. `audiolib_passages` stores offsets and vectors only (text is sliced from the chapter table); `audiolib_summaries` stores the generated summary text and is never vector searched (`SPEC.md` §13.4). Both are written by one database function in a single transaction so a chapter appears all at once (RAG-ING-10).
* **Spec today:** `engine.py:774-781` calls `store.upsert_section_nodes(nodes=[passages...] + summaries.unit_nodes, embeddings=[passage vectors...] + summaries.unit_vectors, summary=summaries.section_summary)`. Unit summaries travel inside `nodes`; the section summary travels in `summary`. Nothing in the port says so, and unit summaries carry no sequence number.
* **Gap:** an adapter with separate tables must walk `nodes`, test `granularity`, `is_summary` and `parent_kind` to tell passages from unit summaries, throw away the summary vectors, and invent `unit_seq` by counting in list order. Getting that wrong is easy: a hard-coded `unit_seq` collides on the unique key.
* **Proposed fix (additive):**
  1. Add `unit_seq: int | None` to `RAGNode` (1-based, set by the engine in document order for unit summaries).
  2. Add an optional port method `upsert_section(write: SectionWrite) -> int`. `SectionWrite` holds `doc_id`, `section_seq`, `passages` and `passage_vectors`, `unit_summaries` (and optional `unit_summary_vectors`), `section_summary` (and optional vector), `embedding_signature`, `fts_signature` (INT-03) and `tenant_or_user_id` plus `scope` (INT-04). The base class implements it by calling today's `upsert_section_nodes` with exactly today's argument shapes, so existing adapters are unaffected. The engine calls `upsert_section`. Adapters with separate tables override it.
  3. Do **not** change the `upsert_section_nodes` signature: the engine calls it with keyword arguments, so any adapter that does not accept a new keyword would fail on its first ingest.
* **Why it helps:** *AudioReading:* a trivial adapter write with no list sorting or counting. *Lib and other apps:* one documented structure that also carries the signatures and the user identity (INT-03, INT-04) without touching the frozen method.
* **Done when:** a store that overrides only `upsert_section` and a store that overrides only `upsert_section_nodes` both ingest a 3-unit section correctly, with `unit_seq` 1..3 and no duplicate keys (`AC-RAG-A33`).

## INT-02 · Make embedding of summaries optional · P1 · §6.3, §13.1

* **Audio use case:** summaries are fetched in order by offset (`fetch_parent_summaries`) and put into the prompt; the app never vector searches them, so it throws their vectors away.
* **Spec today:** `ingestion/nodes.py` embeds the section summary (line 105-108) and every unit summary (line 112-116) in a stage named `summary_embedding`. The retrieval strategies never use those vectors; only a store that indexes summaries does.
* **Gap:** every embedded chapter pays for extra embedding calls and extra budget (and counts against the user's daily limit) for vectors nobody reads.
* **Proposed fix:** `profile.ingestion.summaries.embed_summaries: bool = True` (default keeps today's behavior). When false, the engine skips the `summary_embedding` stage and passes no summary vectors (`None` in `SectionWrite`, INT-01). Because the legacy `upsert_section_nodes` expects one vector per node, engine startup validation raises `ProfileConfigError` if `embed_summaries=False` is combined with a store that does not override `upsert_section`.
* **Why it helps:** *AudioReading:* lower ingestion cost on every chapter. *Lib and other apps:* stores that never index summaries stop paying for them.
* **Done when:** with `embed_summaries=False` the gateway records no `summary_embedding` calls for a section and `SectionWrite` carries no summary vectors; with a legacy-only store the engine refuses to start (`AC-RAG-A34`).

## INT-03 · Stamp signatures on what the store persists; define node ids · P1 · §4.4, §6.6

* **Audio use case:** `needs_reindex(doc_id)` and `sync_document` must tell whether stored vectors and the keyword index were produced with today's embedding model, splitter and tokenizer. AudioReading stores `embedding_signature`, `fts_signature` and `node_id` as columns on `audiolib_passages`.
* **Spec today:** `ingestion/nodes.py` defines the formulas (`embedding_signature(...)` = `model|dims|task_type|splitter_version`; `fts_signature(...)` = `analyzer_name|analyzer_version`) and `needs_reindex` (`engine.py:864-876`) compares stored values with them. But `finalize_node` stamps only `content_hash`, `token_estimate` and `search_tokens`; `RAGNode` has no signature fields; `upsert_section_nodes` receives none. RAG_SPEC §6.6 says the engine persists both.
* **Gap:** a store cannot know the right strings, so it invents them. AudioReading wrote `gemini-embedding-2:768` and `cjk_v1`; the engine expects `gemini-embedding-2|768|RETRIEVAL_DOCUMENT|1.0.0` and `cjk_aware|1.0.0`. Result: `needs_reindex` reports every document stale, and `sync_document` would re-embed everything. Separately, node ids (`{doc}#s{seq}p{idx}`, `#s{seq}_sum`, `#s{seq}_u{n}_sum`) are an unwritten contract, and `delete_nodes(node_ids)` has no document id, so adapters split the id on `#`.
* **Proposed fix:** (1) stamp `embedding_signature` and `fts_signature` (optional fields on `RAGNode`) in `finalize_node`, and carry them in `SectionWrite` (INT-01); (2) document that SQL-backed stores declaring `supports_fetch_nodes` or `supports_fingerprints` persist `node_id`, `embedding_signature`, `fts_signature` as columns; (3) state the node id formats as part of the contract and add `parse_node_id(node_id) -> NodeIdParts(doc_id, section_seq, kind, index)` plus `format_node_id(...)` in `ports.py`.
* **Why it helps:** *AudioReading:* correct re-index detection with no guessing and no id parsing in the adapter. *Lib and other apps:* every adapter persists the same fingerprints.
* **Done when:** after `ingest_document` on `InMemoryRAGStore`, `needs_reindex` returns not stale; changing the embedding model or analyzer version makes exactly the matching flag true; `parse_node_id(format_node_id(...))` round-trips (`AC-RAG-A35`).

## INT-04 · Ingestion carries no user identity (the budget guard is blind) · P0 · §13.1, §6.6, RAG-SEC-5

* **Audio use case:** embedding and summary spend counts against the user's daily budget (`SPEC.md` §17.8). The guard (`BudgetGuard.before_call`) reads the user from `ctx.tenant_or_user_id`. The store is called with the service-role key, so the database enforces nothing about who may do what: identity has to travel with the call.
* **Spec today:** reads carry identity in `RAGFilterSpec.tenant_or_user_id`. Ingestion does not: `engine.py:692-694` builds `UsageTracker(request_id, doc_ids=[doc_id], profile_name=...)` without a tenant or scope, and `upsert_section_nodes` receives none.
* **Gap:** during `ingest_document` and `sync_document` every `CallContext` has `tenant_or_user_id=None`, so a per-user guard cannot decide (AudioReading's guard returns "allow" when the user is missing), `BudgetExceeded` cannot fire mid-ingestion, and the store cannot verify that the document belongs to the caller before writing.
* **Proposed fix:** add `tenant_or_user_id` and `scope` (the required filter keys, as `RAGFilterSpec` uses them) to `ingest_document`, `sync_document` and `prepare_nodes` (or as fields on `RAGDocumentInput`). Pass them to the `UsageTracker`, so `CallContext` has them, and to the store through `SectionWrite` (INT-01). Reject ingestion when the store declares `required_filter_keys` and the identity is missing (`MissingRequiredScopeError`), the same rule as reads.
* **Why it helps:** *AudioReading:* the daily budget covers embedding and summary spend. *Lib and other apps:* multi-tenant stores get the same isolation guarantee on writes as on reads.
* **Done when:** a guard test sees `ctx.tenant_or_user_id` during `ingest_document` and stops the third paid call; ingestion without the required scope raises `MissingRequiredScopeError` (`AC-RAG-A36`).

## INT-05 · Contract kit cannot run on a store that enforces required scope · P1 · §4.6, NF-23

* **Audio use case:** AudioReading's store sets `required_filter_keys={"tenant_or_user_id","doc_ids"}` and raises `MissingRequiredScopeError` otherwise, because the service-role client bypasses row-level security and the adapter is a line of defense.
* **Spec today:** `testing/contract.py` builds `RAGFilterSpec()` or `RAGFilterSpec(doc_ids=[...])` with no tenant in most checks (for example lines 82, 121, 131, 157, 165, 181, 192, 221, 235, 271), while check 2 requires the store to reject a query without required keys.
* **Gap:** a store that does the right thing fails most checks. The only way to run the kit is to build the test store with `required_filter_keys=frozenset()`, so the configuration that ships is never tested by it.
* **Proposed fix:** a `make_filters(doc_id=None, **extra) -> RAGFilterSpec` hook on `RAGStoreContractTests` (default: `RAGFilterSpec(doc_ids=[doc_id])`) that every check uses; apps override it to add their tenant. The kit must run against the store's real `capabilities()`. Keep check 2 as is.
* **Why it helps:** *AudioReading:* the kit tests the configuration that ships. *Lib and other apps:* any tenant-enforcing store can use the kit.
* **Done when:** a store with `required_filter_keys` set passes the full kit using an overridden `make_filters` (`AC-RAG-A37`).

## INT-06 · `scope_stats` must measure the source text · P2 · §4.2, §8.2, §8.12

* **Audio use case:** short articles and early parts of novels are answered in full-context mode (Tier 0) **before anything is embedded**, so no passages exist yet.
* **Spec today:** `strategies/adaptive.py` `fits_full_context` asks the store `scope_stats(filters)` and chooses Tier 0 only when `0 < stats.total_tokens <= full_context_max_tokens`. `ScopeStats.est_tokens` is documented as "sum of node.token_estimate when stored".
* **Gap:** an adapter that implements `scope_stats` over indexed passage rows reports zero for an unembedded article, so the engine plans a retrieval tier and asks for an index that is not needed. The engine behaves correctly given what the store reports; the contract does not say that `scope_stats` must describe the text `fetch_full_scope` would return.
* **Proposed fix:** the store stays the source of truth for Tier 0 eligibility; we do not propose that the engine substitute `request.doc_total_chars`, because that hint is unvalidated caller input and ignores filters other than the position limit. Instead: (1) state in §4.2 and the port docstring that `scope_stats` must describe exactly the content `fetch_full_scope` would return (source text, not indexed rows); (2) add a kit check for stores with `supports_full_scope_fetch`: `scope_stats(f).total_tokens` is consistent with the text returned by `fetch_full_scope(f)`; (3) a diagnostic, not a behavior change: when the hint says the document has text but the store reports zero tokens, log a WARNING and record a trace note pointing at `scope_stats`.
* **Why it helps:** the next lazy-indexing adapter finds the trap in a test instead of in production.
* **Done when:** the kit check fails for a store that returns zero from `scope_stats` while `fetch_full_scope` returns text, and the diagnostic appears in the trace in that case (`AC-RAG-A38`).

## INT-07 · `fetch_full_scope` returns one node per section · P1 · §5.4, §8.2

* **Audio use case:** in full-context mode the answer's citations `[1]`, `[2]` are shown as links that seek the player to the cited passage.
* **Spec today:** the port docstring says only "Fetch all content in scope in document order" (`ports.py:198-200`). The assembler treats each returned `full_doc` node as a whole section: its own marker, its own span, its place in the cacheable prefix (`context/assembler.py` around line 219; RAG_SPEC §5.4 describes "whole sections" and "the partial current section").
* **Gap:** the contract does not say "one node per section". An adapter that returns the whole visible text as one node (`section_seq=0`, `char_start=0`) is accepted, and then every answer has exactly one citation covering everything, clicking it seeks to offset 0, and the prompt has no chapter structure.
* **Proposed fix:** document in the port docstring and §8.2 that `fetch_full_scope` returns one `full_doc` node per natural unit with its real `section_seq`, `char_start`, `char_end` and title, in document order; add a kit check that nodes are ordered and contiguous, and that a scope with several sections returns several nodes.
* **Why it helps:** correct citations and chapter labels in full-context mode for any multi-section document.
* **Done when:** the kit fails a store that returns a single node for a two-section scope (`AC-RAG-A39`).

## INT-08 · `RAGStream` lifecycle is easy to get wrong · P1 · §10.8, RAG-OPS-2

* **Audio use case:** the client can disconnect mid-answer; the app must stop the engine, record the spend of the cut-off call, and keep the stream's usage total.
* **Spec today:** `RAGStream` (`engine.py:77-97`) has `close()` and `usage_total`; §10.8 says closing stops all further calls and marks the generation `cancelled`.
* **Gap:** the natural pattern `for event in stream:` does not close the stream when the loop is left early, so the engine's cleanup (marking the call cancelled, stopping model calls) runs only at garbage collection. Reading `usage_total` before the stream is closed under-counts the cut-off spend.
* **Proposed fix:** make `RAGStream` a context manager (`with engine.stream_query(req) as stream:` closes on exit), show that pattern in §10.8, and state that `usage_total` is final only once the stream has ended or been closed.
* **Why it helps:** every streaming endpoint is cancel-safe by default.
* **Done when:** leaving a `with` block after one delta makes no further model call and `usage_total` shows the cancelled call (`AC-RAG-A40`).

## INT-09 · Streaming contract for callers: `answer_final`, `answer_found`, `message_id` · P2 · §10.5, §10.8

* **Audio use case:** the chat route streams text live and must store the final, cleaned answer, not the concatenated stream.
* **Spec today:** `delta` events carry raw text (including markers); the `done` event carries `answer_final`, `answer_found`, `grounded` and `message_id` (`engine.py:652-668`). RAG_SPEC §10.5 says a missing or invalid flag gives `answer_found=None`. `message_id` is `request.metadata.get("message_id")`.
* **Gap:** three things are easy to miss: (1) the concatenated deltas can contain unverified markers or out-of-range citations, so only `answer_final` is canonical; (2) `answer_found` can be `None`, and converting it with `str()` stores the string `"None"`; (3) `message_id` is `None` unless the caller sets `metadata["message_id"]`.
* **Proposed fix:** add a "Contract for callers" box to §10.8: persist and render `done.answer_final`; `answer_found` is `"yes" | "no" | "partial" | null`; `message_id` echoes the caller's `metadata["message_id"]` or is `null`. Optionally make `message_id` a first-class field on `RAGQueryRequest`.
* **Why it helps:** callers store the right text and handle a missing flag.
* **Done when:** the spec table states the three points and a test asserts that `done.data["answer_found"]` is `None` when the model omits the flag (`AC-RAG-A41`).
