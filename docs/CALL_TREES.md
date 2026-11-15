# Call trees

These trees show what each public call does, step by step, so you can find the code that does the work. They are a map, not a contract: the contract is `specs/RAG_SPEC.md`. The same trees sit in the docstrings of the code they describe, so you can read them next to the code.

How to read them:

- A word followed by parentheses, like `_begin()`, is a function or method you can search for. A test (`tests/test_docs_call_trees.py`) checks that every such name below still exists in the package, so a rename that leaves this map wrong fails the build.
- Indentation means "calls". Order top to bottom is the order of execution.
- `(optional)` marks a step that runs only for some requests.
- Section numbers (§) point to `specs/RAG_SPEC.md`.

Files named below are in `cody_ai_shared_lib/rag/`.

---

## 1. `query()`: answer a question (engine.py)

```
query()
├─ _begin()                          per-request state: id, usage tracker, degradations, deadline, trace
├─ _resolve_intent()                 decide what the question means
│  ├─ clamp_intent()                 (caller supplied an intent) force every value into its legal range
│  └─ route()                        (otherwise) the router reads the question; then temporal.resolve() fixes dates
├─ _prepare_generation()             everything before the model call
│  ├─ run()                          retrieval pipeline, see section 2
│  ├─ _clarification()               mark an unclear question, no model call (§7.10)
│  └─ assemble_prompt()              fence the passages, fit the token budget, add service notes
├─ _asks()                           opt-in only: if true, ask_response() returns the question and the call ends
└─ _generate_and_finish()
   ├─ generate()                     the one model call
   ├─ feed()                         FlagStreamBuffer strips the hidden answer-found flag from the visible text
   └─ finish()                       answering.py: the response, see section 3
```

On the `no_retrieval` route (small talk) `_prepare_generation()` skips `run()` and `_clarification()` and uses `assemble_conversational()`.

Any exception goes through `_fail()` (attach spend and degradations to the error, tell observers) and is re-raised.

---

## 2. The retrieval pipeline: `run()` (retrieval_pipeline.py)

```
run()
├─ embedding_space()                 refuse a store whose vectors came from another embedding model
├─ scoped_filters()                  filters + guards, validated before any store call (§5.6)
│  ├─ effective_filters()            tenant, documents, inferred narrowing, similarity floor, soft date window
│  └─ turn_guards()                  scope, tenant, temporal and sequential-boundary guards for this turn
│        (empty scope? -> _empty_scope_result(), no store call at all)
├─ build_context()                   one RetrievalContext carried through every stage
├─ _variants()                       the primary query plus secondary queries (optional, dropped on a short deadline)
├─ _execute()                        find candidates, then run the post stages
│  ├─ _prior_nodes()                 (follow-up) re-fetch last turn's passages under today's filters
│  ├─ _reuse_alone_suffices()        (follow-up) no new terms? then skip the search entirely
│  ├─ _fetch()                       the strategy, once per query variant, in parallel
│  │  ├─ retrieve()                  the strategy itself: dense, hybrid, keyword, full context, find-then-read...
│  │  └─ fuse_variants()             fuse the ranked results of the variants
│  ├─ _merge_prior()                 give old and new passages a fair share of the slots
│  └─ process()                      each post stage in turn: rerank, dedup, threshold, rule boost, top_k
│        guards are re-applied after every step (_enforce_guards)
├─ _relax()                          (optional) too few results: loosen soft constraints, run _execute() again
├─ limit_ranked()                    cut to the size the prompt can hold
└─ _completeness()                   for a list question: "whole_scope", "searched" or "partial" (§8.13)
```

Which strategy runs is decided by the profile (`adaptive` picks a tier from what the store can do and how big the scope is, §8.12). `plan()` evaluates that decision without searching.

---

## 3. The ending of a request: `finish()` (answering.py)

```
finish()
├─ extract()                         read [n] markers, check each one against a passage the model really saw
├─ verify()                          check quoted words occur in the cited passage
├─ sanitize()                        remove fence markers, unsafe links, leaked internals
├─ list_completeness()               downgrade a list to "partial" if the prompt budget cut read text
├─ next_state()                      the conversation state handed back to the app (§7.7)
│  └─ remember_entities()            names the user asked about that a shown passage confirms (§7.10)
├─ build()                           the inspectable trace
└─ complete()                        notify observers, write the one INFO summary line
```

`ask_response()` is the alternative ending: it builds the same kind of response around the clarifying question, with `answer_found=None` and no citations (§7.10).

---

## 4. `stream_query()`: the same steps as events (engine.py)

```
stream_query()
├─ _begin()
└─ _stream_events()                  a generator; each step below yields events
   ├─ _resolve_intent()              -> event "intent"
   ├─ _prepare_generation()          -> events "status", "retrieval"
   ├─ (opt-in ask)                   -> "delta" (the question), "clarification", "citations" (empty), "done"
   ├─ generate() or stream()         -> "delta" per chunk (buffered: all at once)
   ├─ finish()
   └─                                -> "citations", "done"
```

Order of events for an answer: `intent`, `status(retrieving)`, `retrieval`, `status(generating)`, `delta...`, `citations`, `done`. `degradation` events appear between steps as they happen. On failure the last event is `error`, then the exception is raised. Closing the stream early stops the model call and records it as cancelled.

---

## 5. `retrieve()`: ranked list only (engine.py)

```
retrieve()
├─ _begin()
├─ _resolve_intent()
├─ run()                             the pipeline of section 2, mode "retrieve"
├─ _clarification()                  a mark only; this call never asks
└─ build()                           trace, then observers are told
```

No prompt is assembled and no model is called for the answer.

---

## 6. `query_batched()`: a list across many documents (batching.py)

```
query_batched()
├─ _begin()
├─ _resolve_intent()
└─ run()                             BatchedListAnswerer.run()
   ├─ _gather()                      the documents to read: the app's doc_ids, or those a wide search hit
   ├─ _read()                        per document, sharing one token budget
   │  ├─ read_around_hits()          documents found by search: section, window or hit, cheapest that fits
   │  └─ read_whole_document()       documents the app named: whole, else section by section
   ├─ pack_batches()                 pack the read text into prompt-sized batches
   ├─ _map()                         one model call per batch, in parallel, always strict grounding
   │  └─ parse_map_items()           bullets -> (text, verified citations); unverified claims are dropped
   ├─ _number()                      drop exact repeats, number the items
   ├─ _merge()                       one model call grouping duplicates; parse_merge() reads the reply
   └─ _respond()                     answer text, final [n] numbers, citations, state, trace
```

A document cap, the batch cap, a failed batch or cut text makes the answer `partial` and is reported in `RAGResponse.batches` (§8.13).

---

## 7. Asking the user to clarify (clarification.py)

```
_clarification()                     engine.py, after retrieval, in every mode
└─ find_clarification()
   ├─ _from_referent()               the router flagged "he/she/it": two remembered names, both confirmed in passages
   │  └─ _confirmed()                which names occur in which visible passages
   └─ _from_hook()                   the app gave a candidate_key: two strong groups, none dominating

_asks()                              mode "ask", an askable reason, a state was sent back, not asked last turn
└─ ask_response()                    answering.py
   ├─ question_text()                a template filled with the option labels (no model call)
   └─ next_state()                   sets clarification_asked, so the next turn never asks again
```

---

## 8. Ingestion (engine.py)

```
ingest_document()                    split, summarize, embed, upsert; one section at a time, resumable
├─ validate_owner()                  every key the store requires must be named, before any call
└─ _ingest_section()                 once per section
   ├─ write_signatures()             what the stored rows will say they were built with
   ├─ summarize_section()            (optional) section and unit summaries
   ├─ build_section_nodes()          split the text into passages
   ├─ finalize_node()                hash, tokens, rendered text
   ├─ embed_nodes()                  embed the passages
   ├─ prepare_summaries()            embed the summaries
   ├─ prepare_passages()             attach the signatures
   └─ upsert_section()               one atomic store write per section

sync_document()                      embed only what changed, delete what vanished, keep node ids (§6.6)
└─ sync()                            DocumentSynchronizer.sync()

prepare_nodes()                      for apps that insert rows themselves (§6.4.1): nothing is written
├─ embed_each_with_reasons()         each node on its own; a bad node is listed in `failed`
└─ finalize_node()                   plus the fingerprint the app stores with the row

needs_reindex()                      compare stored fingerprints with what the profile would write now
```

---

## 9. Routing: reading the question (intent/router.py)

```
route()                              HybridIntentRouter, the "auto" router; stops at the first layer that decides
├─ classify()                        heuristic regex rules, 0 ms: small talk and "summarize this" end here
├─ route()                           (optional) JevGateRouter: a fast classifier; decides only when confident
├─ route()                           LLMIntentRouter, when `models.intent` is set: the usual path for real questions
│  ├─ generate()                     one low-temperature model call, fenced question and history
│  ├─ extract_json_block()           pull the JSON out of the model's text
│  ├─ extract()                      TemporalResolver: "last week" -> a date range
│  ├─ clamp_intent()                 force every value into its legal range (DG-05)
│  └─ _must_not_rules()              keep an exclusion only if the question really contains it
└─ extract()                         no model configured: add date words to the heuristic or raw-question intent
```

Other routers: `none` (PassthroughRouter) returns the question as written; `heuristic` stops after `classify()`; `llm` always uses the intent model.

---

## 10. Choosing a strategy: the adaptive ladder (strategies/adaptive.py)

```
retrieve()                           AdaptiveRetrievalStrategy
└─ decide_tier()                     the first rule that is true, whose strategy the store can run
   ├─ fits_full_context()            rule 3: does the whole scope fit the Tier 0 gate? (asks scope_stats)
   └─ the rules in order             override, no_retrieval, full_context, find_then_read (lists),
                                     hierarchical (overview), keyword_first (exact terms), hybrid_rrf,
                                     dense_vector, keyword_only
```

`plan()` on the engine calls `decide_tier()` alone, so the decision can be inspected without searching. The chosen strategy's `retrieve()` then runs; most of them end in `ranked_store_call()`:

```
ranked_store_call()                  every vector, keyword and hybrid search goes through here (shared.py)
├─ query_token_groups()              the question as analyzer tokens, for stores that search tokens
├─ resolve_group()                   split text rules into a narrow set and a wider "any of these names" set
└─ _checked_call()                   one store call: the store applies the rules it declares, the engine checks the rest
   └─ satisfies()                    the engine-side check of the text rules (text/rules.py)
```

---

## 11. Building the prompt (context/assembler.py)

```
assemble_prompt()
├─ _effective_grounding()            strict, or labelled fallback (a guard such as the spoiler guard forces strict)
├─ _fence_for()                      the fence id, rotated if any content contains it
│  └─ ensure_collision_free_id()     (context/fence.py)
├─ allocate()                        budgeter: split the token window into history, summaries, window, passages
├─ select_history()                  budgeter: recent turns always, older turns by relevance
├─ fit()                             once per zone: render nodes until the zone's budget is spent
├─ _render_node_xml()                one passage inside the data fence, text sanitized
│  └─ sanitize_content()             neutralize anything that looks like one of our own tags
└─ stable / volatile split           stable text first, so the provider can cache that prefix
```

---

## 12. Guards (guards.py)

```
scoped_filters()                     retrieval_pipeline.py, before any store call
└─ turn_guards()
   └─ build_turn_guards()            one guard per filter that is set, plus app guards
      ├─ SequentialBoundaryGuard     the reader's position (spoiler prevention)
      ├─ TemporalGuard               the caller's hard date range
      ├─ TenantGuard                 one tenant's data
      └─ ScopeGuard                  documents, domain, categories

apply_guards()                       after every step that adds text, never once at the end
└─ verify()                          per guard, per node: keep, drop, or return a shortened copy
```

Each guard also offers `to_filter()` (ask the store for the allowed rows), `verify_citation()` and `prompt_rules()`.

---

## 13. Every model call: the gateway (gateway.py)

```
generate()                           the same shape for the router, the answer and the summarizer
└─ for each model, in order
   ├─ before_call()                  the app's BudgetGuard: allow, skip (optional stages only) or raise
   ├─ generate_with_usage()          the provider call
   │  ├─ success                     record() the usage, return the text
   │  ├─ finish "safety"             GenerationError (blocked); never retried on another model
   │  ├─ finish "length"             DG-09: one retry with double the output allowance
   │  └─ a provider error            record() the attempt (maybe billed), go on to the next model
   └─ a next model exists            DG-02 recorded; after the last: GenerationError (all_models_failed)

embed()                              cache lookup -> before_call() -> provider -> record() -> cache write
rerank()                             returns None on any failure; the caller keeps its order (DG-06)
stream()                             like generate(); a fallback is possible only before the first text was sent
```

Usage is recorded on every path, including failures: `total()` on the `UsageTracker` always matches what was spent.

---

## 14. Starting an engine: configuration checks (engine.py, profile.py)

```
RAGEngine.__init__()
└─ _raise_on_startup_problems()
   ├─ check()                        EngineProfile: bindings, token budget, zone ratios, thresholds, fallbacks, names
   │  └─ has()                       every name the profile uses must exist in the ComponentRegistry
   └─ _engine_problems()             store and model checks: embedding space, renderer, strategy capabilities, keys

validate()                           the same checks, returned as a ValidationReport instead of raised
```

A bad profile or an incapable store raises `ProfileConfigError` or `StoreCapabilityError` here, before any request is served.

---

## 15. Syncing a changed document (ingestion/sync.py)

```
sync()                               DocumentSynchronizer; compares what is stored with what the text would produce
├─ get_fingerprints()                node id -> how each stored row was made
├─ build_section_nodes()             the passages the current text gives, per section (offsets are document-wide)
│  └─ finalize_node()                content hash, token estimate, search tokens
├─ _sections_with_changed_text()     sections whose text differs, or that lost a passage
├─ per section
│  ├─ _is_stale()                    new or changed passage, or one built under an old model or analyzer
│  ├─ _stale_summary_ids()           an unchanged section whose summary has an old signature
│  │  └─ _refresh_summaries()        re-embed and re-sign the stored summary text; no new summarizing call
│  ├─ embed_nodes()                  embed only the stale passages (all or nothing)
│  ├─ summarize_section()            only when the section's text changed
│  └─ upsert_section()               one atomic write for the section
└─ delete_nodes()                    last: rows whose text no longer exists
```

---

## 16. Checking an answer: citations, quotes, links (postgen/)

```
finish()                             answering.py, after the model call
├─ extract()                         CitationExtractor
│  ├─ parse()                        find markers such as [2] outside code spans
│  ├─ resolve()                      look each marker up in this turn's marker map
│  └─ verify_citation()              each guard may refuse a citation that points at forbidden text
├─ verify()                          QuoteValidator: quoted words must occur in the cited passage
│  └─ quote_match_ratio()            best-aligned similarity of a quote against its source
├─ compute_coverage()                share of the answer's sentences that carry a verified citation
└─ sanitize()                        OutputSanitizer: drop untrusted links and images, fence tags, the fence id, stray flags
```

While streaming, `feed()` on `FlagStreamBuffer` removes the leading `[[answer_found=...]]` flag from the text before the client sees it.

---

## 17. Testing a store: the conformance kit (testing/contract.py)

```
class TestMyStore(RAGStoreContractTests)       the app's own test file
├─ make_store()                      the only required hook: a clean store
└─ every test_check_*                inherited; each one returns early when the store does not declare the capability
   ├─ write()                        writes nodes the way the engine does: one section, fingerprints, owner
   └─ make_filters()                 read filters built from the owner, so a required scope is always stated
```

`BrokenStore` has known defects and must fail the kit; if it ever passed, a check would have gone blind. `InMemoryRAGStore` is the reference store that passes all of it.

---

## 18. The older top-level modules (`cody_ai_shared_lib/*.py`, outside `rag/`)

These are used by the apps directly and by the RAG engine underneath. Paths are relative to `cody_ai_shared_lib/`.

### Fetching an article (fetcher.py, url_validator.py, pdf_text.py)

```
fetch_article()
├─ normalize_url()                   canonical form
├─ validate_public_url()             refuse anything that is not a public web address (SSRF guard)
├─ is_pdf_url()?  yes
│  └─ _fetch_pdf()
│     ├─ safe_get()                  download; every redirect hop is validated; size counted while streaming
│     └─ extract_pdf_text()          PdfTextReader, one page at a time; repeated headers and footers removed
└─ no
   ├─ _fetch_via_jina()              Jina Reader returns Markdown; a 422 on the target selector retries without it
   └─ _has_empty_jina_content()      a 200 with an empty body is an error (paywall, bot wall), never stored silently
```

### One model call (llm.py, pricing.py)

```
generate_with_usage()                routes by the model name's prefix
├─ _gemini_generate()                "gemini-*"
├─ _claude_generate()                "claude-*"   (cacheable prompt blocks marked for the provider)
└─ _openai_generate()                "gpt-*"
   each: build the request -> retry 429 and 5xx with doubling waits -> read the provider's usage
         -> compute_cost() from the one price table -> return text, usage, normalized finish reason
```

The RAG gateway (section 13) wraps these calls with model fallback lists and the budget guard. `generate_embedding()` follows its own ladder: chunked batch, one retry, then per-item calls, with a zero vector for any item that still fails.

### Speech (tts.py)

```
synthesize()                         one Edge-TTS call, one retry, never another voice
├─ clean_text_for_tts()              (the caller's step first) strip Markdown, code, links, emoji
└─ align_cues()                      Edge's word timings -> character offsets for highlighting; [] if they do not line up
TTSCache                             get_key() -> get() / put(): the same text and voice is read from disk the second time
```
