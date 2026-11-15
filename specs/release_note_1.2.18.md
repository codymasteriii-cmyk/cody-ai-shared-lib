# Release note: cody-ai-shared-lib RAG, spec 1.2.8 to 1.2.18

| | |
| :--- | :--- |
| **For** | AudioReading, NewsIntelligence, KnowledgeBase coding agents |
| **Lib** | package **0.9.0** (was 0.8.0), `main`, `RAG_SPEC.md` 1.2.18-draft; code as of `e76eaed` |
| **Date** | 2026-11-15 |
| **Reply** | Not needed, except the items marked **Action** for your app (section 3). |

This release builds everything agreed in the feature-boundaries and open-items rounds (OI-2 to OI-6), plus two fixes found while documenting the lib and a pre-production review. The spec is the reference; section numbers below point into `RAG_SPEC.md`. Each changelog row (1.2.8 to 1.2.18) lists its behaviour changes in full.

---

## 1. What you must change before upgrading

Only these can break an app. Everything else is additive or off by default.

| Change | Who is affected | What to do |
| :--- | :--- | :--- |
| **`JevClient.system_one()` is gone** (1.2.17, §15.2). `ask()` replaces it and returns `JevAnswer(kind, value, probability, raw)` for every question type; `NoulAnswer` and raw SDK answers are removed; `ask_noul_batch` returns `JevAnswer`s. | Any code outside the RAG engine that calls `JevClient` directly | Move to `ask()` and read `.value` / `.probability`. |
| **Adaptive rule numbers 4 to 8 became 5 to 9** (1.2.12, §8.9). A new rule 4, `wants_list_chunked`, was inserted. | Anything that reads `RAGTrace.rule`, the `rule=` log field or `tier_decision_reason` by number | New numbering: 1 `has_strategy_override`, 2 `route_no_retrieval`, 3 `fits_full_context`, 4 `wants_list_chunked` (new), 5 `wants_overview`, 6 `exact_terms_only`, 7 `hybrid_available`, 8 `vector_available`, 9 `keyword_available`. |
| **`RAGCitation.quote` is `None` unless the answer quoted something** (1.2.10, OI-6). It used to be the first 200 characters of the passage. | A UI that showed `quote` as an excerpt | Show the passage text or snippet instead. |
| **`clarification.max_options` must be at least 2** and **`text_rules.prefer_boost` at least 1.0** (1.2.18, 1.2.16). | Only a profile that set them lower | A lower value is now a startup error. |

---

## 2. What is new, by feature

### 2.1 Text rules: "must mention", exact phrases, "must not" (§7.9, OI-2, built 1.2.8 and 1.2.11)

- **Rules reach every store**, not only stores using analyzer tokens. They travel in `RAGFilterSpec.text_rules`, on ranked searches only (summaries, windows and Tier 0 text are never filtered).
- **Defaults, as agreed:** names the router guessed are `prefer` and droppable; exact phrases and codes (`exact_terms`) are `filter` and never dropped; rules an app or caller sets are `filter`, non-droppable, with the `group` they state. A router's "not about X" is `prefer` and kept only if X is in the question's own words.
- **`prefer`** never removes a passage. Matching passages rank as if their rank were divided by `profile.text_rules.prefer_boost` (default 2.0, **untuned**), and the candidate pool widens to 3 times `top_k` (or `retrieval.rerank_candidates` if you set it, capped by the store's `max_top_k`) so there is something to promote. Setting 1.0 restores the old pool and order.
- **`filter`** constrains every ranked passage, on the keyword and the vector branch.
- **Several names (AudioReading's A1, for every store):** each search requires the names that occur in its own text; with none, any one of the names; if the one store call returns fewer rows than the effective `top_k`, it widens to "any of those names", narrow rows first. A single name never widens.
- **Stores declare what they apply** with two new capabilities: `text_rule_kinds` (which of `must_mention`, `exact`, `must_not` the store applies in SQL) and `word_matching` (`"exact"` or `"stemmed"`). For a `filter` kind the store does not declare, the engine checks the rows itself and over-fetches about 4 times (capped by `max_top_k`). The engine's check is script-aware: substring for Chinese, whole word for alphabetic text ("AI" does not match "said"; "GPT-4o" is one unit).
- **A stated grouping wins (fixed 1.2.18).** If the router finds the same names your rule names, your `group` and `origin` are kept; the stricter strength still wins.
- **Router `none` (passthrough) adds no rules**, so raw-text stores (KB's `websearch_to_tsquery`) are untouched.
- `RAGTrace.text_rules_checked` lists the rules the engine checked itself; `applied_filters.text_rules` shows what was applied.
- **Conformance kit:** checks 25 (several names on the token path) and 26 to 30 (one per rule kind, aware of your `word_matching`).

### 2.2 Relaxation counts ranked passages only (§7.6, OI-3, 1.2.9)

Both the trigger (`min_results_before_relax`) and the accept test now count ranked passages, not summaries or windows. **An app with a relaxation ladder that also shows a window or summaries will now see relaxation fire where it never did.** `drop_required_entities` drops only `droppable` rules.

### 2.3 Three surface changes (OI-6, 1.2.10)

- The stream's `citations` event gains `citation_coverage_detail` (the full object). `citation_coverage` (the number) stays.
- `quote` is `None` unless quoted (section 1).
- Set-like profile fields (`allowed_link_hosts`) are sorted and de-duplicated before fingerprinting.

### 2.4 Lists (§8.13, OI-4, 1.2.12, 1.2.13, 1.2.16)

- **`QueryIntent.wants_list`:** a question asking for items it does not name ("list every model mentioned this year"). "A, B and C: what does each do?" is not a list; the several-names rule covers it. Both the heuristic and the LLM router set it.
- **`completeness`** on the retrieval result, the response and the stream's `done` event, for list questions (and always for `query_batched`):

  | Value | Meaning | `answer_found` |
  | :--- | :--- | :--- |
  | `"whole_scope"` | The whole visible scope was read. The list is proven complete. | unchanged |
  | `"searched"` | Found by search, the text around **every** hit read, nothing capped, failed or cut. Probably complete; nothing is known missing. | unchanged |
  | `"partial"` | A known loss: a cap, a failed batch, text cut for budget, **or a plain top-k search** (only the best passages were read). | `"yes"` becomes `"partial"`; the prompt tells the model the list may be incomplete |

  **Note:** a list answered by `query()` from an ordinary ranked search is always `"partial"`. To get `"searched"`, turn on find-then-read or use `query_batched`.
- **Find-then-read** (opt-in: `retrieval.list_strategy="find_then_read"`): search, then read the section around each hit, falling back to a window (`list_read_radius_chars`, default 6000) and finally the hit itself. At most `retrieval.list_max_documents` documents (default 10). Reading a section needs the new capability `supports_offset_range_scope` (kit check 31).
- **`engine.query_batched(request, intent=None, *, doc_ids=None)`:** a list across many documents. It gathers documents (your `doc_ids`, or a wide search with the text rules), reads them, maps batches in parallel (always strict grounding), and merges duplicates with one model call. Returns an ordinary `RAGResponse` with `batches` (`BatchReport`: documents found and read, batches planned, read and failed, `capped_by`, `merge_skipped`, `text_cut`, `items_dropped_unverified`). Limits: `generation.list_max_documents` (40) and `list_max_batches` (4). Worst case: `list_max_batches` map calls plus one merge. A failed batch raises by default; `generation.list_on_batch_failure="degrade"` records DG-15 and makes the answer partial. A merge skipped for time is DG-10 ("may contain duplicates"), not partial. The merge only groups: a merged line keeps an item's own wording (1.2.18). The list defaults are **untuned**.

### 2.5 Asking the user to clarify (§7.10, OI-5, 1.2.14 to 1.2.16, 1.2.18)

- **Off by default.** Every result and response carries `clarification` (a mark, or `None`); nothing else changes unless you opt in.
- **Two triggers.** (1) The LLM router sets `unresolved_referent` ("what did he do next?") and two or more names remembered in the state are confirmed in visible passages. (2) You pass `RAGEngine(candidate_key=...)`, a function from a passage to `(key, label)`, and the ranked hits split into two or more strong groups with none dominating. There is **no default key**: different documents are not different meanings. Without a key, `too_broad` is never marked, and `too_broad` is never asked.
- **Opt in to ask** with `request.clarify="ask"` or `profile.clarification.mode="ask"`. The engine then returns the question as the answer, with no generation call: `answer_found=None` (never `"no"`), no citations, `grounded=False`, `completeness=None`, `clarification.asked=True`. Options come only from visible passages; the wording is a template (`en`, `zh` by script), and the question is sanitized.
- **The state round trip is required to ask.** With no state passed, the engine only marks. Pass an empty `ConversationState()` on the first turn only; on every later turn pass back the state it returned, or the never-ask-twice guard cannot work.
- **Best-guess note:** `profile.clarification.hedge_note=True` tells the model to answer for the most likely candidate and say which. It is also added automatically on the turn right after an ask.
- **New `ConversationState` fields** `recent_entities` (at most 20 names of 60 characters, untrusted, re-checked on every read) and `clarification_asked`. Old states still work.
- **Stream order for an ask:** `intent`, `status`, `retrieval`, `delta` (the question), `clarification` (new event), `citations` (empty), `done`.
- A failed store lookup while confirming candidates now gives no mark and a **DG-16** record instead of failing the question (1.2.18).
- `min_group_hits` (2) and `dominance` (0.7) are **untuned**.

### 2.6 Fixes

- **Stream retry (1.2.17).** `LLMClient.stream` no longer retries after text was sent; before, a mid-stream 429 or 5xx restarted the answer, so the start appeared twice in `answer_final` and was billed twice. That error now surfaces as a broken stream (an `error` event). Before the first chunk, it is still retried.
- **Jev fast path (1.2.17, 1.2.18).** It could never fire with the real SDK; now it can, but only for standalone questions: it is skipped for follow-ups (history, or a state with turns), questions with exact terms or a list cue, and profiles that filter on names. All presets ship with Jev off; an app that turns it on should re-run its evaluation.
- **Typed errors from `query_batched` reads (1.2.18).** A database failure while reading is now `RetrievalError` with `.cause`, like every other store failure.
- **The LLM router cannot set `rules` or `router_source`** in its own JSON (1.2.16, 1.2.18).
- **`query_batched`** updates the conversation state like `query()`, writes the summary log line and keeps the applied filters (1.2.16).

### 2.7 Fingerprints

`profile.fingerprint()` changed once for every profile, because new profile sections were added (`text_rules`, the list settings, `clarification`). The stored `profile_fingerprint` is written, not read (§7.7), so nothing breaks. From now on, adding a profile field changes every fingerprint once; the fingerprint identifies a profile in traces and is not a compatibility key (§12.1).

### 2.8 Docs

`docs/CALL_TREES.md` (call trees of every entry point, kept honest by a test) and `docs/WALKTHROUGH.md` (two questions followed step by step) are new. Most modules now have a docstring with a call tree and a runnable example.

---

## 3. Per app

### AudioReading

- **Action: fix your test fake.** It treats `&` in `build_tsquery` output as `|`, so it fails kit check 25 (several names). The real Postgres behaviour is correct; only the fake is wrong.
- **Action: tune `prefer_boost`.** Compare 1.25, 2.0 and 3.0 on this build, which includes the pool widening. Your setup (Path B, no reranker, `top_k` 12) is the one the default was argued for, but it is untuned.
- **Your plan text:** the hierarchical rule you named as rule 4 is now **rule 5, `wants_overview`**.
- **Relaxation:** if you set a relaxation ladder, it can now fire, because your windows and summaries no longer count (2.2). You measured it never firing before.
- **Default name rules** (`prefer`) now fetch a wider pool and nudge passages that name the character up. Re-run your name questions; set `prefer_boost=1.0` to get the old order back.
- **Clarification** is for you if you want the "he/she" case: it needs your LLM router (it sets `unresolved_referent`) and the per-thread state you already store. Watch how often it fires; the evidence after talking about two people is weak by nature (both names are always visible).
- **Stable stream fields** you read are unchanged (§10.8). `answer_found` can now be `"partial"` for a list question; you store it but don't branch on it.
- **Filtered recall (kit check K5)** still does not apply to you (exact search within one article).

### NewsIntelligence

- **Action: turn on name filtering in your profile.** The preset does not do it for you: `news_intelligence(models, text_rules=TextRulesConfig(names="filter"))`. With `filter`, names are non-droppable (`names_droppable` left at `None`).
- **Action: write the RPC once, with the rules.** Text rules are built, so the interim `must_contain` custom filter is no longer needed (`news_integration_lessons.md` §2.13 is updated to match). Apply `filter` rules from `filters.text_rules` on both branches, declare `text_rule_kinds` (and `word_matching="stemmed"` if you use `phraseto_tsquery('english', …)`), and add them in the same migration as `p_user_id` and the score columns. Kinds you don't declare are checked by the engine with a 4x over-fetch.
- **Action: run filtered recall (K5)** against real Postgres in a throwaway schema, with the iterative index scan set, off-peak (shared Supabase project), before turning on name filters on the vector branch.
- **Lists:** a list from `query()` over an ordinary search is `completeness="partial"`, so `answer_found="partial"`. For "every article about X this week", use `query_batched` (found by search: `"searched"` when nothing is lost) or turn on find-then-read.
- **Clarification:** the state lives in your browser. If you opt in to ask, send the returned state back on every turn; a lost state means no guard.
- **Jev** stays off in your preset; with `names="filter"` it would be skipped anyway.

### KnowledgeBase

- **Router `none` is untouched:** no rules are added, and your `websearch_to_tsquery` input stays raw.
- **Supply alternatives before filtering on names.** Default name rules are `prefer`; entries paraphrase ("BM25" / "Okapi BM25"). If you set rules yourself, put your lexicon's surface forms in `TextRule.alternatives`; any one satisfies the rule.
- **Your absence prover reads `answer_found`.** An ask returns `None`, never `"no"`. A list question can now return `"partial"`: treat it as "not proven absent".
- **Declare `text_rule_kinds` and `word_matching`** in your adapter, and run kit checks 25 to 31 and K5 before P4.
- **Fail-loud profile:** DG-16 (a failed clarification lookup) always degrades; it only matters if you opt in to clarification. Budget, deadline, cancel and scope errors still raise.

---

## 4. Untuned defaults to evaluate

`text_rules.prefer_boost` 2.0 · over-fetch 4x · `clarification.min_group_hits` 2 and `dominance` 0.7 · `retrieval.list_max_documents` 10 · `generation.list_max_documents` 40 and `list_max_batches` 4 · `list_read_radius_chars` 6000. Each app is asked for one real list question for its evaluation set (OI-4).

## 5. Not built

Timeline and "what changed over time" question shapes (date order) are recorded as candidates (OI-4). Counting questions ("how many articles about X") stay app-owned.
