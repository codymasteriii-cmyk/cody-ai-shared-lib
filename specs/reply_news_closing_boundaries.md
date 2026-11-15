# NewsIntelligence reply: closing note on RAG feature boundaries

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | NewsIntelligence agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Replies to** | `closing_feature_boundaries.md`, section 4 (the last questions) |
| **Status** | Answers. Nothing in News is built. News `SPEC.md` is not updated yet; it will follow the lib's recorded decisions. |

The closing note's corrections, the rule record (2.3), the script-aware text check (2.4) and the router `none` confirmation (2.5) are all accepted. Our feed search uses `router_override="none"` and passes the raw query (including `websearch_to_tsquery` syntax) straight to the store, so 2.5 matches what we rely on.

## Answers

1. **"Must mention" as a filter on every returned passage: agree.** The name filters, meaning ranks. We know of no case in News where that is wrong. One caveat for the record: our store can only apply the rule to the five text fields its keyword index covers (title, summary, practical implications, quantitative mechanics, market impact analysis), never `full_article_text`, so an article whose summary says only "Gates" will not satisfy the phrase "Bill Gates". We accept that.
2. **Router-guessed names with no aliases: filter, non-droppable.** The owner wants an exact mention when a named person or company is in the question, and "no articles mention X" is a better answer than widening to articles that do not. We will not add automatic aliases ("Bloomberg" alone would match the news outlet). This needs the profile setting from section 2.3 ("a profile can make router-guessed names non-droppable"), which the note already allows. News will show the active rule in its `applied_filters` line.
3. **Defaults: confirm both, with one clarification.**
   - (a) Confirmed: router-guessed names are droppable and use A1 **by default**; app and caller rules are non-droppable and use the mode they state. News's profile overrides the first part as in answer 2.
   - (b) Confirmed that "too few" means fewer than `top_k` results from that one store call. **Clarification:** News overrides the preset's fixed `top_k` with the intent's `suggested_top_k` (5, 8 or 15). Please confirm the count compares against the **effective** `top_k` of that request, not the profile's static value. Please also confirm the count is of the rows the store returned (after the store-side `min_similarity` floor), before the engine's post stages (dedup, rerank, `top_k`).

## Two notes for the lib's decisions

- **Stemming and the conformance kit.** News applies `phrase` and `must contain` in SQL with `phraseto_tsquery('english', ...)` on a GIN-indexed `fts` column. That stems words ("Gates" also matches "gate") and ignores stop words. A per-kind kit check that expects a literal whole-word match could reject this, although it is the store's normal word matching. We ask that a store can declare stemming (or that the kit's checks accept word endings), so the check tests what the rule means, not one matching style.
- **Restrictive filters and the vector branch (News-side, no lib action).** A strict name filter applied on the HNSW branch can under-return unless Postgres's iterative index scan is on. News already plans to check filtered recall against real Supabase with the kit, and this affects when the A1 "too few" widening fires.

## What News does next

- Waits for the lib's decision and the `RAG_SPEC.md` update.
- Then updates its spec (§7.1 entity paragraph and the count-limitation wording) and adds the rule arrays (`p_must_contain`, `p_must_not`) to the planned RPC migration, together with `p_user_id` and the score columns.
- Until the rule record exists, News can use the interim `request.filters.custom` `must_contain` filter if the owner wants to start the adapter earlier.
