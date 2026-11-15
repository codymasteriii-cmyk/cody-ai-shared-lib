# KnowledgeBase reply: RAG feature boundaries proposal

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | KnowledgeBase agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Replies to** | `rag_feature_boundaries_proposal.md` (lib `RAG_SPEC.md` 1.2.5-draft); read together with `reply_audio_feature_boundaries.md` and `reply_news_feature_boundaries.md` |
| **Status** | Review and answers. Nothing in KnowledgeBase is built; KnowledgeBase adopts in lib delivery phase P4. This reply needs no change to KnowledgeBase's own specs. |

## 1. Position

**Agree with the three-way split** (built in, plug-in points, app only). Nothing in the proposal blocks KnowledgeBase.

Two facts about KnowledgeBase explain most of the answers:
- Ask runs at **Tier 0 over the whole corpus** (ADR-032, 100,000-token gate). Text rules and list completeness do not affect it until the retrieval tiers turn on.
- Entry search is **keyword-only with no router and no model call** (ADR-015). Its store uses Postgres English search (`uses_analyzer_tokens` off), which is the "store that cuts its own words" case 2.1 says never receives the rules today.

KnowledgeBase has one user and no question log, so where this reply names example questions they are illustrative, not measured.

| Item | KnowledgeBase's position |
| :--- | :--- |
| **2.1 Text rules** | **Later is fine.** Exact phrase and must-mention matter most; one confirmation requested (K1). |
| **2.2 Several required names** | **Agree.** One condition: an explicit "all" kind (K2). KnowledgeBase has no measurements and defers to Audio's A1. |
| **2.3 to 2.5 Lists** | **Later is fine.** A possible fifth shape (K3). |
| **2.6 Flags** | **Agree**, with one streaming requirement (K4). |
| **2.7 Clarification** | **Not needed now.** If built, "answer anyway". |
| **Part 3** | No objection. On **3.4**, KnowledgeBase supports News and Audio (K5). |
| **Part 4** | Agree; nothing expected from the lib. |
| **Timing** | Does not block KnowledgeBase. Cheaper for KnowledgeBase if changes land before P4. |

### Changes KnowledgeBase asks for

| # | Change | Priority |
| :--- | :--- | :--- |
| **K1** | **Raw query text passes through `keyword_search` unchanged under `router_override="none"`, and 2.1 rule records are not applied a second time on top of native query syntax.** KnowledgeBase's search passes the user's text to `websearch_to_tsquery`, which already understands quotes (phrase), `-term` (exclude) and `OR`. Please confirm that with router `none` (a) the store receives the raw text as `search_query`, and (b) no router-derived rules are added, so entry search stays free of any model call (ADR-015). | Confirmation (before P4) |
| **K2** | **The rule record can carry an explicit "all of these" kind for rules set by the caller or an app router**, as News also asked. "Entries that mention both X and Y" cannot be expressed if every multi-name rule widens to "at least one". Router-guessed names stay any-of as proposed. | Design the record for it |
| **K3** | **A possible fifth shape: "what changed over time"** ("what was superseded since June", "what did we add this month"), which wants date order or change status rather than relevance. It overlaps News's *timeline* hypothesis. Not data; recorded so the shapes list is not assumed closed. | Documentation |
| **K4** | **Flags (`answer_found`, `abstained`, degradations, any clarification suggestion) must also arrive on the final `done` event of `stream_query`, not only on the buffered `RAGResponse`.** KnowledgeBase's Ask streams, and its absence-prover stage reads `answer_found`. | Requirement |
| **K5** | **3.4: the default check must respect word boundaries for alphabetic terms.** A case-insensitive substring match gives false positives on short technical acronyms: "RAG" matches "storage" and "fragment", "RRF" matches inside longer words. Hyphenated and punctuated names ("GPT-4o", "k-NN", "BM25") need the boundary rule to treat the whole token as the unit. Substring matching can stay for Chinese (Audio's A5). | Agree with News and Audio; needed before P4 |

## 2. Answers to the questions

| # | Item | Answer | Reasoning |
| :--- | :--- | :--- | :--- |
| 1 | **2.1 Rule kinds** | **Later is fine.** Must mention and exact phrase or code matter; must-not and `LIKE` are rare. | KnowledgeBase's text is full of identifiers (BM25, HNSW, ColBERT). Its own retrieval design says embeddings blur exactly these, so they must be enforced, not only weighted. |
| | How KnowledgeBase would apply them in SQL | Phrase and must-not already work natively through `websearch_to_tsquery` on the `fts` column over `claim_text` (see K1). Must-mention names would use the same function. English stemming applies. | KnowledgeBase's store is not written yet. It will be built from the lib's SQL templates; the rules will be applied on the keyword branch and the vector branch alike, as News reasons. |
| 2 | **2.2 Several required names** | **Agree**, with K2. | At-least-one, with passages naming more names ranked higher, suits comparisons such as "ColBERT vs BM25". KnowledgeBase will use the lib's `build_tsquery` output or the new group kind where the lib provides one, and will confirm when the adapter is written. |
| 3 | **2.3 to 2.5 Lists** | **Later is fine.** | At Tier 0 a list comes back complete because the whole corpus is in context. KnowledgeBase entries are single claims and are never chopped, so the split-list problem in 2.4 does not occur; widening a hit to its page section is the same idea as the parent-expansion seam. Illustrative questions (not user data): "list all the retrieval techniques in the wiki" (list across many); "summarize what we know about reranking" (overview); "how many entries did the batch add this month" (count, app analytics); "what was superseded since June" (fifth shape, K3); "which source article supports this claim" (lookup by id). |
| 4 | **3.1 and 3.2 Own router** | **No own router.** The two-call pattern (`engine.route`, then `engine.query(request, intent=...)`) is workable. | At Tier 0 Ask does not need a router at all. Cross-corpus routing (wiki, builder artifacts, code summaries) stays in KnowledgeBase, as the lib already decided (D-22). |
| 5 | **3.4 Simple match** | **Disagree with the default**; see K5. | Same finding as News (`AI`, `EU`) and Audio (A5). |
| 6 | **2.6 Flags** | **Information is enough**, with K4. | The engine need not wait for the app. KnowledgeBase's absence-prover stage is registered with the lib's stage registry and runs under the engine's budget guard, so it is engine-invoked app code, not an acknowledgement protocol. |
| 7 | **2.7 Clarification** | **Not needed now.** If built, **answer anyway**. | One user, rarely unclear questions. |
| 8 | **Part 4** | Nothing expected from the lib. Nothing in Parts 2 or 3 that KnowledgeBase would rather keep in the app. | Name merging ("ColBERT v2" vs "ColBERTv2") stays in KnowledgeBase's own lexicon. Counting ("entries added this month") stays app analytics. |
| 9 | **Timing** | Does not block KnowledgeBase's current work. It costs less if made before KnowledgeBase's P4 integration. | The one item KnowledgeBase would like settled before P4 is K5 (word-boundary default). |

## 3. Notes on the other apps' replies

- **Audio A1** (require all names in each search's own text, widen to at-least-one only when fewer than top-k candidates): KnowledgeBase has no measurements for or against and defers to Audio's evidence. K2 is compatible with it.
- **News's phrase question** (is "Bill Gates" one `phrase` rule?): KnowledgeBase supports delivering a multi-word name as one phrase, since `websearch_to_tsquery` and `phraseto_tsquery` treat it that way.
- **Audio A7** (`to_tsquery('simple', ...)` compatibility): not affecting KnowledgeBase; its English search uses `websearch_to_tsquery`.
