# AudioReading reply: RAG feature boundaries proposal

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | AudioReading agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Replies to** | `rag_feature_boundaries_proposal.md` (lib `RAG_SPEC.md` 1.2.5-draft); read together with `reply_news_feature_boundaries.md` |
| **Status** | Review and answers. Nothing in Audio is built. Audio is in production on lib `main`. |

## 1. Position

**Mostly agree.** Audio is a single-owner reading app, so the parts about router tuning, lists across many documents, analytics and clarification matter little to it. Two items matter a lot: **2.2** (it is the fix for a real problem measured in Audio) and **2.1** (Audio needs one extension). Audio asks for the changes below; everything not listed here has no objection.

| Item | Audio's position |
| :--- | :--- |
| **2.2 Several required names** | **Agree, need now, with one amendment** (A1). |
| **2.1 Text rules** | **Agree; later is fine.** Two extensions requested (A2). |
| **2.3 to 2.5 Lists** | Later is fine. Two wording points (A3). |
| **2.6 Flags** | Agree. |
| **2.7 Clarification** | Later is fine. One constraint (A4). |
| **Part 3** | No objection. 3.5 is where A1 lands. On 3.4 Audio supports News's request (A5). |
| **Part 4** | Agree. Alias handling stays in the app; see A2. |
| **Relaxation to-do** | Agree, and confirmed in Audio's setup (A6). |
| **`build_tsquery` output** | Please keep it valid for `to_tsquery('simple', ...)` (A7). |

### Changes Audio asks for

| # | Change | Priority |
| :--- | :--- | :--- |
| **A1** | **2.2: make "at least one" the widening step, not the only rule.** For each search (the primary query and each secondary query), require **all** the names that occur in that search's own text (a relationship search keeps its meaning; a comparison's secondary search names one subject). Widen to "at least one of those names" only when that returns fewer than top-k candidates. If none of the names occurs in the search's text, use "at least one of all names". See section 2 for the measurement, including a case where "at least one" alone made results worse. | **Need now** (ship with 2.2) |
| **A2** | **2.1: a rule can carry alternatives (aliases) supplied by the app, and a per-rule or per-profile "droppable" flag.** A name has several surface forms in a novel (a full name, a short form, a title). Requiring the exact full name misses many passages. | Later, but design the record for it |
| **A3** | **2.3: define "complete" relative to the visible scope.** An app with a visibility boundary (spoiler guard, access limits) can only be complete within what the user may see. Also consider naming a shape "per-subject lookup" ("A, B and C: what does each do?"), which is not a list and not a comparison, and is covered by 2.2. | Documentation |
| **A4** | **2.7: options in a clarification question must come only from passages inside the visible scope**, never from the router's own knowledge of the document. | Constraint if built |
| **A5** | **3.4: the default check should depend on the script.** Substring match is right for Chinese (no word boundaries). Alphabetic text needs a word boundary (News's point). One rule per script covers both. | Agree with News |
| **A6** | **Relaxation should count only ranked passages** (agrees with the proposal's own to-do). In Audio it can never fire: summaries and the reading window alone always exceed `min_results_before_relax`. | To-do, as proposed |
| **A7** | **Keep `build_tsquery` output valid for `to_tsquery('simple', ...)`**, and announce any new operator. Audio passes the string straight to that function. | Compatibility |

## 2. Evidence for 2.2 and A1

**Method.** Read-only. Audio's real keyword function (`to_tsquery('simple', q)` ordered by `ts_rank`, over an engine-tokenised column with overlapping bigrams) was called with three query shapes for the same search text, on one Chinese-language novel of about 1.5M characters, reading position 22.5% (about 345k characters read, so full-context mode does not apply and a search runs). Names are shown as A to E.

- **A**: 4 characters, appears in 95% of passage-sized windows (the protagonist).
- **B**, **C**: 2 characters each. B and C share 2 of their 106 windows.
- **D**: 4 characters, 15 windows (its short form appears in 28).
- **E**: 3 characters.

The query shapes: **today** (`free & name1 & name2`, every name required), **at-least-one** (`free & (name1 | name2)`, the proposal), **A1** (`free & <all names that occur in this search's own text>`, widened to `free & (<those names joined with |>)` when that returns fewer than 8 candidates). "Candidates" is the size of the keyword result (up to 100). The numbers below count, among the top 8 by `ts_rank`, the passages that name the search's own subject (first table) or both people (second table).

**Today's rule on names that rarely share a passage.** The keyword search returned **0 candidates** for every pair of names with no shared passage, and 2 candidates for three names that share two passages.

**Per-search result, own subject in the top 8:**

| Search (secondary query of a comparison or list) | No name rule | At least one | **A1** |
| :--- | :--- | :--- | :--- |
| subject C, with D in the question | 6 | **0** | 8 |
| subject D, with C in the question | 8 | 8 | 8 |
| subject C, compared with E | 8 | 7 | 8 |
| subject E, compared with C | 8 | 8 | 8 |
| subject A, with B and C in the question | 8 | 8 | 8 |
| subject B, with A and C in the question | 8 | 7 | 8 |
| subject C, with A and B in the question | 8 | 7 | 8 |

**A relationship question** (one search that names both people), passages naming **both** people in the top 8:

| Names | Windows sharing both | Today (all required) | At least one | **A1** |
| :--- | :--- | :--- | :--- | :--- |
| A and F (A is in 95% of windows) | 59 | 8 (65 candidates) | **2** | 8 |
| B and C | 2 | 2 (only 2 candidates) | 2 | 2 (widened) |

**What this shows.**

1. "At least one" fixes the zero-result problem (candidates went from 0 to 30 to 100) and Audio's SQL accepts the `|` form without change.
2. **It is not neutral, and it can lose evidence in two ways.**
   - *A longer name crowds out a shorter one in the same search.* In the first row of the first table, the search for C returned **no passage naming C** and 8 naming D. `ts_rank` scores a passage higher when more query terms match, and D is a three-bigram phrase while C is one bigram. English multi-word names against single-word names could behave the same way.
   - *A name that is everywhere dilutes a relationship question.* In the second table, with A in 95% of windows, "A or F" matches almost everything, and the passages naming both people fell from 8 (today's rule) to **2**. So the proposal's statement that passages naming both "still rank first" does not hold in general: `ts_rank` does not reward the number of distinct names matched.
3. **A1 avoided both** in every search tested. Requiring all the names that occur in the search's own text keeps a relationship question's meaning (8 of 8) and keeps a comparison's secondary search on its own subject (8 of 8 in all seven searches). It widens to "at least one" only when that is thin, which is exactly the zero-result case (and the 2-shared-window case).
4. The check in A1 is deterministic, not a guess: the router's secondary queries name their subject (seen in every comparison and list question traced), and a relationship search is expected to name both people, so a substring test says which names belong to which search. When no name occurs in a search's text, the fallback is the proposal's "at least one of all names".
5. **A "must not" query also runs** on Audio's SQL (`free & name1 & !(name2)` returned 100 candidates, none naming the excluded name), so 2.1's must-not could be applied in Audio's SQL.

**Router frequency (the open question from the earlier trace).** The router set `entity_mode="require"` for 10 of 10 questions naming two or more people (relationships, comparisons, per-person lists) and for 8 of 12 questions overall. It chose "ignore" for no-name questions and for two single-name questions. The router varied between runs (the same question was labelled a comparison in one run and not in another); `require` was stable.

**Limits.** One book, one position, three question families plus two relationship questions, one router sample per question (the comparison and list search texts are the router's own output from earlier runs; the two relationship search texts were written by hand, because the router's own relationship query was not recorded in these runs). A1's threshold of 8 candidates is the top-k used in these runs, not a tuned value. This covers the keyword search only. It does not cover the vector search, the fusion of the two, or answer generation.

## 3. Answers to the questions

| # | Item | Answer | Reasoning |
| :--- | :--- | :--- | :--- |
| 1 | **2.1 Rule kinds** | **Later is fine.** In practice only "must mention" occurs. Audio's SQL can apply phrase, any-of and NOT in `to_tsquery`. `like` is not practical in SQL (passages store offsets; text is sliced from a chunk table), so the engine's check is right for Audio. | Audio's 13 evaluation questions are plain lookups; none asks for a must-not, a code or a pattern. A name rule must accept alternatives (A2): in the tested novel one character is named by its full form in 15 windows and by a short form in 28. |
| 2 | **2.2 Several required names** | **Agree that two or more names must not simply be ANDed across the whole question; need now, with A1.** | Audio's measured result: plain "at least one" fixes zero results but can crowd out a shorter name or dilute a relationship question (section 2). A1 (require the names in each search's own text, widen to "at least one" when thin) was best in every case tested. Audio uses `build_tsquery` output, so it gets the fix by upgrading. News's request for an explicit "all" on the rule record fits A1; Audio would use it as the default for a search that names several people. |
| 3 | **2.3 to 2.5 Lists** | **Later is fine.** | No complete-list question in Audio's data. The owner's multi-person questions ("A, B and C: what does each do?") are per-subject lookups (A3), covered by 2.2. Counting stays in the app. |
| 4 | **3.1 and 3.2** | **Not needed.** Audio uses the engine's router and plans no router of its own. The two-call pattern is workable. | Audio already calls `engine.plan(request)` before streaming, with no intent, so the plan uses a passthrough intent and cannot see the router's `route` or `scope`. Harmless today. |
| 5 | **3.4 Simple match** | **Agree with News (A5).** | For Chinese, substring is right. The real problem is aliases and short forms, not matching. |
| 6 | **2.6 Flags** | **Information is enough.** | Audio reads `answer_found` (nullable), `grounded`, degradations and typed errors. Please keep documenting that `answer_found` can be `None`. |
| 7 | **2.7 Clarification** | **Later is fine; "answer anyway".** | No frequency data (single owner). Constraint in A4: with a spoiler boundary, an option that names something beyond the boundary would leak it. |
| 8 | **Part 4** | **Agree.** | Alias merging stays in Audio (character cards are deferred there). Audio would use a lib helper only if a rule can carry alias groups (A2). Nothing in Parts 2 or 3 should move back into Audio. |
| 9 | **Timing** | **Does not block.** 2.2 with A1 is the one worth doing first, as its own small release. | Upgrading is code-free for Audio, but it changes retrieval, so Audio will re-run its retrieval evaluation tool and the contract kit afterwards. 2.1 can wait. |

## 4. Relation to the News reply

- **Aligned:** 2.2 agreed; information-only flags; lists across documents later; the engine's router; clarification later; alias help welcome.
- **No conflict:** News wants an explicit "all" on the rule record and a non-droppable entity rule per profile. Audio supports both options existing; Audio would leave its entity rules droppable (the meaning search keeps working when a name rule is dropped). A1 uses the same per-search rules idea as 3.5.
- **Script difference:** News needs a word-boundary check for short English names; Audio needs substring matching for Chinese. A5 (one rule per script) serves both.

## 5. What Audio does next

- Nothing is built. When the lib ships 2.2, Audio upgrades, re-runs its retrieval tool (`tools/chat_eval`, including its entity trace tool) and the contract kit, and records the result.
