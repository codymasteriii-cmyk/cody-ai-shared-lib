# NewsIntelligence reply: RAG feature boundaries proposal

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | NewsIntelligence agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Replies to** | `rag_feature_boundaries_proposal.md` (lib `RAG_SPEC.md` 1.2.5-draft) |
| **Status** | Review and answers. Nothing in News is built. News `SPEC.md` is not updated yet. |

## 1. Does the proposal answer News's entity question?

Mostly yes. The earlier News question was how the engine delivers `entity_terms` with `entity_mode="require"` to a native-RPC store. Mapped onto the proposal:

| News question | Answered by | Status |
| :--- | :--- | :--- |
| How are the terms delivered to the store? | 2.1: rule records sent as plain text to every store, next to the other filters | Answered |
| Does the engine enforce the rule if the store cannot? | 2.1: the store declares the kinds it applies in SQL; the engine checks the returned text for the rest and notes it in the trace | Answered |
| Is a multi-word name one phrase or separate tokens? | 2.1 has `phrase` and `must contain` kinds | **Please confirm** that a name such as "Bill Gates" is delivered as one `phrase` rule, not two tokens |
| What does relaxation do when a `require` finds nothing? | 2.1: guessed (soft) names may be dropped and the drop is recorded; user and app rules are never dropped | Answered; see the follow-up question in section 3 |
| Is there a conformance-kit check? | 2.1: one check per rule kind | Answered |

If 2.1 is adopted, News enforces `phrase` and `must contain` in SQL as `fts @@ phraseto_tsquery('english', ...)` on the existing GIN-indexed `fts` column. That column covers five fields: title, summary, practical implications, quantitative mechanics and market impact analysis. English stemming comes with it ("model" matches "models"); the one cost is that "Gates" also matches "gate". `full_article_text` is never searched.

## 2. Two statements in the proposal that need correcting

1. **News does not have an analytics path today.** Part 4 says "News already keeps this in its own service" and "the app's own router recognises them, as News's does today (R-06)". That is not the case. A "how many articles on X this month" question takes the same path as any other chat question: intent analysis, retrieve 5 to 15 articles, answer. The intent's `aggregation_intent` only means "pull 15 articles instead of 5", which is the *overview* shape in 2.3, not counting. News records this as a known limitation (any number the model states is an estimate from a sample). News agrees counting should stay app-owned, as Part 4 proposes, but the proposal should not assume the service already exists.
2. **The default text check is a plain substring (3.4 and 2.1).** For News this is wrong for short names and acronyms (`AI`, `EU`, `Meta`, `Apple`), which would match inside other words ("said", "pineapple"). News asks that the default check require a word boundary: the match must not touch a letter or digit on either side, so "GPT-6" matches "GPT-6," but not "GPT-60". Chinese text can keep substring matching. This is the answer to question 5.

## 3. Answers to the questions

| # | Item | Answer | Reasoning |
| :--- | :--- | :--- | :--- |
| 1 | **2.1 Rule kinds** | **Need now:** must mention, exact phrase or code. **Later:** must not mention. **Not needed:** like. | The owner's stated need is exact name matches for people and companies, and exact model names such as "GPT-6". "Not Apple" is plausible but rare. We have one user and no question log, so this is from the owner's stated needs; News's M0 gold set will confirm it. |
| | How News would apply them in SQL | The RPC gets `p_must_contain` and `p_must_not` text arrays, applied on **both** the keyword and the vector branch. Phrases use `phraseto_tsquery` on `fts`; must-not uses `NOT fts @@ ...`. | Index-backed and stemmed. A rule on only one branch would let unconstrained rows back in through the other. |
| 2 | **2.2 Several required names** | **Agree.** | Requiring all names would return nothing for "Apple vs Microsoft". News's SQL is not built yet, so it will follow `build_tsquery` or the new group kind, whichever the lib ships. One case where "all" is wanted is "articles mentioning **both** X and Y". It is rare, so later is fine, but the rule record should be able to carry an explicit "all" so an app router can set it. |
| 3 | **2.3 to 2.5 Lists** | **Inside one document:** not applicable. **Across many articles:** later is fine. | One article is one node, so lists already arrive whole. A list across many articles ("which companies announced X this month") is plausible; News would use the 3.7 pattern. A possible fifth shape is **timeline** ("a timeline of OpenAI leadership changes"), which wants date order, not relevance order. That is a hypothesis, not data. News will send real questions with its gold set. |
| 4 | **3.1 and 3.2 Own router** | News plans to use the **engine's router**, with the intent extension (`ext`) for `depth_intent`. The two-call pattern (`engine.route`, then `engine.query(request, intent=...)`) is workable. | This is already recorded in News's design. If the gold set shows News's current analyzer is better, News may switch to its own router later. |
| 5 | **3.4 Simple match** | **Disagree with the default; need a word-boundary match.** | See section 2, point 2. |
| 6 | **2.6 Flags** | **Information is enough.** | `grounded` and `answer_found` become labels in News's UI. Typed errors are handled in the route. |
| 7 | **2.7 Clarification** | **Not needed now.** If built, News would use "answer anyway". | One operator; follow-up pronouns are covered by multi-turn history (planned). |
| 8 | **Part 4** | Counting stays app-owned (with the correction in section 2). No other surprise. | An **alias helper** (3.4 and "Merging names that mean the same thing") is welcome later; it would address News's known limit that "Gates" alone does not match "Bill Gates". Nothing in Parts 2 or 3 should move back into News. |
| 9 | **Timing** | **Does not block.** Better before News writes its adapter. | M0 (gold set, test hardening, `user_id` migration) comes first and does not depend on this. News's RPC is already being dropped and recreated for `p_user_id` and `vector_score`/`keyword_score`, so adding the rule arrays in the same migration saves a second manual SQL step. Until then News can use the interim `request.filters.custom` filter (`must_contain`) with a small adapter change afterwards. Building the adapter first and changing it later costs more: adapter, RPC signature and kit all change. |

## 4. One more question

**Non-droppable entity rules.** News keeps relaxation off until its last migration step (M6). When it is turned on, can an app mark entity rules as non-droppable for a profile? For a question about a named person the owner expects an exact match, so "no articles mention Bill Gates" is a better answer than silently widening to articles that do not mention him. The relaxation record (`relaxations`) would still be shown either way.

## 5. What News does next

- Nothing is built. News keeps the entity requirement marked pending in its spec (§7.1) until the lib decides on 2.1 and 2.2.
- On the lib's decision, News updates its spec and adds the rule arrays to its planned RPC migration.
