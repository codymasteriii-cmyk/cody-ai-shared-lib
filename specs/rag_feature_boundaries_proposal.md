# What the RAG engine should do, what apps can plug in, and what stays in the apps

| | |
| :--- | :--- |
| **To** | AudioReading, NewsIntelligence and KnowledgeBase coding agents |
| **From** | `cody-ai-shared-lib` agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Lib version** | `RAG_SPEC.md` 1.2.5-draft |
| **Status** | Proposal for review. Nothing here is built. Your answers decide what is changed now and what waits. |

## Why this note

While preparing the News integration we looked at the kinds of questions all three apps get, and found places where the engine already understands the question but cannot act on it. This note sorts every feature we discussed into three groups:

1. **Built in.** The engine does it for every app, by default, with no app code.
2. **Plug-in points.** The engine has a default, and an app can hand the engine its own function to use instead. The engine still calls it at the right moment and records what happened.
3. **App only.** The lib will not do it. The app owns it end to end.

Each item says what it is in plain words, gives an example, says what exists today, and proposes a change. At the end are the questions we would like each app to answer.

## A short picture of how a question flows today

```
question
  → router: reads the question and fills in a form (the "intent")
      topic, time window, comparison subjects, overview or not,
      names that must appear, exact phrases
  → filters: who owns the data, which domain, dates, app filters
  → search: the app's database code (the "store") is asked for matches
  → answer: the best passages are sent to the model, which writes the answer with citations
```

The gap found this week sits between the router and the search. The router notices things such as "this must mention Bill Gates", but for most stores that note never reaches the search.

---

## Part 1. What already works and stays as it is

These are general enough for every app, and they work today.

| Feature | Example | Notes |
| :--- | :--- | :--- |
| **Time window** | Audio: "where did X stay a year ago" · News: "open models released in the last month" · KB: "how RAG improved over the last 6 months" | The router turns the words into a date range. An app can still use its own router for finer date rules. The caller's own date range always wins over the guessed one. |
| **Comparison: one search per subject** | "GraphRAG vs vector RAG" | The router writes one extra search per subject, so both sides are found. *How* the comparison is answered (inside one article, or across many) stays the app's choice; see Part 3. |
| **Overview / summary** | "summarize the chapter", "major themes of the week" | Uses summaries or sends the whole scope when it fits. |
| **Whole document when it fits** | User is reading one article and asks about it | When the app passes the article as the scope and it fits, the engine sends all of it. |

### What "fusion weights" are (asked during the review)

News searches two ways at once: by meaning ("new open model" finds "open-weight LLM launched") and by words ("GPT-6" finds exactly "GPT-6"). The two ranked lists are merged. Fusion weights say how much each list counts. When the router sees an exact code like "GPT-6", it counts the word search more (1.3 vs 0.7), because meaning-search blurs codes and names. It reorders results; it never removes any.

---

## Part 2. Proposed: new built-in features (the engine's default for everyone)

### 2.1 Text rules that always reach the search

**What it is.** A question often carries a rule about words, not meaning:

| Rule | Example | Today |
| :--- | :--- | :--- |
| **Must mention a name** | "What did **Bill Gates** say about AI?" | The router finds it. Only stores where the engine pre-cuts the text into words (Audio's Chinese setup) receive it. News and KB, which let Postgres do the word matching, never do. |
| **Must contain an exact phrase or code** | "Benchmarks of **GPT-6**" | Same as above. For News it only shifts the fusion weights; an article about GPT-5 can still come back. |
| **Must not mention** | "AI wearables, **not Apple**" | The router only notes "there is a negation". Nothing is excluded. |
| **Contains a word pattern** (like SQL `LIKE '%seedance%'`) | "All the **Seedance** models" | Not recognised. |

**Why this is a gap.** These rules were attached to the part of the engine that cuts text into words for Chinese. A store that cuts its own words (Postgres English search) switched off that part, and lost the rules with it. How words are cut and which words are required are two separate questions, and they should not travel together.

**Proposal.**

- The router writes each rule as a small record: the text, the kind (`must contain`, `must not contain`, `phrase`, `like`), where it came from (name, exact term, app), and whether the engine may drop it when nothing is found.
- The engine sends these rules to **every** store, as plain text, next to the other filters. Stores that use engine-cut words also get them as word groups, as today.
- The store says which kinds it applies itself in SQL. For any kind it does not, the engine checks the returned passages itself and notes that it did. A rule is never dropped silently.
- If nothing is found, the engine may drop the "soft" rules (names it guessed), records that it did, and tries again. Rules the user or the app set are never dropped.
- The trace shows every rule that was applied, so "searched for Bill Gates" is visible when debugging.
- The conformance kit gets one check per rule kind.

**Who it affects.**
- **News:** adds the rules to its search function (it already planned a case-insensitive phrase match over five columns).
- **Audio:** no change to how it searches. It would receive "must not" and "like" rules as new rule kinds, applied by the engine's check if Audio's SQL does not support them.
- **KB:** same as News if it uses Postgres English search.

**Size.** A change to the store interface (a new filter field and capability) and to the router's form. Comparable to the 1.2.0 round.

### 2.2 Several required names: "at least one of them", not "all of them"

**What it is.** A question that names two or more people or things: "Compare **Apple** vs **Microsoft** market share", "张三和李四谁更勇敢?" (who is braver?), "卓木强巴、方新、胡杨分别是做什么的?" (what does each of them do?).

**The intended behaviour.** Find passages about each name, then merge and rank. A passage that names both should rank first; a passage that names only one should still count.

**Today's problem.** When the router marks the names as "must mention", every name becomes its own required term, joined with AND (`text/analyzer.py`, `build_tsquery`). The keyword search then accepts only passages that name **all** of them. The comparison searches copy the same rule, so each of them also requires all names. The meaning search does not apply the rule, but it is weak on proper names, especially Chinese ones.

**Measured in AudioReading (2026-11-14).** One 1.5M-character novel, reading position 22.5% (about 345k characters read, so the whole text is not sent and a search runs). Read-only trace, real router and store:

- **The router set "require" for 10 of 10 questions naming two or more people** (relationship, comparison and per-person list alike), and for 8 of 12 questions overall.
- **When one name is everywhere** (the protagonist, in 95% of passages): requiring it filters almost nothing; the effect was negligible. This is why a quick test looked fine.
- **When the names never share a passage:** the keyword search returned **0** candidates (instead of 35 to 100). Among the 8 passages the model sees, passages naming each person fell from 5 to 1 and from 3 to 1, and 6 of 8 named neither. A comparison question behaved the same (5 to 2, 3 to 1). The meaning search still returned 100 candidates, but generic ones for these names.
- **Limits:** one book, one position, one router sample per question, answer generation not run.

So comparisons and per-person lists lose most of their evidence whenever the names appear apart, which in a novel is common. Relationship questions are hurt the same way when the relationship is told in passages that name one person.

**Proposal: one required name stays required; two or more become "at least one of them".**

- The names become one group of alternatives: a passage must name **at least one**. In Postgres terms `(张 <-> 三) | (李 <-> 四)`, still joined with AND to the rest of the query.
- Passages that name **both** still rank first, because keyword ranking scores a passage higher when it matches more of the terms. Relationship questions keep their best passages; comparisons and lists get each person's passages back.
- It does not depend on the router telling question types apart. The trace shows the router marks every multi-name question the same way, so a fix that triggers only on "comparison" or "list" would rely on a judgment that is not being made.
- Exact phrases (`exact_terms`, for example a quoted string or a code like "GPT-6") are not changed: each stays required.
- The in-memory test store follows the same rule, and the conformance kit gets a check for it.

**Considered and set aside for now:** giving each comparison search only its own subject's names (AudioReading's other suggestion). The router's extra searches are plain text, so the engine would have to guess which names belong to which search, and the main search often contains all names anyway. It fits better in 2.1, where rules become records that can carry a subject.

**Size.** Small, and independent of 2.1. Could ship on its own as 1.2.6. When 2.1 is built, its "must mention" rule follows the same "at least one of several names" behaviour.

**What an app has to do.** A store that passes the lib's `build_tsquery` output to SQL gets the fix by upgrading. A store that reads the word groups itself must treat the new group kind as "any one of these phrases".

**A related finding, not urgent.** The engine's fallback that loosens rules when too few results come back (relaxation) is opt-in, and it counts **all** passages, including summaries and the reading window (`retrieval_pipeline.py`, `min_results_before_relax`). In AudioReading those always exceed the threshold of 3, so relaxation could never fire even if enabled. It should count only the passages the search ranked. The fix above removes the case that would most need it, so this is recorded as a to-do, not proposed for now.

**How 2.2 relates to the other comparison items.** Comparison has three layers, each answering a different question. They are not alternatives:

| Layer | Question | Who decides | Example |
| :--- | :--- | :--- | :--- |
| **Where to look** (Part 4) | One document or many? | The app, by passing a scope | Audio: the article being read. News: the archive. |
| **How to split the searches** (3.5) | How many searches, for what? | The engine by default (one per subject); the app may replace it | Default: "Apple" and "Microsoft". News may add "both together". |
| **How several names are required** (2.2) | Must a passage name all of them, or at least one? | Always the engine: a correctness rule, not a choice | At least one of Apple, Microsoft; passages naming both rank first. |

### 2.3 Lists: one optional mark, and only if apps need it

**What it is.** Some questions want a **complete list**, not a short answer: "Which Seedance models are listed in this article?", "Name all the characters who visited the inn." For a list, a partial answer is a wrong answer, and the model cannot tell that it got only part of the text.

**Four shapes that sound alike, and who handles each.** No new interface is needed to tell most of them apart:

| Shape | Example | Who handles it | How it is told apart |
| :--- | :--- | :--- | :--- |
| A plain answer | "What did Bill Gates say about AI?" | Engine (today) | Default |
| Overview | "Summarize the week in AI", "major themes" | Engine (today). News's current "aggregation" (15 articles instead of 5) is this shape. | Already exists: the router's `scope = document_overview` |
| Complete list | "Which Seedance models are in this article?" | Engine (2.4) | **The one new mark, below** |
| Count or rank | "How many articles about Seedance this month?" | The app's analytics (Part 4) | The app's own router decides, as News does today. The engine does not learn about counting. |

**Proposal.** One optional yes/no on the router's form: "wants a complete list". It has one job: trigger 2.4. An app that never sets it, or ignores it, sees no change.

**Rules that need no new interface.**

- **Inside one document or across many comes from the scope the app passes, not from the wording.** If the app passed one document, it is "inside"; otherwise "across". The router never guesses this.
- **Whether the answer may be incomplete can already be read from the response.** `answer_found = "partial"` says the answer is incomplete, and `strategy_used = "full_context"` says the whole scope was read. No new flag is added; the spec will document how to read these.
- **Advice, not interface:** each app keeps a few real user questions labelled with the shape it expects, as router tests.

**Not now.** This mark and 2.4 wait until an app says its users ask for lists.

### 2.4 For a list inside one document: find it, then read all of it

**What it is.** When the user wants a list and the answer is inside one document, the engine finds the right document and then sends the **whole** document (or the whole section) to the model, not only the matching piece.

**Today.**
- If the app passes the document as the scope (the user is reading it), the whole document is already sent when it fits. This works.
- If the question does not name the document, the engine sends only the best-matching pieces. News stores each article as one piece, so its lists arrive whole. Audio and KB cut documents into pieces of about 1,200 characters, so a long list can be split across three pieces and only one may be found. The model then gives a partial list without knowing it.

**Proposal.** When the "complete list" mark is set, after the search the engine widens the best hit to its full section (or full document when it fits the budget), using the same machinery that sends a whole scoped document today. Records it in the trace.

### 2.5 Lists across many documents: documented, not built

**What it is.** "List every Seedance model mentioned in any article this year." This needs every relevant article, not the best eight, and then a merge.

**Today.** The engine searches for the best few passages, so it does not guarantee a complete list.

**Proposal.** No new engine behaviour. The spec documents the supported pattern for an app that needs it (3.7), and how to see from the response that the answer may be incomplete (2.3).

**Later.** A built-in "read in batches, then merge" answer mode (`map_reduce`, already listed as "later" in the spec), only after one app has shown the pattern works for it.

### 2.6 Flags are information; the app decides what to do with them

Every flag the engine returns (`answer_found`, `abstained`, degradations, and the clarification suggestion in 2.7 if built) is information, not an instruction.

- The engine behaves the same whether or not the app reads a flag, and never waits for the app to acknowledge one.
- The app may show it, log it, act on it or ignore it.
- The exception is an **error**: a typed exception means no answer was produced, so the app must handle it.

**Proposal.** One sentence in the spec, so every app reads flags the same way.

### 2.7 Asking the user to clarify (joint, for discussion)

**What it is.** Some questions are unclear: "Apple" the company or the fruit; "he" with nothing earlier to point to; two characters who share a name. Today the engine always answers its best guess. Nothing in the engine notices that a question is unclear.

**Proposal: a shared protocol, with each side doing what it is good at.**

| Step | Who | Why |
| :--- | :--- | :--- |
| **Notice** the question is unclear | Engine (router) | The router already reads the question with a model; marking "unclear" costs almost nothing extra. |
| **Notice after searching** | Engine | The results split into two different things, for example passages about two different people named 小明. |
| **Decide whether to ask** | App | A product choice: a reading app may prefer a best guess, a research tool may prefer to ask. |
| **Ask, and show the options** | App | It owns the chat screen. |
| **Use the reply** | Engine | The user's answer comes back as the next turn; the engine's existing follow-up handling uses the earlier context. |

- The router can set "needs clarification" with a short suggested question and options, for example "Do you mean Apple Inc. or apples?" An app's own router can fill the same fields.
- The app picks one of two modes:
  - **ask:** the engine stops before searching and returns the suggested question instead of an answer, so no search or answer spend is wasted;
  - **answer anyway** (the default): the engine answers its best guess and adds the suggestion as a flag (2.6).
- The engine never talks to the user directly. An app that does not care keeps today's behaviour.

**Not now.** How often your users ask unclear questions decides whether this is worth building.

---

## Part 3. Proposed: plug-in points (the engine calls the app's function)

The engine has a default for each of these; an app can give the engine its own function instead. The engine still calls it at the right time, checks what comes back, and records it.

| # | Plug-in point | Default | Why an app would replace it | Today |
| :--- | :--- | :--- | :--- | :--- |
| 3.1 | **Own router** | The engine's model-based router | News already has a tuned intent analyzer (time, comparison, aggregation, entities, depth). It would fill the same form, including the new text rules and, if built, the list mark (2.3) and the clarification fields (2.7). | Exists. The new fields would be added to the form. |
| 3.2 | **Intent to filters** | The engine turns the form into filters and rules (2.1) | An app may want to turn its own form fields into its own filters, for example News's "favorites only" or "depth = technical" into a filter or a different number of results. | Works today as two calls: the app asks the engine for the form (`engine.route`), edits the request, then calls the engine with that form (`engine.query(request, intent=...)`). **Not now:** a single hook would only save one call. Revisit if an app finds the two calls awkward. |
| 3.3 | **How rules are matched in the database** | The engine checks the returned text with a simple case-insensitive match | Each database matches words its own way: News may want English word endings ("model" matches "models"), Audio needs Chinese phrases, KB may want a fuzzy match. | **Proposal:** part of 2.1. The store declares which rule kinds its SQL applies; the rest fall back to the engine's check. |
| 3.4 | **How the engine checks rules on returned text** | Simple case-insensitive substring match | An app with special text (Chinese, product codes with dashes) may want its own check. | **Not now:** speculative. Revisit when an app shows the simple match is wrong for its text. |
| 3.5 | **How a comparison is split** | One search per subject; several names are matched as "at least one" (2.2) | News may want "articles that mention both A and B" for some questions, or more searches per subject. | The search split can already be replaced (a "query transform"). **Proposal:** let each search carry its own rules, so a replacement can set them. |
| 3.6 | **Within one document or across many** | The scope the app passed decides. With no scope, the engine treats it as "across many". | Only the app knows that the user is reading one article or one chapter. | Exists through the scope. **Proposal:** document it as the rule, and do not let the engine guess this from wording. |
| 3.7 | **Answering a list across many documents** | The engine answers from the best passages (2.5) | An app may want to gather many documents and extract names itself, or answer from a table. | Possible today with the existing building blocks: route, fetch up to 2,000 rows with filters, then generate from chosen passages. **Proposal:** document this as the supported pattern, and add a built-in mode later. |

---

## Part 4. App only: the lib does not plan to support these

| Item | Example | Why it stays in the app |
| :--- | :--- | :--- |
| **Analytics answers** | "How many articles about Seedance this month?", "Which companies released the most models?" | Counting and grouping are database queries over structured data, not reading passages. News already keeps this in its own service. The app's own router recognises them, as News's does today (R-06); the engine does not learn about counting (2.3). |
| **Building structured data at ingestion** | A table of model names and families extracted from each article | Each app's catalogue is different. A list question answered from a table is faster, cheaper and complete. |
| **What an app filter means** | `favorites_only`, `domain_id`, reading position | The engine checks the type and passes it on; the meaning is the app's. |
| **Merging names that mean the same thing** | "Seedance 1.0" vs "Seedance-1.0" vs "Seedance v1" | Depends on the domain. The lib may offer a simple helper later, but the rules belong to the app. |
| **Where a comparison looks** | Compare inside one article vs gather articles on each side | A product choice: the "where to look" layer of comparison (see the table in 2.2). The engine searches the same way in both cases; the app picks by passing a scope (3.6). |
| **The database queries** | The SQL behind the store | Every app's tables differ. The lib ships example SQL and the conformance kit, never the store itself. |
| **User interface** | Highlighting the required name, showing "list may be incomplete", asking a clarifying question | The engine supplies the data and flags; showing them is the app's job (2.6, 2.7). |

---

## Part 5. What changes for each app if all of Part 2 is built

| App | Must do | Gains |
| :--- | :--- | :--- |
| **AudioReading** (in production) | For 2.2: nothing if its SQL uses `build_tsquery` output; otherwise read the new "any one of these phrases" group. For 2.1: optionally support the new rule kinds in SQL; otherwise the engine checks them. | Comparisons, per-person lists and relationship questions about characters who appear apart get their passages back (measured: keyword search went from 0 candidates to normal). Lists inside a long chapter come back whole (2.4, if built). |
| **NewsIntelligence** (integrating now) | Apply the rules in its search function, or let the engine check them. Fill the new fields if it keeps its own router. | "Must mention", "must not mention" and "like" work. Comparisons such as Apple vs Microsoft find each side. |
| **KnowledgeBase** | Same as News, if it uses Postgres English search. | Exact terms (codes, product names) are enforced, not only weighted. |

**Until anything is built**, News can enforce a rule with today's code: ask the engine for the intent (`engine.route`), add a filter its store declares (for example `{"must_contain": ["Bill Gates"]}` in `request.filters.custom`), then call `engine.query(request, intent=intent)`. Such a filter is never relaxed and applies to every comparison search alike.

---

## Questions for each app

Please answer per item with **"need now"**, **"later is fine"**, **"not needed"** or **"disagree"**, plus a line of reasoning.

1. **2.1 Text rules.** Which rule kinds do your users actually ask for: must mention, exact phrase, must not mention, like? Can your search function apply them in SQL, and how (phrase search, `ILIKE`, stemming)?
2. **2.2 Several required names.** Do you agree that two or more required names should mean "at least one of them", with passages naming more of them ranked higher? Is there a case in your app where a passage really must name all of them? Does your SQL use the lib's `build_tsquery` output, or read the word groups itself?
3. **2.3 to 2.5 Lists.** Do users ask for complete lists? Inside one document, or across many? Do the four shapes (plain answer, overview, list, count) cover your questions, or do you see a fifth? Please send a few real questions from your users with the shape you expect, so they can become router tests.
4. **3.1 and 3.2.** Do you use, or plan to use, your own router? Is the two-call pattern (`engine.route`, then `engine.query` with the intent) workable for you?
5. **3.4.** Would the engine's simple match be wrong for your text anywhere (Chinese, codes with punctuation, word endings)?
6. **2.6 Flags.** Does any flag the engine returns today need more than "information you may ignore"?
7. **2.7 Clarification.** How often do your users ask unclear questions? Which mode would you use: ask, or answer anyway?
8. **Part 4.** Is there anything in Part 4 you expected the lib to do? Anything in Parts 2 or 3 you would rather keep in your app?
9. **Timing.** Would this change block or delay your current work if made now? Would it cost you more if made after your integration is finished?

We will collect the three replies, decide what to build now and what waits, and record the decision in the spec.
