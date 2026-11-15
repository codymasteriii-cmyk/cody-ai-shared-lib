# Walkthrough: one question, step by step

This follows two dummy questions through the engine and shows what goes in and out of each step. The data is made up and simplified so it fits on a page; field names are the real ones from `rag/models.py`. For the call structure see `docs/CALL_TREES.md`; for the promises see `specs/RAG_SPEC.md`.

The setting: a reading app. The user has one novel loaded. The novel has two characters, Li Wei and Li Ming.

---

## Example 1: an ordinary question

**Request** (what the app sends): `RAGQueryRequest(question="When did the festival start?", tenant_id="u1", document_id="novel-1")`

| Step | What happens | Output (dummy) |
|---|---|---|
| `_begin()` | A request id and fresh trackers are made. Nothing is shared between requests. | `request_id="req-3f9a..."` |
| `_resolve_intent()` | The router reads the question. | `QueryIntent(route="rag", scope="local_lookup", search_query="festival start date", is_followup=False)` |
| `scoped_filters()` | Request filters, the document and the guards combine. A reader's boundary would hide text past their position here. | filters: `tenant u1`, `doc_ids ["novel-1"]` |
| `_fetch()` | The strategy searches the store with the query. Say it returns 20 candidates. | 20 ranked passages |
| post stages | Rerank, remove near-duplicates, drop weak matches, keep the top few. Guards run again after each. | 6 passages, best first |
| `_clarification()` | No router flag and no app hook, so no check runs. | `None` |
| `assemble_prompt()` | The 6 passages are fenced as untrusted data and fitted to the budget. All 6 fit. | prompt with 6 numbered passages |
| `generate()` | One model call. The model writes the answer with markers and a hidden answer-found flag. | `"The festival began on the first of May [2]."` + flag `yes` |
| `finish()` | `[2]` is checked against passage 2, which the model really saw. The flag is read. A new state is built. | see below |

**Response** (abridged):

```
answer_final   "The festival began on the first of May [2]."
citations      [RAGCitation(marker="[2]", node_id=<passage 2>, verified=True)]
answer_found   "yes"
completeness   None                  (not a list question)
state          ConversationState(turn_count=1, prior_nodes=[the 6 passages], recent_entities=[], ...)
```

The app stores `state` and sends it back with the next question. That is the only thing that carries the conversation.

---

## Example 2: an unclear question, with the opt-in ask

The user first asked about both characters. The engine remembered their names. Now:

**Turn 1:** "Tell me about Li Wei and Li Ming."

`finish()` calls `next_state()`, which calls `remember_entities()`. The router found two names; both occur in passages shown this turn, so both are kept:

```
state.recent_entities = ["Li Wei", "Li Ming"]
```

A name that is *not* in a shown passage is never stored, even if the router "knows" it. This is how the engine avoids revealing anything from outside the user's visible text.

**Turn 2:** "What did he do the following week?" with the state from turn 1 and `clarify="ask"`.

| Step | What happens | Output (dummy) |
|---|---|---|
| `_resolve_intent()` | The router sees "he" and two possible people. | `QueryIntent(is_followup=True, unresolved_referent=True, ...)` |
| `run()` | The search returns passages about both characters. | 8 passages |
| `find_clarification()` | `_from_referent()` reads `state.recent_entities`, then `_confirmed()` checks each name against the visible passages. Both occur. | two confirmed candidates |
| result | `Clarification(reason="unresolved_referent", options=[Li Wei, Li Ming])` | |
| `_asks()` | Mode is `ask`, the reason is askable, a state was sent, and the engine did not ask last turn. | `True` |
| `ask_response()` | The question is built from a template and the two labels. **No model call.** | `"Which one do you mean: Li Wei or Li Ming?"` |

**Response** (abridged). It is shaped so it cannot be mistaken for an answer:

```
answer_final   "Which one do you mean: Li Wei or Li Ming?"
answer_found   None                  (never "no": a reader of "no" would treat it as proof of absence)
citations      []
grounded       False
clarification  Clarification(reason="unresolved_referent", asked=True, options=[...])
state          clarification_asked=True
usage          no generation call
```

The app shows the question. The user answers "Li Wei".

**Turn 3:** "Li Wei."

The new state has `clarification_asked=True`, so the engine **never asks twice in a row**: even if the question were still unclear, it would answer with its best guess and add a prompt note ("The question may refer to ...; answer for the most likely one and say which."). Here the user named the person, so the router no longer flags a missing referent and the turn is an ordinary answer, as in Example 1.

---

## What to take from the two examples

- One request, one `RequestRun`: nothing is stored on the engine, so many requests can run at once.
- Guards and filters are applied again after every step that can add text. A hidden passage cannot enter later.
- The model sees only fenced passages the engine chose, and can cite only what it saw.
- The state is the whole memory of the conversation. The app owns it; the engine never keeps it.
- The ask costs no model call, and its wording comes from visible passages, so no name comes from the model's own knowledge.
