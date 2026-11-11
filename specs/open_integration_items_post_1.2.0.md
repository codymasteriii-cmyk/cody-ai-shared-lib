# Open integration items after 1.2.0 (for app review)

| | |
| :--- | :--- |
| **To** | AudioReading, News and KnowledgeBase coding agents |
| **From** | `cody-ai-shared-lib` agent, with the owner (Cody) |
| **Date** | 2026-11-11 |
| **Status** | Not scheduled. Three small changes the lib has deliberately **not** made, because each changes something an app can see. We want each app's feedback first, then one change after the apps have adopted 1.2.0. |
| **Origin** | Second pass over `specs/RAG_CODE_REVIEW_2026-11-11.md`. Everything else in the review is fixed or settled; these are the leftovers. |

None of the three is a bug that loses data or breaks isolation. All three concern the surface the apps consume (stream events, stored quotes, profile fingerprints), so the apps are the right judges of the cost of changing them.

## 1. The streamed `citations` event carries a bare coverage number

- **Today:** the `citations` stream event has `"citation_coverage": 0.83` (`engine.py`, in `stream_query`). The non-streaming `RAGResponse.citation_coverage` is the full `CitationCoverage` object (`sentences_total`, `sentences_cited`, `ratio`, `uncited_sentences`, `total_citations`, `verified_citations`). The spec (§3.10) uses the object.
- **Effect:** a client that reads the stream gets less than one that reads the final response. It cannot show which sentences lack a citation.
- **Proposed change:** send `coverage.model_dump()` in the event. The payload shape changes (number to object), so a client that reads the number breaks unless it adapts. An additive alternative: keep `citation_coverage` as the number and add `citation_coverage_detail` with the object.
- **Questions for each app:**
  1. Do you read `citation_coverage` from the stream today? In what form?
  2. Would you use the uncited sentences (for example to highlight them)?
  3. Breaking change (object replaces number) or additive (new field)?

## 2. `RAGCitation.quote` is always pre-filled with the first 200 characters of the source

- **Today:** every citation has `quote = node.text[:200]` before verification (`postgen/citations.py`). Verification no longer uses it: when the answer quotes nothing next to a marker, the quote check is skipped (1.2.0), and when it does, `quote` is replaced by what the answer quoted.
- **Effect:** a UI that shows `citation.quote` shows an excerpt of the source when the model quoted nothing, or something different from what it quoted. It is an excerpt, not the model's words, and nothing in the field says which.
- **Proposed change:** leave `quote` as `None` unless the answer actually quoted something. A UI that wants an excerpt can slice the source itself.
- **Questions for each app:**
  1. Do you display or store `citation.quote`? As the excerpt or as the model's quote?
  2. Would an empty `quote` break anything (a non-null column, a template)?
  3. Would you rather have a separate `excerpt` field, so `quote` can mean only "what the model quoted"?

## 3. `allowed_link_hosts` is a `list`; the spec says `frozenset`

- **Today:** `EngineProfile.allowed_link_hosts: list[str]` (`profile.py`). The sanitizer already lower-cases it into a set. The profile fingerprint hashes the list as written, so the same hosts in a different order give a different fingerprint.
- **Effect:** none on behavior. The fingerprint differs only if an app builds the list in a non-deterministic order.
- **Proposed change (pick one):** (a) sort and de-duplicate the list when fingerprinting; no type change, no visible change; (b) change the type to `frozenset[str]` (a change for any app that passes or reads a list); (c) change the spec to say `list`.
- **Questions for each app:**
  1. Do you set `allowed_link_hosts`? From a fixed literal or from config or a database?
  2. Do you store or compare profile fingerprints (for example to invalidate caches)?
  3. Is option (a) enough for you?

## What we ask

One reply per app, by item, with "no impact", "prefer X" or "blocks us". We will make one change covering all three once the apps have adopted 1.2.0, and record it in the spec changelog.

## Not in this note

- **K5** (filtered-recall check): decided as not fixing. It only means something against real Postgres; Audio's rerun of the kit against Supabase is the test.
- **M11** (sequential section and summary embedding during ingestion): performance only. Owner decision: no parallelism unless ingestion time becomes a real problem.
