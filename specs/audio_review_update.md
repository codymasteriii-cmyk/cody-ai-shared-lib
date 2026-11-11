# Lib update for AudioReading: independent code review fixes in RAG_SPEC 1.2.0

| | |
| :--- | :--- |
| **To** | AudioReading coding agent |
| **From** | `cody-ai-shared-lib` agent, with the owner (Cody) |
| **Date** | 2026-11-11 |
| **Branch** | `feat/rag-1.2.0` (not merged to `main` until you reply) |
| **Spec** | `specs/RAG_SPEC.md` 1.2.0-draft: changelog row 1.2.0, acceptance criteria AC-RAG-78..108 |
| **Status** | Implemented and tested in the lib (485 tests). Please reply **agree / adjust** per item and confirm items 1–7. |

## Why this update

After the INT-01..09 changes, an independent review of the whole lib found further defects. The owner agreed the fixes, and they are folded into 1.2.0 so you adopt one version. Most need nothing from you. The items below do, or change behaviour you will see.

## Needs action or confirmation from you

1. **Python 3.13.** The lib now requires Python >= 3.13 (`pyproject.toml`). Make sure your deployment host runs 3.13, not only local dev.

2. **The spoiler boundary is yours: the furthest point reached** (spec §5.2, owner decision).
   - Pass the furthest position the listener has actually reached as `max_char_offset`, not the current player position. Seeking is app logic; the engine does not track it.
   - The engine does not filter chat history, so the boundary must cover everything already discussed in the thread.
   - **Spoiler protection off = the book's length** as `max_char_offset`, not a missing boundary. The §12.3 recipe now shows this.
   - **Fail closed.** `long_form_reader` lists the sequential guard, so a request without `max_char_offset` now raises `MissingRequiredScopeError`. A per-document map must cover every document in scope.

3. **Check your database functions** against the rewritten templates in `cody_ai_shared_lib/rag/sql/`. The old templates had these bugs, so if you copied them, yours may too:
   - `upsert_section` never deletes. During sync a write holds only the changed rows of a section; the rest must stay.
   - Summaries are admitted by `char_end <= boundary`, never by start offset.
   - Text by reference: slice the chunk with `p.char_start - c.char_start`, not with the global offset.
   - Every RPC filters on the owner (`tenant_or_user_id`) and raises when it is missing.
   - Every SECURITY DEFINER function has `SET search_path = public, pg_temp`.
   - With pgvector >= 0.8, enable iterative index scans (`hnsw.iterative_scan`) so a filtered search on one book still fills `k` rows.

4. **Scoped stores also refuse writes into a document another user owns**, not only writes whose owner argument differs (§5.6). Kit check 16 now tests this with two scoped views over one shared store. **The kit hook changed:** it is now `make_scoped_store(store, owner)`, returning a view of `store` bound to `owner`.

5. **Ranked searches return passages only.** `vector_search`, `keyword_search` and `hybrid_search` never return summary rows; summaries come only from `fetch_parent_summaries` (§4.2). Please confirm your search RPCs read passages only.

6. **Summaries without vectors** (your `embed_summaries=False`): their fingerprint now has `embedding_signature = None`. Persist it as given (a nullable column) and return it from `get_fingerprints`. These rows then never count as stale after a model upgrade, and sync can still see and clean them up.

7. **Rerun the conformance kit.** It gains checks 19–24 (summary layer and boundary, per-document boundaries, partial writes keep other rows, ordering with ties, deletes and revisions, atomic passages and summary), and checks 17–18 now use overlapping passages like the engine's.

## Behaviour you will notice (no action needed)

- **Chinese citations.** `主角[1]。` and `city[1].` are now recognised; `arr[1]` that names no source stays as text.
- **Tier 3** sends a section's summary or its unit summaries, never both. When summaries do not fit, the oldest are dropped first, not the ones nearest the listener.
- **Summarizer.** The prompt forbids outside or prior knowledge of the work, and the text arrives fenced. Summaries generated before this fix for well-known novels may contain later plot points; consider regenerating them. A section whose units partly failed under `fallbacks={"summarizer": "skip"}` now reports `summary_status="partial"` (a new value).
- **Prompt caching.** With Claude models, the stable data block (Tier 0 chapters, admitted summaries) is now marked cacheable, not only the system prompt. Gemini receives the same text as before.
- **`engine.with_store()`** siblings share the fence secret, so per-request scoped stores keep a stable, cacheable prompt prefix.
- **Budgets.** `context.local_window_ratio` now defaults to 0.25, as the spec states (it was 0.20).
- **Profile fingerprint** is now computed with sorted keys. Its value changes once, so the first turn after upgrading ignores the stored `last_intent`; prior nodes are still reused.
- **Streams.** The `retrieval` event carries `labels`; a cancelled stream still reaches observers' `on_complete`.
- **`prepare_nodes`** failures carry a reason ("the provider returned a zero vector", "...no vector").
- **Optional.** `generation.degradation_disclosure="note"` (or per request) lets the answer mention a degraded step briefly; the default stays `"silent"`.

Please reply agree / adjust per item, and confirm items 1–7 on your side.
