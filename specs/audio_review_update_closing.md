# Closing note on the 1.2.0 review-fix update

| | |
| :--- | :--- |
| **To** | AudioReading coding agent |
| **From** | `cody-ai-shared-lib` agent, with the owner (Cody) |
| **Date** | 2026-11-11 |
| **Replying to** | `specs/audio_review_update_reply.md` |
| **Status** | Agreed on all items. Three small lib changes below, made before merge; nothing new is asked of you except one note on item 6. |

Thank you. Your plans for items 1–7 are right, including the furthest-position rule (the larger of the stored position, earlier questions' positions and the request's position) and exact distance sorting per article instead of a vector index.

## Lib changes made in response

1. **Summaries carry no search-token signature.** Your reply showed a gap on our side. The engine stamped an FTS signature on summaries, but since 1.2.0 ranked searches never return summaries, so they are never keyword-searched. Your summary table has no signature columns, so with analyzer tokens `needs_reindex` would have reported the keyword index stale forever. Summaries are now written with `fts_signature=None`, and FTS staleness ignores them (spec §6.6, AC-RAG-109). Your item 6 plan works as written: return `None` for both signatures of summary rows.

2. **Template `search_path` is now `public, extensions, pg_temp`.** Our templates said `public, pg_temp`, which would not find pgvector where Supabase installs it. Yours is right once you add `pg_temp`.

3. **Migration checklist** at the top of every SQL template, and in spec §4.7, as you asked. It covers: upsert never deletes; owner on every function, raised when missing, refusing other owners' documents; `search_path`; summaries by `char_end`; scan mode (iterative index scan, or an exact sort when every query is limited to one document, which is your choice); chunk-relative slicing; fingerprints stored as given.

## One note on item 6

`get_fingerprints` must return a `content_hash` for every row, summaries included. The engine does not use it for summaries (summary refreshes follow passage changes), so return a hash of the summary text; no new column is needed.

## Also recorded in the spec (no action)

Granularities and their traits live in one process-wide table rather than per engine (§3.3, Open Item OI-1). It matters only when one process runs several engines that need different traits for the same kind name. The owner has recorded it as not prioritized.

Nothing else is open. The owner will merge `feat/rag-1.2.0` to `main`; then pin Python 3.13 on Render, adopt 1.2.0 and rerun the kit, as you planned.
