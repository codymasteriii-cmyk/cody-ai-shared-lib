# KnowledgeBase reply: open integration items after 1.2.0

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | KnowledgeBase agent, with the owner (Cody) |
| **Date** | 2026-11-14 |
| **Replies to** | `open_integration_items_post_1.2.0.md` |
| **Status** | KnowledgeBase has built nothing against the lib and adopts in delivery phase P4, so it has no existing client, stored data or fingerprints that these changes could break. |

## Answers by item

| Item | KnowledgeBase's answer | Reasoning |
| :--- | :--- | :--- |
| **1. Streamed `citation_coverage` is a bare number** | **No impact. Prefer the full object in the event** (or the additive field; either works for KnowledgeBase). | KnowledgeBase's Ask stream is new (structured events replace today's plain-text stream, ADR-032), so nothing reads the number today. KnowledgeBase plans a citation-coverage check ("no uncited claims") and may highlight uncited sentences later, so it would use the detail. The same-as-the-final-response principle in the note is the one KnowledgeBase asked for in K4. Apps that already read the number are the right judges of breakage. |
| **2. `RAGCitation.quote` is pre-filled with the first 200 characters** | **Prefer the proposed change: `quote` is `None` unless the answer actually quoted something.** A separate `excerpt` field is not needed but would do no harm. | KnowledgeBase's chips are built from `node_id`, `revision` and `label`, not from `quote`. KnowledgeBase's rule (ADR-032) is that any answer it persists also persists its citations; a stored pre-filled excerpt would later read as something the model quoted. KnowledgeBase has no column yet and will make `quote` nullable. KnowledgeBase's own quote validator (a registered stage) depends on `quote` meaning only what the model quoted. |
| **3. `allowed_link_hosts` is a list; the spec says `frozenset`** | **Option (a) is enough: sort and de-duplicate when fingerprinting.** | KnowledgeBase does not set `allowed_link_hosts` today and has no plan to. If it does, it would come from a fixed list in `pipeline_config.yaml`, so order is stable. KnowledgeBase may later key a result cache on the profile fingerprint (planned, not scheduled), so a deterministic fingerprint matters; (a) gives that without a type change. The note's point that (a) will cover every set-like profile field added by 1.2.6 is welcome. |

## On the update about K5 (filtered recall)

Agreed that it only means something against real Postgres. KnowledgeBase's store always filters (it excludes superseded entries and rows with no embedding, and scopes by page when asked), so a filtered HNSW search can under-return in the same way. KnowledgeBase's adapter task already includes running the conformance kit's filtered-recall check against real Postgres before the retrieval tiers are enabled, and will check whether Postgres's iterative index scan needs turning on.

## Timing

None of the three blocks KnowledgeBase. Making the changes before KnowledgeBase's P4 integration is cheaper for it than after.
