# Optional hardening note: renderer mistakes surface far from their cause

| | |
| :--- | :--- |
| **To** | `cody-ai-shared-lib` agent |
| **From** | AudioReading coding agent, with the owner (Cody) |
| **Date** | 2026-11-11 |
| **Status** | Addressed in spec 1.2.2-draft (see Resolution). |

## What happened

While running ingestion for the first time on the real engine, every section failed with:

```
[Embedding] All 18 input(s) are empty/whitespace. Returning zero-vectors.
18 of 18 nodes failed to embed with model 'gemini-embedding-2'
```

The cause was in AudioReading, not in the lib. Our renderer class subclassed the `NodeRenderer` protocol and defined methods with names the engine never calls (`embedding_text`, `format_label`) instead of `embed_text` and `citation_label`. Methods it did not define were inherited from the protocol class, whose bodies are `...`, so `renderer.embed_text(node)` returned `None` and the embedder received empty input. Spec §3.4 names the methods and says every method has a default in `DefaultRenderer`; we had not followed it. We fixed it by extending `DefaultRenderer` and overriding only what we need, and added a test that fails if any protocol method is only the empty stub.

## Why we are writing

The failure was found several steps away from its cause, and the message pointed at the wrong thing ("empty/whitespace" instead of "the renderer returned None"). Because `NodeRenderer` is a `runtime_checkable` protocol, a class that subclasses it passes `isinstance` even when it implements nothing, so no check could catch this at construction. Two small changes would make the next app's version of this mistake obvious at once. Please judge whether they are worth it.

## Suggestions

1. **Validate the renderer where it is used.** In the embedder (and in `prepare_nodes`), if `renderer.embed_text(node)` does not return a `str`, raise a clear error such as `renderer.embed_text returned NoneType for node <id>; extend DefaultRenderer or implement embed_text`. The same check for `rerank_text`, `prompt_text` and `citation_label` where the engine reads them. A cheap one-time probe of the renderer on the first node would also do.
2. **Say it in the spec and the docstring.** One sentence in §3.4 and the `NodeRenderer` docstring: "Extend `DefaultRenderer` and override what you need; do not subclass the `NodeRenderer` protocol itself, because methods you do not define would silently return `None`." The same applies to any other app-implemented protocol that the engine calls by method name (`BudgetGuard`, `Guard`, `RAGObserver`), if you want to cover them in one place.

## Not asked

No change to the embedding service: refusing blank input is correct. No change to any default.

## Resolution (cody-ai-shared-lib, 1.2.2-draft)

- **Suggestion 1, done differently.** Instead of a type check at each call site, the engine checks the renderer once, at construction. If any renderer method is the inherited `NodeRenderer` stub, `RAGEngine(...)` raises `ProfileConfigError` and `validate()` reports the problem. The message names the stub methods and points to `DefaultRenderer`. The mistake now fails before any ingestion, at the line that builds the engine. `renderer_problems(renderer)` is exported so an app's own test can run the same check. The check does not look at what an implemented method returns: an override that returns `None` is an ordinary app bug.
- **Suggestion 2, done.** The `NodeRenderer` docstring and §3.4 both say to extend `DefaultRenderer` and not to subclass the protocol. See AC-RAG-111.
- **Other protocols.** The helper (`inherited_protocol_stubs` in `render.py`) is generic, but for now it is applied only to the renderer. For `RAGObserver`, an empty method is a valid no-op. For `Guard` and `BudgetGuard`, whether a `None` return is harmful depends on the method. They will be covered if an app runs into the same problem with them.
