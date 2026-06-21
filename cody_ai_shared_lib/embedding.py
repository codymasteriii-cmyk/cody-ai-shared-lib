"""Embedding Service — Generates 768-dimensional vectors via Gemini Embedding API.

Supports both document ingestion (RETRIEVAL_DOCUMENT task type) and query
embedding (RETRIEVAL_QUERY task type) for RAG pipelines. All vectors are
compatible with Supabase pgvector cosine distance matching.

Failure strategy:
  1. Empty/whitespace inputs are pre-filtered and assigned zero-vectors to
     maintain index alignment. Callers are responsible for zero-vector guards
     before writing to any vector store.
  2. Valid inputs are sent to the API in chunks of CHUNK_SIZE (default 8).
     The synchronous embed_content API has a ~20,000 total-token ceiling per
     request. At ~2,000 tokens per input, 8 items × 2,000 tokens = 16,000
     tokens — safely under the ceiling.
  3. Each chunk is retried once (with a 3-second delay) on failure.
  4. Any chunk that fails both attempts falls back to per-item individual
     calls, truncated to 8,000 characters.
"""
import os
import time
import logging
from google import genai
from google.genai import types
from typing import List, Union

logger = logging.getLogger("service-embedding")

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMENSIONS = 768

# Hard character ceiling for individual-fallback embedding calls.
# Applied only in Phase 2 (per-item fallback); batch callers should
# pre-truncate inputs before calling this service.
_MAX_INDIVIDUAL_CHARS = 8_000

# Number of items per synchronous embed_content batch call.
# Sized to stay safely under the ~20,000 total-token ceiling.
_CHUNK_SIZE = 8


def generate_embedding(
    text: Union[str, List[str]],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> Union[List[float], List[List[float]]]:
    """Generates 768-dimensional embeddings using Gemini gemini-embedding-2-preview.

    Supports both single strings and batches (List[str]).
    Preserves input length: returns zero-vectors for empty/whitespace-only
    strings to maintain index alignment in batch pipelines.

    Failure handling: chunked batch → per-chunk retry → per-item fallback → zero-vector.
    Guards against silent partial failures where the API returns fewer embeddings
    than requested (observed on cloud deployments when batches exceeded token ceilings).

    Args:
        text: Input string or list of strings.
        task_type: "RETRIEVAL_DOCUMENT" (default) for storage, "RETRIEVAL_QUERY" for search.

    Returns:
        Single embedding (List[float]) for string input, or List[List[float]] for batch.
    """
    if not text:
        return []

    is_batch = isinstance(text, list)
    input_list = text if is_batch else [text]

    valid_indices = [i for i, t in enumerate(input_list) if t and t.strip()]
    results = [[0.0] * EMBEDDING_DIMENSIONS for _ in range(len(input_list))]

    if not valid_indices:
        logger.warning(
            f"[Embedding] All {len(input_list)} input(s) are empty/whitespace. "
            "Returning zero-vectors. Caller should filter before writing to vector store."
        )
        return results if is_batch else results[0]

    if len(valid_indices) < len(input_list):
        invalid_indices = [i for i in range(len(input_list)) if i not in valid_indices]
        for i in invalid_indices:
            stripped = input_list[i].strip() if input_list[i] else ""
            preview = f"'{stripped[:50]}...'" if stripped else "EMPTY/WHITESPACE STRING"
            logger.warning(f"[Embedding] Pre-filtered index {i} ({preview}): empty/whitespace.")

    valid_texts = [input_list[i] for i in valid_indices]
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), http_options={"timeout": 60000})

    def _embed_chunk(chunk_texts: List[str], chunk_indices: List[int]) -> bool:
        """Issues one batch call with a length-check guard and one retry.

        Returns True if all embeddings were written to results, False if both
        the initial call and the retry failed.
        """
        try:
            contents_wrapped = [types.Content(parts=[types.Part(text=t)]) for t in chunk_texts]
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents_wrapped,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            # Guard against silent partial failures (API returns fewer results than requested).
            if len(resp.embeddings) != len(chunk_texts):
                raise ValueError(
                    f"Mismatched embedding count: requested {len(chunk_texts)}, "
                    f"got {len(resp.embeddings)}"
                )
            for idx, embedding in zip(chunk_indices, resp.embeddings):
                results[idx] = embedding.values
            logger.info(f"[Embedding] Chunk succeeded: {len(chunk_texts)} item(s).")
            return True

        except Exception as e:
            logger.warning(f"[Embedding] Chunk of {len(chunk_texts)} failed: {e}. Retrying after 3s...")
            time.sleep(3)
            try:
                contents_wrapped = [types.Content(parts=[types.Part(text=t)]) for t in chunk_texts]
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents_wrapped,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
                )
                if len(resp.embeddings) != len(chunk_texts):
                    raise ValueError(
                        f"Mismatched count on retry: requested {len(chunk_texts)}, "
                        f"got {len(resp.embeddings)}"
                    )
                for idx, embedding in zip(chunk_indices, resp.embeddings):
                    results[idx] = embedding.values
                logger.info(f"[Embedding] Chunk retry succeeded: {len(chunk_texts)} item(s).")
                return True

            except Exception as e2:
                logger.error(
                    f"[Embedding] Chunk retry failed: {e2}. "
                    f"Indices {chunk_indices} falling back to per-item calls."
                )
                return False

    # Phase 1: Chunked batch processing (attempt + retry per chunk)
    failed_indices = []
    for chunk_start in range(0, len(valid_texts), _CHUNK_SIZE):
        chunk_texts = valid_texts[chunk_start : chunk_start + _CHUNK_SIZE]
        chunk_indices = valid_indices[chunk_start : chunk_start + _CHUNK_SIZE]
        if not _embed_chunk(chunk_texts, chunk_indices):
            failed_indices.extend(chunk_indices)

    if not failed_indices:
        logger.info(f"[Embedding] Complete: all {len(valid_indices)} item(s) embedded.")
        return results if is_batch else results[0]

    # Phase 2: Per-item fallback for failed chunks only
    logger.error(f"[Embedding] Per-item fallback for {len(failed_indices)} item(s).")
    time.sleep(3)
    recovered = 0

    for i in failed_indices:
        truncated = input_list[i][:_MAX_INDIVIDUAL_CHARS]
        try:
            resp = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[truncated],
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIMENSIONS),
            )
            results[i] = resp.embeddings[0].values
            recovered += 1
        except Exception as e3:
            # Leave results[i] as zero-vector; caller must check before writing.
            logger.error(
                f"[Embedding] Per-item fallback failed for index {i} "
                f"(preview: '{input_list[i][:80].strip()}...'): {e3}. Zero-vector assigned."
            )
        time.sleep(1.5)

    logger.info(
        f"[Embedding] Fallback complete: {recovered}/{len(failed_indices)} recovered. "
        f"{len(failed_indices) - recovered} zero-vector(s) assigned."
    )
    return results if is_batch else results[0]
