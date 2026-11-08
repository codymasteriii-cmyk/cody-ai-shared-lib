"""Canary test helper for sequential boundary verification (§5.2, §5.3, AC-RAG-23).

Ensures that canary tokens placed past the sequential boundary B appear nowhere
in retrieved nodes, assembled prompt, streaming deltas, answer text, citations, or trace.
"""
from __future__ import annotations

from typing import Any, Sequence
from ..models import RAGCitation, RAGNode, RAGResponse, RAGStreamEvent


def assert_no_text_past_boundary(
    nodes: Sequence[RAGNode],
    boundary: int,
    canary_token: str | None = None,
) -> None:
    """Assert that no node in the list violates sequential boundary B.

    Checks:
      1. No node has span.char_start >= boundary.
      2. No node has span.char_end > boundary.
      3. If canary_token is provided, canary_token is absent from every node's text.
    """
    for n in nodes:
        span = getattr(n, "span", None)
        if span is not None:
            if span.char_start >= boundary:
                raise AssertionError(
                    f"Node {n.node_id} leaked past boundary: char_start {span.char_start} >= boundary {boundary}"
                )
            if span.char_end > boundary:
                raise AssertionError(
                    f"Node {n.node_id} leaked past boundary: char_end {span.char_end} > boundary {boundary}"
                )
        if canary_token and canary_token in n.text:
            raise AssertionError(
                f"Canary token {canary_token!r} leaked in node {n.node_id}: {n.text!r}"
            )


def assert_canary_not_leaked(
    target: RAGResponse | Sequence[RAGStreamEvent] | str,
    canary_token: str,
) -> None:
    """Assert canary token appears nowhere in response, stream events, or prompt text."""
    if isinstance(target, str):
        if canary_token in target:
            raise AssertionError(f"Canary token {canary_token!r} found in text: {target[:200]!r}")
        return

    if isinstance(target, RAGResponse):
        if canary_token in target.answer:
            raise AssertionError(f"Canary token {canary_token!r} found in response answer")
        if canary_token in target.answer_final:
            raise AssertionError(f"Canary token {canary_token!r} found in response answer_final")
        nodes = target.retrieved.nodes if hasattr(target.retrieved, "nodes") else getattr(target, "retrieved", [])
        for node in nodes:
            if hasattr(node, "text") and canary_token in node.text:
                raise AssertionError(f"Canary token {canary_token!r} found in retrieved node {node.node_id}")
        for source in target.sources:
            if canary_token in getattr(source, "text", ""):
                raise AssertionError(f"Canary token {canary_token!r} found in source")
        return

    # Sequence of stream events
    for event in target:
        data = event.data if isinstance(event, RAGStreamEvent) else event.get("data", {})
        data_str = str(data)
        if canary_token in data_str:
            raise AssertionError(f"Canary token {canary_token!r} found in stream event {getattr(event, 'event', '')}")
