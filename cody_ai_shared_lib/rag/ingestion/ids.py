"""Engine-built node ids (§6.6).

The engine names the nodes it builds with one format. Stores treat every node id as an opaque key:
they store it, look rows up and delete by it, and never parse it. Apps that build their own nodes
(for `prepare_nodes`) may use any id; `parse_node_id` returns None for those.
"""
from __future__ import annotations

import re
from typing import Literal, NamedTuple

NodeKind = Literal["passage", "section_summary", "unit_summary"]


class NodeIdParts(NamedTuple):
    """The parts of an engine-built node id."""
    section_seq: int
    kind: NodeKind
    number: int | None         # passage index or unit_seq; None for the section summary

    @property
    def is_summary(self) -> bool:
        return self.kind != "passage"


def format_node_id(doc_id: str, section_seq: int, kind: NodeKind, index: int | None = None) -> str:
    """`{doc}#s{seq}p{i}` for passages, `{doc}#s{seq}_sum` and `{doc}#s{seq}_u{n}_sum` for summaries."""
    if kind == "passage":
        return f"{doc_id}#s{section_seq}p{index}"
    if kind == "unit_summary":
        return f"{doc_id}#s{section_seq}_u{index}_sum"
    return f"{doc_id}#s{section_seq}_sum"


_NODE_ID_TAIL = re.compile(r"#s(\d+)(?:p(\d+)|_u(\d+)_sum|(_sum))")


def parse_node_id(doc_id: str, node_id: str) -> NodeIdParts | None:
    """Parse an engine-built id, or return None for any other id.

    `doc_id` is required because a doc id may itself contain `#`.
    """
    if not node_id.startswith(doc_id):
        return None
    match = _NODE_ID_TAIL.fullmatch(node_id[len(doc_id):])
    if match is None:
        return None
    seq = int(match.group(1))
    if match.group(2) is not None:
        return NodeIdParts(seq, "passage", int(match.group(2)))
    if match.group(3) is not None:
        return NodeIdParts(seq, "unit_summary", int(match.group(3)))
    return NodeIdParts(seq, "section_summary", None)
