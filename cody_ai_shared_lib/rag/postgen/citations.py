"""Citation formatters, extraction and verification (§11.1, §11.2, AC-RAG-41)."""
from __future__ import annotations

import re
from typing import Any, ClassVar, Generic, Literal, Protocol, Sequence, runtime_checkable

from ..models import MetaT, RAGCitation, RAGNode, SourceSpan
from ..render import DefaultRenderer, NodeRenderer


@runtime_checkable
class CitationFormatter(Protocol):
    """Decides what citation markers look like and how they map back to nodes (§11.1)."""
    name: str

    def marker(self, node: RAGNode[Any], index: int) -> str:
        ...

    def parse(self, text: str) -> list[tuple[str, int, int]]:
        ...

    def resolve(self, marker: str, turn_map: dict[str, str]) -> str | None:
        ...

    def label(self, node: RAGNode[Any]) -> str:
        ...

    def instructions(self) -> str:
        ...


class _BaseFormatter:
    pattern: ClassVar[re.Pattern[str]]

    def __init__(self, renderer: NodeRenderer[Any] | None = None):
        self.renderer = renderer or DefaultRenderer()

    def parse(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in self.pattern.finditer(text)]

    def resolve(self, marker: str, turn_map: dict[str, str]) -> str | None:
        return turn_map.get(marker)

    def label(self, node: RAGNode[Any]) -> str:
        return self.renderer.citation_label(node) or node.span.doc_id

    def instructions(self) -> str:
        return (
            "Cite every factual claim by copying the ref attribute of the node that supports it, "
            "placed right after the claim."
        )

    def expand(self, marker: str) -> list[str]:
        """Split a parsed marker that cites several nodes into single markers."""
        return [marker]


class RefCitationFormatter(_BaseFormatter):
    """Short per-turn markers: `[1]`, `[2]`. Four or more digits are not markers, so `[2024]` stays text.

    Any bracketed number list is a candidate, wherever it appears, so `主角[1]。` and `city[1].` are
    recognised. Whether a candidate is a citation is decided by the extractor against the turn's
    citation map (§11.2): one glued to an ASCII word (`arr[1]`) that resolves to nothing stays text.
    """
    name = "ref"
    pattern = re.compile(r"\[\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\]")

    def marker(self, node: RAGNode[Any], index: int) -> str:
        return f"[{index}]"

    def expand(self, marker: str) -> list[str]:
        return [f"[{number}]" for number in re.findall(r"\d+", marker)]


class OffsetCitationFormatter(_BaseFormatter):
    """Character-range markers: `[c{seq}:{start}-{end}]`. A cited range snaps to its node (§11.2)."""
    name = "offset"
    pattern = re.compile(r"\[c(\d+):(\d+)-(\d+)\]")

    def marker(self, node: RAGNode[Any], index: int) -> str:
        return f"[c{node.span.section_seq}:{node.span.char_start}-{node.span.char_end}]"

    def locate(self, marker: str, nodes: Sequence[RAGNode[Any]]) -> tuple[RAGNode[Any], SourceSpan] | None:
        match = self.pattern.fullmatch(marker)
        if match is None:
            return None
        seq, start, end = (int(g) for g in match.groups())
        hits = [
            node for node in nodes
            if node.span.section_seq == seq and node.span.char_end > start and node.span.char_start < end
        ]
        if len({node.span.doc_id for node in hits}) != 1:
            return None   # no match, or the same range exists in several documents: cannot be verified
        node = hits[0]
        span = node.span.model_copy(
            update={"char_start": max(node.span.char_start, start), "char_end": min(node.span.char_end, end)}
        )
        return node, span


class DocumentCitationFormatter(_BaseFormatter):
    """Whole-document markers: `[doc:{doc_id}]`."""
    name = "document"
    pattern = re.compile(r"\[doc:[^\]]+\]")

    def marker(self, node: RAGNode[Any], index: int) -> str:
        return f"[doc:{node.span.doc_id}]"


BUILTIN_FORMATTERS = {
    "ref": RefCitationFormatter,
    "offset": OffsetCitationFormatter,
    "document": DocumentCitationFormatter,
}


_CODE_RE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


_GLUED_BEFORE = re.compile(r"[A-Za-z0-9_\])]")


def _glued_to_ascii_word(text: str, start: int) -> bool:
    """True when the marker directly follows an ASCII word character or a closing bracket (`arr[1]`, `f(x)[2]`)."""
    return start > 0 and bool(_GLUED_BEFORE.fullmatch(text[start - 1]))


def _markers_outside_code(text: str, formatter: CitationFormatter) -> list[tuple[str, int, int]]:
    """Parsed markers, leaving anything inside a code block or code span alone."""
    code = [match.span() for match in _CODE_RE.finditer(text)]
    return [
        (raw, start, end)
        for raw, start, end in formatter.parse(text)
        if not any(code_start <= start < code_end for code_start, code_end in code)
    ]


def _rebuild(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply marker replacements, tidying spaces only where a marker was removed."""
    parts: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        before = text[cursor:start]
        cursor = end
        if replacement:
            parts.append(before + replacement)
            continue
        parts.append(before.rstrip(" "))
        written = "".join(parts)
        if not written or written.endswith("\n"):  # at a line start, the following spaces go too
            while cursor < len(text) and text[cursor] == " ":
                cursor += 1
    parts.append(text[cursor:])
    return "".join(parts).strip()


def strip_markers(text: str, formatter: CitationFormatter) -> str:
    """Remove citation markers, for a UI that needs clean text while streaming (§10.8).

    Without the turn's citation map it cannot tell `arr[1]` from a citation, so it leaves markers glued
    to an ASCII word alone; `done.answer_final` is the canonical clean text.
    """
    return _rebuild(text, [
        (start, end, "") for _, start, end in _markers_outside_code(text, formatter)
        if not _glued_to_ascii_word(text, start)
    ])


class CitationExtractor(Generic[MetaT]):
    """Resolves markers against this turn's fenced nodes and applies the unverified policy (§11.2)."""

    def __init__(
        self,
        on_unverified: Literal["filter", "flag"] = "filter",
        formatter: CitationFormatter | None = None,
    ):
        self.on_unverified = on_unverified
        self.formatter = formatter or RefCitationFormatter()

    def extract(
        self,
        text: str,
        nodes: list[RAGNode[MetaT]],
        guards: Sequence[Any] | None = None,
        current_revisions: dict[str, str] | None = None,
        on_unverified: Literal["filter", "flag"] | None = None,
        turn_map: dict[str, str] | None = None,
    ) -> tuple[str, list[RAGCitation]]:
        """Return the answer with unverified markers handled, and one citation per distinct marker."""
        policy = on_unverified or self.on_unverified
        markers_to_nodes = turn_map if turn_map is not None else {
            self.formatter.marker(node, i): node.node_id for i, node in enumerate(nodes, start=1)
        }
        nodes_by_id = {node.node_id: node for node in nodes}
        by_marker: dict[str, RAGCitation] = {}

        def citation_for(marker: str) -> RAGCitation:
            if marker not in by_marker:
                by_marker[marker] = self._build_citation(
                    marker, markers_to_nodes, nodes_by_id, guards or (), current_revisions, len(by_marker) + 1
                )
            return by_marker[marker]

        replacements: list[tuple[int, int, str]] = []
        for raw, start, end in _markers_outside_code(text, self.formatter):
            markers = self._expand(raw)
            resolves = all(self.formatter.resolve(m, markers_to_nodes) for m in markers)
            if len(markers) > 1 and not resolves:
                continue  # a list of numbers such as [7, 8, 9] that are not all sources is ordinary text
            if not resolves and _glued_to_ascii_word(text, start):
                continue  # `arr[1]`, `x[2]`: an index, not a citation, unless it names a source
            outputs = []
            for marker in markers:
                citation = citation_for(marker)
                if citation.verified:
                    outputs.append(marker)
                elif policy == "flag":
                    outputs.append("[?]")
            replacements.append((start, end, " ".join(outputs)))

        return _rebuild(text, replacements), list(by_marker.values())

    def _expand(self, marker: str) -> list[str]:
        expand = getattr(self.formatter, "expand", None)
        return expand(marker) if expand is not None else [marker]

    def _build_citation(
        self,
        marker: str,
        turn_map: dict[str, str],
        nodes_by_id: dict[str, RAGNode[MetaT]],
        guards: Sequence[Any],
        current_revisions: dict[str, str] | None,
        ordinal: int,
    ) -> RAGCitation:
        digits = re.fullmatch(r"\[(\d+)\]", marker)
        index = int(digits.group(1)) if digits else ordinal

        node_id = self.formatter.resolve(marker, turn_map)
        node = nodes_by_id.get(node_id) if node_id is not None else None
        span = node.span if node is not None else None

        locate = getattr(self.formatter, "locate", None)
        if node is None and locate is not None:
            located = locate(marker, list(nodes_by_id.values()))
            if located is not None:
                node, span = located

        if node is None:
            return RAGCitation(
                index=index, marker=marker, label="Unverified", node_id="unverified",
                verified=False, verification_score=0.0,
            )

        citation = RAGCitation(
            index=index,
            marker=marker,
            label=self.formatter.label(node),
            node_id=node.node_id,
            document_id=node.span.doc_id,
            section_id=node.span.section_id,
            revision=node.revision,
            quote=node.text[:200],
            span=span,
        )
        if not all(guard.verify_citation(citation) for guard in guards):
            citation.verified = False
            citation.verification_score = 0.0
        if current_revisions is not None and node.revision is not None:
            current = current_revisions.get(node.node_id)
            citation.stale = current is not None and current != node.revision
        return citation
