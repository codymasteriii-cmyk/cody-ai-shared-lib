"""Citation formatters, extraction and verification (§11.1, §11.2, AC-RAG-41).

How a citation gets from the model's text to a verified `RAGCitation`:

    1. the prompt shows each passage with a marker, e.g. <node ref="[2]" ...>        (the formatter's `marker`)
    2. the model writes "founded in 1999 [2]"
    3. `CitationExtractor.extract()` finds markers in the answer                      (the formatter's `parse`)
    4. each marker is looked up in the turn map {marker -> node id}                  (the formatter's `resolve`)
    5. a marker that resolves to a node the model was really shown, and that the guards allow, is verified;
       anything else is unverified and, by policy, removed from the text ("filter") or shown as "[?]" ("flag")

The point of step 4: a model can invent "[9]". A marker only counts if it maps to a passage in THIS turn's prompt.

Three formatters: `ref` ("[1]", the default), `offset` ("[c2:1500-1620]", a character range that snaps to its node) and
`document` ("[doc:novel-7]"). An app picks one with `profile.citations.formatter`.
"""
from __future__ import annotations

import re
from typing import Any, ClassVar, Generic, Literal, Protocol, Sequence, runtime_checkable

from ..models import MetaT, RAGCitation, RAGNode, SourceSpan
from ..render import DefaultRenderer, NodeRenderer


@runtime_checkable
class CitationFormatter(Protocol):
    """Decides what citation markers look like and how they map back to nodes (§11.1).

    `marker` writes one for the prompt; `parse` finds them in the answer; `resolve` maps one back to a node id;
    `label` is the human name shown for a source; `instructions` is the rule given to the model about how to cite.
    """
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
    """What the three built-in formatters share: finding markers with a regex and looking them up in the turn map."""
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

    >>> RefCitationFormatter().parse("A [1] and [2, 3] but [2024]")        # four digits is a year, not a marker
    [('[1]', 2, 5), ('[2, 3]', 10, 16)]
    >>> RefCitationFormatter().expand("[2, 3]")
    ['[2]', '[3]']
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
        """Find the node a character range falls in, and narrow the span to the cited range.

        None when no shown node overlaps the range, or when the same range exists in several documents (it cannot be
        told which was meant, so it is not verified).
        """
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
    """Parsed markers, leaving anything inside a code block or code span alone.

    Code often contains "[1]" as an array index; it must never be read as a citation.
    """
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

    >>> strip_markers("Sky is blue [1]. See arr[1] too.", RefCitationFormatter())
    'Sky is blue. See arr[1] too.'
    """
    return _rebuild(text, [
        (start, end, "") for _, start, end in _markers_outside_code(text, formatter)
        if not _glued_to_ascii_word(text, start)
    ])


class CitationExtractor(Generic[MetaT]):
    """Resolves markers against this turn's fenced nodes and applies the unverified policy (§11.2).

    >>> nodes = [RAGNode(node_id="n1", text="Founded in 1999."), RAGNode(node_id="n2", text="Moved in 2005.")]
    >>> clean, cites = CitationExtractor().extract("It was founded in 1999 [1]. A made-up claim [9].", nodes)
    >>> clean
    'It was founded in 1999 [1]. A made-up claim.'
    >>> [(c.marker, c.verified) for c in cites]
    [('[1]', True), ('[9]', False)]
    """

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
        """Return the answer with unverified markers handled, and one citation per distinct marker.

        `nodes` must be the passages that were actually in the prompt (`AssembledPrompt.included_nodes`), and `turn_map` the
        prompt's marker-to-node map; with neither, markers are numbered by position. `guards` may reject a citation that
        points at forbidden text. `current_revisions` lets a citation be marked stale.
        """
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
        """Split "[1, 2]" into "[1]" and "[2]" when the formatter supports it; other formats stay as one marker."""
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
        """Build the citation for one marker: resolve it, then check the guards and staleness.

        A marker that resolves to nothing gives an unverified placeholder; one that resolves but is refused by a guard
        is marked unverified too, so a citation can never point at text the reader may not see.
        """
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
            span=span,
        )
        if not all(guard.verify_citation(citation) for guard in guards):
            citation.verified = False
            citation.verification_score = 0.0
        if current_revisions is not None and node.revision is not None:
            current = current_revisions.get(node.node_id)
            citation.stale = current is not None and current != node.revision
        return citation
