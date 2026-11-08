"""Citation extraction and mapping (§11.3, AC-RAG-41)."""
from __future__ import annotations

import re
from typing import Generic
from ..models import MetaT, RAGCitation, RAGNode


class CitationExtractor(Generic[MetaT]):
    """Extracts inline citation markers and resolves them to retrieved nodes."""

    def extract(
        self,
        text: str,
        nodes: list[RAGNode[MetaT]],
    ) -> tuple[str, list[RAGCitation]]:
        """Extract citations and return (clean_text, citations_list)."""
        citations: list[RAGCitation] = []
        node_map_by_idx = {i + 1: node for i, node in enumerate(nodes)}
        node_map_by_id = {node.id: node for node in nodes}

        # 1. Look for numeric citations [1], [2], [1, 2], [1, 3]
        def replace_numeric(match: re.Match) -> str:
            raw_nums = match.group(1)
            found_refs: list[str] = []
            for piece in raw_nums.split(","):
                piece = piece.strip()
                if piece.isdigit():
                    num = int(piece)
                    if num in node_map_by_idx:
                        node = node_map_by_idx[num]
                        cite = RAGCitation(
                            index=len(citations) + 1,
                            node_id=node.id,
                            document_id=node.document_id,
                            section_id=node.section_id,
                            quote=node.text[:200],
                            span=node.span,
                            verified=True,
                        )
                        citations.append(cite)
                        found_refs.append(f"[{cite.index}]")
            return " ".join(found_refs) if found_refs else match.group(0)

        # Match [1], [2], [1, 2]
        processed_text = re.sub(r"\[([0-9,\s]+)\]", replace_numeric, text)

        # 2. Look for offset citations [c{seq}:{start}-{end}] (AC-RAG-30, §11.2)
        def replace_offset(match: re.Match) -> str:
            seq = int(match.group(1))
            start = int(match.group(2))
            end = int(match.group(3))
            # Match containing node in the fence
            matching_node = next(
                (n for n in nodes if n.span.section_seq == seq and n.span.char_start <= start and n.span.char_end >= end),
                None,
            )
            if not matching_node:
                # Search for any overlapping node in that section
                matching_node = next(
                    (n for n in nodes if n.span.section_seq == seq and n.span.char_end > start and n.span.char_start < end),
                    None,
                )
            if matching_node:
                cite = RAGCitation(
                    index=len(citations) + 1,
                    node_id=matching_node.id,
                    document_id=matching_node.document_id,
                    section_id=matching_node.section_id,
                    span=matching_node.span.model_copy(update={
                        "char_start": matching_node.span.char_start,
                        "char_end": matching_node.span.char_end,
                    }),
                    quote=matching_node.text[:200],
                    verified=True,
                )
                citations.append(cite)
                return f"[{cite.index}]"
            return match.group(0)

        processed_text = re.sub(r"\[c(\d+):(\d+)-(\d+)\]", replace_offset, processed_text)

        # 3. Look for ref citations [ref:node-id]
        def replace_ref(match: re.Match) -> str:
            nid = match.group(1).strip()
            if nid in node_map_by_id:
                node = node_map_by_id[nid]
                cite = RAGCitation(
                    index=len(citations) + 1,
                    node_id=node.id,
                    document_id=node.document_id,
                    section_id=node.section_id,
                    quote=node.text[:200],
                    span=node.span,
                    verified=True,
                )
                citations.append(cite)
                return f"[{cite.index}]"
            return match.group(0)

        processed_text = re.sub(r"\[ref:([^\]]+)\]", replace_ref, processed_text)

        return processed_text, citations
