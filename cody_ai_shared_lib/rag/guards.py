"""Built-in Guards enforcing boundaries, spoiler prevention, and scope isolation (§5).

Guards translate into filter patches store-side, and verify/truncate nodes and citations
in-memory for defense in depth (D-04, D-05, D-24, AC-RAG-23, AC-RAG-24, AC-RAG-73).
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Sequence

from .granularity import TRAIT_OFFSET_ADDRESSABLE, has_trait
from .models import (
    FilterPatch,
    RAGCitation,
    RAGFilterSpec,
    RAGNode,
    as_utc,
)
from .text.analyzer import CjkAwareAnalyzer, TextAnalyzer


class SequentialBoundaryGuard:
    """Enforces spoiler prevention by cutting or dropping text beyond reading boundary B (§5.2)."""
    name: str = "sequential_boundary"
    hard: bool = True

    def __init__(
        self,
        max_char_offset: int | None = None,
        max_char_offset_by_doc: dict[str, int] | None = None,
        max_snap_back_chars: int = 300,
        analyzer: TextAnalyzer | None = None,
    ):
        self.max_char_offset = max_char_offset
        self.max_char_offset_by_doc = dict(max_char_offset_by_doc or {})
        self.max_snap_back_chars = max_snap_back_chars
        self.analyzer = analyzer or CjkAwareAnalyzer()

    @property
    def is_active(self) -> bool:
        return self.max_char_offset is not None or bool(self.max_char_offset_by_doc)

    @property
    def forces_strict_grounding(self) -> bool:
        return self.is_active

    def get_boundary_for_doc(self, doc_id: str) -> int | None:
        """Resolve the effective boundary B for a specific document."""
        bound = self.max_char_offset_by_doc.get(doc_id)
        if bound is not None:
            return min(bound, self.max_char_offset) if self.max_char_offset is not None else bound
        return self.max_char_offset

    def to_filter(self) -> FilterPatch:
        fields: dict[str, Any] = {}
        if self.max_char_offset is not None:
            fields["max_char_offset"] = self.max_char_offset
        if self.max_char_offset_by_doc:
            fields["max_char_offset_by_doc"] = self.max_char_offset_by_doc
        return FilterPatch(fields=fields)

    def verify(self, node: RAGNode[Any]) -> RAGNode[Any] | None:
        """Verify, drop, or sentence-truncate node against boundary B (§5.2, AC-RAG-73)."""
        b = self.get_boundary_for_doc(node.span.doc_id)
        if b is None:
            return node

        # Rule 1: A node starting at or after B is dropped
        if node.span.char_start >= b:
            return None

        # Rule 3: A node without offset_addressable (summary, entity facts) is kept only if char_end <= B
        is_offset_addressable = has_trait(node.granularity, TRAIT_OFFSET_ADDRESSABLE) and not node.is_summary
        if not is_offset_addressable:
            if node.span.char_end <= b:
                return node
            return None

        # Rule 2: Offset-addressable node with char_end >= B is cut at sentence end
        if node.span.char_end >= b:
            limit = b - node.span.char_start
            prefix = node.text[:limit]
            last_end = self.analyzer.last_sentence_end(prefix)

            if last_end is None or (limit - last_end) > self.max_snap_back_chars:
                cut = limit  # Hard cut at B
            else:
                cut = last_end  # Cut at sentence end

            cut = max(0, min(cut, limit))
            if cut == 0:
                return None
            if cut == len(node.text):
                return node

            new_node = copy.deepcopy(node)
            new_node.text = prefix[:cut]
            new_node.span.char_end = new_node.span.char_start + cut
            if "truncated_by_guard" not in new_node.flags:
                new_node.flags.append("truncated_by_guard")
            return new_node

        return node

    def verify_citation(self, citation: RAGCitation) -> bool:
        """Reject citations referring to content past boundary B."""
        b = self.get_boundary_for_doc(citation.span.doc_id)
        if b is None:
            return True
        return citation.span.char_end <= b

    def prompt_rules(self) -> list[str]:
        if not self.is_active:
            return []
        if self.max_char_offset is not None:
            b_desc = str(self.max_char_offset)
        else:
            b_desc = ", ".join(f"{k}:{v}" for k, v in sorted(self.max_char_offset_by_doc.items()))
        return [
            f"SPOILER GUARD ACTIVE: The user has read only up to character offset {b_desc}. "
            "Everything in the <data-...> block is at or before that offset. "
            "You MUST NOT mention, imply, foreshadow, or deduce any event, character fate, "
            "relationship, or reveal that occurs after that point, even if you know the work from pre-training."
        ]


class TemporalGuard:
    """Enforces strict caller-specified date bounds (§5.1)."""
    name: str = "temporal"
    hard: bool = True
    forces_strict_grounding: bool = False

    def __init__(
        self,
        date_range_utc: tuple[datetime | None, datetime | None] | None = None,
        date_field: str | None = None,
    ):
        self.date_range_utc = date_range_utc
        self.date_field = date_field or "created_at"

    def to_filter(self) -> FilterPatch:
        fields: dict[str, Any] = {}
        if self.date_range_utc is not None:
            fields["date_range_utc"] = self.date_range_utc
        if self.date_field is not None:
            fields["date_field"] = self.date_field
        return FilterPatch(fields=fields)

    def verify(self, node: RAGNode[Any]) -> RAGNode[Any] | None:
        if self.date_range_utc is None:
            return node
        dt_start, dt_end = self.date_range_utc
        extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
        if isinstance(node.metadata, dict):
            node_dt = node.metadata.get(self.date_field)
        else:
            node_dt = getattr(node.metadata, self.date_field, None) or (
                extra.get(self.date_field) if isinstance(extra, dict) else None
            )
        if node_dt is not None and isinstance(node_dt, datetime):
            if dt_start is not None and as_utc(node_dt) < as_utc(dt_start):
                return None
            if dt_end is not None and as_utc(node_dt) > as_utc(dt_end):
                return None
        return node

    def verify_citation(self, citation: RAGCitation) -> bool:
        return True

    def prompt_rules(self) -> list[str]:
        return []


class TenantGuard:
    """Drops nodes whose metadata names a different tenant (§5.1); required scope is checked separately (§5.6)."""
    name: str = "tenant"
    hard: bool = True
    forces_strict_grounding: bool = False

    def __init__(self, tenant_or_user_id: str | None = None):
        self.tenant_or_user_id = tenant_or_user_id

    def to_filter(self) -> FilterPatch:
        fields: dict[str, Any] = {}
        if self.tenant_or_user_id is not None:
            fields["tenant_or_user_id"] = self.tenant_or_user_id
        return FilterPatch(fields=fields)

    def verify(self, node: RAGNode[Any]) -> RAGNode[Any] | None:
        if self.tenant_or_user_id is None:
            return node
        extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
        if isinstance(node.metadata, dict):
            node_tenant = node.metadata.get("tenant_id")
        else:
            node_tenant = getattr(node.metadata, "tenant_id", None) or (
                extra.get("tenant_id") if isinstance(extra, dict) else None
            )
        if node_tenant is not None and node_tenant != self.tenant_or_user_id:
            return None
        return node

    def verify_citation(self, citation: RAGCitation) -> bool:
        return True

    def prompt_rules(self) -> list[str]:
        return []


class ScopeGuard:
    """Enforces caller scope constraints (doc_ids, domain, categories, custom)."""
    name: str = "scope"
    hard: bool = True
    forces_strict_grounding: bool = False

    def __init__(
        self,
        doc_ids: list[str] | None = None,
        domain_id: str | None = None,
        categories: list[str] | None = None,
        custom: dict[str, Any] | None = None,
    ):
        self.doc_ids = doc_ids
        self.domain_id = domain_id
        self.categories = categories
        self.custom = dict(custom or {})

    def to_filter(self) -> FilterPatch:
        fields: dict[str, Any] = {}
        if self.doc_ids is not None:
            fields["doc_ids"] = self.doc_ids
        if self.domain_id is not None:
            fields["domain_id"] = self.domain_id
        if self.categories is not None:
            fields["categories"] = self.categories
        if self.custom:
            fields["custom"] = self.custom
        return FilterPatch(fields=fields)

    def verify(self, node: RAGNode[Any]) -> RAGNode[Any] | None:
        if self.doc_ids is not None and node.span.doc_id not in self.doc_ids:
            return None
        extra = getattr(node.metadata, "extra", {}) if hasattr(node.metadata, "extra") else {}
        if self.domain_id is not None:
            if isinstance(node.metadata, dict):
                node_domain = node.metadata.get("domain_id")
            else:
                node_domain = getattr(node.metadata, "domain_id", None) or (
                    extra.get("domain_id") if isinstance(extra, dict) else None
                )
            if node_domain is not None and node_domain != self.domain_id:
                return None
        if self.categories is not None:
            if isinstance(node.metadata, dict):
                node_cat = node.metadata.get("category")
            else:
                node_cat = getattr(node.metadata, "category", None) or (
                    extra.get("category") if isinstance(extra, dict) else None
                )
            if node_cat is not None and node_cat not in self.categories:
                return None
        for k, v in self.custom.items():
            if isinstance(node.metadata, dict):
                node_v = node.metadata.get(k)
            else:
                node_v = getattr(node.metadata, k, None) or (extra.get(k) if isinstance(extra, dict) else None)
            if node_v is not None and node_v != v:
                return None
        return node

    def verify_citation(self, citation: RAGCitation) -> bool:
        if self.doc_ids is not None and citation.span.doc_id and citation.span.doc_id not in self.doc_ids:
            return False
        return True

    def prompt_rules(self) -> list[str]:
        return []


def build_turn_guards(
    filters: RAGFilterSpec,
    analyzer: TextAnalyzer | None = None,
    profile_guards: Sequence[str] = (),
    extra_guards: Sequence[Any] = (),
) -> list[Any]:
    """Build the turn's guards from the effective filters, plus app guards, which only ever add (§5.3)."""
    eff_analyzer = analyzer or CjkAwareAnalyzer()
    active: list[Any] = []

    # 1. Sequential boundary guard from filters or profile
    if (
        filters.max_char_offset is not None
        or filters.max_char_offset_by_doc
        or "sequential_boundary" in profile_guards
    ):
        active.append(
            SequentialBoundaryGuard(
                max_char_offset=filters.max_char_offset,
                max_char_offset_by_doc=filters.max_char_offset_by_doc,
                analyzer=eff_analyzer,
            )
        )

    # 2. Caller hard temporal guard (only when caller set date_range_utc; soft intent time window is NOT a guard per D-04)
    if filters.date_range_utc is not None:
        active.append(
            TemporalGuard(
                date_range_utc=filters.date_range_utc,
                date_field=filters.date_field,
            )
        )

    # 3. Tenant guard
    if filters.tenant_or_user_id is not None:
        active.append(TenantGuard(tenant_or_user_id=filters.tenant_or_user_id))

    # 4. Scope guard
    if (
        filters.doc_ids is not None
        or filters.domain_id is not None
        or filters.categories is not None
        or filters.custom
    ):
        active.append(
            ScopeGuard(
                doc_ids=filters.doc_ids,
                domain_id=filters.domain_id,
                categories=filters.categories,
                custom=filters.custom,
            )
        )

    # 5. App-supplied guards
    active.extend(extra_guards)

    return active


# Names built from the turn's filters; any other profile guard name comes from the registry.
BUILTIN_GUARDS: dict[str, Any] = {
    SequentialBoundaryGuard.name: SequentialBoundaryGuard,
    TemporalGuard.name: TemporalGuard,
    TenantGuard.name: TenantGuard,
    ScopeGuard.name: ScopeGuard,
}
