"""In-memory guards read metadata exactly (§5.1, RAG-SEC-5, AC-RAG-114).

1. A present value of False, 0 or "" is a value that must match, not a missing field. The guards read
   `getattr(...) or extra.get(...)`, so such a node was kept against a filter it did not match.
2. The temporal guard reads ISO strings and dates, not only datetime objects. A node dated "2026-06-01T..."
   was treated as undated and dropped, which emptied answers for a caller date range.
3. A failed Jev call is recorded with billing="unknown" (RAG-OPS-1), like every other failed call.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from cody_ai_shared_lib.rag import DegradationTracker, RAGNode, SourceSpan, UsageTracker
from cody_ai_shared_lib.rag.guards import ScopeGuard, TemporalGuard, TenantGuard
from cody_ai_shared_lib.rag.intent.jev_gate import JevGateRouter
from cody_ai_shared_lib.rag.models import DefaultNodeMeta
from cody_ai_shared_lib.rag.render import parse_timestamp


def node(**meta) -> RAGNode:
    return RAGNode(node_id="a", text="t", span=SourceSpan(doc_id="d"), metadata=DefaultNodeMeta(**meta))


def dict_node(meta: dict) -> RAGNode:
    return RAGNode(node_id="a", text="t", span=SourceSpan(doc_id="d"), metadata=meta)


# ── 1. falsy values are values

@pytest.mark.parametrize(
    "wanted, has, kept",
    [
        (True, True, True),
        (True, False, False),      # the bug: a draft passed a published-only filter
        (False, True, False),
        (2, 0, False),             # the bug: tier 0 passed a tier-2 filter
        (0, 0, True),
    ],
)
def test_custom_filter_compares_falsy_values(wanted, has, kept):
    guard = ScopeGuard(custom={"flag": wanted})
    assert (guard.verify(node(flag=has)) is not None) is kept
    assert (guard.verify(dict_node({"flag": has})) is not None) is kept


def test_a_node_without_the_field_is_still_kept():
    assert ScopeGuard(custom={"flag": True}).verify(node()) is not None


def test_an_empty_domain_or_category_does_not_pass_a_filter_for_another():
    assert ScopeGuard(domain_id="sales").verify(node(domain_id="")) is None
    assert ScopeGuard(categories=["news"]).verify(node(category="")) is None


def test_a_field_in_extra_is_read_too():
    meta = DefaultNodeMeta(extra={"flag": False})
    assert ScopeGuard(custom={"flag": True}).verify(RAGNode(node_id="a", text="t", span=SourceSpan(doc_id="d"), metadata=meta)) is None


def test_tenant_guard_treats_an_empty_tenant_as_a_value():
    assert TenantGuard("t1").verify(node(tenant_id="")) is None
    assert TenantGuard("t1").verify(node(tenant_id="t1")) is not None
    assert TenantGuard("t1").verify(node()) is not None


# ── 2. dates as text

RANGE = (datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 12, 31, tzinfo=timezone.utc))


@pytest.mark.parametrize(
    "value, kept",
    [
        ("2026-06-01T00:00:00+00:00", True),
        ("2026-06-01T00:00:00Z", True),
        ("2026-06-01", True),
        (date(2026, 6, 1), True),
        (datetime(2026, 6, 1, tzinfo=timezone.utc), True),
        ("2025-06-01T00:00:00Z", False),   # outside the range
        ("not a date", False),             # unreadable: treated as undated, so dropped (fail closed)
    ],
)
def test_temporal_guard_reads_strings_and_dates(value, kept):
    assert (TemporalGuard(RANGE).verify(dict_node({"created_at": value})) is not None) is kept


def test_temporal_guard_falls_back_to_the_renderer_when_the_field_is_unreadable():
    renderer = MagicMock()
    renderer.timestamp.return_value = "2026-03-01T00:00:00Z"
    guard = TemporalGuard(RANGE, renderer=renderer)
    assert guard.verify(dict_node({"created_at": "garbage"})) is not None


def test_parse_timestamp_handles_each_shape():
    assert parse_timestamp("2026-06-01T10:00:00Z") == datetime(2026, 6, 1, 10, tzinfo=timezone.utc)
    assert parse_timestamp(date(2026, 6, 1)) == datetime(2026, 6, 1)
    assert parse_timestamp("") is None and parse_timestamp(None) is None and parse_timestamp(5) is None


# ── 3. a failed Jev call is recorded

def test_a_failed_jev_call_is_recorded_as_unknown_billing():
    gateway = MagicMock()
    gateway.jev_client.system_one.side_effect = RuntimeError("Jev timeout")
    tracker = UsageTracker("req")
    degradations = DegradationTracker()
    assert JevGateRouter(gateway=gateway, api_key="k").route("a question", tracker, degradations) is None
    [record] = tracker.stage_usages
    assert record.stage == "intent_jev" and record.billing == "unknown" and record.outcome == "error"
    assert record.estimated and record.calls == 1
    assert "DG-03" in [d.id for d in degradations.degradations]
