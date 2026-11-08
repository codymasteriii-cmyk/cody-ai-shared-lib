"""Temporal query extraction and resolution (§7.6, AC-RAG-17).

Resolves calendar expressions in questions relative to an injected Clock:
  - "yesterday", "today"
  - "last week", "this week"
  - "past N days", "last N days"
  - "this month", "last month"
  - Year/Quarter/Month patterns ("2026-Q1", "October 2026", "2026")
  - Range repair if range_start_utc > range_end_utc
"""
from __future__ import annotations

import re
from datetime import datetime, time, timedelta, timezone
from typing import Any

from ..runtime import Clock, SystemClock


class TemporalExtractor:
    """Extracts and resolves temporal expressions against an injected Clock."""

    def __init__(self, clock: Clock | None = None):
        self.clock = clock or SystemClock()

    def extract(self, query: str) -> dict[str, Any]:
        """Extract temporal bounds and return dict with resolved fields.

        Returns:
            dict containing:
              - has_time_expression: bool
              - temporal_scope: "none" | "recent" | "historical" | "specific_range"
              - date_window_days: int | None
              - range_start_utc: datetime | None
              - range_end_utc: datetime | None
        """
        now = self.clock.now_utc()
        text = query.lower()

        # 1. "last week" / "previous week"
        if re.search(r"\b(last|previous)\s+week\b", text):
            # Calendar week: Monday to Sunday of previous week
            # weekday: Monday is 0, Sunday is 6
            start_of_current_week = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            start_of_last_week = start_of_current_week - timedelta(days=7)
            end_of_last_week = start_of_current_week - timedelta(microseconds=1)
            return {
                "has_time_expression": True,
                "temporal_scope": "recent",
                "date_window_days": 7,
                "range_start_utc": start_of_last_week,
                "range_end_utc": end_of_last_week,
            }

        # 2. "this week"
        if re.search(r"\bthis\s+week\b", text):
            start_of_current_week = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return {
                "has_time_expression": True,
                "temporal_scope": "recent",
                "date_window_days": (now - start_of_current_week).days + 1,
                "range_start_utc": start_of_current_week,
                "range_end_utc": now,
            }

        # 3. "yesterday" / "昨天"
        if re.search(r"\byesterday\b", text) or "昨天" in text:
            yest_date = (now - timedelta(days=1)).date()
            start = datetime.combine(yest_date, time.min, tzinfo=timezone.utc)
            end = datetime.combine(yest_date, time.max, tzinfo=timezone.utc)
            return {
                "has_time_expression": True,
                "temporal_scope": "recent",
                "date_window_days": 1,
                "range_start_utc": start,
                "range_end_utc": end,
            }

        # 4. "today" / "今天"
        if re.search(r"\btoday\b", text) or "今天" in text:
            today_date = now.date()
            start = datetime.combine(today_date, time.min, tzinfo=timezone.utc)
            return {
                "has_time_expression": True,
                "temporal_scope": "recent",
                "date_window_days": 1,
                "range_start_utc": start,
                "range_end_utc": now,
            }

        # 5. "past/last N days" / "最近N天"
        m_days = re.search(r"\b(past|last)\s+(\d+)\s+days?\b", text)
        if not m_days:
            m_days = re.search(r"最近\s*(\d+)\s*天", text)
        if m_days:
            days = min(max(int(m_days.group(1 if m_days.group(1).isdigit() else 2)), 1), 3650)
            start = now - timedelta(days=days)
            return {
                "has_time_expression": True,
                "temporal_scope": "recent",
                "date_window_days": days,
                "range_start_utc": start,
                "range_end_utc": now,
            }

        # 6. Quarter: e.g. 2026-Q1, Q1 2026
        m_q = re.search(r"\b(\d{4})[-/ ]?[qQ]([1-4])\b|\b[qQ]([1-4])[-/ ]?(\d{4})\b", text)
        if m_q:
            year = int(m_q.group(1) or m_q.group(4))
            quarter = int(m_q.group(2) or m_q.group(3))
            start_month = (quarter - 1) * 3 + 1
            start = datetime(year, start_month, 1, 0, 0, 0, tzinfo=timezone.utc)
            if quarter == 4:
                end = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(microseconds=1)
            else:
                end = datetime(year, start_month + 3, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(microseconds=1)
            return {
                "has_time_expression": True,
                "temporal_scope": "specific_range",
                "date_window_days": (end - start).days,
                "range_start_utc": start,
                "range_end_utc": end,
            }

        # 7. Year: e.g. "in 2025", "2024年"
        m_yr = re.search(r"\b(in|during|for)\s+(20\d\d)\b|(20\d\d)年", text)
        if m_yr:
            year = int(m_yr.group(2) or m_yr.group(3))
            start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            end = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc) - timedelta(microseconds=1)
            return {
                "has_time_expression": True,
                "temporal_scope": "specific_range",
                "date_window_days": 365,
                "range_start_utc": start,
                "range_end_utc": end,
            }

        return {
            "has_time_expression": False,
            "temporal_scope": "none",
            "date_window_days": None,
            "range_start_utc": None,
            "range_end_utc": None,
        }

    @staticmethod
    def repair_range(
        start: datetime | None,
        end: datetime | None,
    ) -> tuple[datetime | None, datetime | None]:
        """Repair reversed range (if start > end, swap them) (§7.6, AC-RAG-17)."""
        if start is not None and end is not None and start > end:
            return end, start
        return start, end
