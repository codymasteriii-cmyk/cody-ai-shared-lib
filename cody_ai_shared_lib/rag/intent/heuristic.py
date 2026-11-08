"""Heuristic rule-based query classifier (§7.3, AC-RAG-13).

Provides instant 0ms classification without model calls for:
  - Document overview prompts (bilingual EN and ZH)
  - Follow-up continuation prompts
  - No-retrieval chat (greetings, thanks)
  - Exact term extraction from quotes and code symbols
  - Comparison queries
"""
from __future__ import annotations

import re
from ..models import QueryIntent


OVERVIEW_PATTERNS = [
    r"\b(summarize|summary|overview|recap|story so far|main characters|plot outline|synopsis)\b",
    r"\b(what('s| is) (this|the) (book|document|article|story|paper) about)\b",
    r"\b(who are the main characters)\b",
    r"(总结|概述|前情提要|主要角色|故事梗概|主要人物|讲了什么|全书梗概|大纲)",
]

FOLLOWUP_PATTERNS = [
    r"\b(what about|why did (he|she|they)|explain that|tell me more|and then|what else|what next|continue)\b",
    r"(那他|那她|然后呢|继续|详细说说|接下来呢|还有呢|为什么这样)",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening|thanks|thank you|thanks a lot|thank you very much|thx|bye|goodbye)[!.,? ]*$",
    r"^(你好|您好|早上好|晚上好|谢谢|多谢|十分感谢|非常感谢|再见|拜拜)[!.,? ？！。]*$",
]

COMPARISON_PATTERNS = [
    r"\b(compare\b.*\b(to|with|and)\b|difference between\b.*\band\b|\bvs\.?\b|\bversus\b)",
    r"(对比|区别|比较|相比)",
]


class HeuristicRouter:
    """Classifies queries via regex patterns and extracts lexical signals."""

    def classify(self, query: str) -> QueryIntent | None:
        """Classify query if matched by high-confidence heuristic rules, else None."""
        text = query.strip()
        lower = text.lower()

        # 1. No-retrieval greeting / thanks check
        for pat in GREETING_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                return QueryIntent(
                    route="no_retrieval",
                    scope="local_lookup",
                    search_query=text,
                    router_source="heuristic",
                )

        # 2. Extract exact terms from quotes
        quotes = re.findall(r'["“](.+?)["”]|[\'‘](.+?)[\'’]', text)
        exact_terms: list[str] = []
        for q1, q2 in quotes:
            term = (q1 or q2).strip()
            if term and len(term) >= 2:
                exact_terms.append(term)

        # 3. Overview detection
        is_overview = False
        for pat in OVERVIEW_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                is_overview = True
                break

        # 4. Follow-up detection
        is_followup = False
        for pat in FOLLOWUP_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                is_followup = True
                break

        # 5. Comparison detection
        is_comp = False
        for pat in COMPARISON_PATTERNS:
            if re.search(pat, lower, re.IGNORECASE):
                is_comp = True
                break

        if is_overview or is_followup or is_comp or exact_terms:
            return QueryIntent(
                route="rag",
                scope="document_overview" if is_overview else "local_lookup",
                search_query=text,
                is_comparison=is_comp,
                is_followup=is_followup,
                exact_terms=exact_terms[:8],
                router_source="heuristic",
            )

        return None
