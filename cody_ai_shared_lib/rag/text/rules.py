"""Text rules: the default check, derivation from an intent, per-search resolution (§7.9).

The engine derives rules from the router's names and exact terms (and from app, caller and router rules),
resolves several names per search (AudioReading's A1), and checks passages itself for any store that does
not apply a kind of rule in SQL.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Sequence

from ...tokens import is_cjk_char
from ..models import QueryIntent, RAGNode, TextRule, merge_text_rules

_CJK_RANGES = "\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af"
# A letter or digit that is not CJK: a Latin form must not touch one on either side. CJK text has no word boundary.
_WORD_CHAR = rf"[^\W_{_CJK_RANGES}]"


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def _form_matches(text: str, form: str) -> bool:
    """One surface form against already-normalized text, by script (§7.9)."""
    form = _norm(form).strip()
    if not form:
        return False
    if any(is_cjk_char(ch) for ch in form):
        return form in text                              # no word boundaries: substring
    pattern = rf"(?<!{_WORD_CHAR}){re.escape(form)}(?!{_WORD_CHAR})"
    return re.search(pattern, text) is not None          # whole word; "GPT-4o", "k-NN", "BM25" are one unit


def mentions(text: str, rule: TextRule) -> bool:
    """True when the text contains the rule's phrase or one of its alternatives."""
    normalized = _norm(text)
    return any(_form_matches(normalized, form) for form in [rule.text, *rule.alternatives])


def holds(text: str, rule: TextRule) -> bool:
    """Whether the text satisfies one rule: it mentions the phrase, or for `must_not` it does not."""
    found = mentions(text, rule)
    return not found if rule.kind == "must_not" else found


def satisfies(text: str, rules: Sequence[TextRule]) -> bool:
    """Every `all` rule holds, and at least one `any` rule holds (when there are any)."""
    if not all(holds(text, rule) for rule in rules if rule.group != "any"):
        return False
    any_rules = [rule for rule in rules if rule.group == "any"]
    return not any_rules or any(holds(text, rule) for rule in any_rules)


def check_nodes(
    nodes: list[RAGNode[Any]], rules: Sequence[TextRule], prompt_text: Callable[[RAGNode[Any]], str]
) -> list[RAGNode[Any]]:
    """Keep the nodes whose prompt text satisfies the rules."""
    return [node for node in nodes if satisfies(prompt_text(node), rules)]


# ── derivation ─────────────────────────────────────────────────────────────


def derive_rules(intent: QueryIntent, caller_rules: Sequence[TextRule], config: Any) -> list[TextRule]:
    """All rules for one request, with the defaults of §7.9. `config` is the profile's TextRulesConfig."""
    derived: list[TextRule] = []
    if intent.entity_mode == "require":
        strength = config.names
        droppable = config.names_droppable if config.names_droppable is not None else strength == "prefer"
        derived += [
            TextRule(text=name, kind="must_mention", group="auto", strength=strength, droppable=droppable, origin="router")
            for name in intent.entity_terms
            if name.strip()
        ]
    derived += [
        TextRule(text=term, kind="exact", group="all", strength="filter", droppable=False, origin="router")
        for term in intent.exact_terms
        if term.strip()
    ]
    for rule in intent.rules:
        if rule.origin == "router" and rule.kind == "must_not":
            rule = rule.model_copy(update={"strength": config.must_not, "droppable": config.must_not == "prefer"})
        derived.append(rule)
    return merge_text_rules(derived, list(caller_rules))


def resolve_group(
    rules: Sequence[TextRule], search_text: str
) -> tuple[list[TextRule], list[TextRule] | None]:
    """Resolve `group=auto` for one search: (narrow rules, widened rules or None).

    Names found in the search's own text are each required; with none found, "any of all the names"; a single
    name never widens (§7.9, A1). Rules that are not `auto` pass through unchanged.
    """
    auto = [rule for rule in rules if rule.group == "auto"]
    other = [rule for rule in rules if rule.group != "auto"]
    if len(auto) <= 1:
        return [*other, *(rule.model_copy(update={"group": "all"}) for rule in auto)], None
    present = [rule for rule in auto if mentions(search_text, rule)]
    if not present:
        return [*other, *(rule.model_copy(update={"group": "any"}) for rule in auto)], None
    narrow = [*other, *(rule.model_copy(update={"group": "all"}) for rule in present)]
    if len(present) < 2:
        return narrow, None
    wide = [*other, *(rule.model_copy(update={"group": "any"}) for rule in present)]
    return narrow, wide
