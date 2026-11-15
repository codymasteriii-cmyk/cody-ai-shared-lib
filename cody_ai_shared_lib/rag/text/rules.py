"""Text rules: the default check, derivation from an intent, per-search resolution (§7.9).

The engine derives rules from the router's names and exact terms (and from app, caller and router rules),
resolves several names per search (AudioReading's A1), and checks passages itself for any store that does
not apply a kind of rule in SQL.

What a text rule is: a condition on the passage's own words, stronger than "this is similar to the question".
The kinds: `must_mention` (the passage must contain a name), `must_not` (it must not), `exact` (a quoted phrase).
The strength: `filter` removes passages that fail, `prefer` only ranks them lower. Rules come from the router (names
the question requires, phrases it quotes, exclusions it read), from the profile's defaults, and from the app.

Dummy example: the question "what did Li Wei and Li Ming do in Beijing?" gives two auto-grouped name rules.
    both names appear in the search text   -> each is required ("all"); if too few passages, widen to "any of them"
    neither appears in the search text      -> "any of them" from the start
    only Li Wei appears                     -> required; a single name never widens
`mentions` is the script-aware test underneath: a whole-word match for Latin text, a substring match for Chinese.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Sequence

from ...tokens import CJK_CHAR_CLASS, is_cjk_char
from ..models import QueryIntent, RAGNode, TextRule, merge_text_rules

# A letter or digit that is not CJK (the same ranges as `is_cjk_char`): a Latin form must not touch one on either
# side. CJK text has no word boundary.
_WORD_CHAR = rf"[^\W_{CJK_CHAR_CLASS}]"


def _norm(text: str) -> str:
    """Fold text for comparison: compatibility normalization (full-width -> normal) and case-folding."""
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
    """True when the text contains the rule's phrase or one of its alternatives.

    Latin forms match as whole words (so "cat" is not found in "category"), and punctuation inside a name such as
    "GPT-4o" keeps it one unit. Chinese forms match as substrings, since Chinese has no word boundaries.

    >>> rule = lambda name: TextRule(text=name, kind="must_mention")
    >>> mentions("The GPT-4o model", rule("GPT-4o")), mentions("A category", rule("cat"))
    (True, False)
    >>> mentions("李明去了北京", rule("李明"))
    True
    """
    normalized = _norm(text)
    return any(_form_matches(normalized, form) for form in [rule.text, *rule.alternatives])


def holds(text: str, rule: TextRule) -> bool:
    """Whether the text satisfies one rule: it mentions the phrase, or for `must_not` it does not.

    >>> holds("a quiet day", TextRule(text="storm", kind="must_not"))
    True
    """
    found = mentions(text, rule)
    return not found if rule.kind == "must_not" else found


def satisfies(text: str, rules: Sequence[TextRule]) -> bool:
    """Every `all` rule holds, and at least one `any` rule holds (when there are any).

    This is the engine-side check used when a store cannot apply a kind of rule itself.
    """
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
    """All rules for one request, with the defaults of §7.9. `config` is the profile's TextRulesConfig.

    Order of sources: the router's required names, the router's quoted exact terms, rules the router or an app
    router set on the intent, then the caller's own rules, which win a conflict (`merge_text_rules`). A router's
    "not about X" is only a preference by default, because a model's exclusion is a guess; the caller's is a filter.
    """
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

    >>> name = lambda text: TextRule(text=text, kind="must_mention", group="auto")
    >>> narrow, wide = resolve_group([name("Li Wei"), name("Li Ming")], "Li Wei and Li Ming")
    >>> [r.group for r in narrow], [r.group for r in wide]
    (['all', 'all'], ['any', 'any'])
    >>> narrow, wide = resolve_group([name("Li Wei"), name("Li Ming")], "Li Wei trip")
    >>> [r.group for r in narrow], wide
    (['all'], None)
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
