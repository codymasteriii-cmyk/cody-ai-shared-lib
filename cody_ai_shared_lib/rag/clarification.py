"""Asking the user to clarify: when to, what to offer, and how to word it (§7.10, OI-5).

The engine never waits for or talks to the user. It marks a question as unclear and, when an app opts in, returns
the question as the answer so the app can show it; the user's reply arrives as the next turn.

Rules this module keeps:
  * The router's suspicion alone never triggers a question. The search (or the conversation's passages) has to
    confirm two or more distinct candidates in passages inside the visible scope.
  * Every option label, and so every name in the question, comes from those passages, never from a model.
  * Nothing is checked unless there is a reason: an unresolved reference the router flagged, or an app hook for
    telling referents apart (in-memory grouping only; with a named entity it is `several_matches`, without one
    `too_broad`, which is a mark and never asked).
  * `recent_entities` in the conversation state is untrusted. It is capped and trimmed on every read, and a name
    is only a hint until a visible passage confirms it.

Call tree (the engine calls `find_clarification` after retrieval; `ask_response` in answering.py calls `question_text`):

    find_clarification()
    ├─ router flagged an unresolved reference  ->  _from_referent()
    │  └─ _confirmed()        which remembered names occur in a visible passage (search hits, then earlier turns)
    └─ the app gave a candidate_key hook       ->  _from_hook()
       └─ groups the ranked hits by candidate; two strong groups, none dominating, is ambiguous

Dummy example: after chatting about Li Wei and Li Ming, the user asks "what did he do the following week".
The router flags `unresolved_referent`; both names are remembered in the state and both occur in visible
passages, so the result is Clarification(reason="unresolved_referent", options=[Li Wei, Li Ming]) and, with
`clarify="ask"`, the answer text is "Which one do you mean: Li Wei or Li Ming?" with no model call.
"""
from __future__ import annotations

from typing import Any, Callable, Literal, Sequence

from ..tokens import is_cjk_char
from .granularity import is_ranked
from .models import Clarification, ClarificationOption, ConversationState, QueryIntent, RAGNode, TextRule
from .text.rules import mentions

MAX_ENTITIES = 20
MAX_ENTITY_CHARS = 60

# node -> (key, label) | None: how an app tells referents apart (two characters both called Li Ming)
CandidateKey = Callable[[RAGNode[Any]], "tuple[str, str] | None"]

DEFAULT_TEMPLATES = {
    "en": "Which one do you mean: {options}?",
    "zh": "你指的是哪一个：{options}？",
}
_JOINERS = {"en": (", ", " or "), "zh": ("、", "还是")}


def clean_entities(raw: Any) -> list[str]:
    """The names a state may carry: strings only, trimmed, at most 60 characters, de-duplicated, the newest 20.

    The state can travel through a browser, so every read goes through here: nothing in it is trusted.

    >>> clean_entities(["Li Wei", " Li  Ming ", "li wei", 7, "x" * 61])   # a repeat moves to the newest slot
    ['Li Ming', 'li wei']
    """
    if not isinstance(raw, (list, tuple)):
        return []
    seen: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, str):
            continue
        name = " ".join(item.split())
        if not name or len(name) > MAX_ENTITY_CHARS:
            continue
        seen.pop(name.casefold(), None)                  # a repeat moves to the newest position
        seen[name.casefold()] = name
    return list(seen.values())[-MAX_ENTITIES:]


def _rule(name: str) -> TextRule:
    """A name as a "must_mention" text rule, so the script-aware check of §7.9 (CJK, word boundaries) is reused."""
    return TextRule(text=name, kind="must_mention")


def remember_entities(
    current: Sequence[str],
    intent: QueryIntent,
    shown: Sequence[RAGNode[Any]],
    prompt_text: Callable[[RAGNode[Any]], str],
) -> list[str]:
    """Add the names the router found that also occur in a passage shown this turn (never the router's own knowledge).

    The passage is read as the model read it (`prompt_text`), the same text a stored name is later confirmed against.
    """
    texts = [prompt_text(node) for node in shown]
    seen = [
        name for name in clean_entities(intent.entity_terms)
        if any(mentions(text, _rule(name)) for text in texts)
    ]
    return clean_entities([*clean_entities(current), *seen])


# ── the decision ───────────────────────────────────────────────────────────


def _confirmed(names: list[str], nodes: Sequence[RAGNode[Any]], prompt_text: Callable[[RAGNode[Any]], str]) -> dict[str, list[str]]:
    """Which of `names` occur in which passages, as the model reads them: {name: [node ids]}. Unseen names are absent."""
    found: dict[str, list[str]] = {}
    for name in names:
        rule = _rule(name)
        ids = [node.node_id for node in nodes if mentions(prompt_text(node), rule)]
        if ids:
            found[name] = ids
    return found


def _from_referent(
    state: ConversationState | None,
    nodes: Sequence[RAGNode[Any]],
    prior: Callable[[], list[RAGNode[Any]]],
    prompt_text: Callable[[RAGNode[Any]], str],
    max_options: int,
) -> Clarification | None:
    """The "he/she/it" case: two or more remembered names, each confirmed in a visible passage (§7.10).

    Confirms against this turn's search hits first; only if that leaves fewer than two does it call `prior()`,
    which makes one store call for the passages of earlier turns.
    """
    names = clean_entities(state.recent_entities) if state is not None else []
    if len(names) < 2:
        return None
    found = _confirmed(names, nodes, prompt_text)
    if len(found) < 2:
        remaining = [n for n in names if n not in found]
        for name, ids in _confirmed(remaining, prior(), prompt_text).items():     # one store call, only when needed
            found[name] = ids
    if len(found) < 2:
        return None
    options = [ClarificationOption(label=name, key=name.casefold(), node_ids=ids[:3]) for name, ids in found.items()]
    return Clarification(reason="unresolved_referent", options=options[:max_options])


def _from_hook(
    intent: QueryIntent,
    nodes: Sequence[RAGNode[Any]],
    candidate_key: CandidateKey,
    min_group_hits: int,
    dominance: float,
    max_options: int,
) -> Clarification | None:
    """The app-hook case: group the ranked hits by what `candidate_key` says each one is about.

    Ambiguous when at least two groups each have `min_group_hits` hits and no group holds more than `dominance`
    of all keyed hits. With a named entity in the question that is `several_matches`; without one, `too_broad`.
    """
    groups: dict[str, tuple[str, list[str]]] = {}
    for node in nodes:
        if not is_ranked(node):
            continue
        keyed = candidate_key(node)
        if keyed is None:
            continue
        key, label = keyed
        groups.setdefault(key, (label, []))[1].append(node.node_id)
    strong = {key: group for key, group in groups.items() if len(group[1]) >= min_group_hits}
    if len(strong) < 2:
        return None
    total = sum(len(ids) for _, ids in groups.values())
    if max(len(ids) for _, ids in strong.values()) / total > dominance:
        return None
    options = [ClarificationOption(label=label, key=key, node_ids=ids[:3]) for key, (label, ids) in strong.items()]
    reason: Literal["several_matches", "too_broad"] = "several_matches" if intent.entity_terms else "too_broad"
    return Clarification(reason=reason, options=options[:max_options])


def find_clarification(
    *,
    state: ConversationState | None,
    intent: QueryIntent,
    nodes: Sequence[RAGNode[Any]],
    prior: Callable[[], list[RAGNode[Any]]],
    candidate_key: CandidateKey | None,
    config: Any,
    prompt_text: Callable[[RAGNode[Any]], str],
) -> Clarification | None:
    """The engine's suggestion that this question is unclear, or None. Runs no check without a reason.

    Two independent triggers, tried in this order: the router's unresolved-reference flag, then the app's
    `candidate_key` hook. With neither, an ordinary question pays nothing and gets None.
    """
    if intent.route == "no_retrieval":
        return None
    if intent.unresolved_referent:
        return _from_referent(state, nodes, prior, prompt_text, config.max_options)
    if candidate_key is not None:
        # An app that can tell referents apart opted in: two or more of them, none dominating, is ambiguous
        return _from_hook(intent, nodes, candidate_key, config.min_group_hits, config.dominance, config.max_options)
    return None


ASKABLE_REASONS = ("several_matches", "unresolved_referent")


def may_ask(clarification: Clarification | None) -> bool:
    """`too_broad` is a mark only: it is never asked (there is no evidence an app needs it)."""
    return clarification is not None and clarification.reason in ASKABLE_REASONS


# ── wording ────────────────────────────────────────────────────────────────


def question_text(clarification: Clarification, user_question: str, templates: dict[str, str]) -> str:
    """A question built from a template and the option labels, which come only from visible passages.

    The language follows the script of the user's question: any CJK character selects the Chinese template.

    >>> option = lambda name: ClarificationOption(label=name, key=name, node_ids=[])
    >>> pick = Clarification(reason="unresolved_referent", options=[option("Li Wei"), option("Li Ming")])
    >>> question_text(pick, "what did he do the following week", {})
    'Which one do you mean: Li Wei or Li Ming?'
    >>> question_text(pick, "他后来做了什么", {})
    '你指的是哪一个：Li Wei还是Li Ming？'
    """
    lang = "zh" if any(is_cjk_char(ch) for ch in user_question) else "en"
    template = templates.get(lang) or DEFAULT_TEMPLATES[lang]
    labels = [option.label for option in clarification.options]
    comma, last = _JOINERS[lang]
    joined = labels[0] if len(labels) == 1 else comma.join(labels[:-1]) + last + labels[-1]
    return template.replace("{options}", joined)


def hedge_note(clarification: Clarification) -> str:
    """The prompt note for a best-guess answer; model-facing, so English whatever the question's language.

    >>> option = lambda name: ClarificationOption(label=name, key=name, node_ids=[])
    >>> hedge_note(Clarification(reason="several_matches", options=[option("A"), option("B"), option("C")]))
    'The question may refer to A, B or C; answer for the most likely one and say which.'
    """
    labels = [option.label for option in clarification.options]
    labels_text = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]
    return f"The question may refer to {labels_text}; answer for the most likely one and say which."
