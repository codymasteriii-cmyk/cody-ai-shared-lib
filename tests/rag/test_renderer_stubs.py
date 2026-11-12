"""A renderer that only subclasses the NodeRenderer protocol is refused at startup (§3.4, AC-RAG-111).

AudioReading's renderer subclassed the protocol and named its methods `embedding_text` and `format_label`.
The real `embed_text` was the inherited `...` body, returned None, and ingestion failed much later as
"all inputs are empty". The engine now names the stub when it is built.
"""
from __future__ import annotations

import pytest

from cody_ai_shared_lib.rag import (
    DefaultRenderer,
    EngineProfile,
    InMemoryRAGStore,
    ModelBindings,
    NodeRenderer,
    ProfileConfigError,
    RAGEngine,
    renderer_problems,
)
from cody_ai_shared_lib.rag.render import RENDERER_METHODS

from .test_blocker_regressions import real_embedding_shape  # noqa: F401  (autouse fixture)


def engine_with(renderer) -> RAGEngine:
    profile = EngineProfile(name="t", models=ModelBindings(generation=["m1"], embedding="gemini-embedding-2"))
    return RAGEngine(profile, InMemoryRAGStore(), renderer=renderer)


class ProtocolSubclass(NodeRenderer):
    """The AudioReading mistake: wrong names, so the engine's methods are the protocol stubs."""

    def embedding_text(self, node):
        return node.text

    def format_label(self, node):
        return node.span.doc_id


class ProtocolFirst(NodeRenderer, DefaultRenderer):
    """Listing the protocol first in the bases puts its stubs ahead of the real defaults."""


class Extended(DefaultRenderer):
    def embed_text(self, node):
        return f"title: {node.text}"


class DuckTyped:
    """Implements every method without inheriting from either class."""


for _name in RENDERER_METHODS:
    setattr(DuckTyped, _name, lambda self, arg, _n=_name: None)


def test_a_protocol_subclass_fails_at_construction_naming_the_stubs():
    with pytest.raises(ProfileConfigError) as err:
        engine_with(ProtocolSubclass())
    message = str(err.value)
    assert "embed_text" in message and "citation_label" in message
    assert "DefaultRenderer" in message
    assert "embedding_text" not in message        # its own extra methods are not stubs


def test_protocol_first_in_the_bases_is_also_caught():
    with pytest.raises(ProfileConfigError, match="embed_text"):
        engine_with(ProtocolFirst())


@pytest.mark.parametrize("renderer", [DefaultRenderer(), Extended(), DuckTyped()], ids=["default", "extended", "duck"])
def test_renderers_that_implement_the_methods_are_accepted(renderer):
    assert renderer_problems(renderer) == []
    assert engine_with(renderer).renderer is renderer


def test_renderer_problems_lists_every_inherited_stub():
    [problem] = renderer_problems(ProtocolSubclass())
    for name in RENDERER_METHODS:
        assert name in problem


def test_the_default_profile_renderer_passes():
    assert engine_with(None).validate().valid
