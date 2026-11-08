"""ContextAssembler combining instructions, history, fenced data, and query (§10.1, §10.2)."""
from __future__ import annotations

from typing import Generic
from ..models import MetaT, RAGNode, RAGQueryRequest
from ..ports import NodeRenderer
from ..profile import EngineProfile
from ..text.analyzer import estimate_tokens
from .fence import ContextFencer


class ContextAssembler(Generic[MetaT]):
    """Assembles prompt payload honoring zone token budgets and ID-tagged fences."""

    def __init__(
        self,
        profile: EngineProfile,
        renderer: NodeRenderer[MetaT],
        fencer: ContextFencer | None = None,
    ):
        self.profile = profile
        self.renderer = renderer
        self.fencer = fencer or ContextFencer()

    def assemble(
        self,
        request: RAGQueryRequest,
        nodes: list[RAGNode[MetaT]],
        fence_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Assemble (system_prompt, user_message, active_fence_id).

        Returns:
            (system_prompt, user_message, active_fence_id)
        """
        # 1. Base system instructions
        base_inst = request.custom_instruction or "Answer the user question accurately based on the provided context."
        grounding_mode = self.profile.generation.grounding_mode

        if grounding_mode == "strict":
            grounding_rule = (
                "You must base your answer ONLY on the provided retrieved context. "
                "If the context does not contain enough information to answer the question, "
                "state clearly that the answer cannot be found in the provided documents."
            )
        elif grounding_mode == "labelled_fallback":
            grounding_rule = (
                "Base your answer primarily on the retrieved context. If providing information "
                f"beyond the context, label it clearly with '{self.profile.generation.fallback_label}'."
            )
        else:
            grounding_rule = "Use the retrieved context to inform your answer."

        system_prompt = f"{base_inst}\n\n{grounding_rule}"

        # 2. Render and fence retrieved context
        rendered_nodes: list[str] = []
        for idx, node in enumerate(nodes):
            header = f"[Source {idx + 1}] (Node ID: {node.id}, Doc: {node.document_id})"
            rendered = self.renderer.render(node)
            rendered_nodes.append(f"{header}\n{rendered}\n")

        raw_context = "\n---\n".join(rendered_nodes) if rendered_nodes else "(No relevant context retrieved)"
        fenced_context, active_fence_id = self.fencer.fence_block(raw_context, fence_id=fence_id)

        # 3. Format user message
        parts: list[str] = [
            "Retrieved Context:",
            fenced_context,
            "",
        ]

        if request.conversation and request.conversation.turns:
            history_lines: list[str] = ["Conversation History:"]
            for turn in request.conversation.turns[-4:]:
                history_lines.append(f"{turn.role.capitalize()}: {turn.text}")
            fenced_hist, _ = self.fencer.fence_block("\n".join(history_lines), fence_id=active_fence_id)
            parts.extend([fenced_hist, ""])

        parts.extend([
            f"Question: {request.query}",
            "",
            "Please provide your answer with citations where appropriate (e.g. [1], [2]).",
        ])

        user_message = "\n".join(parts)
        return system_prompt, user_message, active_fence_id
