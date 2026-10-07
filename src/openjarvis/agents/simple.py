"""SimpleAgent — single-turn query-to-response agent (no tool calling)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.core.registry import AgentRegistry


@AgentRegistry.register("simple")
class SimpleAgent(BaseAgent):
    """Single-turn agent: query -> model -> response.  No tool calling."""

    agent_id = "simple"
    supports_managed_tool_fallback = True

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Single-turn: build messages, call engine, return result."""
        self._emit_turn_start(input)

        messages = self._build_messages(input, context)
        result = self._generate(messages)
        content = result.get("content", "")

        self._emit_turn_end(content_length=len(content))

        return AgentResult(content=content, turns=1)

    async def stream(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a simple response token-by-token."""
        self._emit_turn_start(input)
        messages = self._build_messages(input, context)
        content_length = 0
        try:
            async for token in self._stream_generate(messages, **kwargs):
                content_length += len(token)
                yield token
        finally:
            self._emit_turn_end(content_length=content_length)


__all__ = ["SimpleAgent"]
