from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, TypedDict


class ContentChunk(TypedDict):
    """A piece of assistant free-text output."""
    type: str  # 'content'
    text: str


class ToolCallChunk(TypedDict):
    """A structured tool/function call requested by the model."""
    type: str  # 'tool_call'
    name: str
    arguments: dict


# A backend yields either content deltas (for streaming text) or tool calls.
# Routing in orion_dialogue consumes tool calls; never string-match the content.
Chunk = ContentChunk | ToolCallChunk


class LLMBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[Chunk]: ...

    @abstractmethod
    def is_available(self) -> bool: ...
