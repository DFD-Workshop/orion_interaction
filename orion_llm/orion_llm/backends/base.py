from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMBackend(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[str]: ...

    @abstractmethod
    def is_available(self) -> bool: ...
