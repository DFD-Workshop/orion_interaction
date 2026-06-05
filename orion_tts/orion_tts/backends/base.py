from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class TTSBackend(ABC):
    @abstractmethod
    async def synthesize(self, text: str, voice: str) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` and return (float32 PCM mono, sample_rate)."""
        ...

    @abstractmethod
    def is_available(self) -> bool: ...
