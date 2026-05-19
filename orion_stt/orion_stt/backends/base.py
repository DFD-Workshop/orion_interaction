from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTBackend(ABC):
    @abstractmethod
    async def transcribe(
        self, audio: np.ndarray, sample_rate: int, language: str
    ) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...
