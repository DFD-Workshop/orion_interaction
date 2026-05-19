from __future__ import annotations

from orion_stt.backends.base import STTBackend
from orion_stt.backends.whisper_backend import WhisperBackend

_REGISTRY: dict[str, type[STTBackend]] = {
    'whisper': WhisperBackend,
}


def get_backend(name: str, config: dict) -> STTBackend:
    if name not in _REGISTRY:
        available = list(_REGISTRY)
        raise ValueError(f'Unknown STT backend: {name!r}. Available: {available}')
    return _REGISTRY[name](config)
