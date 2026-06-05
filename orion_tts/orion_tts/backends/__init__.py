from __future__ import annotations

from orion_tts.backends.base import TTSBackend
from orion_tts.backends.edge_tts_backend import EdgeTTSBackend

_REGISTRY: dict[str, type[TTSBackend]] = {
    'edge_tts': EdgeTTSBackend,
}


def get_backend(name: str, config: dict) -> TTSBackend:
    if name not in _REGISTRY:
        available = list(_REGISTRY)
        raise ValueError(f'Unknown TTS backend: {name!r}. Available: {available}')
    return _REGISTRY[name](config)
