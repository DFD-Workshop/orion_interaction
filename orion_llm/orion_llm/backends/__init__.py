from __future__ import annotations

from orion_llm.backends.base import LLMBackend


def get_backend(name: str, config: dict) -> LLMBackend:
    if name == 'ollama':
        from orion_llm.backends.ollama_backend import OllamaBackend
        return OllamaBackend(config)
    raise ValueError(f'Unknown LLM backend: {name!r}. Available: ollama')
