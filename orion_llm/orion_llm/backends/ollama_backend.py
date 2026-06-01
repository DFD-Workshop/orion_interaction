from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from orion_llm.backends.base import LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self, config: dict) -> None:
        self._host: str = config.get('host', 'localhost:11434')
        self._model: str = config.get('model', 'gemma3:12b')
        self._stream: bool = config.get('stream', True)
        self._max_tokens: int = config.get('max_tokens', 1024)
        self._base_url = f'http://{self._host}'

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f'{self._base_url}/api/tags')
                return r.status_code == 200
        except Exception:
            return False

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> AsyncIterator[str]:
        payload: dict = {
            'model': self._model,
            'messages': messages,
            'stream': stream,
            'options': {'num_predict': self._max_tokens},
        }
        if tools:
            payload['tools'] = tools

        async with httpx.AsyncClient(timeout=180.0) as client:
            if stream:
                async with client.stream(
                    'POST', f'{self._base_url}/api/chat', json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        delta = chunk.get('message', {}).get('content', '')
                        if delta:
                            yield delta
            else:
                resp = await client.post(
                    f'{self._base_url}/api/chat', json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                yield data.get('message', {}).get('content', '')
