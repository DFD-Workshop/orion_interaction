from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from orion_llm.backends.base import Chunk, LLMBackend


class OllamaBackend(LLMBackend):
    def __init__(self, config: dict) -> None:
        self._host: str = config.get('host', 'localhost:11434')
        self._model: str = config.get('model', 'gemma3:12b')
        self._stream: bool = config.get('stream', True)
        self._max_tokens: int = config.get('max_tokens', 1024)
        self._num_ctx: int = config.get('num_ctx', 8192)
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
    ) -> AsyncIterator[Chunk]:
        # Ollama returns tool calls only on non-streamed responses, so when tools
        # are offered we force a single-shot call and surface them explicitly.
        if tools:
            stream = False

        payload: dict = {
            'model': self._model,
            'messages': messages,
            'stream': stream,
            'options': {'num_predict': self._max_tokens, 'num_ctx': self._num_ctx},
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
                            yield {'type': 'content', 'text': delta}
            else:
                resp = await client.post(
                    f'{self._base_url}/api/chat', json=payload
                )
                resp.raise_for_status()
                message = resp.json().get('message', {})

                content = message.get('content', '')
                if content:
                    yield {'type': 'content', 'text': content}

                for call in message.get('tool_calls', []):
                    fn = call.get('function', {})
                    args = fn.get('arguments', {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    yield {
                        'type': 'tool_call',
                        'name': fn.get('name', ''),
                        'arguments': args if isinstance(args, dict) else {},
                    }
