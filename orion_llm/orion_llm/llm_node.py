from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError

import rclpy
from rclpy.node import Node

from orion_interfaces.srv import LLMChat
from orion_llm.backends import get_backend
from orion_llm.backends.base import LLMBackend


class LLMNode(Node):
    def __init__(self) -> None:
        super().__init__('llm')

        self.declare_parameter('backend', 'ollama')
        self.declare_parameter('model', 'qwen2.5:7b')
        self.declare_parameter('host', 'localhost:11434')
        self.declare_parameter('stream', True)
        self.declare_parameter('max_tokens', 200)
        self.declare_parameter('num_ctx', 8192)
        self.declare_parameter('temperature', 0.2)
        self.declare_parameter('system_prompt', '')
        self.declare_parameter('request_timeout', 200.0)

        self._system_prompt: str = self.get_parameter('system_prompt').value
        self._request_timeout: float = self.get_parameter('request_timeout').value
        self._stream: bool = self.get_parameter('stream').value

        backend_name: str = self.get_parameter('backend').value
        config = {
            'model': self.get_parameter('model').value,
            'host': self.get_parameter('host').value,
            'stream': self._stream,
            'max_tokens': self.get_parameter('max_tokens').value,
            'num_ctx': self.get_parameter('num_ctx').value,
            'temperature': self.get_parameter('temperature').value,
        }
        self._backend: LLMBackend = get_backend(backend_name, config)

        if not self._backend.is_available():
            self.get_logger().error(f'LLM backend {backend_name!r} is not available')
            raise RuntimeError(f'LLM backend {backend_name!r} unavailable')

        self._srv = self.create_service(LLMChat, '/llm/request', self._handle_request)
        self._loop: asyncio.AbstractEventLoop | None = None

        self.get_logger().info(
            f'LLMNode ready — backend={backend_name}, '
            f'model={config["model"]}, host={config["host"]}'
        )

    def _handle_request(
        self, request: LLMChat.Request, response: LLMChat.Response
    ) -> LLMChat.Response:
        if self._loop is None:
            response.success = False
            response.error_msg = 'Event loop not initialized'
            return response

        future = asyncio.run_coroutine_threadsafe(
            self._run_chat(request), self._loop
        )
        try:
            return future.result(timeout=self._request_timeout)
        except FutureTimeoutError:
            future.cancel()
            response.success = False
            response.error_msg = (
                f'LLM request timed out after {self._request_timeout:.0f}s '
                '(model too slow or still loading)'
            )
            return response
        except Exception as exc:
            response.success = False
            response.error_msg = str(exc) or repr(exc)
            return response

    async def _run_chat(self, request: LLMChat.Request) -> LLMChat.Response:
        response = LLMChat.Response()
        try:
            messages: list[dict] = [json.loads(m) for m in request.messages_json]
        except json.JSONDecodeError as exc:
            response.success = False
            response.error_msg = f'Invalid messages_json: {exc}'
            return response

        if self._system_prompt:
            messages = [{'role': 'system', 'content': self._system_prompt}] + messages

        try:
            tools: list[dict] = [json.loads(t) for t in request.tools_json]
        except json.JSONDecodeError as exc:
            response.success = False
            response.error_msg = f'Invalid tools_json: {exc}'
            return response

        try:
            text_parts: list[str] = []
            tool_calls: list[str] = []
            async for chunk in self._backend.chat(
                messages, tools=tools or None, stream=self._stream
            ):
                if chunk['type'] == 'content':
                    text_parts.append(chunk['text'])
                elif chunk['type'] == 'tool_call':
                    tool_calls.append(json.dumps(
                        {'name': chunk['name'], 'arguments': chunk['arguments']}
                    ))
            response.response = ''.join(text_parts)
            response.tool_calls_json = tool_calls
            response.success = True
            self.get_logger().info(
                f'[LLM] → "{response.response[:80]}..." '
                f'(+{len(tool_calls)} tool call(s))'
            )
        except Exception as exc:
            response.success = False
            response.error_msg = str(exc)

        return response

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        ros_thread.start()
        self._loop.run_forever()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LLMNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
