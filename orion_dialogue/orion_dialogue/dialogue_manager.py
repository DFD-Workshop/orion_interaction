from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile

from orion_interfaces.msg import AssistantResponse, TTSRequest, UserInput
from orion_interfaces.srv import LLMChat

_CONTEXT_SECTIONS = ('persona', 'background', 'environment', 'capabilities', 'guidelines')


def _load_system_prompt(context_file: str, fallback: str) -> str:
    if not context_file:
        return fallback

    path = Path(context_file)
    if not path.exists():
        return fallback

    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return fallback

    parts = [str(data[s]).strip() for s in _CONTEXT_SECTIONS if s in data]
    return '\n\n'.join(parts)


class DialogueManager(Node):
    def __init__(self) -> None:
        super().__init__('dialogue_manager')

        self.declare_parameter('context_file', '')
        self.declare_parameter('system_prompt', 'Eres ORION, un robot asistente amigable.')
        self.declare_parameter('max_history', 20)
        self.declare_parameter('tts_voice', '')

        context_file: str = self.get_parameter('context_file').value
        fallback: str = self.get_parameter('system_prompt').value
        self._system_prompt: str = _load_system_prompt(context_file, fallback)
        self._max_history: int = self.get_parameter('max_history').value
        # Empty voice -> orion_tts falls back to its own default voice
        self._tts_voice: str = self.get_parameter('tts_voice').value

        if context_file:
            self.get_logger().info(f'System prompt loaded from: {context_file}')
        else:
            self.get_logger().info('Using inline system_prompt parameter')

        # Conversation history — alternating user/assistant turns
        self._history: list[dict] = []

        qos = QoSProfile(depth=10)
        self.create_subscription(UserInput, '/dialogue/user_input', self._input_cb, qos)
        self._pub = self.create_publisher(AssistantResponse, '/dialogue/assistant_response', qos)
        self._tts_pub = self.create_publisher(TTSRequest, '/tts/speak', qos)
        self._llm_client = self.create_client(LLMChat, '/llm/request')

        self._queue: asyncio.Queue[UserInput] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self.get_logger().info(
            f'DialogueManager ready — max_history={self._max_history}'
        )

    def _input_cb(self, msg: UserInput) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._queue.put(msg), self._loop)

    async def _wait_for_llm(self) -> None:
        loop = asyncio.get_running_loop()
        while rclpy.ok():
            available = await loop.run_in_executor(
                None, lambda: self._llm_client.wait_for_service(timeout_sec=1.0)
            )
            if available:
                self.get_logger().info('Connected to /llm/request')
                return
            self.get_logger().info('Waiting for /llm/request service...')

    async def _call_llm(self, messages: list[dict]) -> str:
        request = LLMChat.Request()
        request.messages_json = [json.dumps(m) for m in messages]

        event = asyncio.Event()
        result: list[LLMChat.Response | None] = [None]

        def _done_cb(future: rclpy.Future) -> None:
            try:
                result[0] = future.result()
            except Exception as exc:
                self.get_logger().error(f'LLM service call failed: {exc}')
            assert self._loop is not None
            self._loop.call_soon_threadsafe(event.set)

        ros_future = self._llm_client.call_async(request)
        ros_future.add_done_callback(_done_cb)
        await event.wait()

        resp = result[0]
        if resp is None:
            return ''
        if not resp.success:
            self.get_logger().error(f'LLM error: {resp.error_msg}')
            return ''
        return resp.response

    async def _process_loop(self) -> None:
        await self._wait_for_llm()
        self.get_logger().info('Dialogue processing loop started')

        while rclpy.ok():
            user_input: UserInput = await self._queue.get()
            user_text = user_input.text.strip()
            if not user_text:
                continue

            self.get_logger().info(f'[User]  "{user_text}"')
            self._history.append({'role': 'user', 'content': user_text})

            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            messages: list[dict] = self._history
            if self._system_prompt:
                messages = [{'role': 'system', 'content': self._system_prompt}] + messages

            response_text = await self._call_llm(messages)
            if not response_text:
                continue

            self._history.append({'role': 'assistant', 'content': response_text})

            pub_msg = AssistantResponse(
                text=response_text,
                tool_calls_json=[],
                is_streaming=False,
            )
            self._pub.publish(pub_msg)

            # Close the voice loop: hand the response to orion_tts to speak.
            self._tts_pub.publish(TTSRequest(text=response_text, voice=self._tts_voice))
            self.get_logger().info(f'[ORION] "{response_text[:100]}..."')

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        ros_thread.start()
        self._loop.run_until_complete(self._process_loop())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DialogueManager()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
