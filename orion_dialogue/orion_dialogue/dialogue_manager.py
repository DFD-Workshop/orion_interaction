from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile

from orion_interfaces.msg import ActionCommand, AssistantResponse, TTSRequest, UserInput
from orion_interfaces.srv import LLMChat

# Tools advertised to the LLM. The model triggers actions by returning a tool call
# (structured output), never by emitting a magic string for us to match on.
AVAILABLE_TOOLS: list[dict] = [
    {
        'type': 'function',
        'function': {
            'name': 'execute_movement',
            'description': 'Mueve la base del robot en una dirección por una duración.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'direction': {
                        'type': 'string',
                        'enum': ['forward', 'backward', 'left', 'right'],
                    },
                    'duration': {
                        'type': 'number',
                        'description': 'Segundos de movimiento.',
                    },
                },
                'required': ['direction'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'stop_movement',
            'description': 'Detiene inmediatamente la base del robot.',
            'parameters': {'type': 'object', 'properties': {}},
        },
    },
]

# Maps an LLM tool name to the ActionCommand.action_type understood by orion_actions.
_TOOL_TO_ACTION = {
    'execute_movement': 'move',
    'stop_movement': 'stop',
}

# Canonical ordering for known sections. Any other section present in the YAML is
# appended afterwards (in file order), so new sections are never silently dropped.
_CONTEXT_SECTIONS = (
    'output_rules', 'persona', 'background', 'environment', 'capabilities', 'guidelines'
)


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

    ordered = [s for s in _CONTEXT_SECTIONS if s in data]
    extras = [k for k in data if k not in _CONTEXT_SECTIONS]
    parts = [str(data[k]).strip() for k in (*ordered, *extras) if str(data[k]).strip()]
    return '\n\n'.join(parts) if parts else fallback


class DialogueManager(Node):
    def __init__(self) -> None:
        super().__init__('dialogue_manager')

        self.declare_parameter('context_file', '')
        self.declare_parameter('system_prompt', 'Eres ORION, un robot asistente amigable.')
        self.declare_parameter('max_history', 20)
        self.declare_parameter('tts_voice', '')
        self.declare_parameter('reply_suffix', '(Máximo 2 oraciones.)')

        context_file: str = self.get_parameter('context_file').value
        fallback: str = self.get_parameter('system_prompt').value
        self._system_prompt: str = _load_system_prompt(context_file, fallback)
        self._max_history: int = self.get_parameter('max_history').value
        self._tts_voice: str = self.get_parameter('tts_voice').value
        self._reply_suffix: str = self.get_parameter('reply_suffix').value

        if context_file:
            self.get_logger().info(f'System prompt loaded from: {context_file}')
        else:
            self.get_logger().info('Using inline system_prompt parameter')

        self._history: list[dict] = []

        qos = QoSProfile(depth=10)
        self.create_subscription(UserInput, '/dialogue/user_input', self._input_cb, qos)
        self._pub = self.create_publisher(AssistantResponse, '/dialogue/assistant_response', qos)
        self._tts_pub = self.create_publisher(TTSRequest, '/tts/speak', qos)
        self._action_pub = self.create_publisher(ActionCommand, '/actions/command', qos)
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

    async def _call_llm(self, messages: list[dict]) -> tuple[str, list[str]]:
        request = LLMChat.Request()
        request.messages_json = [json.dumps(m) for m in messages]
        request.tools_json = [json.dumps(t) for t in AVAILABLE_TOOLS]

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
            return '', []
        if not resp.success:
            self.get_logger().error(f'LLM error: {resp.error_msg}')
            return '', []
        return resp.response, list(resp.tool_calls_json)

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

            messages: list[dict] = list(self._history)
            if self._reply_suffix and messages:
                last = dict(messages[-1])
                last['content'] = f'{last["content"]} {self._reply_suffix}'
                messages[-1] = last
            if self._system_prompt:
                messages = [{'role': 'system', 'content': self._system_prompt}] + messages

            response_text, tool_calls = await self._call_llm(messages)
            if not response_text and not tool_calls:
                continue

            if response_text:
                self._history.append({'role': 'assistant', 'content': response_text})

            self._pub.publish(AssistantResponse(
                text=response_text,
                tool_calls_json=tool_calls,
                is_streaming=False,
            ))

            self._route_tool_calls(tool_calls)

            if response_text:
                self._tts_pub.publish(
                    TTSRequest(text=response_text, voice=self._tts_voice)
                )
                self.get_logger().info(f'[ORION] "{response_text[:100]}..."')

    def _route_tool_calls(self, tool_calls: list[str]) -> None:
        """Turn LLM tool calls into ActionCommand messages for orion_actions."""
        for raw in tool_calls:
            try:
                call = json.loads(raw)
            except json.JSONDecodeError as exc:
                self.get_logger().error(f'Bad tool call JSON: {exc}')
                continue

            name = call.get('name', '')
            action_type = _TOOL_TO_ACTION.get(name)
            if action_type is None:
                self.get_logger().warning(f'Unknown tool call: {name!r}')
                continue

            cmd = ActionCommand(
                action_type=action_type,
                payload_json=json.dumps(call.get('arguments', {})),
            )
            self._action_pub.publish(cmd)
            self.get_logger().info(
                f'[Action] {name} → {action_type} {cmd.payload_json}'
            )

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
