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
    {
        'type': 'function',
        'function': {
            'name': 'set_emotion',
            'description': (
                'Muestra una emoción en la cara de ORION acorde al tono de la '
                'respuesta. Vuelve a neutral sola al terminar de hablar.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'emotion': {
                        'type': 'string',
                        'enum': ['angry', 'disgust', 'fear', 'happy',
                                 'neutral', 'sad', 'surprise', 'wink'],
                    },
                },
                'required': ['emotion'],
            },
        },
    },
]

# Maps an LLM tool name to the ActionCommand.action_type understood by orion_actions.
_TOOL_TO_ACTION = {
    'execute_movement': 'move',
    'stop_movement': 'stop',
    'set_emotion': 'emotion',
}

# Valid emotion names (mirrors the set_emotion enum and the ESP32 screen).
EMOTIONS = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'wink')

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
        # Empty by default: brevity is already enforced by the context's
        # output_rules. Appending a per-turn suffix distracts the model away from
        # tool calling (it stops emitting set_emotion / actions).
        self.declare_parameter('reply_suffix', '')
        # After executing tool calls, re-query the LLM (with the tool results in
        # history) so it can produce the spoken reply that accompanies the action.
        # Cap the loop to avoid the model chaining tool calls forever.
        self.declare_parameter('max_tool_iterations', 3)
        # When the model does NOT call set_emotion, infer one from the spoken
        # reply via a tiny LLM classification so the face is always expressive.
        # An explicit set_emotion tool call always takes priority over this.
        self.declare_parameter('emotion_fallback', True)

        context_file: str = self.get_parameter('context_file').value
        fallback: str = self.get_parameter('system_prompt').value
        self._system_prompt: str = _load_system_prompt(context_file, fallback)
        self._max_history: int = self.get_parameter('max_history').value
        self._tts_voice: str = self.get_parameter('tts_voice').value
        self._reply_suffix: str = self.get_parameter('reply_suffix').value
        self._max_tool_iterations: int = self.get_parameter('max_tool_iterations').value
        self._emotion_fallback: bool = self.get_parameter('emotion_fallback').value

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

    async def _call_llm(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> tuple[str, list[str]]:
        request = LLMChat.Request()
        request.messages_json = [json.dumps(m) for m in messages]
        offered = AVAILABLE_TOOLS if tools is None else tools
        request.tools_json = [json.dumps(t) for t in offered]

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
            self._truncate_history()

            await self._handle_turn()

    async def _handle_turn(self) -> None:
        """Run the LLM/tool loop for one user turn.

        Each iteration: query the LLM, execute any tool calls, and record the
        result in history. If the model called tools, loop again (with the tool
        results now in context) so it can produce the spoken reply that goes with
        the action. Stops once the model answers with no further tool calls.
        """
        spoken_text = ''
        emotion_set = False
        for _ in range(max(1, self._max_tool_iterations)):
            response_text, tool_calls = await self._call_llm(self._build_messages())
            if not response_text and not tool_calls:
                break

            self._record_history(response_text, tool_calls)
            self._truncate_history()

            self._pub.publish(AssistantResponse(
                text=response_text,
                tool_calls_json=tool_calls,
                is_streaming=False,
            ))
            routed = self._route_tool_calls(tool_calls)
            emotion_set = emotion_set or 'set_emotion' in routed

            if response_text:
                spoken_text = response_text
            if not tool_calls:
                break  # final answer reached

        # Hybrid expressiveness: if the model never set an emotion, infer one from
        # the spoken reply so the face still reflects the mood. Publish it before
        # the TTS so it shows while ORION talks.
        if self._emotion_fallback and not emotion_set and spoken_text:
            emotion = await self._classify_emotion(spoken_text)
            if emotion and emotion != 'neutral':
                self._publish_emotion(emotion)

        if spoken_text:
            self._tts_pub.publish(
                TTSRequest(text=spoken_text, voice=self._tts_voice)
            )
            self.get_logger().info(f'[ORION] "{spoken_text[:100]}..."')

    async def _classify_emotion(self, text: str) -> str | None:
        """Infer one of the 8 emotions from a reply via a tiny LLM call (no tools)."""
        messages = [
            {'role': 'system', 'content': (
                'Classify the emotion conveyed by the assistant line into exactly '
                'one of: ' + ', '.join(EMOTIONS) + '. Reply with ONLY that single '
                'word, lowercase, nothing else.'
            )},
            {'role': 'user', 'content': text},
        ]
        response, _ = await self._call_llm(messages, tools=[])
        word = response.strip().lower().strip('.!"\'')
        if word in EMOTIONS:
            self.get_logger().info(f'[Emotion fallback] "{text[:40]}..." → {word}')
            return word
        self.get_logger().warning(f'Emotion fallback got unparseable: {response!r}')
        return None

    def _publish_emotion(self, emotion: str) -> None:
        self._action_pub.publish(ActionCommand(
            action_type='emotion',
            payload_json=json.dumps({'emotion': emotion}),
        ))
        self.get_logger().info(f'[Action] emotion (fallback) → {emotion}')

    def _truncate_history(self) -> None:
        if len(self._history) <= self._max_history:
            return
        self._history = self._history[-self._max_history:]
        # Never start the window on an orphaned 'tool' result (a tool message
        # must follow the assistant tool_call that produced it), or Ollama
        # rejects the request.
        while self._history and self._history[0]['role'] == 'tool':
            self._history.pop(0)

    def _build_messages(self) -> list[dict]:
        messages: list[dict] = list(self._history)
        # Nudge brevity only on a real user turn (not on tool-result follow-ups).
        if self._reply_suffix and messages and messages[-1]['role'] == 'user':
            last = dict(messages[-1])
            last['content'] = f'{last["content"]} {self._reply_suffix}'
            messages[-1] = last
        if self._system_prompt:
            messages = [{'role': 'system', 'content': self._system_prompt}] + messages
        return messages

    def _record_history(self, response_text: str, tool_calls: list[str]) -> None:
        """Append the assistant turn using the native tool-call protocol.

        The assistant message carries the structured tool_calls (not free text),
        and each call is followed by a 'tool' result message. This keeps the
        history well-formed without feeding back narration text the model would
        otherwise imitate instead of actually calling the tool.
        """
        parsed: list[dict] = []
        for raw in tool_calls:
            try:
                parsed.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

        assistant_msg: dict = {'role': 'assistant', 'content': response_text}
        if parsed:
            assistant_msg['tool_calls'] = [
                {'function': {'name': c.get('name', ''),
                              'arguments': c.get('arguments', {})}}
                for c in parsed
            ]
        self._history.append(assistant_msg)

        for c in parsed:
            self._history.append({
                'role': 'tool',
                'tool_name': c.get('name', ''),
                'content': 'ok',
            })

    def _route_tool_calls(self, tool_calls: list[str]) -> list[str]:
        """Turn LLM tool calls into ActionCommand messages for orion_actions.

        Returns the list of tool names actually routed (used to detect whether
        the model already set an emotion this turn).
        """
        routed: list[str] = []
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
            routed.append(name)
            self.get_logger().info(
                f'[Action] {name} → {action_type} {cmd.payload_json}'
            )
        return routed

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
