from __future__ import annotations

import asyncio
import re
import threading

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from rclpy.qos import QoSProfile

from std_msgs.msg import Bool

from orion_interfaces.msg import TTSRequest
from orion_tts.backends import get_backend
from orion_tts.backends.base import TTSBackend

# Emoji and pictographic ranges — stripped so the TTS does not read them aloud
# (e.g. edge-TTS would otherwise speak "emoji de cara sonriente").
_EMOJI_RE = re.compile(
    '['
    '\U0001F300-\U0001FAFF'  # symbols, pictographs, emoji extensions
    '\U00002600-\U000027BF'  # misc symbols + dingbats
    '\U0001F000-\U0001F0FF'  # mahjong, dominoes, playing cards
    '\U0000FE00-\U0000FE0F'  # variation selectors
    '\U00002190-\U000021FF'  # arrows
    '\U00002300-\U000023FF'  # misc technical
    '\U0000200D'             # zero-width joiner
    ']+',
    flags=re.UNICODE,
)

# Markdown markup omission
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')  # [text](url) -> text
_MD_CHARS_RE = re.compile(r'[*_`#>~]')              # emphasis/code/heading/quote/strike


def _sanitize(text: str) -> str:
    """Strip emojis and markdown markup, then collapse leftover whitespace."""
    text = _EMOJI_RE.sub('', text)
    text = _MD_LINK_RE.sub(r'\1', text)
    text = _MD_CHARS_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


class TTSNode(Node):
    def __init__(self) -> None:
        super().__init__('tts')

        self.declare_parameter('backend', 'edge_tts')
        self.declare_parameter('voice', 'es-VE-SebastianNeural')
        # English alternative: self.declare_parameter('voice', 'en-US-GuyNeural')
        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('device_index', -1)

        self._default_voice: str = self.get_parameter('voice').value
        self._sample_rate: int = self.get_parameter('sample_rate').value
        device_index: int = self.get_parameter('device_index').value
        self._device = None if device_index < 0 else device_index

        config = {
            'voice': self._default_voice,
            'sample_rate': self._sample_rate,
        }
        backend_name: str = self.get_parameter('backend').value
        self._backend: TTSBackend = get_backend(backend_name, config)

        if not self._backend.is_available():
            self.get_logger().error(f'TTS backend {backend_name!r} is not available')
            raise RuntimeError(f'TTS backend {backend_name!r} unavailable')

        qos = QoSProfile(depth=10)
        self.create_subscription(TTSRequest, '/tts/speak', self._speak_cb, qos)
        # True while audio is playing, False when idle — drives gesture sync.
        self._speaking_pub = self.create_publisher(Bool, '/tts/speaking', qos)

        # asyncio.Queue populated from ROS callbacks, drained by _process_loop
        self._queue: asyncio.Queue[TTSRequest] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self.get_logger().info(
            f'TTSNode ready — backend={backend_name}, voice={self._default_voice}'
        )

    def _speak_cb(self, msg: TTSRequest) -> None:
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._queue.put(msg), self._loop)

    def _play(self, audio: np.ndarray, sample_rate: int) -> None:
        sd.play(audio, samplerate=sample_rate, device=self._device)
        sd.wait()

    async def _process_loop(self) -> None:
        self.get_logger().info('TTS processing loop started')
        loop = asyncio.get_running_loop()
        while rclpy.ok():
            req = await self._queue.get()
            text = _sanitize(req.text)
            if not text:
                continue
            voice = req.voice or self._default_voice
            try:
                audio, sample_rate = await self._backend.synthesize(text, voice)
            except Exception as exc:
                self.get_logger().error(f'Synthesis error: {exc}')
                continue

            if audio.size == 0:
                continue

            self.get_logger().info(f'[TTS] "{text}"')
            # Playback is blocking — run it off the event loop, serialized by the queue.
            self._speaking_pub.publish(Bool(data=True))
            try:
                await loop.run_in_executor(None, self._play, audio, sample_rate)
            except Exception as exc:
                self.get_logger().error(f'Playback error: {exc}')
            finally:
                self._speaking_pub.publish(Bool(data=False))

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        ros_thread.start()
        self._loop.run_until_complete(self._process_loop())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TTSNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
