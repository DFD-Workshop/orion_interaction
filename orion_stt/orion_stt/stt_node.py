from __future__ import annotations

import asyncio
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from orion_interfaces.msg import AudioData, AudioInfo, UserInput
from orion_stt.backends import get_backend
from orion_stt.backends.base import STTBackend


class STTNode(Node):
    def __init__(self) -> None:
        super().__init__('stt')

        self.declare_parameter('backend', 'whisper')
        self.declare_parameter('model', 'small')
        self.declare_parameter('language', 'es')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('no_speech_threshold', 0.6)
        self.declare_parameter('log_prob_threshold', -1.0)
        self.declare_parameter('beam_size', 5)

        self._language: str = self.get_parameter('language').value

        config = {
            'model': self.get_parameter('model').value,
            'device': self.get_parameter('device').value,
            'no_speech_threshold': self.get_parameter('no_speech_threshold').value,
            'log_prob_threshold': self.get_parameter('log_prob_threshold').value,
            'beam_size': self.get_parameter('beam_size').value,
        }
        backend_name: str = self.get_parameter('backend').value
        self._backend: STTBackend = get_backend(backend_name, config)

        if not self._backend.is_available():
            self.get_logger().error(f'STT backend {backend_name!r} is not available')
            raise RuntimeError(f'STT backend {backend_name!r} unavailable')

        qos = QoSProfile(depth=10)
        self.create_subscription(AudioInfo, '/audio/utterance/info', self._info_cb, qos)
        self.create_subscription(AudioData, '/audio/utterance', self._data_cb, qos)
        self._pub = self.create_publisher(UserInput, '/dialogue/user_input', qos)

        self._last_info: AudioInfo | None = None
        # asyncio.Queue populated from ROS callbacks, drained by _process_loop
        self._queue: asyncio.Queue[tuple[AudioInfo, AudioData]] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None

        self.get_logger().info(
            f'STTNode ready — backend={backend_name}, '
            f'model={config["model"]}, language={self._language}'
        )

    def _info_cb(self, msg: AudioInfo) -> None:
        self._last_info = msg

    def _data_cb(self, msg: AudioData) -> None:
        if self._last_info is None:
            self.get_logger().warning('AudioData received but no AudioInfo cached yet — skipping')
            return
        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._queue.put((self._last_info, msg)), self._loop
            )

    async def _process_loop(self) -> None:
        self.get_logger().info('STT processing loop started')
        while rclpy.ok():
            info, data = await self._queue.get()
            audio = np.array(data.data, dtype=np.float32)
            try:
                text = await self._backend.transcribe(audio, info.sample_rate, self._language)
            except Exception as exc:
                self.get_logger().error(f'Transcription error: {exc}')
                continue

            text = text.strip()
            if not text:
                continue

            msg = UserInput(
                text=text,
                confidence=1.0,
                stamp=self.get_clock().now().to_msg(),
            )
            self._pub.publish(msg)
            self.get_logger().info(f'[STT] "{text}"')

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        ros_thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        ros_thread.start()
        self._loop.run_until_complete(self._process_loop())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = STTNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
