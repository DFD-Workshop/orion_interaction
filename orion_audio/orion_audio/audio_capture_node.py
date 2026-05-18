from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import rclpy
import sounddevice as sd
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from orion_interfaces.msg import AudioData, AudioInfo


class AudioCaptureNode(Node):
    def __init__(self) -> None:
        super().__init__('audio_capture')

        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('device_index', -1)
        self.declare_parameter('frame_duration_ms', 30)

        self._sample_rate: int = self.get_parameter('sample_rate').value
        self._channels: int = self.get_parameter('channels').value
        self._device_index: int = self.get_parameter('device_index').value
        self._frame_duration_ms: int = self.get_parameter('frame_duration_ms').value
        self._frame_size: int = int(self._sample_rate * self._frame_duration_ms / 1000)

        raw_qos = QoSProfile(depth=50)
        transient_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._pub_raw = self.create_publisher(AudioData, '/audio/raw', raw_qos)
        self._pub_info = self.create_publisher(AudioInfo, '/audio/info', transient_qos)

        # Thread-safe bridge between sounddevice C thread and asyncio loop
        self._sd_queue: queue.Queue[bytes] = queue.Queue()

        self._publish_info()
        self.get_logger().info(
            f'AudioCaptureNode ready — {self._sample_rate} Hz, '
            f'{self._channels} ch, frame {self._frame_duration_ms} ms'
        )

    def _publish_info(self) -> None:
        msg = AudioInfo(
            num_channels=self._channels,
            sample_rate=self._sample_rate,
            subtype='int16',
            uuid='',
        )
        self._pub_info.publish(msg)

    def _sd_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self.get_logger().warning(f'PortAudio: {status}')
        self._sd_queue.put_nowait(bytes(indata))

    async def _capture_loop(self) -> None:
        loop = asyncio.get_running_loop()
        device = None if self._device_index < 0 else self._device_index

        stream = sd.RawInputStream(
            samplerate=self._sample_rate,
            blocksize=self._frame_size,
            dtype='int16',
            channels=self._channels,
            device=device,
            callback=self._sd_callback,
        )

        with stream:
            self.get_logger().info('Audio stream started')
            while rclpy.ok():
                raw = await loop.run_in_executor(None, self._sd_queue.get)
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                msg = AudioData(data=pcm.flatten().tolist())
                self._pub_raw.publish(msg)

    def run(self) -> None:
        ros_thread = threading.Thread(
            target=rclpy.spin, args=(self,), daemon=True
        )
        ros_thread.start()
        asyncio.run(self._capture_loop())


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AudioCaptureNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
