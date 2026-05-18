from __future__ import annotations

import uuid

import numpy as np
import rclpy
import webrtcvad
from rclpy.node import Node
from rclpy.qos import QoSProfile

from orion_interfaces.msg import AudioData, AudioInfo

_VALID_FRAME_MS = (10, 20, 30)


class VadNode(Node):
    def __init__(self) -> None:
        super().__init__('vad')

        self.declare_parameter('sample_rate', 48000)
        self.declare_parameter('frame_duration_ms', 30)
        self.declare_parameter('vad_aggressiveness', 2)
        self.declare_parameter('silence_ms', 800)
        self.declare_parameter('min_speech_ms', 100)
        self.declare_parameter('energy_threshold', 0.03)

        self._sample_rate: int = self.get_parameter('sample_rate').value
        self._frame_ms: int = self.get_parameter('frame_duration_ms').value
        self._energy_threshold: float = self.get_parameter('energy_threshold').value

        if self._frame_ms not in _VALID_FRAME_MS:
            raise ValueError(
                f'frame_duration_ms must be one of {_VALID_FRAME_MS}, got {self._frame_ms}'
            )

        aggressiveness: int = self.get_parameter('vad_aggressiveness').value
        silence_ms: int = self.get_parameter('silence_ms').value
        min_speech_ms: int = self.get_parameter('min_speech_ms').value

        self._vad = webrtcvad.Vad(aggressiveness)
        self._max_silence_frames: int = max(1, int(silence_ms / self._frame_ms))
        self._min_speech_frames: int = max(1, int(min_speech_ms / self._frame_ms))

        # State machine
        self._in_speech: bool = False
        self._speech_count: int = 0
        self._silence_count: int = 0
        self._ring_buffer: list[np.ndarray] = []
        self._speech_frames: list[np.ndarray] = []
        self._uid: str = ''

        qos = QoSProfile(depth=50)
        utterance_qos = QoSProfile(depth=10)

        self.create_subscription(AudioData, '/audio/raw', self._raw_cb, qos)
        self._pub_info = self.create_publisher(AudioInfo, '/audio/utterance/info', utterance_qos)
        self._pub_data = self.create_publisher(AudioData, '/audio/utterance', utterance_qos)

        self.get_logger().info(
            f'VadNode ready — aggressiveness={aggressiveness}, '
            f'silence={silence_ms} ms, min_speech={min_speech_ms} ms'
        )

    def _raw_cb(self, msg: AudioData) -> None:
        pcm_f32 = np.array(msg.data, dtype=np.float32)
        pcm_i16 = (pcm_f32 * 32768.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_i16.tobytes()

        energy = float(np.max(np.abs(pcm_f32)))

        try:
            vad_flag = self._vad.is_speech(pcm_bytes, self._sample_rate)
        except Exception:
            vad_flag = False

        is_speech = vad_flag and energy > self._energy_threshold

        if not self._in_speech:
            self._ring_buffer.append(pcm_f32)
            if len(self._ring_buffer) > self._min_speech_frames:
                self._ring_buffer.pop(0)

            if is_speech:
                self._speech_count += 1
                if self._speech_count >= self._min_speech_frames:
                    self._start_utterance()
            else:
                self._speech_count = 0
        else:
            self._speech_frames.append(pcm_f32)
            if not is_speech:
                self._silence_count += 1
                if self._silence_count >= self._max_silence_frames:
                    self._end_utterance()
            else:
                self._silence_count = 0

    def _start_utterance(self) -> None:
        self._uid = str(uuid.uuid4())
        self._speech_frames = list(self._ring_buffer)
        self._in_speech = True
        self._silence_count = 0
        self.get_logger().info(f'Utterance START  uuid={self._uid}')

    def _end_utterance(self) -> None:
        # Trim trailing silence frames before publishing
        trimmed = self._speech_frames[: -self._max_silence_frames]
        if not trimmed:
            self._reset()
            return

        data = np.concatenate(trimmed)
        duration_ms = len(trimmed) * self._frame_ms

        info_msg = AudioInfo(
            num_channels=1,
            sample_rate=self._sample_rate,
            subtype='float32',
            uuid=self._uid,
        )
        data_msg = AudioData(data=data.flatten().tolist())

        self._pub_info.publish(info_msg)
        self._pub_data.publish(data_msg)

        self.get_logger().info(
            f'Utterance END    uuid={self._uid}, '
            f'frames={len(trimmed)}, duration={duration_ms} ms'
        )
        self._reset()

    def _reset(self) -> None:
        self._in_speech = False
        self._speech_count = 0
        self._silence_count = 0
        self._ring_buffer.clear()
        self._speech_frames.clear()
        self._uid = ''


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VadNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
