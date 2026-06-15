from __future__ import annotations

import json
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool

from orion_interfaces.msg import ActionCommand

# Resting pose both arms return to when ORION stops speaking.
_REST = 0.0


class GestureSync(Node):
    """Animates the arms while ORION speaks, synced to the TTS playback signal.

    Listens to /tts/speaking and, while it is True, periodically emits arm-pose
    ActionCommands. It never touches the hardware directly — the action_executor
    remains the sole owner of the controllers (it consumes /actions/command).
    """

    def __init__(self) -> None:
        super().__init__('gesture_sync')

        # random: pick fresh poses while speaking. disabled: do nothing.
        self.declare_parameter('backend', 'random')
        # How often a new gesture pose is emitted while speaking (seconds).
        self.declare_parameter('gesture_period', 0.6)
        # Max |angle| for a gesture, kept within the servo limit (±pi/3 rad).
        self.declare_parameter('gesture_amplitude', 0.6)

        self._backend: str = self.get_parameter('backend').value
        self._amplitude: float = abs(self.get_parameter('gesture_amplitude').value)
        period: float = self.get_parameter('gesture_period').value

        qos = QoSProfile(depth=10)
        self._action_pub = self.create_publisher(ActionCommand, '/actions/command', qos)
        self.create_subscription(Bool, '/tts/speaking', self._speaking_cb, qos)

        self._speaking = False
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'GestureSync ready — backend={self._backend}, '
            f'period={period:.2f}s, amplitude={self._amplitude:.2f}'
        )

    def _speaking_cb(self, msg: Bool) -> None:
        if msg.data == self._speaking:
            return
        self._speaking = msg.data
        if not self._speaking:
            # Speech ended: relax arms and let the face rest back to neutral
            # (the expressive emotion set by the LLM lasts only while speaking).
            self._send_arms(_REST, _REST)
            self._send_emotion('neutral')

    def _send_emotion(self, emotion: str) -> None:
        self._action_pub.publish(ActionCommand(
            action_type='emotion',
            payload_json=json.dumps({'emotion': emotion}),
        ))

    def _tick(self) -> None:
        if not self._speaking or self._backend != 'random':
            return
        amp = self._amplitude
        self._send_arms(random.uniform(-amp, amp), random.uniform(-amp, amp))

    def _send_arms(self, left: float, right: float) -> None:
        cmd = ActionCommand(
            action_type='arm',
            payload_json=json.dumps({'left': left, 'right': right}),
        )
        self._action_pub.publish(cmd)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GestureSync()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
