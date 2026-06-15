from __future__ import annotations

import json

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float64MultiArray, Int32

from orion_interfaces.msg import ActionCommand

# Maps an LLM/dialogue movement direction onto unit (linear, angular) signs.
# Actual magnitudes come from the linear_speed / angular_speed parameters.
_DIRECTIONS = {
    'forward': (1.0, 0.0),
    'backward': (-1.0, 0.0),
    'left': (0.0, 1.0),
    'right': (0.0, -1.0),
}

# Emotion name -> screen index, mirrored exactly from the ESP32 firmware
# (orion_interaction_micro_ros: emotions.hpp, epd_bitmap_allArray order).
_EMOTIONS = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6,
    'wink': 7,
}


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ActionExecutor(Node):
    """Sole owner of the robot hardware: turns ActionCommand into controller msgs.

    Movement is streamed: a periodic timer republishes the active Twist until its
    duration elapses, because the diff_drive_controller stops the base if it stops
    receiving commands within cmd_vel_timeout (1.0 s).
    """

    def __init__(self) -> None:
        super().__init__('action_executor')

        self.declare_parameter('cmd_vel_topic', '/mobile_base_controller/cmd_vel')
        self.declare_parameter('left_arm_topic', '/simple_left_arm_controller/commands')
        self.declare_parameter('right_arm_topic', '/simple_right_arm_controller/commands')

        # Streaming rate for cmd_vel — must stay above the controller's
        # cmd_vel_timeout (1.0 s) so the base does not stall mid-motion.
        self.declare_parameter('publish_rate', 10.0)

        # Velocity caps (mobile_base_controller limits: ±0.28 m/s, ±0.56 rad/s).
        self.declare_parameter('max_linear', 0.28)
        self.declare_parameter('max_angular', 0.56)
        # Default speeds used for a movement command (conservative, within caps).
        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.4)
        self.declare_parameter('default_duration', 1.0)
        self.declare_parameter('max_duration', 10.0)

        # Arm joint limit (servo_conn_*_joint: ±pi/3 rad).
        self.declare_parameter('arm_limit', 1.0472)

        # Emotion screen (ESP32 interaction board subscribes to this Int32 topic).
        self.declare_parameter('emotion_topic', '/emotion/int')

        self._cmd_vel_topic: str = self.get_parameter('cmd_vel_topic').value
        self._publish_rate: float = self.get_parameter('publish_rate').value
        self._max_linear: float = self.get_parameter('max_linear').value
        self._max_angular: float = self.get_parameter('max_angular').value
        self._linear_speed: float = self.get_parameter('linear_speed').value
        self._angular_speed: float = self.get_parameter('angular_speed').value
        self._default_duration: float = self.get_parameter('default_duration').value
        self._max_duration: float = self.get_parameter('max_duration').value
        self._arm_limit: float = self.get_parameter('arm_limit').value

        qos = QoSProfile(depth=10)
        self._cmd_vel_pub = self.create_publisher(TwistStamped, self._cmd_vel_topic, qos)
        self._left_arm_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter('left_arm_topic').value, qos
        )
        self._right_arm_pub = self.create_publisher(
            Float64MultiArray, self.get_parameter('right_arm_topic').value, qos
        )
        self._emotion_pub = self.create_publisher(
            Int32, self.get_parameter('emotion_topic').value, qos
        )

        self.create_subscription(ActionCommand, '/actions/command', self._command_cb, qos)

        # Active movement state, driven by the streaming timer.
        self._linear = 0.0
        self._angular = 0.0
        self._stop_time: float | None = None  # ROS time (sec) when motion ends
        self._timer = self.create_timer(1.0 / self._publish_rate, self._stream_cmd_vel)

        self.get_logger().info(
            f'ActionExecutor ready — cmd_vel={self._cmd_vel_topic} @ '
            f'{self._publish_rate:.0f} Hz'
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _command_cb(self, msg: ActionCommand) -> None:
        try:
            payload = json.loads(msg.payload_json) if msg.payload_json else {}
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Bad payload_json: {exc}')
            return
        if not isinstance(payload, dict):
            self.get_logger().error(f'payload_json must be an object, got {payload!r}')
            return

        handler = {
            'move': self._handle_move,
            'stop': self._handle_stop,
            'arm': self._handle_arm,
            'emotion': self._handle_emotion,
        }.get(msg.action_type)

        if handler is None:
            self.get_logger().warning(f'Unknown action_type: {msg.action_type!r}')
            return
        handler(payload)

    def _handle_move(self, payload: dict) -> None:
        direction = str(payload.get('direction', '')).lower()
        signs = _DIRECTIONS.get(direction)
        if signs is None:
            self.get_logger().warning(
                f'Ignoring move with invalid direction: {direction!r}'
            )
            return

        try:
            duration = float(payload.get('duration', self._default_duration))
        except (TypeError, ValueError):
            duration = self._default_duration
        duration = max(0.0, min(duration, self._max_duration))

        lin_sign, ang_sign = signs
        self._linear = _clamp(lin_sign * self._linear_speed, self._max_linear)
        self._angular = _clamp(ang_sign * self._angular_speed, self._max_angular)
        self._stop_time = self._now() + duration

        self.get_logger().info(
            f'[move] {direction} for {duration:.1f}s '
            f'(lin={self._linear:.2f}, ang={self._angular:.2f})'
        )

    def _handle_stop(self, _payload: dict) -> None:
        self._linear = 0.0
        self._angular = 0.0
        self._stop_time = None
        self._cmd_vel_pub.publish(self._make_twist(0.0, 0.0))
        self.get_logger().info('[stop]')

    def _handle_emotion(self, payload: dict) -> None:
        name = str(payload.get('emotion', '')).lower()
        index = _EMOTIONS.get(name)
        if index is None:
            self.get_logger().warning(f'Ignoring unknown emotion: {name!r}')
            return
        self._emotion_pub.publish(Int32(data=index))
        self.get_logger().info(f'[emotion] {name} ({index})')

    def _handle_arm(self, payload: dict) -> None:
        for key, pub in (('left', self._left_arm_pub), ('right', self._right_arm_pub)):
            if key not in payload:
                continue
            try:
                pos = _clamp(float(payload[key]), self._arm_limit)
            except (TypeError, ValueError):
                self.get_logger().warning(f'Ignoring non-numeric arm {key}: {payload[key]!r}')
                continue
            pub.publish(Float64MultiArray(data=[pos]))
            self.get_logger().info(f'[arm] {key}={pos:.3f}')

    def _make_twist(self, linear: float, angular: float) -> TwistStamped:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.twist.linear.x = linear
        twist.twist.angular.z = angular
        return twist

    def _stream_cmd_vel(self) -> None:
        if self._stop_time is None:
            return  # idle — nothing to stream
        if self._now() >= self._stop_time:
            self._linear = 0.0
            self._angular = 0.0
            self._stop_time = None
            self._cmd_vel_pub.publish(self._make_twist(0.0, 0.0))
            self.get_logger().info('[move] done')
            return
        self._cmd_vel_pub.publish(self._make_twist(self._linear, self._angular))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ActionExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
