"""Driver-node logic tests with rclpy stubbed out.

The sandbox/CI box has no ROS 2, but the safety logic (arming, deadman,
estop, overtemp) is too important to ship untested — so rclpy and the message
modules are replaced with minimal fakes and the callbacks/timers are invoked
directly. On a real ROS 2 machine the same file runs unchanged.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------
# fake rclpy + message stubs (installed before importing driver_node)
# --------------------------------------------------------------------------

class _FakeClockTime:
    def __init__(self, clock):
        self.nanoseconds = int(clock.t * 1e9)

    def to_msg(self):
        return self.nanoseconds


class FakeClock:
    def __init__(self):
        self.t = 1000.0  # seconds

    def now(self):
        return _FakeClockTime(self)


class FakeLogger:
    def __init__(self):
        self.lines = []

    def _log(self, level, msg):
        self.lines.append((level, msg))

    def info(self, msg):
        self._log("info", msg)

    def warning(self, msg):
        self._log("warning", msg)

    def error(self, msg):
        self._log("error", msg)


class _Param:
    def __init__(self, value):
        self.value = value


class FakeNode:
    def __init__(self, name):
        self._params = {}
        self._clock = FakeClock()
        self._logger = FakeLogger()
        self.publishers = {}      # topic -> list of published msgs
        self.subscriptions = {}   # topic -> callback
        self.timers = []

    def declare_parameter(self, name, default):
        self._params[name] = _Param(default)

    def get_parameter(self, name):
        return self._params[name]

    def create_publisher(self, _type, topic, _qos):
        store = self.publishers.setdefault(topic, [])

        class _Pub:
            def publish(_self, msg):
                store.append(msg)
        return _Pub()

    def create_subscription(self, _type, topic, cb, _qos):
        self.subscriptions[topic] = cb

    def create_timer(self, period, cb):
        self.timers.append((period, cb))

    def get_clock(self):
        return self._clock

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        pass


def _module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


class _Vec3:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Quat(_Vec3):
    def __init__(self):
        super().__init__()
        self.w = 1.0


class _Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class JointState:
    def __init__(self):
        self.header = _Header()
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []


class Imu:
    def __init__(self):
        self.header = _Header()
        self.orientation = _Quat()
        self.angular_velocity = _Vec3()
        self.linear_acceleration = _Vec3()


class Twist:
    def __init__(self):
        self.linear = _Vec3()
        self.angular = _Vec3()


class _Data:
    def __init__(self, data=None):
        self.data = data


class Bool(_Data):
    def __init__(self, data=False):
        super().__init__(data)


class Float64(_Data):
    def __init__(self, data=0.0):
        super().__init__(data)


class Float32MultiArray(_Data):
    def __init__(self):
        super().__init__([])


class Float64MultiArray(_Data):
    def __init__(self):
        super().__init__([])


if "rclpy" not in sys.modules or not hasattr(sys.modules["rclpy"], "spin"):
    _module("rclpy", init=lambda **k: None, spin=lambda n: None,
            shutdown=lambda: None)
    _module("rclpy.node", Node=FakeNode)
    _module("sensor_msgs")
    _module("sensor_msgs.msg", JointState=JointState, Imu=Imu)
    _module("std_msgs")
    _module("std_msgs.msg", Bool=Bool, Float64=Float64,
            Float32MultiArray=Float32MultiArray,
            Float64MultiArray=Float64MultiArray)
    _module("geometry_msgs")
    _module("geometry_msgs.msg", Twist=Twist)

from skate_ros2 import names                          # noqa: E402
from skate_ros2 import shared_classes_def as SCD      # noqa: E402
from skate_ros2.driver_node import SkateDriver        # noqa: E402
from skate_ros2.protocol import TelemetryState        # noqa: E402


class FakeLink:
    """Captures outgoing commands; exposes a real TelemetryState."""

    def __init__(self):
        self.state = TelemetryState()
        self.sent = []  # (targ, vel, height, deadman)

    def poll(self):
        return 0

    def send_command(self, targ, vel, height, deadman):
        self.sent.append((np.array(targ), np.array(vel), height, deadman))
        return True

    def close(self):
        pass

    @property
    def connected(self):
        return self.state.connected


def make_node():
    node = SkateDriver()
    node.link.close()
    node.link = FakeLink()
    return node


def feed_state(node, pos, temps=None):
    se = SCD.state_est()
    se.dof_pos = names.vector_to_can_dict(pos)
    node.link.state.update(2, se)
    if temps is not None:
        ms = SCD.motor_state()
        ms.motor_temp = names.vector_to_can_dict(temps)
        node.link.state.update(1, ms)


def test_arms_at_measured_pose_and_publishes_joint_states():
    node = make_node()
    measured = np.linspace(-0.5, 0.5, 26)
    feed_state(node, measured)
    node.rx_tick()
    assert node.targ is not None
    assert np.allclose(node.targ, measured)
    js = node.publishers["joint_states"][-1]
    assert js.name == list(names.JOINT_NAMES)
    assert np.allclose(js.position, measured)
    # armed but no user command yet -> deadman (0,0,0)
    node.tx_tick()
    assert node.link.sent[-1][3] == (0, 0, 0)


def test_deadman_follows_command_freshness():
    node = make_node()
    feed_state(node, np.zeros(26))
    node.rx_tick()
    msg = Float64MultiArray()
    msg.data = list(names.DEFAULT_POSE)
    node.subscriptions["skate/joint_position_cmd_raw"](msg)
    node.tx_tick()
    assert node.link.sent[-1][3] == (1, 1, 1)
    # 0.5 s later with no new commands -> dampen
    node._clock.t += 0.5
    node.tx_tick()
    assert node.link.sent[-1][3] == (0, 0, 0)


def test_estop_latches():
    node = make_node()
    feed_state(node, np.zeros(26))
    node.rx_tick()
    node.subscriptions["skate/joint_position_cmd_raw"](
        type("M", (), {"data": [0.0] * 26})())
    node.subscriptions["skate/estop"](Bool(True))
    node.tx_tick()
    assert node.link.sent[-1][3] == (0, 0, 0)
    node.subscriptions["skate/estop"](Bool(False))
    node.tx_tick()
    assert node.link.sent[-1][3] == (1, 1, 1)


def test_overtemp_latch_and_hysteresis():
    node = make_node()
    temps = np.full(26, 30.0)
    temps[12] = 61.0
    feed_state(node, np.zeros(26), temps=temps)
    node.rx_tick()
    node.slow_tick()
    assert node.overtemp
    node.subscriptions["skate/joint_position_cmd_raw"](
        type("M", (), {"data": [0.0] * 26})())
    node.tx_tick()
    assert node.link.sent[-1][3] == (0, 0, 0)
    # cooling below 58-5 releases the latch
    temps[12] = 50.0
    feed_state(node, np.zeros(26), temps=temps)
    node.slow_tick()
    assert not node.overtemp
    node.tx_tick()
    assert node.link.sent[-1][3] == (1, 1, 1)


def test_joint_cmd_by_name_merges_and_ignores_unknown():
    node = make_node()
    measured = np.linspace(0.1, 0.2, 26)
    feed_state(node, measured)
    node.rx_tick()
    msg = JointState()
    msg.name = ["a3_armL_a11", "definitely_not_a_joint"]
    msg.position = [1.0, 9.9]
    node.subscriptions["skate/joint_position_cmd"](msg)
    assert node.targ[11] == 1.0
    # everything else stays at the measured (armed) pose
    assert np.allclose(np.delete(node.targ, 11), np.delete(measured, 11))
    assert any("definitely_not_a_joint" in line
               for _lvl, line in node._logger.lines)


def test_joint_cmd_ignored_before_arming():
    """Commands before first telemetry must not invent a base pose."""
    node = make_node()
    msg = JointState()
    msg.name = ["a3_armL_a11"]
    msg.position = [1.0]
    node.subscriptions["skate/joint_position_cmd"](msg)
    assert node.targ is None                      # still unarmed
    assert any("no telemetry yet" in line
               for _lvl, line in node._logger.lines)
    # after telemetry the driver arms at the MEASURED pose, not the command
    measured = np.full(26, 0.3)
    feed_state(node, measured)
    node.rx_tick()
    assert np.allclose(node.targ, measured)


def test_stale_cmd_vel_decays_to_zero():
    """A single old Twist must not keep driving the base forever."""
    node = make_node()
    feed_state(node, np.zeros(26))
    node.rx_tick()
    tw = Twist()
    tw.linear.x = 0.3
    node.subscriptions["skate/cmd_vel"](tw)
    node.tx_tick()
    assert np.allclose(node.link.sent[-1][1], [0.3, 0.0, 0.0])
    # joint commands keep the deadman alive, but the old Twist must die
    node._clock.t += 0.5
    node.subscriptions["skate/joint_position_cmd_raw"](
        type("M", (), {"data": [0.0] * 26})())
    node.tx_tick()
    assert node.link.sent[-1][3] == (1, 1, 1)     # deadman alive
    assert np.allclose(node.link.sent[-1][1], [0.0, 0.0, 0.0])  # vel decayed


def test_raw_cmd_rejects_wrong_length():
    node = make_node()
    node.subscriptions["skate/joint_position_cmd_raw"](
        type("M", (), {"data": [0.0] * 7})())
    assert node.targ is None


def test_arms_from_motor_pos_when_state_est_missing():
    """If the calibrated state-estimate stream never arrives (e.g. its packets
    are dropped by a low-MTU link), the driver still arms from raw motor
    telemetry and publishes joint states."""
    node = make_node()
    measured = np.linspace(-0.3, 0.4, 26)
    ms = SCD.motor_state()
    ms.motor_pos = names.vector_to_can_dict(measured)
    node.link.state.update(1, ms)                 # only motor_state, no state_est
    assert node.link.state.dof_pos() is None       # calibrated stream absent
    assert node.link.state.motor_pos() is not None
    node.rx_tick()
    assert node.targ is not None
    assert np.allclose(node.targ, measured)
    js = node.publishers["joint_states"][-1]
    assert np.allclose(js.position, measured)


if __name__ == "__main__":
    for f in [test_arms_at_measured_pose_and_publishes_joint_states,
              test_deadman_follows_command_freshness,
              test_estop_latches,
              test_overtemp_latch_and_hysteresis,
              test_joint_cmd_by_name_merges_and_ignores_unknown,
              test_joint_cmd_ignored_before_arming,
              test_stale_cmd_vel_decays_to_zero,
              test_raw_cmd_rejects_wrong_length,
              test_arms_from_motor_pos_when_state_est_missing]:
        f()
        print(f"PASS {f.__name__}")
