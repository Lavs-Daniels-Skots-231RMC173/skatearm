"""skate_driver — ROS 2 bridge node over Skate's native UDP protocol.

Thin by design: all wire logic lives in :mod:`skate_ros2.protocol` (pure
Python, tested without ROS); this node only converts between topics and the
link object.

Published topics
    joint_states            sensor_msgs/JointState   calibrated 26-DoF state
    skate/imu               sensor_msgs/Imu          INS stream passthrough
    skate/temperatures      std_msgs/Float32MultiArray  per-motor °C
    skate/connected         std_msgs/Bool            telemetry freshness

Subscribed topics
    skate/joint_position_cmd      sensor_msgs/JointState   by-name, partial OK
    skate/joint_position_cmd_raw  std_msgs/Float64MultiArray  full 26-vector
    skate/cmd_vel                 geometry_msgs/Twist      base velocity
    skate/height_cmd              std_msgs/Float64         crouch height
    skate/estop                   std_msgs/Bool            True = dampen, latched

Safety model (mirrors the firmware deadman):
* until the first telemetry arrives, nothing is commanded — and incoming
  joint commands are IGNORED, so the robot can never jump to a guessed pose
  the moment it comes online;
* the first target is the robot's own measured pose (no jump-to-zero);
* a stale ``skate/cmd_vel`` decays to zero after ``cmd_timeout`` — joint
  commands can't keep an old base velocity alive;
* deadman flags are (1,1,1) only while subscriber commands are fresher than
  ``cmd_timeout`` and no estop/overtemp is latched — stop publishing commands
  and the robot dampens, exactly like releasing the VR deadman button;
* motors over ``overtemp_c`` (58 °C default, PETG limit from the official
  docs) latch a whole-body estop with 5 °C release hysteresis.
"""

from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32MultiArray, Float64, Float64MultiArray

from . import names
from .protocol import SkateLink


class SkateDriver(Node):
    def __init__(self):
        super().__init__("skate_driver")
        self.declare_parameter("robot_host", "r.local")
        self.declare_parameter("robot_port", 2000)
        self.declare_parameter("tx_rate", 60.0)
        self.declare_parameter("rx_rate", 60.0)
        self.declare_parameter("cmd_timeout", 0.3)
        self.declare_parameter("auto_deadman", True)
        self.declare_parameter("overtemp_c", 58.0)

        host = self.get_parameter("robot_host").value
        port = int(self.get_parameter("robot_port").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.auto_deadman = bool(self.get_parameter("auto_deadman").value)
        self.overtemp_c = float(self.get_parameter("overtemp_c").value)

        self.link = SkateLink(host, port)

        # command state
        self.targ = None              # np.float64[26] or None until armed
        self.vel_cmd = np.zeros(3)
        self.height_cmd = 1.0
        self.last_cmd_time = None     # node clock seconds of last subscriber cmd
        self.last_vel_time = None     # node clock seconds of last cmd_vel
        self.estop = False            # latched via skate/estop
        self.overtemp = False         # latched with hysteresis
        self._warned_names = set()
        self._warned_not_armed = False
        self._last_connected = None
        self._last_state_count = -1

        # pubs
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_imu = self.create_publisher(Imu, "skate/imu", 10)
        self.pub_temp = self.create_publisher(
            Float32MultiArray, "skate/temperatures", 10)
        self.pub_conn = self.create_publisher(Bool, "skate/connected", 10)

        # subs
        self.create_subscription(
            JointState, "skate/joint_position_cmd", self.on_joint_cmd, 10)
        self.create_subscription(
            Float64MultiArray, "skate/joint_position_cmd_raw",
            self.on_joint_cmd_raw, 10)
        self.create_subscription(Twist, "skate/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64, "skate/height_cmd",
                                 self.on_height, 10)
        self.create_subscription(Bool, "skate/estop", self.on_estop, 10)

        tx_rate = float(self.get_parameter("tx_rate").value)
        rx_rate = float(self.get_parameter("rx_rate").value)
        self.create_timer(1.0 / tx_rate, self.tx_tick)
        self.create_timer(1.0 / rx_rate, self.rx_tick)
        self.create_timer(1.0, self.slow_tick)

        self.get_logger().info(
            f"skate_driver -> {host}:{port} "
            f"(tx {tx_rate:.0f} Hz, rx {rx_rate:.0f} Hz, "
            f"auto_deadman={self.auto_deadman})")

    # -- helpers --------------------------------------------------------------
    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _mark_cmd(self):
        self.last_cmd_time = self._now_s()

    def _cmd_fresh(self):
        return (self.last_cmd_time is not None
                and self._now_s() - self.last_cmd_time < self.cmd_timeout)

    # -- subscriber callbacks ---------------------------------------------------
    def on_joint_cmd(self, msg: JointState):
        if self.targ is None:
            # Not armed yet: merging into a guessed base pose could make the
            # robot jump to it when telemetry finally comes up. Refuse.
            if not self._warned_not_armed:
                self._warned_not_armed = True
                self.get_logger().warning(
                    "joint command ignored — no telemetry yet; the driver "
                    "arms at the robot's measured pose first")
            return
        for name, pos in zip(msg.name, msg.position):
            idx = names.INDEX.get(name)
            if idx is None:
                if name not in self._warned_names:
                    self._warned_names.add(name)
                    self.get_logger().warning(
                        f"unknown joint '{name}' ignored "
                        f"(expected skt_v3 URDF names like 'a3_armL_a11')")
                continue
            self.targ[idx] = float(pos)
        self._mark_cmd()

    def on_joint_cmd_raw(self, msg: Float64MultiArray):
        if len(msg.data) != names.N_JOINTS:
            self.get_logger().warning(
                f"joint_position_cmd_raw needs {names.N_JOINTS} values, "
                f"got {len(msg.data)} — ignored")
            return
        self.targ = np.asarray(msg.data, dtype=np.float64).copy()
        self._mark_cmd()

    def on_cmd_vel(self, msg: Twist):
        self.vel_cmd = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.last_vel_time = self._now_s()
        self._mark_cmd()

    def on_height(self, msg: Float64):
        self.height_cmd = float(msg.data)
        self._mark_cmd()

    def on_estop(self, msg: Bool):
        if bool(msg.data) != self.estop:
            self.estop = bool(msg.data)
            level = "ESTOP LATCHED — robot dampened" if self.estop \
                else "estop released"
            self.get_logger().warning(level)

    # -- timers -----------------------------------------------------------------
    def tx_tick(self):
        if self.targ is None:
            # not armed yet: poll() heartbeats keep telemetry flowing
            return
        live = (not self.estop and not self.overtemp
                and (self._cmd_fresh() or not self.auto_deadman))
        deadman = (1, 1, 1) if live else (0, 0, 0)
        # a stale Twist must not keep driving the base while joint commands
        # keep the deadman alive — decay it to zero after cmd_timeout
        vel_fresh = (self.last_vel_time is not None
                     and self._now_s() - self.last_vel_time < self.cmd_timeout)
        vel = self.vel_cmd if vel_fresh else np.zeros(3)
        self.link.send_command(self.targ, vel, self.height_cmd, deadman)

    def rx_tick(self):
        self.link.poll()
        st = self.link.state
        if st.n_packets == self._last_state_count:
            return
        self._last_state_count = st.n_packets

        pos = st.dof_pos()
        if pos is None:
            # Fall back to raw motor telemetry when the calibrated state-estimate
            # stream is unavailable (e.g. dropped by a low-MTU link), so the
            # driver still arms and publishes joint states.
            pos = st.motor_pos()
        if pos is not None:
            if self.targ is None:
                # arm at the measured pose so the first command can't jump
                self.targ = np.asarray(pos, dtype=np.float64).copy()
                self.get_logger().info(
                    "telemetry up — armed at current robot pose (dampened "
                    "until commands stream)")
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(names.JOINT_NAMES)
            js.position = [float(x) for x in pos]
            vel = st.dof_vel()
            tau = st.dof_torque()
            if vel is not None:
                js.velocity = [float(x) for x in vel]
            if tau is not None:
                js.effort = [float(x) for x in tau]
            self.pub_js.publish(js)

        if st.ins is not None:
            imu = Imu()
            imu.header.stamp = self.get_clock().now().to_msg()
            imu.header.frame_id = "base_link"
            q = np.asarray(st.ins.out_quat, dtype=np.float64).ravel()
            if q.shape == (4,) and np.linalg.norm(q) > 1e-6:
                # firmware order (w, x, y, z) -> ROS (x, y, z, w)
                imu.orientation.w = float(q[0])
                imu.orientation.x = float(q[1])
                imu.orientation.y = float(q[2])
                imu.orientation.z = float(q[3])
            gyr = np.asarray(st.ins.out_gyr, dtype=np.float64).ravel()
            acc = np.asarray(st.ins.out_acc, dtype=np.float64).ravel()
            if gyr.shape == (3,):
                imu.angular_velocity.x = float(gyr[0])
                imu.angular_velocity.y = float(gyr[1])
                imu.angular_velocity.z = float(gyr[2])
            if acc.shape == (3,):
                imu.linear_acceleration.x = float(acc[0])
                imu.linear_acceleration.y = float(acc[1])
                imu.linear_acceleration.z = float(acc[2])
            self.pub_imu.publish(imu)

    def slow_tick(self):
        st = self.link.state

        temps = st.motor_temps()
        if temps is not None:
            msg = Float32MultiArray()
            msg.data = [float(t) for t in temps]
            self.pub_temp.publish(msg)
            tmax = max(temps)
            if not self.overtemp and tmax > self.overtemp_c:
                self.overtemp = True
                self.get_logger().error(
                    f"OVERTEMP {tmax:.1f}°C > {self.overtemp_c}°C — "
                    "whole-body deadman dropped")
            elif self.overtemp and tmax < self.overtemp_c - 5.0:
                self.overtemp = False
                self.get_logger().info(
                    f"temperature back to {tmax:.1f}°C — overtemp released")

        connected = self.link.connected
        if connected != self._last_connected:
            self._last_connected = connected
            self.get_logger().info(
                "robot telemetry LIVE" if connected
                else "robot telemetry LOST (robot dampens itself after 0.3 s)")
        msg = Bool()
        msg.data = bool(connected)
        self.pub_conn.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SkateDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.link.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
