# skate_ros2_control — ros2_control for the Skate arms

A **C++ `hardware_interface::SystemInterface`** that exposes the Skate's 14
structural arm joints to [ros2_control], plus
`JointTrajectoryController` configs whose names **match
`skate_moveit_config`** — so MoveIt drives the standard ros2_control stack
with zero MoveIt-side changes.

```
MoveIt 2 ─▶ JointTrajectoryController ─▶ controller_manager
                                             │ read()/write()
                                     SkateSystem (C++ plugin)
                                             │ topics
                     /joint_states ◀── skate_ros2 driver ──▶ /skate/joint_position_cmd
                                             │ UDP (pickle wire)
                                    MuJoCo sim endpoint / real Skate
```

## Design: why topics, not a second UDP client

The transport between the plugin and the robot is the **existing
`skate_ros2` driver**, on purpose:

* the driver already implements the firmware safety contract —
  arm-at-measured-pose, command-freshness **deadman**, e-stop, overtemp
  latch — and ros2_control inherits it instead of re-implementing it in C++;
* the wire stays in one place (the driver), so sim and hardware behave
  identically on this path;
* a UDP-native C++ SystemInterface (speaking the pickle wire directly)
  remains the hardware-era option if the extra hop ever measures as a
  bottleneck.

The plugin details, honestly:

* `read()` copies the latest `/joint_states` telemetry (position + velocity)
  into the state interfaces; `write()` publishes the position commands as a
  by-name `sensor_msgs/JointState` **every controller cycle**, which is
  exactly what the driver's command-freshness deadman wants — while
  controllers are active the robot is armed, when the controller_manager
  stops the stream the driver dampens it firmware-style;
* `on_activate` refuses to run until real telemetry has been seen, then
  initialises every command at the **measured pose** — activating a
  controller can never jump the arm;
* grippers are excluded (they get a dedicated controller when the real
  gripper interface lands); legs/head stay with the firmware balance chain.

## Quick start (WSL2 / any ROS 2 Jazzy)

```bash
sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers
cd ~/skate_ws && colcon build && source install/setup.bash

# a sim endpoint must be listening on :2000 (e.g. Skate Commander's, or:
#   python3 -m skate_ros2.sim_endpoint --model .../skt_v3_control.xml)

ros2 launch skate_ros2_control control.launch.py \
    model_path:=/path/to/skate_teleop/skt_v3 robot_host:=127.0.0.1
```

Then plan and execute with MoveIt exactly as with the Python bridge — the
controller names (`skate_left_arm_controller` / `skate_right_arm_controller`,
`follow_joint_trajectory`) are identical, so `skate_moveit_config` needs no
changes; just don't start the `moveit_bridge` node on this path.

## Files

| file | what |
|---|---|
| `src/skate_system.cpp` + `include/…/skate_system.hpp` | the SystemInterface plugin |
| `skate_ros2_control.xml` | pluginlib export |
| `config/controllers.yaml` | 2× JTC (MoveIt-matching names); no JSB — the driver publishes `/joint_states` |
| `launch/control.launch.py` | RSP + controller_manager + spawners + driver |
| `make_control_urdf.py` | inserts the `<ros2_control>` block into `skt_v3.urdf` (joint list derived from `skate_ros2.names`) |
| `test/test_make_control_urdf.py` | ROS-free unit tests for the generator |

## Notes

* The launch file feeds `robot_description` through `robot_state_publisher`
  (bonus: TF from the driver's `/joint_states`); the controller_manager picks
  it up from the topic.
* `update_rate: 100` — command stream comfortably above the 0.3 s deadman.
* Sim first: verified end-to-end against the MuJoCo sim endpoint (see the
  repo ROADMAP for the verification log); hardware validation follows the
  Skate's arrival, same as the rest of the stack.

[ros2_control]: https://control.ros.org
