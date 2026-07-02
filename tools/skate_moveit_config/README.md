# skate_moveit_config — MoveIt 2 for the Skate bimanual upper body

MoveIt 2 motion-planning configuration for the [R.Botic Skate](https://www.rboticlabs.com/shop/p/skate-upper-body-v2)
(skt_v3), driven through the [`skate_ros2`](../skate_ros2/) UDP bridge — so you
can plan bimanual motions in RViz against the MuJoCo sim endpoint today, and
re-point the same stack at the real robot when it arrives.

Part of the [SkateArm](../../README.md) project.

> **Status — built &amp; end-to-end-verified on ROS 2 Jazzy** (Ubuntu 24.04 / WSL2):
> `colcon build` is clean, `move_group` loads this config (URDF `r04` + this
> SRDF + OMPL) and reports *"You can start planning now!"*, **MoveItPy plans
> collision-free bimanual trajectories** for `left_arm` and `right_arm`
> (~13–15 waypoints, with the SRDF collision matrix active), and the **full
> execution loop runs — MoveItPy `execute()` → the `FollowJointTrajectory`
> bridge → the driver → the sim, with the sim arm visibly moving to the planned
> pose** (a shoulder joint drove 0 → ~1.14 rad, commands streamed, zero driver
> rejects). Five real issues were caught &amp; fixed during the live bring-up:
> the SRDF `<robot>` name must match the URDF (`r04`); the URDF's scheme-less
> mesh paths need `file://` URIs; the Jazzy planning-pipeline params are
> **lists** (`planning_plugins`, request/response adapters); WSL2 needs a
> CycloneDDS loopback-**unicast** profile for cross-process discovery; and WSL2
> loopback drops UDP datagrams over ~1472 B, so the sim's ~2 kB `state_est`
> telemetry is now fragmented and reassembled by the UDP transport. The
> SRDF↔URDF consistency, trajectory interpolation, the UDP
> fragmentation/reassembly and the driver arm-fallback are all unit-tested
> without ROS. Real-hardware validation waits for the Skate.

## Planning groups

| Group | Kind | Joints / chain |
|---|---|---|
| `left_arm` | KDL chain | `base_link2` → `wrist_a2_1` (protocol joints 8–14) |
| `right_arm` | KDL chain | `base_link2` → `wrist_a2_Mirror__1` (protocol joints 16–22) |
| `both_arms` | subgroups | `left_arm` + `right_arm` (14 DoF, joint-space) |
| `left_gripper` / `right_gripper` | single joint | `a7_armL_a15` / `a7_armR_a23` |

Joint and link names are the exact skt_v3 URDF names (see
[`skate_ros2/names.py`](../skate_ros2/skate_ros2/names.py)). `home` = elbows
bent 90°, matching `names.DEFAULT_POSE`.

## How execution works

MoveIt's simple controller manager sends each arm a **FollowJointTrajectory**
goal. Those action servers are the `moveit_bridge` node in `skate_ros2`: it
interpolates the planned trajectory (`skate_ros2/traj_interp.py`, pure-Python,
unit-tested) and streams the setpoints to `skate/joint_position_cmd`. The
existing `skate_driver` then does the UDP wire and the deadman / e-stop /
overtemp safety — MoveIt never touches the robot directly and inherits the
audited safety model instead of duplicating it.

```
  MoveIt move_group ──FollowJointTrajectory──▶ moveit_bridge ──JointState──▶ skate_driver ──UDP──▶ sim / robot
```

The production alternative to the Python bridge now exists:
[`skate_ros2_control`](../skate_ros2_control/) — a C++ `ros2_control`
`SystemInterface` + per-arm `JointTrajectoryController`s under the **same
controller names**, so this MoveIt config drives either backend unchanged
(just don't start `moveit_bridge` on the ros2_control path). The
FollowJointTrajectory contract above is identical either way.

## Files

```
skate_moveit_config/
├── make_srdf.py                 # generates config/skate.srdf FROM the URDF
├── config/
│   ├── skate.srdf               # groups, states, end-effectors, 197-pair ACM
│   ├── kinematics.yaml          # KDL IK per arm
│   ├── joint_limits.yaml        # velocity / acceleration for time-parameterization
│   ├── ompl_planning.yaml       # OMPL planners (RRTConnect default)
│   └── moveit_controllers.yaml  # simple manager → 2× FollowJointTrajectory
├── launch/demo.launch.py        # rsp + move_group + driver + bridge + RViz
├── examples/plan_bimanual.py    # MoveItPy: plan & execute a bimanual motion
└── test/test_srdf.py            # SRDF↔URDF consistency (no ROS)
```

The SRDF is **generated** (`make_srdf.py`), not hand-typed, so its collision
matrix stays in lockstep with the model and with
`sim/make_collision_model.py`'s structural excludes. Inter-arm and arm↔torso
pairs stay collision-checked (that is the bimanual-safety point); regenerate the
fully sampled ACM with the MoveIt Setup Assistant on a ROS 2 box before hardware
use.

```bash
python make_srdf.py --model /path/to/skate_teleop/skt_v3   # regenerate config/skate.srdf
```

## Build & launch (on a ROS 2 machine)

```bash
mkdir -p ~/skate_ws/src && cp -r tools/skate_ros2 tools/skate_moveit_config ~/skate_ws/src/
cd ~/skate_ws && colcon build && source install/setup.bash

# terminal 1 — a "robot" on UDP :2000
python -m skate_ros2.sim_endpoint --model /path/to/skt_v3/skt_v3_control.xml

# terminal 2 — MoveIt + the bridge + the driver + RViz
ros2 launch skate_moveit_config demo.launch.py \
    model_path:=/path/to/skate_teleop/skt_v3 robot_host:=127.0.0.1
```

Plan in the RViz **MotionPlanning** panel (pick `left_arm` / `right_arm` /
`both_arms`), or drive it from code — see [`examples/plan_bimanual.py`](examples/plan_bimanual.py).

## Tests

```bash
SKT_DIR=/path/to/skt_v3 python test/test_srdf.py          # SRDF↔URDF consistency
python ../skate_ros2/test/test_moveit_bridge.py           # trajectory interpolation + validation
```

## License

MIT — part of [SkateArm](https://github.com/dsl-robotics/skatearm). The skt_v3
URDF/meshes belong to [Rbotic/skate_teleop](https://github.com/Rbotic/skate_teleop).
