# skate_ros2 — ROS 2 bridge for the R.Botic Skate

ROS 2 driver for the [R.Botic Skate](https://www.rboticlabs.com/shop/p/skate-upper-body-v2)
bimanual robot over its **native UDP protocol** — plus a **MuJoCo sim endpoint
that speaks the exact same protocol**, so you can develop and test your whole
ROS 2 stack before the robot is even out of the box.

First community tool from the [SkateArm](../../README.md) project.

![wire demo](../../docs/img/ros2_wire_demo.gif)

*A scripted client teleoperates the MuJoCo endpoint over real UDP packets —
60 Hz commands in, ~190 telemetry pkt/s out. At t=11 s the client goes silent
and the firmware-style deadman watchdog dampens the robot in 0.3 s.*

## Why

The Skate ships with a clean Python teleop example
([Rbotic/skate_teleop](https://github.com/Rbotic/skate_teleop)) but no ROS 2
integration. This package bridges the gap with three thin layers:

| Layer | File | Needs |
|---|---|---|
| Wire protocol client | `skate_ros2/protocol.py` | numpy only |
| Sim endpoint (fake robot) | `skate_ros2/sim_endpoint.py` | + mujoco |
| ROS 2 driver node | `skate_ros2/driver_node.py` | + ROS 2 |

The protocol and safety logic are pure Python and fully unit-tested without
ROS; the rclpy node is a thin shell around them.

## The wire protocol (documented)

Reverse-engineered from the official client and confirmed against the official
Skate docs. Everything is UDP to/from port **2000** on the robot (`r.local`):

**Telemetry** (robot → you, streamed to whoever spoke last):
`pickle.dumps((id, obj))` with the classes from the official
`shared_classes_def.py` (vendored here, Apache-2.0):

| id | object | content | internal rate |
|---|---|---|---|
| 0 | `motor_command` | commands actually sent to motors | 200 Hz |
| 1 | `motor_state` | raw pos/vel/torque/error/**temp** per motor | 200 Hz |
| 2 | `state_est` | calibrated dof pos/vel/torque + base estimates | 200 Hz |
| 3 | `INS_fusion_state` | IMU fusion: quat, gyro, acc, global pose | 400 Hz |
| 4 | controller states | internal | 200 Hz |

**Command** (you → robot): `pickle.dumps((5, payload))` where payload is

```python
(targ_pos,        # np.float64[26], radians, see ordering below
 vel_cmd,         # np.float64[3]: base x, y, yaw rate — the driver feeds
                  #   skate/cmd_vel (Twist) straight through, i.e. m/s & rad/s
 height_cmd,      # float: stance height factor, official default 1.0
                  #   (lower = crouch; treated as dimensionless upstream)
 (wb, la, ra))    # deadman flags: 0 = dampen (whole body / left / right arm)
```

**Keepalive / watchdog:** the robot streams to the last address it heard from;
the official client pings `b"yo"` every 0.3 s. If the robot hears *nothing*
for 0.3 s it assumes deadman `(0,0,0)` and dampens itself. Any packet resets
the watchdog; flags come from the last command.

**26-DoF ordering** — the URDF joint names encode the protocol index directly
(`a3_armL_a11` = index 11), so the mapping in `skate_ros2/names.py` is exact:

| indices | group | URDF names |
|---|---|---|
| 0–7 | lower chain (legs) | `a0` … `a7` |
| 8–15 | left arm (15 = gripper) | `a0_armL_a8` … `a7_armL_a15` |
| 16–23 | right arm (23 = gripper) | `a0_armR_a16` … `a7_armR_a23` |
| 24–25 | head | `a0_head_a24`, `a1_head_a25` |

> ⚠️ The wire format is Python pickle — deserializing it can execute code.
> That's the firmware's design; use it on a trusted LAN only (the official
> stack makes the same assumption). `decode_packet` hardens the *decoding*
> side, and `SKATE_AUTH` (below) can authenticate a link where **both** ends
> are ours — but a real robot has no key, so on that link the trusted LAN is
> still the whole of the security model.

## Quick start — no hardware, no ROS

```bash
# 1. generate the control-ready MJCF over your skate_teleop clone
python3 sim/make_control_model.py /path/to/skate_teleop/skt_v3

# 2. terminal 1: a "robot" appears on UDP :2000
python3 -m skate_ros2.sim_endpoint --model .../skt_v3_control.xml

# 3. terminal 2: stream a bimanual wave to it
python3 examples/wave_no_ros.py --host 127.0.0.1
```

Anything written against `127.0.0.1` here talks to the real robot by swapping
the host for `r.local`. That's the whole point.

## Where it listens, and authenticating the link (`SKATE_AUTH`)

Anyone who can reach the sim endpoint's port can command motion, so it binds
**`127.0.0.1` by default**: on a fresh checkout nothing outside the machine can
reach it. Serving any other address is something you have to ask for, and
asking requires a key:

```bash
# refused, by design:
python3 -m skate_ros2.sim_endpoint --model .../skt_v3_control.xml --bind 0.0.0.0
# [sim_endpoint] refusing to serve motion commands on '0.0.0.0' without a key:
# anyone who can reach this port could drive the arms. Set $SKATE_AUTH to the
# same secret on BOTH ends, or bind 127.0.0.1. (...)

# works — the same secret exported on both ends:
export SKATE_AUTH='a long random string, 16+ bytes'
python3 -m skate_ros2.sim_endpoint --model .../skt_v3_control.xml --bind 0.0.0.0
```

With `SKATE_AUTH` set, every datagram in both directions is wrapped in a keyed
envelope

```
b"SKA1" | nonce (8 B) | HMAC-SHA256(key, nonce|body)[:16] | body
```

that a forged, stale or replayed packet can't produce. Three properties are
worth knowing:

* **The envelope sits outside fragmentation.** Every datagram verifies on its
  own, so a forged fragment is dropped before it can occupy a reassembly slot —
  and its 28 B still fit under the ~1472 B ceiling that WSL2 loopback enforces
  (note 2 below): `FRAG_MAX_DGRAM + AUTH_OVERHEAD` = 1400 + 28 ≤ 1472, pinned
  by a test so the two limits can't drift apart.
* **The endpoint verifies before it registers the sender**, so an unkeyed peer
  gets no *telemetry* either, not merely no commands. Authentication that only
  guards the command path is not authentication.
* **The nonce carries a wall-clock stamp**, accepted inside a ±5 s window —
  that window is what bounds replay, and it is wall clock because the two ends
  are separate processes whose monotonic clocks aren't comparable. Two machines
  whose clocks disagree by more than that will see *everything* refused: the
  endpoint's periodic log grows a `rejected=N` field and `SkateLink.auth_errors`
  counts the same on the client. Fix the clocks rather than the window.

Clients read the key from the environment too, so exporting `SKATE_AUTH` before
starting the cockpit, the driver or MoveIt is enough (`SkateLink(host, port)`
picks it up; pass `key=` to be explicit). There is deliberately **no `--key`
flag** — argv is world-readable in `ps`. The key must be at least 16 bytes: it
is raw HMAC key material with no KDF, so a shorter secret is brute-forceable
from a single captured datagram.

**Against a real Skate, leave `SKATE_AUTH` unset.** The firmware has no
authentication and cannot be given one; it would drop the envelope as garbage.
Authentication protects the links this project owns on both ends — on the link
to a real robot, the trusted LAN is still the whole security model. See
[SECURITY.md](../../SECURITY.md).

## Quick start — ROS 2 (target: Jazzy)

> **Honesty note:** the wire protocol, sim endpoint, and all driver safety
> logic are verified by tests that run without ROS (rclpy is stubbed). A
> `colcon build` on a live ROS 2 install has **not** been run yet — that's
> the first task once a ROS 2 machine enters the loop.

```bash
mkdir -p ~/skate_ws/src
cp -r tools/skate_ros2 ~/skate_ws/src/
cd ~/skate_ws && colcon build && source install/setup.bash

# real robot:
ros2 launch skate_ros2 skate_bridge.launch.py
# or against the sim endpoint:
ros2 launch skate_ros2 skate_bridge.launch.py robot_host:=127.0.0.1
```

### Topics

| topic | type | dir | notes |
|---|---|---|---|
| `joint_states` | sensor_msgs/JointState | pub | calibrated 26-DoF pos/vel/effort, URDF names — feed `robot_state_publisher` + `skt_v3.urdf` for RViz |
| `skate/imu` | sensor_msgs/Imu | pub | INS passthrough |
| `skate/temperatures` | std_msgs/Float32MultiArray | pub | per-motor °C, protocol order |
| `skate/connected` | std_msgs/Bool | pub | telemetry freshness |
| `skate/joint_position_cmd` | sensor_msgs/JointState | sub | by-name, partial messages fine |
| `skate/joint_position_cmd_raw` | std_msgs/Float64MultiArray | sub | full 26-vector |
| `skate/cmd_vel` | geometry_msgs/Twist | sub | base velocity |
| `skate/height_cmd` | std_msgs/Float64 | sub | crouch height |
| `skate/estop` | std_msgs/Bool | sub | `true` = dampen, latched |

### Safety model (mirrors the firmware)

* Nothing is commanded until telemetry arrives; the driver then **arms at the
  robot's own measured pose** — a fresh bridge can never jump the robot.
  Joint commands received before arming are ignored (with a warning), so a
  robot that comes online late can't jump to a guessed base pose either.
* A stale `skate/cmd_vel` decays to zero after `cmd_timeout` — joint commands
  can't keep an old base velocity alive.
* Deadman flags are `(1,1,1)` only while your commands are fresher than
  `cmd_timeout` (0.3 s default). Stop publishing → the robot dampens, exactly
  like releasing the deadman button in the official VR teleop.
* Any motor over `overtemp_c` (58 °C, the PETG limit from the official docs)
  latches a whole-body dampen with 5 °C release hysteresis. Temperatures are
  evaluated on the 1 Hz slow tick, so a spike can take **up to ~1 s** to
  latch — the thermal time constant of the motors is far longer than that.
* `skate/estop` `true` dampens immediately and stays latched until `false`.

Parameters: `robot_host`, `robot_port`, `tx_rate` (60), `rx_rate` (60),
`cmd_timeout` (0.3), `auto_deadman` (true), `overtemp_c` (58.0) — all
exposed as launch arguments of `skate_bridge.launch.py` too.

## Running on Windows (WSL2)

There is no native ROS 2 on Windows — use **WSL2 Ubuntu 24.04** (= ROS 2
**Jazzy**). This is the exact environment the MoveIt config was built and
end-to-end-verified on (planning **and** execution → the sim arm moves).

**The whole path, at a glance:** ① `wsl --install -d Ubuntu-24.04` → ② install
ROS 2 Jazzy (apt) → ③ `colcon build` the two packages → ④ apply the one-time
WSL2 DDS fix (copy [`cyclonedds.xml`](cyclonedds.xml) + two env vars) → ⑤ start
the sim endpoint on `:2000` → ⑥ `ros2 launch skate_moveit_config demo.launch.py`
→ the RViz **MotionPlanning** window opens on your desktop; set a goal,
**Plan**, **Execute**. Each step is spelled out below.

```powershell
# once, in Windows PowerShell:
wsl --install -d Ubuntu-24.04     # then set up the Linux user it prompts for
```

```bash
# inside the WSL Ubuntu shell — add the ROS 2 apt repo + install Jazzy:
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update && sudo apt install -y \
  ros-jazzy-desktop ros-jazzy-moveit ros-jazzy-moveit-py \
  python3-colcon-common-extensions python3-pip
pip install mujoco --break-system-packages   # the sim endpoint needs it

# build the workspace (your repo shows up under /mnt/c/... from WSL):
mkdir -p ~/skate_ws/src
cp -r /mnt/c/path/to/skatearm/tools/skate_ros2 \
      /mnt/c/path/to/skatearm/tools/skate_moveit_config ~/skate_ws/src/
cd ~/skate_ws && colcon build && source install/setup.bash

# DDS discovery fix (WSL2 only — see note 1): use the ready-made profile
cp /mnt/c/path/to/skatearm/tools/skate_ros2/cyclonedds.xml ~/cyclonedds.xml
sudo apt install -y ros-jazzy-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/cyclonedds.xml

# run it (see the MoveIt 2 section below):
python3 -m skate_ros2.sim_endpoint --model /mnt/c/.../skt_v3/skt_v3_control.xml &
ros2 launch skate_moveit_config demo.launch.py \
    model_path:=/mnt/c/.../skt_v3 robot_host:=127.0.0.1
```

On **Windows 11 the RViz window opens on your desktop automatically** — WSLg
provides the GUI, no X server to install. (On Windows 10 without WSLg, run an
X server such as VcXsrv and `export DISPLAY=…`.) The MoveIt **MotionPlanning**
panel is where you set a goal and click **Plan** / **Execute**.

> **WSL2 setup notes (two one-time fixes; a native ROS 2 Linux box needs
> neither).** The full multi-node execution loop is verified on WSL2 once these
> are in place:
>
> 1. **DDS discovery.** WSL2's default networking breaks ROS 2 cross-process
>    discovery — two nodes can't see each other's topics/actions and
>    `ros2 topic list` comes up empty. Fix: CycloneDDS pinned to loopback
>    **unicast**, shipped ready-made as [`cyclonedds.xml`](cyclonedds.xml) —
>    just install the RMW, copy the file and point the two env vars at it (the
>    three commands are in the Quick start above). It sets multicast off, the
>    `127.0.0.1` interface only, and `localhost` as an explicit discovery peer.
>    (On WSL2, `ros2 topic list` needs `--no-daemon` — the CLI daemon's
>    XML-RPC socket hangs there; direct rclpy pub/sub is unaffected.)
>
> 2. **Large UDP telemetry.** WSL2 loopback silently drops any UDP datagram over
>    ~1472 B (its `lo` reports MTU 65536, but the vNIC enforces ~1500 and
>    fragmentation is unreliable at any MTU), so the sim's ~2 kB `state_est`
>    packet never reached the driver and it never armed. The sim transport now
>    fragments oversized telemetry into ≤1400 B chunks and `SkateLink`
>    reassembles them (`protocol.pack_datagrams` / `Reassembler`); packets that
>    already fit — including real firmware's raw pickles — go out unchanged, so
>    the wire stays compatible. As a belt-and-braces fallback the driver also
>    arms from raw `motor_state` when the calibrated `state_est` stream is
>    absent.

## MoveIt 2 (bimanual planning)

[`skate_moveit_config`](../skate_moveit_config/) is a MoveIt 2 config for the
two arms: `left_arm` / `right_arm` / `both_arms` planning groups, an SRDF
**generated from the URDF** (`make_srdf.py`, so the 197-pair collision matrix
tracks the model and `sim/make_collision_model.py`'s excludes), OMPL planning,
and per-arm `FollowJointTrajectory` controllers.

Execution reuses **this** driver instead of a second control path: the
`moveit_bridge` node runs one action server per arm, interpolates the planned
trajectory (`skate_ros2/traj_interp.py` — pure Python, unit-tested) and streams
it to `skate/joint_position_cmd`, so MoveIt inherits the driver's
arm-at-measured-pose / deadman / estop / overtemp safety instead of
re-implementing it.

Prefer the standard controller stack? [`skate_ros2_control`](../skate_ros2_control/)
serves the **same** FollowJointTrajectory actions from real
`ros2_control` `JointTrajectoryController`s over a C++ `SystemInterface`
that still rides this driver's topics — swap backends without touching the
MoveIt config.

```bash
# sim endpoint on :2000, then:
ros2 launch skate_moveit_config demo.launch.py \
    model_path:=/path/to/skate_teleop/skt_v3 robot_host:=127.0.0.1
# plan in the RViz MotionPlanning panel: move_group -> moveit_bridge -> driver -> sim
```

> **Status — built &amp; end-to-end-verified on ROS 2 Jazzy** (Ubuntu 24.04 /
> WSL2): `colcon build` clean, `move_group` loads the config, **MoveItPy plans
> collision-free bimanual trajectories** (~13–15 waypoints), and the **full
> execution loop runs — MoveItPy `execute()` → `moveit_bridge` →
> `skate/joint_position_cmd` → driver → sim, with the sim arm visibly moving to
> the planned pose** (a shoulder joint drove 0 → ~1.14 rad, commands streamed,
> zero driver rejects). Five real bugs were caught &amp; fixed during the live
> bring-up (URDF/SRDF robot-name match, `file://` mesh URIs, Jazzy list-form
> planning-pipeline params, and the WSL2 DDS + UDP-MTU fixes noted above).
> SRDF↔URDF consistency, trajectory interpolation, the UDP
> fragmentation/reassembly and the driver arm-fallback are all unit-tested
> without ROS. On hardware, a `ros2_control` `JointTrajectoryController` + a
> Skate `SystemInterface` is the production alternative to the Python bridge.

### Watch a MoveIt execution live in the cockpit

The sim endpoint streams telemetry to **every** client heard from recently
(3 s TTL), not just the last sender — so two stacks can share one sim. That
turns [Skate Commander](../skate_commander/) into a live viewer for MoveIt:
start the cockpit (its spawned sim endpoint binds `127.0.0.1:2000`), click
**OBSERVE** in its menu bar (the cockpit keeps receiving telemetry but stops
transmitting commands), then run the driver + MoveIt against the same
endpoint — the browser twin mirrors the executed trajectory in real time and
an **EXTERNAL** chip lights while it moves. On WSL2 with mirrored networking
`robot_host:=127.0.0.1` reaches the Windows-hosted sim directly. Without it the
two stacks sit on different addresses, so the endpoint has to listen
off-loopback and that now needs a key: export `SKATE_AUTH` everywhere, start
the endpoint yourself with `--bind 0.0.0.0`, and point the cockpit's
`--sim-host` (it connects as a client instead of spawning its own) and MoveIt's
`robot_host:` at the host's IP. Verified live: a 15-waypoint MoveItPy plan
executed SUCCEEDED on the shared sim with the cockpit twin sweeping the full
trajectory — zero driver rejects.

## Sim endpoint: honest approximations

The emulator is faithful on the wire (port, packet layout, watchdog timing,
telemetry classes) but approximates the physics-side behavior:

* "dampen" freezes position targets at the current pose instead of dropping
  to damping-only torques;
* motor temperatures are **synthetic** (warm with |τ|, cool to 25 °C) so the
  temperature plumbing can be exercised;
* `vel_cmd` / `height_cmd` are accepted but ignored (fixed-base model);
* INS reports a static upright pose;
* telemetry goes to every recently-heard client (the firmware streams only to
  the last sender), and the deadman watchdog follows the **command** stream
  rather than any packet — with several peers attached, an observer's
  heartbeats must never keep a dead commander's last command alive. With a
  single commanding client the behavior is unchanged.

## Tests

```bash
python3 test/test_names.py              # ordering / CAN layout invariants
python3 test/test_protocol_loopback.py  # wire contract over localhost UDP
python3 test/test_driver_logic.py       # arming/deadman/estop/overtemp logic
                                        # (runs WITHOUT ROS via stubbed rclpy)
pytest -q test/test_wire_safe.py        # restricted unpickler vs hostile packets
pytest -q test/test_wire_auth.py        # keyed envelope: forgery, replay, a
                                        # stale nonce, the bind gate, and
                                        # verify-before-register
SKATE_MJCF=.../skt_v3_control.xml python3 test/test_e2e_sim.py    # full e2e
SKATE_MJCF=.../skt_v3_control.xml python3 test/test_multipeer.py  # two peers,
                                        # one of them without the key
```

Verified end-to-end in CI-like conditions: 60 Hz commands, ~190 telemetry
pkt/s, elbow tracking error 0.015 rad, watchdog dampen < 0.3 s
([stats plot](../../docs/img/ros2_wire_stats.png)).

## Roadmap

* MoveIt 2 config + `FollowJointTrajectory` bridge for the bimanual chains —
  **authored & structurally validated** ([`tools/skate_moveit_config`](../skate_moveit_config/),
  see the MoveIt 2 section above); the live `colcon build` + planning run awaits
  a ROS 2 machine, and a native `ros2_control` hardware interface is the
  production alternative to the Python bridge;
* gripper action server;
* real-hardware validation when the Skate lands — wire numbers above
  are sim-endpoint numbers until then.

## Credits

Wire classes and protocol from [Rbotic/skate_teleop](https://github.com/Rbotic/skate_teleop)
(Apache-2.0, vendored with attribution in `skate_ros2/shared_classes_def.py`).
Everything else MIT, part of [SkateArm](https://github.com/dsl-robotics/skatearm).
