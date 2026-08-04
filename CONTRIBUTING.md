# Contributing to SkateArm

Thanks for your interest! SkateArm is an open, **sim-first** bimanual work-cell and tool
ecosystem for the [R.Botic Skate](https://www.rboticlabs.com/shop/p/skate-upper-body-v2).
You don't need the robot — everything here runs in MuJoCo simulation today.

## Ways to contribute

- **Report a bug** — open a [bug report](../../issues/new?template=bug_report.yml).
- **Request a feature or tool** — open a [feature request](../../issues/new?template=feature_request.yml).
  Ideas and requests from other Skate owners are especially welcome.
- **Improve the docs** — typos, clarifications, or a missing step in a quick-start.
- **Send a pull request** — see below.

## Good first issues

New here? These are scoped, self-contained starter tasks — all doable in simulation, no
robot needed:

- **URDF / config validator** — a small `tools/` script that sanity-checks a Skate model
  directory (joint count, limits, mesh references) and prints a friendly report.
- **A new sim demo** — add a `sim/demo_*.py` for a bimanual primitive, following the
  existing demos, with a still + GIF. The two-handed hand-off that used to be the
  example here now ships as a controller and a benchmark task (`sim/handoff.py`,
  `benchmark.py --tasks handoff`) but has no demo script or GIF of its own, so
  rendering that one is a good first pass at the pattern.
- **Getting-started handbook** — expand the quick-start into a step-by-step `docs/` page,
  from clone to first browser-driven pose.
- **More tests** — add a hardware-free unit test for an untested helper (e.g. in
  `kinematics.py` or `detect.py`).

Open an issue to claim one (or propose your own) before you start, so work isn't duplicated.

## Project layout

| Path | What's there |
|---|---|
| `sim/` | MuJoCo model builders (actuators, collision capsules), demos, telemetry |
| `tools/skate_commander/` | Web cockpit — FastAPI + WebSocket server, Three.js twin, tests |
| `tools/skate_ros2/` | ROS 2 driver over Skate's native UDP + a protocol-true sim endpoint |
| `dashboard/` | Flask + SQLite SCADA dashboard for the work-cell |
| `docs/` | Landing page (GitHub Pages), architecture, roadmap, media |

## Development setup

```bash
git clone https://github.com/dsl-robotics/skatearm.git
cd skatearm
pip install numpy pytest      # core test deps — no MuJoCo / ROS needed
pip install mujoco imageio    # only for the sim demos / renders
```

> **Windows:** use `py` instead of `python` / `python3` (the bare names may open the
> Microsoft Store stub).

## Running the tests

Most of the suite is **hardware-free** — no MuJoCo or ROS required:

```bash
pytest -q tools/skate_commander/test tools/skate_ros2/test
```

The model-gated tests (collision guard, RRT planner, IK, URDF) **skip** unless the `skt_v3`
model is present. To run them too, build the models once and point `SKT_DIR` at them:

```bash
py sim/make_control_model.py path/to/skt_v3      # + actuators
py sim/make_collision_model.py path/to/skt_v3    # + collision capsules
SKT_DIR=path/to/skt_v3 pytest -q tools/skate_commander/test
```

CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the wire-protocol,
joint-map, driver-safety and natural-language-parser tests on every push and pull request.
Please make sure `pytest -q` is green locally before opening a PR, and add tests for new
behaviour.

## Pull requests

1. Fork and branch from `main` (`feature/...` or `fix/...`).
2. Keep changes focused — one logical change per PR.
3. Match the surrounding style; don't introduce new warnings.
4. Update the docs / READMEs when behaviour changes.
5. Make sure the tests pass and fill in the PR template.

## Safety-critical code

This project drives (or soon will drive) a real two-armed robot. Changes to the
**collision guard, deadman / E-STOP, UDP protocol or joint mapping** get extra scrutiny —
never weaken a safety check just to make a demo pass. The core invariants must hold:
arm-at-measured-pose on connect, a command-freshness deadman, and the overtemp latch.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE). The `skt_v3` model and meshes belong to
[Rbotic/skate_teleop](https://github.com/Rbotic/skate_teleop) and are not redistributed here.
