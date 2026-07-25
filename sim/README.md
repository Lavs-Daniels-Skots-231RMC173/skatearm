# sim — MuJoCo digital twin

Phase 0 result: the official `skt_v3` model loads, poses and renders headlessly.

## Setup

```bash
git clone https://github.com/Rbotic/skate_teleop.git
pip install mujoco numpy imageio
python render_skate.py --model skate_teleop/skt_v3 --out renders
```

On a headless Linux box use `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, `apt install libosmesa6`).

## What we know about the model (verified 11.06.2026, MuJoCo 3.9)

- `skt_v3_converted.xml`: 26 named hinge joints + free base joint (nq=33)
  - `a0…a7` — lower chain (hips/legs/wheels of the full Skate)
  - `a0_armL_a8 … a7_armL_a15` — left arm (8 DoF)
  - `a0_armR_a16 … a7_armR_a23` — right arm (8 DoF)
  - `a0_head_a24`, `a1_head_a25` — head pan/tilt
- Mirrored arm chains take the **same sign** for a symmetric pose.
- **Respect joint ranges** — they are asymmetric: e.g. `a1` (abduction) −0.79…2.36, `a3` (elbow) 0…2.64 — the elbow can't bend backwards. Out-of-range qpos silently interpenetrates meshes and looks broken; `render_skate.py` clamps as a guard.
- **No actuators (`nu=0`), no sensors** — it is a visualization/teleop model. A control-ready MJCF (position servos + sensors) is the next sim task and a community-tool candidate.
- Base free joint origin is mid-body: lift `qpos[2] ≈ 0.95` to put wheels on a floor plane.

## Control-ready model

`make_control_model.py` turns the visualization-only MJCF into a controllable one:

- 26 **position actuators** (kp=100), ctrlrange = joint ranges, forcerange ±28 N·m from the URDF
- joint **damping 2.0 + armature 0.05** (the original has none — undamped servos oscillate)
- **fixed base** (freejoint removed) — work-cell configuration
- **contacts disabled**: the raw converted meshes interpenetrate at the shoulder mounts and jam the joints — the shoulder servo saturates at its 28 N·m force limit just fighting the contact. Free-space control is correct without contacts; real collision geometry is a roadmap task.

Verified: holds RELAXED and WORK poses with **max error < 0.03 rad (~1.5°)**, settles to zero velocity, no divergence (MuJoCo 3.9, 2 ms timestep).

`demo_wave.py` runs a closed-loop sequence (independent left/right arm trajectories + head pan) and writes the GIF in the main README.

## Collision model

`make_collision_model.py` builds `skt_v3_collision.xml` on top of the control model:

- mesh geoms become **visual-only**; each body gets an auto-fitted **capsule** (a **sphere** when the link is near-isotropic) from the compiled model's AABB (`m.geom_aabb` — compiled values respect MuJoCo's mesh re-centering). Capsules hug the elongated arm links far better than boxes; `--boxes` keeps the old box behaviour. The current model is 22 capsules + 5 spheres.
- residual home-pose overlaps (link mounts: torso↔shoulders/hips, wrists↔hips at hanging pose) are **auto-excluded** — 11 pairs
- contacts **re-enabled**: poses still hold at < 0.03 rad; commanded arm-crossing now *blocks on the hips* instead of tunneling; a staged out→up→together trajectory ends in a stable wrist↔wrist contact

Known limitation: the AABB fit still slightly overestimates the L-shaped wrist links (a per-link shrink of 0.62 tightens them), so hands can "touch" a hair early and the hands-together route stays OUT → UP → MEET. Convex-hull decomposition of the bulky torso is a possible future refinement.

## Files

- `render_skate.py` — patch scene (floor/light/framebuffer), set bimanual pose, render PNGs. Outputs in [../docs/img/](../docs/img/).
- `make_control_model.py` — generate `skt_v3_control.xml` (actuators + damping + fixed base, contacts off).
- `make_collision_model.py` — generate `skt_v3_collision.xml` (box collision layer, contacts on).
- `demo_wave.py` — physics demo: arm trajectories under position control → GIF or MP4 (format follows the `--out` extension; MP4 needs `pip install imageio-ffmpeg`).
- `demo_selfcollision.py` — hands-meet demo with the collision layer revealed mid-clip → GIF or MP4.
- `telemetry_demo.py` — log sensors during the wave trajectory → tracking/torque/EE plot (+ optional CSV).
- `make_cell_scene.py` — generate `skt_v3_cell.xml`: work table, base part (60×40×25 mm, 45 g, square 22 mm pocket as v1 bore stand-in), peg (Ø20×40, 12 g), accept/reject bins.
- `primitives.py` — task-space primitives: `reach()` = closed-loop damped-least-squares IK on the 8-DoF arm chains, servoed through position actuators (physics stays honest, no qpos writes); optional gravity feed-forward (`reach(grav_ff=True)`, `mj_rne` at qvel=0) cancels the standing servo sag (30 → 1 mm, `test_gravity_ff.py`).
- `demo_cell_reach.py` — Phase 1 demo: bimanual hover → descend → lift over the parts → GIF/MP4.
- `demo_cell_pick.py` — Phase 1 demo: full bimanual pick & place (grasp → carry → place → release). The grasp is a **weld-constraint stand-in** (`primitives.grasp/release`): engaged at the part's current relative pose so nothing snaps; replaced by real gripper geometry once the hardware arrives.
- `demo_cell_assemble.py` — Phase 1 capstone: full bimanual assembly (fixture + align + force-guarded insert + place). Insertion know-how documented in the script docstring: lateral-offset grasps, orientation-locked carries (`Arm.lock_orientation` + `ik_step6`), relative servoing, τ watchdog.
- `sequencer.py` — GRAFCET-style soft-PLC engine + the demonstrator cycle S0–S7. Receptivities are sensor predicates (poses, grasp state, insertion depth, τ), never timers. QC verify is a v1 pose oracle — the camera pipeline replaces it. Cycle log → JSON (see `../logs/cycle_001.json`).
- `demo_cell_cycle.py` — run the full automatic cycle and render it with an HMI overlay (live GRAFCET step + sensor metrics). Reference cycle: 42.6 s (`../logs/cycle_001.json`).
- `qc.py` — camera QC pipeline: `measure()` renders qc_top/qc_side and returns peg presence, alignment (mm) and insertion-depth estimate; `verdict()` applies the spec thresholds; `annotate()` saves inspection images.
- `benchmark.py` — **bimanual benchmark suite**: headless, seeded, quantitative metrics over N trials for four tasks — reach · carry · peg-insert · force-regulated peg-insert (`insert_m2`) — reusing the same primitives; `--json` writes a full report (sample: `benchmark_results.json`). `test_benchmark.py` is the smoke test.
- `eval_data/` — the raw JSON written by `eval_insertion.py` / `eval_admittance.py` / `eval_gripper.py` (`--json PATH`). Every manipulation figure quoted in `../README.md` and `../docs/MANIPULATION.md` is re-derived from these files — plus `benchmark_results.json` and `../logs/*.json` — by `test_manipulation_numbers.py`, which runs in the **hardware-free** CI job (stdlib + pytest, no MuJoCo). Re-run an eval without updating the prose and CI goes red.

## QC vision lessons (all measured, not guessed)

1. **Fix the camera roll explicitly** — `zaxis="1 0 0"` leaves the roll free and
   MuJoCo picked up = −y: the whole image was rotated 90° and the vertical peg
   "pointed sideways". Use `xyaxes` to pin image right/up.
2. **Fixed inspection window (ROI)** — a specular glint on the wrist 150 px away
   matched the "yellow peg" threshold and dragged the centroid 148 mm off.
   Industrial answer: the part is always presented at the same pose, so analyze
   only a centered window.
3. **The wooden table defeats naive thresholds** — lit table RGB (247,194,138)
   sits 4 units under a G>190 "yellow" threshold. Thresholds must be validated
   against EVERY background surface, not just the part.
4. **Reference = the feature, not the part** — wrist occlusion biased the
   whole-block centroid ~7 mm; alignment is measured against the POCKET RIM
   ring around the peg (what concentricity actually means here).
5. **Present the part to the camera** — the final fix was choreography, not CV:
   the left arm grasps the block with a FRONT (−y) offset so the overhead
   camera sees the pocket unoccluded at the verify station; grasp offsets on
   both arms widened to 8 cm so the wrists clear each other at the meet point.
6. Residuals vs sim ground truth: alignment ±1.6 mm, depth ±3.4 mm (480p,
   1 px ≈ 0.66 mm overhead). Tilt needs higher resolution — explicitly v2.

## 6-DOF carry notes

`Arm.ik_step6` holds the EE orientation captured by `lock_orientation()` while
tracking position. Tuning matters: orientation must **dominate** (rot_weight
2.0, position step capped at 2 cm/cycle). Letting tilt accumulate and fixing it
later does NOT work — a 60° correction demands wrist excursions beyond the
±90° joint limits; holding from the start keeps the wrist mid-range (≤2° tilt
over a 16 cm carry, measured). The orientation error is computed with
`mju_subQuat` (local frame) and rotated to world to match `mj_jacSite`'s jacr.

## Benchmark suite

`benchmark.py` runs repeatable **bimanual** tasks headlessly under physics — the
same task-space primitives as the cockpit and the cell demos — and reports
quantitative metrics over N seeded trials (`--json` writes the full report).

```bash
python make_control_model.py   skate_teleop/skt_v3
python make_collision_model.py skate_teleop/skt_v3
python make_cell_scene.py      skate_teleop/skt_v3
python benchmark.py --model skate_teleop/skt_v3 --trials 5 --json results.json
```

| Task | What it measures | Result (5 trials, seed 0) |
|---|---|---|
| `reach` | both arms servo to random reachable target pairs | **5/5**, max EE error 0.2–0.4 mm |
| `carry` | both arms grasp an object each and carry them together (6-DoF, orientation-locked) | **5/5**, objects lifted &amp; carried ~11 cm, peg tilt 1.8° |
| `insert` | full bimanual peg-in-hole (offset grasps, carry, align, force-guarded descent) | **5/5**, depth 18.7 mm (target 18), peg tilt 1.2–1.4°, no τ-abort |
| `insert_m2` | the same staging, but the M2 **force-regulated** controller with spiral search absorbs an injected residual xy misalignment (≤2.5 mm) | **5/5**, peg-in-base 23.7 mm, peg tilt 0.7–0.9°, peak wrench 4.7–4.9 N (abort 9), no abort |

Committed report: [`benchmark_results.json`](benchmark_results.json) — regenerate
with the command above (defaults to all four tasks, seed 0).

A true weld-transfer **hand-off** is deferred: the sim grasp is a magnetic weld
stand-in, so passing one object between two welds is a hardware-era task (the
robust co-carry above stands in for the two-arm-coordination metric).
`test_benchmark.py` is the smoke test (one trial of each task).

## Workspace notes (measured)

EE site reach (fixed base): x ±0.33 m, y up to ~0.54 m forward, z −0.13…0.42 m.
Table top at z = 0.03, front edge at y = 0.38, parts at y = 0.44; IK converges
to ≤ 2.5 cm under physics (gravity sag of the kp=100 servos is the limit).

## Motion-quality lessons (paid for in debugging hours)

The first cell demo was visibly jerky. Measured causes and the fixes, in order:

1. **Goal-jump commands** — feeding the IK the final goal directly produced
   ~5 m/s EE whips at segment starts. Fix: the commanded target *glides* from
   the current EE pose to the goal on a smoothstep profile (`reach(ease=True)`).
2. **Catch-up whip in the settle phase** — tracking lag released as one violent
   step (a0 hit 9 rad/s). Fix: task-space step clamp (`Arm.max_step`).
3. **Intra-arm jams** — grandparent collision boxes (wrist_a1↔wrist_a3) overlap
   during articulation and lock the wrist. Fix: structural excludes in
   `make_collision_model.py` (intra-arm + arm↔lower-body).
4. **Table-edge geometry** — the arm is long: any straight path from the hanging
   rest pose crosses the table plane while the hand is still below the top.
   Fix: fold-elbows → raise route in joint space (`move_joints`), and the table
   front edge moved to y = 0.38 in the scene.
5. **Controller stability** — integrating IK updates on `d.ctrl` winds up;
   `qfrc_bias/kp` feedforward feeds Coriolis terms back (unstable). Final law:
   plain P on qpos + weighted DLS (distal joints de-weighted) + small null-space
   posture bias + the step clamp. Residual ~2 cm gravity sag is an accepted v1
   limitation (future: gravity-only feedforward via `mj_rne` with qvel=0).

Verified profile of the final demo: peak EE speed 0.61 m/s, peak accel ~11 m/s²,
final speed 0.000 m/s (the clip ends at rest, not mid-motion).

## Sensors

Both generated models carry 82 sensors: `qpos_<joint>`, `qvel_<joint>`,
`tau_<joint>` for all 26 joints, plus `ee_left`/`ee_right` wrist sites with
`framepos`/`framequat`. The naming is the telemetry schema for everything
downstream (dashboard, datasets, the real Skate's state stream).

## Media convention

Every milestone gets **three kinds of media**: stills (`docs/img/*.png`), a small GIF for inline README preview, and an HD MP4 (`docs/video/*.mp4`, 1280x960/30fps).

To get an *embedded video player* in the GitHub README (instead of a file link): open README.md in the GitHub web editor and **drag the .mp4 into the edit area** — GitHub uploads it to user-attachments and inserts a URL that renders as a player. Repo-stored .mp4 files only render a player on their own file page, not inside README.
