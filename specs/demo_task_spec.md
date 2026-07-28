# Demonstrator task spec — v1

> Status: **v1 approved 11.06.2026.** Decisions 1–3 signed off (fit progression
> H9/d9 → H7/g6; PETG print first, one CNC aluminium pair later; takt ≤ 60 s,
> stretch 30 s). Decision 4 (gripper jaw vs Ø20 peg) stays open until the robot
> arrives. Sim implementation can start.

## 1. Task statement

Two-handed small-parts assembly with in-cell quality inspection:

1. Feeder (AVR node) presents a **base part**.
2. **Left arm** picks the base part and holds it in a fixturing pose at chest height.
3. **Right arm** picks the **peg** and inserts it into the base part's bore (peg-in-hole).
4. In-cell **QC station** measures the assembly → accept / reject.
5. Left arm places the assembly into the accept or reject bin; cycle logs to the dashboard.

The task is deliberately *bimanual-essential*: the base part has no flat resting
face usable for one-armed insertion, so holding is a functional requirement,
not a gimmick.

## 2. Part definition

| Item | v1 value | Rationale |
|---|---|---|
| Base part | 60×40×25 mm block, central vertical bore Ø20 H9, 3D-printed (PETG) | big enough for the Skate gripper, printable overnight |
| Peg | Ø20 d9 × 40 mm, 3D-printed, 2 mm 45° lead-in chamfer | loose running fit — sim-realistic for v1 |
| Fit class | **Ø20 H9/d9** (clearance ≈ 65–169 µm) | DECISION: v1 starts loose; v2 tightens to H7/g6 (clearance ≈ 7–41 µm) once arm repeatability is measured |
| Mass | base ≈ 45 g, peg ≈ 12 g | « 3 kg bimanual payload, ample margin |
| Material v2+ | one CNC-milled aluminium pair (links to CNC/CAD portfolio projects) | DECISION: make when fixtures are designed |

**Why printable parts first:** the bottleneck in v1 is manipulation accuracy, not
part precision; printed parts iterate in hours and the GD&T story stays honest —
we *measure* printed parts (they vary!) instead of assuming nominals.

## 3. Cycle definition

GRAFCET top level (full diagram to be drawn in the sequencer work package):

```
S0 idle ─ start ─→ S1 feeder presents base
S1 ─ base detected (camera) ─→ S2 L-arm pick base
S2 ─ grasp confirmed (torque sensor τ > threshold) ─→ S3 L-arm to fixture pose
S3 ─ pose settled (qvel < ε) ─→ S4 feeder presents peg → R-arm pick peg
S4 ─ grasp confirmed ─→ S5 R-arm insert (guarded move, force-limited)
S5 ─ insertion depth reached ─→ S6 QC measure
S5 ─ τ exceeds jam limit ─→ S8 reject path (no QC)
S6 ─ accept ─→ S7a place to accept bin    S6 ─ reject ─→ S7b place to reject bin
S7a/S7b ─→ S0
```

- **Target takt (v1 sim): ≤ 60 s/cycle**, stretch 30 s. (WPL/RTK takt methodology
  applies once the real cell layout exists.)
- Every transition condition is a sensor read (camera, qpos/qvel, τ) — no open-loop timers
  except feeder settling.

## 4. Inspection (QC)

Measured characteristics, v1 (sim camera + ground-truth assists where honest):

| Characteristic | Method (sim) | Method (real, v2+) | Accept threshold |
|---|---|---|---|
| Peg presence | camera (color/shape) | camera | present |
| Insertion depth | EE site z vs base pose | depth gauge / camera scale | ≥ 35 mm |
| Peg perpendicularity to base top face | relative quat of parts | camera + reference square | ⊥ tol 1.0 mm (GD&T callout) |
| Insertion force history | τ log during S5 | servo current log | no spike > limit |

Accept = all four pass. The GD&T callouts (⊥, position of bore) go on a proper
drawing — links to the metrology/engineering-drawing portfolio projects.

## 5. Success metrics

- **Sim v1:** ≥ 20 consecutive successful cycles; ≥ 90 % success over 100
  randomized trials (part pose jitter ±5 mm / ±5°).
- **Sim v2 (tight fit):** ≥ 70 % success over 100 trials with H7/g6.
- **Real (Phase 2+):** every motion validated joint-by-joint before autonomous
  cycles (SO-101 lesson: green RViz ≠ safe robot); then the same metrics re-run.
- All runs logged with the telemetry schema (`qpos_*`, `tau_*`, `ee_*`) → dashboard.

## 6. Safety

- **Sim-first gate:** no motion runs on hardware that hasn't passed sim.
- **Guarded moves:** insertion is force-limited (τ watchdog aborts to S8).
- Real cell (Phase 2+): e-stop chain in series with servo power, reduced-speed
  mode for any human-attended run, workspace fencing per the drilling-PLC
  project's AV.STOP pattern (relay + dump), ISO 10218/13849 category to be
  assessed in the safety work package.

## 7. DECISIONs

1. ✅ Fit class progression H9/d9 → H7/g6 — approved 11.06.2026.
2. ✅ PETG print v1, one aluminium CNC pair later — approved 11.06.2026.
3. ✅ Takt target ≤ 60 s (stretch 30 s) — approved 11.06.2026.
4. ⏳ Gripper jaw geometry vs Ø20 peg — **open**, to verify when the robot arrives; sim proceeds without a detailed gripper model until then.
   - *Annotated 28.07.2026, decision still open.* The second clause stopped
     being true at M4: the sim now carries a detailed gripper — a 90° V-groove
     parallel jaw on both wrists, its pads tangent to the Ø20 peg so the peg
     self-centres on four line contacts (`sim/make_cell_scene.py::_vee_jaw`).
     Every figure that geometry publishes — 17.4 mm tip gap at rest, 41.61 mm
     at the stop, ±11.3 mm from pad centre to the contact lines — is rebuilt
     from the emitted geoms on every push, by
     `sim/test_manipulation_numbers.py::test_jaw_geometry_matches_the_prose_that_states_it`.
     The DECISION itself does not move: whether these jaws hold the part at the
     H9/d9 fit is a measurement on the real arm, not a sim result, and the
     jaw dimensions are still a guess at what the delivered gripper will be.
