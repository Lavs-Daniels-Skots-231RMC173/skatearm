# Manipulation core — scope & phased plan

> **Status: M1 + M2 in the tree; M3–M5 plan.** M1's wrist force/torque sensor and M2's
> force-regulated insertion are implemented and tested (see their status below); M3–M5 are
> plan, not implementation. This scopes how SkateArm's manipulation evolves from the original
> *scripted position control + joint-torque watchdog + weld-constraint grasp* into genuine
> **contact-force manipulation** — wrist force/torque sensing, force-regulated and compliant
> control, and a real grasp — sim-first and honest about the hardware boundary. It exists
> because the portfolio review correctly flagged the manipulation core as the project's single
> largest remaining gap.

## 1 · Where we are today (grounded in the code)

| Concern | Today | Verdict |
|---|---|---|
| Arm control | MuJoCo `position` servos (kp = 100, joint damping 2.0, armature 0.05); DLS-IK to Cartesian targets; gravity via an open-loop `mj_rne` feed-forward | **real, but position-only** — no force feedback |
| Grasp | a pre-declared MuJoCo `weld` equality toggled active at pickup (`primitives.grasp()`); a documented "magnetic" stand-in. The gripper DOF (`a7`) exists but is excluded from IK and unused for grasping | **stand-in**, not a gripper |
| Peg-in-hole | a scripted descent (−1.4 mm / control cycle, 6-DoF IK with live xy re-centring, stop at 18 mm depth) guarded by a **joint-torque watchdog**: the sum of \|`actuatorfrc`\| over the 8 right-arm servos, tripping at baseline + 25 N·m (`sequencer.py`) | **scripted trajectory + real contact + threshold guard** |
| Contact | the work-cell scene runs on the collision model with contacts **enabled** — a real peg cylinder, a square pocket, table friction — so the peg↔pocket reaction is genuine MuJoCo physics, and that reaction is what loads the servos the watchdog reads. (The control/ACT scene disables contacts because the raw meshes self-jam at the shoulders.) | **real contact, but not force-regulated** |
| Force sensing | **none at the wrist.** No MuJoCo `touch` / `force` / `torque` sensor anywhere. "Force" is either the servo-effort sum (the watchdog) or a software 3-DoF TCP-force *estimate* from joint torques in the cockpit (`F = (J·Jᵀ)⁻¹·J·τ`, position-only) | **estimate, not measurement** |
| Compliance | **none on the arms** — pure position servos. The only admittance-like code is a 1-D load-equaliser in the dual-carry demo; the cockpit's contact reflex *latches a soft-stop* (it stops, it does not yield) | **absent** |
| Collision geometry | primitive capsules (boxes via `--boxes`) fitted per link; used for **both** self-collision **and** environment/object contact; plus a stricter pre-send kinematic guard in the cockpit | real, reused for cell contact |

**Honest one-liner:** today's *force-guarding* is a joint-torque threshold on a scripted motion, not force control; *grasping* is a weld, not a gripper. That is a legitimate v1 for a sim-first demonstrator — but it is exactly the gap between "assembles a peg when perfectly aligned in sim" and "does compliant, contact-rich assembly."

## 2 · The target

A manipulation core that **(1) measures** the contact wrench at the wrist, **(2) regulates** it — compliant / hybrid position–force control — instead of thresholding it, and **(3) grasps** with a real actuated gripper. Everything below is sim-first; each phase names its hardware boundary explicitly, and each ships with a verification metric in the repo (matching the project's existing test/eval culture).

## 3 · Phased plan

### M1 — Wrist force/torque sensing in sim  *(foundation, low-risk)*
- **Status — shipped.** The `force`+`torque` sensors are in `make_control_model.py` (inherited by the collision + cell scenes) and validated by `sim/test_ft_sensor.py` (both wrists read a known static load to < 0.05 N / N·m; run in CI). The wrench is now the control signal for M2's force-regulated insertion below — which supersedes the "re-express the guard as a threshold" idea with true force *regulation*. (Surfacing the wrench in the cockpit telemetry/plots remains an optional follow-up.)
- **Goal:** a true 6-axis contact wrench at each wrist, replacing the actuator-torque proxy.
- **Do:** add a MuJoCo `force`+`torque` sensor pair on a site at each wrist flange; surface the wrench in the telemetry schema and the cockpit plots; re-express the insertion guard as a *wrench* threshold.
- **Replaces:** the `tau_R()` sum-of-`actuatorfrc` watchdog → a real wrench signal.
- **Unlocks:** all of M2–M4 (you cannot regulate a force you cannot read).
- **Sim ↔ hardware:** the sim sensor is exact; the real Skate needs an actual wrist F/T sensor *or* a joint-torque-based estimator — design the M1 interface so either backs it.
- **Verify:** static-load calibration test (apply a known mass/wrench, sensor vs analytic); confirm the re-expressed guard reproduces today's insertion behaviour.
- **Risk:** MuJoCo sensor frame conventions; add a noise model so the signal stays honest for sim-to-real. **Effort:** small–medium; no new control theory.

### M2 — Force-regulated insertion  *(hybrid position–force)* — **SHIPPED**
- **Status — shipped.** `sim/insertion.py` (`Insertion`) regulates the axial contact force during the descent instead of pushing open-loop until a threshold trips, and searches for the bore under misalignment. Verified by `sim/test_insertion.py` (in CI) and swept by `sim/eval_insertion.py`.
- **How:** axial admittance on an accumulating z-setpoint — `z_cmd += clip(kf·(f_ax − f_target), −vz, vz)` — so the wrist descends while the measured axial force (the M1 wrist wrench) is below `f_target` (3 N) and eases off once it reaches it; the force is *regulated*, not rammed. `lead_cap` bounds how far the setpoint may lead the actual wrist so a stalled peg cannot wind up a large command. A **spiral hole-search** with a *pause-and-seat* state machine recovers misalignment: the wrist xy spirals out from the assumed centre, freezes the instant the peg starts to drop so it can seat, and resumes if it stalls. Gravity feed-forward (`mj_rne`, the `reach()` trick) holds height; a wrench-magnitude abort (`w_abort` = 9 N) is the safety backstop. The controller uses only the wrench + wrist proprioception + the assumed centre — never the live peg pose (the sim peg pose is an oracle used only to *score* the eval).
- **Depends:** M1.
- **Sim ↔ hardware:** the controller is identical in sim and on hardware; only the wrench *source* differs (a real F/T sensor or a joint-torque estimator behind the M1 interface).
- **Verify — misalignment-tolerance curve** (rigid fixture via the `fixture_base` weld, 6 directions/offset): with search, 0–4 mm **6/6**, 6 mm 5/6, 8 mm 3/6; **without search the open-loop descent jams almost everywhere (≤1/6)**. Peak axial force ~3 N (target 3, abort 9) — regulated, not rammed. This replaces the old single "5/5 under nominal alignment" with success *vs* initial xy error. A **peg-tilt (θ) tolerance** sweep (`--theta 0,3,6,9,12`) shows initial tilts up to ~9° are levelled to <2° and seated — the controller holds the target orientation upright (`relock=False`), so the 6-DoF IK rights the peg as it inserts. Reproduce: `python sim/eval_insertion.py --model .../skt_v3 --offsets 0,2,4,6,8 --dirs 6 --no-search-baseline` (and `--theta 0,3,6,9,12`).
- **Live cycle — integrated.** The controller now drives S4 of the demonstrator's GRAFCET cycle (`sim/sequencer.py`): the right arm force-regulates the insertion (peak wrench ~2.6 N) while the left keeps holding the base at the meet point (`hold_arms`); the full cycle seats, QC-accepts and places the unit, the peg staying centred in the bore through placement. So the *rendered demo and the cockpit run on M2's force control*, not the old open-loop `tau`-watchdog descent.
- **Also shipped:** a `benchmark.py` `insert_m2` task on the new controller (bimanual, scored on the base-recoil-invariant oracle), the peg-tilt (θ) tolerance above, and the spec's **round chamfered H9 bore** — `make_cell_scene.py --round-bore` generates a faceted-cylinder bore (20 facets, 10.4 mm inradius → ≈0.4 mm radial clearance on the D20 peg) with a wider lead-in mouth. The M2 controller seats in it with **no retuning**: misalignment sweep (search on) 0 mm 1/1, 2 mm 4/4, 4 mm 4/4, 6 mm 3/4, peak axial force ≤5.4 N — locked in by `test_round_bore_seats` in CI (generated to a *side* model so the square v1 pocket stays the default and the other tests' shared staging is untouched). So the earlier "small remaining" gap is closed; M2 is complete. **Risk (addressed):** contact stiffness (`solref`/`solimp`) tuning — the rigid `fixture_base` weld keeps the sweep deterministic. **Effort:** medium; core contact-control work.

### M3 — Cartesian compliant (admittance/impedance) arm control
- **Goal:** the arm *yields* to unexpected contact rather than latching a soft-stop — a general controller, not the 1-D dual-carry heuristic.
- **Do:** wrap the position servos in a Cartesian admittance loop at the TCP (measured wrench → commanded pose/velocity offset), tunable stiffness/damping per axis, integrated with the DLS-IK; give the cockpit selectable *compliant* vs *stop* contact modes.
- **Depends:** M1; benefits from M2.
- **Sim ↔ hardware:** admittance-on-position-servos ports cleanly to a real position-controlled arm (which the Skate is).
- **Verify:** a push-and-yield test (external wrench → TCP moves along the compliant axes at the commanded stiffness); a bimanual compliant carry that generalises the dual-carry demo.
- **Risk:** admittance stability vs the inner position-loop bandwidth (the existing gravity feed-forward helps here). **Effort:** medium; the main new control theory.

### M4 — Real grasp model  *(actuated gripper)*
- **Goal:** replace the weld stand-in with an actuated parallel-jaw gripper closing on friction contacts, with grasp-force control. (The spec's round chamfered H9 bore is already in the tree from M2 — `make_cell_scene.py --round-bore`, verified by `test_round_bore_seats` — so M4 is now purely the gripper.)
- **Do:** model jaws as contacting geoms driven by the (currently decorative) `a7` gripper DOF; grasp-force target via the wrench or a tendon; wire the cockpit's point-cloud grasp synthesiser (`grasp.py`, already real and unit-tested) to *execute* — which needs the IK to gain wrist **orientation** (today's IK is position-only, gripper excluded).
- **Depends:** M1 (grasp force); needs oriented IK.
- **Sim ↔ hardware:** the real Skate gripper geometry is unknown until hardware — build M4's interface to swap geometry, mirroring how the weld stand-in is already documented as replaceable.
- **Verify:** grasp a part by friction and carry it through the full cycle *without the weld*; a grasp-slip curve (payload until slip vs grasp force).
- **Risk:** friction-grasp stability in MuJoCo (notoriously stiff); jaw/part contact tuning; IK reach under orientation constraints. **Effort:** medium–large; the most sim-tuning-heavy phase.

### M5 — Hardware bring-up  *(Phase 2, on the real Skate)*
- **Goal:** the sim controllers run on the physical arm.
- **Do:** source the wrist wrench on hardware (real F/T sensor or joint-torque estimator behind the M1 interface); re-tune admittance gains to the real dynamics; calibrate; re-run the misalignment-tolerance and grasp-slip metrics on hardware.
- **Depends:** M1–M4 and the Skate arriving.
- **Honest boundary:** everything above is sim-only until then. The sim-to-real gap (contact stiffness, sensor noise, latency) is exactly what M1's noise model and the prior SO-101 real-hardware bring-up experience are meant to de-risk.

## 4 · Sequencing & the smallest useful slice

M1 was the keystone — low-risk, and nothing regulates force without it. The smallest end-to-end slice that actually moves the review needle is **M1 + M2**, and it is now **delivered**: a *force-regulated* insertion reported with a *misalignment-tolerance curve*, replacing the scripted-push-plus-threshold. That result converts "assembles when perfectly aligned" into "assembles under realistic misalignment via contact force" — the real manipulation claim. M3/M4 next broaden it from insertion to general compliant manipulation and a real grasp.

## 5 · Honesty rules

M1 and M2 are in the tree with their tests; **M3–M5 are still a plan, not implementation.** This doc keeps SkateArm's honesty rules: each phase states its sim-vs-hardware boundary, ships with a verification test that lands in the repo, and claims nothing until its metric is committed. Nothing in M3–M5 should be described as done, in progress, or working until the corresponding phase is in the tree with its test — and everything here is sim-only until the M5 hardware bring-up.

---

*Grounding: current-state facts above are from a code read of `sim/{sequencer,primitives,make_control_model,make_collision_model,make_cell_scene,demo_cell_assemble,demo_dual_carry,benchmark,qc}.py` and `tools/skate_commander/skate_commander/{bridge,grasp,kinematics}.py` at the time of writing.*
