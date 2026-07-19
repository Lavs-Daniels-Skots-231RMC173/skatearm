# Manipulation core — scope & phased plan

> **Status: mostly plan.** M1's wrist force/torque sensor is now in the tree (see M1's status below); everything else here is plan, not implementation. It scopes how SkateArm's
> manipulation would evolve from today's *scripted position control + joint-torque watchdog + weld-constraint
> grasp* into genuine **contact-force manipulation** — wrist force/torque sensing, compliant control, and a real
> grasp — sim-first and honest about the hardware boundary. It exists because the portfolio review correctly
> flagged the manipulation core as the project's single largest remaining gap.

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
- **Status — foundation shipped.** The `force`+`torque` sensors are in `make_control_model.py` and validated by `sim/test_ft_sensor.py` (both wrists read a known static load to < 0.05 N / N·m; run in CI). Remaining M1: surface the wrench in the telemetry / cockpit plots and re-express the insertion guard as a wrench threshold.
- **Goal:** a true 6-axis contact wrench at each wrist, replacing the actuator-torque proxy.
- **Do:** add a MuJoCo `force`+`torque` sensor pair on a site at each wrist flange; surface the wrench in the telemetry schema and the cockpit plots; re-express the insertion guard as a *wrench* threshold.
- **Replaces:** the `tau_R()` sum-of-`actuatorfrc` watchdog → a real wrench signal.
- **Unlocks:** all of M2–M4 (you cannot regulate a force you cannot read).
- **Sim ↔ hardware:** the sim sensor is exact; the real Skate needs an actual wrist F/T sensor *or* a joint-torque-based estimator — design the M1 interface so either backs it.
- **Verify:** static-load calibration test (apply a known mass/wrench, sensor vs analytic); confirm the re-expressed guard reproduces today's insertion behaviour.
- **Risk:** MuJoCo sensor frame conventions; add a noise model so the signal stays honest for sim-to-real. **Effort:** small–medium; no new control theory.

### M2 — Force-regulated insertion  *(hybrid position–force)*
- **Goal:** regulate the axial contact force during the descent instead of pushing open-loop until a threshold trips, and search for the bore under small misalignment.
- **Do:** hybrid control — position along the approach until contact, then force control (target axial force) with position/orientation held on the other DoF; add a spiral/Lissajous hole-search; keep the wrench abort as a safety backstop.
- **Depends:** M1.
- **Sim ↔ hardware:** the controller is identical in sim and on hardware; only the wrench *source* differs.
- **Verify:** the metric that proves this earns its keep — a **misalignment-tolerance curve** (insertion success vs initial xy / θ error), replacing today's single "5/5 under nominal alignment."
- **Risk:** contact stiffness (`solref`/`solimp`) tuning so sim contact is neither mushy nor explosive. **Effort:** medium; core contact-control work.

### M3 — Cartesian compliant (admittance/impedance) arm control
- **Goal:** the arm *yields* to unexpected contact rather than latching a soft-stop — a general controller, not the 1-D dual-carry heuristic.
- **Do:** wrap the position servos in a Cartesian admittance loop at the TCP (measured wrench → commanded pose/velocity offset), tunable stiffness/damping per axis, integrated with the DLS-IK; give the cockpit selectable *compliant* vs *stop* contact modes.
- **Depends:** M1; benefits from M2.
- **Sim ↔ hardware:** admittance-on-position-servos ports cleanly to a real position-controlled arm (which the Skate is).
- **Verify:** a push-and-yield test (external wrench → TCP moves along the compliant axes at the commanded stiffness); a bimanual compliant carry that generalises the dual-carry demo.
- **Risk:** admittance stability vs the inner position-loop bandwidth (the existing gravity feed-forward helps here). **Effort:** medium; the main new control theory.

### M4 — Real grasp model  *(actuated gripper + round bore)*
- **Goal:** replace the weld stand-in with an actuated parallel-jaw gripper closing on friction contacts, with grasp-force control; swap the square pocket for the spec's round H9 bore.
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

M1 is the keystone — low-risk, and nothing regulates force without it. The smallest end-to-end slice that would actually move the review needle is **M1 + M2**: a *force-regulated* insertion reported with a *misalignment-tolerance curve*, replacing the scripted-push-plus-threshold. That single result converts "assembles when perfectly aligned" into "assembles under realistic misalignment via contact force" — which is the real manipulation claim. M3/M4 then broaden it from insertion to general compliant manipulation and a real grasp.

## 5 · What this is not

Not implemented — a plan. It keeps SkateArm's honesty rules: each phase states its sim-vs-hardware boundary, ships with a verification test that lands in the repo, and claims nothing until its metric is committed. Nothing here should be described as done, in progress, or working until the corresponding phase is in the tree with its test.

---

*Grounding: current-state facts above are from a code read of `sim/{sequencer,primitives,make_control_model,make_collision_model,make_cell_scene,demo_cell_assemble,demo_dual_carry,benchmark,qc}.py` and `tools/skate_commander/skate_commander/{bridge,grasp,kinematics}.py` at the time of writing.*
