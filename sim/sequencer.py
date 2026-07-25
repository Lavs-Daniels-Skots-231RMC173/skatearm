"""GRAFCET-style soft-PLC sequencer for the SkateArm demonstrator cycle.

Engine: ordered steps; each step runs its ACTION to completion, then waits for
its RECEPTIVITY (a sensor predicate — never a timer, per the task spec) before
the marked transition fires. A guard violation during a guarded action (tau
watchdog) diverts to the reject branch. Every transition is logged with sim
time and telemetry — the log is the seed of the SCADA dashboard.

v1 QC note: the VERIFY step reads part poses directly from the simulator (an
"oracle"); on the real cell this is the camera + metrology station's job.

GRIPPER PATH (opt-in, detected — never configured). When the loaded scene
carries M4's actuated jaws (a ``grip`` actuator, i.e. a model built with
``make_cell_scene.py --gripper``), the RIGHT hand's ``grasp_right`` weld is
replaced by a real force-servoed grip: S1 closes the jaws on the peg under
force control and S4 releases by opening them, not by dropping an equality.
The LEFT hand keeps its weld — there is only one gripper on the robot, on the
right wrist. A model without a ``grip`` actuator runs exactly the code it
always did, so every existing demo, test and log stays on the weld path.
"""
import json
import time

import mujoco
import numpy as np

from primitives import reach, hold, move_joints, grasp, release, Arm, smoothstep
from insertion import Insertion

# --- gripper path constants -------------------------------------------------
GRIP_OPEN = -8.0          # motor command that parks the jaws wide open
GRIP_TARGET_N = 12.0      # closing force regulated on the pad touch sensor
GRIP_MIN_N = 1.0          # "this hand is holding something" receptivity
GRIP_REL = 0.019          # pad-centre height above the peg centre at first close
# Re-taught LEFT approach: park the wrist 140 mm behind the base instead of 80.
# The right tool is 45 mm of jaw plus 15 mm of pad longer than the bare wrist,
# and from the old point the left wrist sits inside the corridor the right one
# has to descend through — measured 4.5 mm of clearance and a
# wrist_a3_Mirror__1|base_part interpenetration during S3. Clearance is bought
# HERE and not by raising the meet point, because the meet point is the QC
# station's calibration pose (see MEET_BASE).
LEFT_DY = -0.06
# Where the BASE PART meets — not the wrist. 0.125 is the repo's fixture pose,
# the one both QC cameras were placed and calibrated around: qc_side sits at
# z 0.13 with fovy 38 and a 0.32 m standoff, and qc.measure() restricts the
# analysis to a centred 300 px ROI = a window from z 0.084 to z 0.176. A unit
# presented higher has its block top outside the inspection window, the segment
# test fails, depth_mm_est comes back None and every part rejects.
MEET_BASE = np.array([0.005, 0.412, 0.125])
R_PARK = [0.22, 0.36, 0.30]           # right wrist parking pose for QC + place
HOME_L = [-0.18, 0.36, 0.26]          # left wrist home
R_DOWN = np.array([[1.0, 0.0, 0.0],   # tool-down wrist orientation for the jaws
                   [0.0, 0.0, 1.0],
                   [0.0, -1.0, 0.0]])
# The gripper-side retune of M2's seat latch. All four are already Insertion
# constructor parameters, so this is a CALL-SITE change, not a change to the
# controller. The shipped defaults (deep_gate 0.040, seat_depth 0.013) cannot
# latch on this path: S3 stages the peg 12 mm above a pocket whose bore is
# 20 mm deep, so the wrist travels 34 mm before the peg even bottoms out.
GRIP_LATCH = dict(f_contact=0.5, seat_depth=0.006, f_target=2.0, deep_gate=0.048)


class Cell:
    """Wraps model/data + sensor predicates for the demonstrator cell."""

    def __init__(self, m, d, on_frame=None, qc_renderer=None):
        self.m, self.d = m, d
        self.on_frame = on_frame
        self.qc_renderer = qc_renderer   # mujoco.Renderer for the QC cameras
        self.armL = Arm(m, d, "left")
        self.armR = Arm(m, d, "right")
        self.bp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_part")
        self.pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.tau_ids = [m.sensor_adr[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SENSOR, f"tau_a{k}_armR_a{16+k}")] for k in range(8)]
        self.log = []
        self.t0 = 0.0
        # M4's jaws, if this scene was built with `make_cell_scene.py --gripper`.
        # No `grip` actuator = the weld cell every other demo and test loads.
        self.grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip")
        self.jaws = self.grip >= 0
        self._f_adr = self._sensor_adr("grip_force")
        self._j_adr = self._sensor_adr("jaw")
        # gripper-path state, carried between steps so run_cycle(steps=[...])
        # can still be called in chunks for chunked rendering
        self.lhold = None        # left wrist pose while it holds the base
        self.offl = None         # base_part position minus left wrist position
        self.gl_hold = None      # left wrist target held through S2..S6
        self.grip_cmd = GRIP_OPEN

    # --- sensors / predicates (no timers!) ---
    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return self.m.sensor_adr[sid] if sid >= 0 else -1

    def grip_force(self):
        """Normal force on the right jaw's pad (N). 0.0 on the weld cell."""
        return float(self.d.sensordata[self._f_adr]) if self._f_adr >= 0 else 0.0

    def jaw_mm(self):
        """Jaw travel from rest (mm; negative = open). NaN on the weld cell."""
        return (float(self.d.sensordata[self._j_adr]) * 1000
                if self._j_adr >= 0 else float("nan"))

    def sim_t(self):
        return self.d.time - self.t0

    def tau_R(self):
        return float(sum(abs(self.d.sensordata[a]) for a in self.tau_ids))

    def part_pose(self, body):
        b = self.bp if body == "base" else self.pg
        return self.d.xpos[b].copy()

    def tilt_deg(self, body):
        b = self.bp if body == "base" else self.pg
        z = self.d.xmat[b].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(min(1.0, z[2]))))

    def parts_on_table(self):
        return abs(self.part_pose("base")[2] - 0.030) < 0.01 and \
               abs(self.part_pose("peg")[2] - 0.050) < 0.012

    def grasped(self, side):
        """Receptivity: is this hand actually holding its part?

        With real jaws the RIGHT hand reads the pad force sensor — an equality
        flag would only say "commanded", the sensor says "gripped". The LEFT
        hand always reads its weld: there is one gripper on the robot and it is
        on the right wrist."""
        if side == "right" and self.jaws:
            return self.grip_force() > GRIP_MIN_N
        eq = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY,
                               {"left": "grasp_left", "right": "grasp_right"}[side])
        return bool(self.d.eq_active[eq])

    def pocket_top(self):
        return self.part_pose("base") + np.array([0, 0, 0.027])

    def peg_bottom(self):
        return self.part_pose("peg") + np.array([0, 0, -0.020])

    def insertion_depth(self):
        return float(self.pocket_top()[2] - self.peg_bottom()[2])

    def align_err_xy(self):
        return float(np.linalg.norm((self.pocket_top() - self.part_pose("peg"))[:2]))

    # --- logging ---
    def event(self, step, msg, **data):
        self.log.append({"t": round(self.sim_t(), 3), "step": step, "msg": msg,
                         **{k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in data.items()}})


def run_cycle(cell, steps=None, state=None):
    """Run the demonstrator GRAFCET cycle (optionally a subset of steps,
    for chunked rendering). Returns the final state string."""
    m, d = cell.m, cell.d
    on_frame = cell.on_frame
    armL, armR = cell.armL, cell.armR
    ALL = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    steps = steps or ALL

    def servo6_both(tL, tR, seconds, tol=0.010):
        sL, sR = armL.ee_pos(), armR.ee_pos()
        gL, gR = np.asarray(tL, float), np.asarray(tR, float)
        n = int(seconds / (m.opt.timestep * 4))
        for i in range(n + 400):
            ss = min(1, (i + 1) / n)
            ss = ss * ss * (3 - 2 * ss)
            qL, _ = armL.ik_step6(sL + (gL - sL) * ss)
            armL.set_ctrl(qL)
            qR, _ = armR.ik_step6(sR + (gR - sR) * ss)
            armR.set_ctrl(qR)
            for _ in range(4):
                mujoco.mj_step(m, d)
            if on_frame:
                on_frame()
            if i >= n and np.linalg.norm(gL - armL.ee_pos()) < tol \
                    and np.linalg.norm(gR - armR.ee_pos()) < tol:
                break

    def servo6_one(arm, goal, seconds, tol=0.010):
        s0 = arm.ee_pos()
        g = np.asarray(goal, float)
        n = int(seconds / (m.opt.timestep * 4))
        for i in range(n + 300):
            ss = min(1, (i + 1) / n)
            ss = ss * ss * (3 - 2 * ss)
            q, _ = arm.ik_step6(s0 + (g - s0) * ss)
            arm.set_ctrl(q)
            for _ in range(4):
                mujoco.mj_step(m, d)
            if on_frame:
                on_frame()
            if i >= n and np.linalg.norm(g - arm.ee_pos()) < tol:
                break

    # ----- gripper path: gravity-compensated stepping ----------------------
    # ik_step6 rebases its command on qpos, so an arm driven by it sags under
    # load unless the mj_rne gravity term is fed forward (primitives.reach does
    # exactly this behind grav_ff=True). On the weld path the sag is
    # survivable. Here the LEFT arm carries the assembled unit past the QC
    # cameras and its standing sag moves the inspection subject, so both arms
    # stay driven and compensated for the whole gripper cycle.
    gmask = np.zeros(m.nv)
    for _arm in (armL, armR):
        for _i in _arm.vadr:
            if m.jnt_type[m.dof_jntid[_i]] == mujoco.mjtJoint.mjJNT_HINGE:
                gmask[_i] = 1.0
    gff = np.zeros(m.nv)

    def step_both(tR, gcmd, tL=None, n=4):
        """One control cycle: both wrists + the grip command, gravity fed forward."""
        armR.set_ctrl(armR.ik_step6(np.asarray(tR, float))[0])
        if tL is not None:
            armL.set_ctrl(armL.ik_step6(np.asarray(tL, float))[0])
        if cell.grip >= 0:
            d.ctrl[cell.grip] = gcmd
        qv = d.qvel.copy()
        d.qvel[:] = 0.0                       # qvel=0 -> gravity only, no Coriolis
        mujoco.mj_rne(m, d, 0, gff)
        d.qvel[:] = qv
        for _ in range(n):
            d.qfrc_applied[:] = gff * gmask
            mujoco.mj_step(m, d)
        d.qfrc_applied[:] = 0.0               # never leak it into a force-guarded descent
        if on_frame:
            on_frame()

    def servo_both(tR, tL, gcmd, seconds, tol=0.010):
        """Eased Cartesian move of BOTH wrists, gravity-compensated throughout."""
        sR, sL = armR.ee_pos().copy(), armL.ee_pos().copy()
        gR, gL = np.asarray(tR, float), np.asarray(tL, float)
        n = int(seconds / (m.opt.timestep * 4))
        for i in range(n + 400):
            s = smoothstep((i + 1) / n)
            step_both(sR + (gR - sR) * s, gcmd, sL + (gL - sL) * s)
            if i >= n and np.linalg.norm(gR - armR.ee_pos()) < tol \
                    and np.linalg.norm(gL - armL.ee_pos()) < tol:
                break

    # ----- S0: idle / home -----
    if "S0" in steps:
        cell.event("S0", "cycle start")
        if cell.jaws:
            # The V's tip gap at rest is 17.4 mm — narrower than the D20 peg.
            # The jaws have to be commanded open before the arm goes anywhere
            # near it, or the pads arrive already touching the part.
            d.ctrl[cell.grip] = GRIP_OPEN
        move_joints(m, d, {"a1": 0.3, "a3": 2.2}, seconds=1.8, on_frame=on_frame)
        move_joints(m, d, {"a0": 0.9, "a1": 0.3, "a3": 1.3}, seconds=1.8, on_frame=on_frame)
        # receptivity: parts detected on the table (camera's job on the real cell)
        assert cell.parts_on_table(), "S0->S1 receptivity failed: parts not on table"
        cell.event("S0", "parts detected on table -> S1",
                   base=list(cell.part_pose("base")), peg=list(cell.part_pose("peg")))

    # ----- S1: approach + grasp both (lateral offsets) -----
    if "S1" in steps:
        cell.event("S1", "approach")
        if cell.jaws:
            # --- S1a: the LEFT hand takes the base ------------------------
            peg0 = cell.part_pose("peg")
            lap = np.array([-0.12, 0.36 + LEFT_DY, 0.115])
            d.ctrl[cell.grip] = GRIP_OPEN
            reach(m, d, {"left": list(lap + [0, 0, 0.085]), "right": [0.20, 0.44, 0.24]},
                  seconds=2.4, on_frame=on_frame, tol=0.012, grav_ff=True)
            reach(m, d, {"left": list(lap), "right": [0.20, 0.44, 0.24]},
                  seconds=2.0, on_frame=on_frame, tol=0.010, grav_ff=True)
            grasp(m, d, "left")
            hold(m, d, 0.4, on_frame=on_frame)
            armL.lock_orientation()
            cell.lhold = armL.ee_pos().copy()
            # MEASURE the grasp offset instead of assuming the old 85 mm: every
            # base-frame target downstream is converted through this.
            cell.offl = cell.part_pose("base") - cell.lhold
            cell.event("S1", "left holds the base",
                       offset_mm=[round(float(v) * 1000, 1) for v in cell.offl])

            # --- S1b: the RIGHT hand grips the peg for real ----------------
            from make_gripper_cell import REACH, PAD_Y     # M4's jaw geometry

            def pad_centre():
                R = d.site_xmat[armR.site].reshape(3, 3)
                return armR.ee_pos() + R @ np.array([0.0, REACH + PAD_Y, 0.0])

            def tool_tilt():
                ax = d.site_xmat[armR.site].reshape(3, 3)[:, 1]
                return float(np.degrees(np.arccos(np.clip(-ax[2], -1, 1))))

            reach(m, d, {"left": list(cell.lhold),
                         "right": [peg0[0], peg0[1],
                                   peg0[2] + GRIP_REL + REACH + PAD_Y + 0.085]},
                  seconds=2.6, on_frame=on_frame, tol=0.012, grav_ff=True)
            # point the tool straight down: the V-groove only self-centres on a
            # peg whose axis lies along the groove
            armR.lock_orientation()
            qd = np.zeros(4)
            mujoco.mju_mat2Quat(qd, R_DOWN.flatten())
            armR.q_lock = qd
            tgt = armR.ee_pos().copy()
            for _ in range(600):
                step_both(tgt, GRIP_OPEN, cell.lhold)
                if tool_tilt() < 0.4:
                    break
            # servo the PAD CENTRE onto the peg, not the wrist: the tool is
            # REACH + PAD_Y long, and it is the pads that have to straddle it
            goal = peg0 + np.array([0.0, 0.0, GRIP_REL])
            for _ in range(1500):
                err = goal - pad_centre()
                step_both(armR.ee_pos() + err, GRIP_OPEN, cell.lhold)
                if np.linalg.norm(err) < 0.0008 and tool_tilt() < 0.5:
                    break
            # close under force control: creep the command up until the pads
            # touch, then regulate the pad sensor to GRIP_TARGET_N with a
            # rate-limited command (a step command bounces the peg out)
            rt = armR.ee_pos().copy()
            cmd = 0.0
            while cell.grip_force() < 0.05 and cmd < 6.0:
                cmd = min(6.0, cmd + 0.25)
                step_both(rt, cmd, cell.lhold, n=8)
            for _ in range(400):
                want = np.clip(cmd + 1.5 * (GRIP_TARGET_N - cell.grip_force()), 0.0, 60.0)
                cmd = float(np.clip(want, cmd - 0.5, cmd + 0.5))
                step_both(rt, cmd, cell.lhold)
            armR.lock_orientation()
            cell.grip_cmd = cmd
            cell.event("S1", "right jaws closed on the peg",
                       grip_force_n=cell.grip_force(), jaw_mm=cell.jaw_mm(),
                       tool_tilt_deg=tool_tilt())
        else:
            reach(m, d, {"left": [-0.12, 0.36, 0.20], "right": [0.20, 0.44, 0.20]},
                  seconds=2.4, on_frame=on_frame, tol=0.012)
            reach(m, d, {"left": [-0.12, 0.36, 0.115], "right": [0.20, 0.44, 0.115]},
                  seconds=2.0, on_frame=on_frame, tol=0.010)
            grasp(m, d, "left")
            grasp(m, d, "right")
            hold(m, d, 0.4, on_frame=on_frame)
            armL.lock_orientation()
            armR.lock_orientation()
        # receptivity: both grasps engaged (the right one reads the pad force
        # sensor when the model has real jaws — see Cell.grasped)
        assert cell.grasped("left") and cell.grasped("right")
        cell.event("S1", "grasps confirmed -> S2")

    # ----- S2: carry to meet point -----
    if "S2" in steps:
        cell.event("S2", "carry to fixture/staging")
        if cell.jaws:
            # The target is expressed on the BASE PART and converted through the
            # measured grasp offset. The weld path's [0.0, 0.33, 0.21] is the
            # same point baked through the OLD offset; re-teaching the approach
            # changes that offset, so the number has to be derived, not copied.
            cell.gl_hold = MEET_BASE - cell.offl
            gr = np.array([0.08, 0.41, 0.30])
            servo_both(gr, cell.gl_hold, cell.grip_cmd, seconds=4.5)
            for i in range(1200):        # settle until the carried unit is still
                lp, rp = armL.ee_pos().copy(), armR.ee_pos().copy()
                step_both(gr, cell.grip_cmd, cell.gl_hold)
                sp = max(np.linalg.norm(armL.ee_pos() - lp),
                         np.linalg.norm(armR.ee_pos() - rp)) / (4 * m.opt.timestep)
                if i > 40 and sp < 0.002 \
                        and np.linalg.norm(cell.gl_hold - armL.ee_pos()) < 0.004 \
                        and np.linalg.norm(gr - armR.ee_pos()) < 0.004:
                    break
        else:
            servo6_both([0.0, 0.33, 0.21], [0.08, 0.41, 0.30], seconds=4.5)
        cell.event("S2", "at meet point -> S3",
                   block_tilt=cell.tilt_deg("base"), peg_tilt=cell.tilt_deg("peg"))

    # ----- S3: align peg over pocket (relative servoing) -----
    if "S3" in steps:
        cell.event("S3", "align", err_xy=cell.align_err_xy())
        if cell.jaws:
            # Same relative servo, but the peg is held by friction now, so the
            # convergence test has to HOLD: a single in-tolerance sample can be
            # the peg swinging through zero rather than settled on it.
            ok_k = 0
            for _ in range(4000):
                err_xy = (cell.pocket_top() - cell.part_pose("peg"))[:2]
                zerr = (cell.pocket_top()[2] + 0.012) - cell.peg_bottom()[2]
                step_both(armR.ee_pos() + np.array([err_xy[0], err_xy[1], zerr]),
                          cell.grip_cmd, cell.gl_hold)
                ok_k = ok_k + 1 if (np.linalg.norm(err_xy) < 0.0035
                                    and abs(zerr) < 0.007) else 0
                if ok_k >= 60:
                    break
        else:
            for _ in range(400):
                err_xy = (cell.pocket_top() - cell.part_pose("peg"))[:2]
                # stage the peg ~12mm above the opening — the M2 force-regulated
                # insertion (S4) does the final descent, so hand it a short approach
                # gap (its deep_gate is 40mm; the old open-loop descent started 30mm up)
                zerr = (cell.pocket_top()[2] + 0.012) - cell.peg_bottom()[2]
                q, _ = armR.ik_step6(armR.ee_pos() + np.array([err_xy[0], err_xy[1], zerr]))
                armR.set_ctrl(q)
                qL, _ = armL.ik_step6(np.array([0.0, 0.33, 0.21]))
                armL.set_ctrl(qL)
                for _ in range(4):
                    mujoco.mj_step(m, d)
                if on_frame:
                    on_frame()
                if np.linalg.norm(err_xy) < 0.0035 and abs(zerr) < 0.007:
                    break
        cell.event("S3", "aligned -> S4", err_xy=cell.align_err_xy())

    # ----- S4: force-regulated insertion (M2) -----
    if "S4" in steps:
        # the M2 controller regulates the wrist contact force and spiral-searches
        # the bore, replacing the open-loop "1.4mm/cycle + tau watchdog" descent.
        # It drives the right arm; the left keeps holding the base at the meet
        # point (hold_arms) so the base stays put under the insertion reaction.
        center = armR.ee_pos()[:2].copy()          # wrist xy aligning the peg over the bore
        cell.event("S4", "insert (force-regulated)",
                   assumed_center=[round(float(v), 4) for v in center])
        if cell.jaws:
            d.ctrl[cell.grip] = cell.grip_cmd      # Insertion drives the arm, not the grip
        res = Insertion(m, d, armR, center,
                        hold_arms=[(armL, list(cell.gl_hold) if cell.jaws
                                    else [0.0, 0.33, 0.21])],
                        on_step=on_frame,
                        **(GRIP_LATCH if cell.jaws else {})).run(search=True)
        # `seated` is not a nicety on the gripper path, it is the ONLY detector
        # for a stalled insert: once the jaws open, a peg left proud of the bore
        # settles the rest of the way under its own weight, so by the time the
        # QC camera looks at it the failure has healed itself (measured).
        aborted = res["aborted"] or not res["seated"]
        # Release the RIGHT tool. This is the one place the two attachments
        # differ irreducibly: the weld drops an equality, the jaws have to be
        # commanded open and the pads have to physically leave the peg.
        if cell.jaws:
            rhold = armR.ee_pos().copy()
            for _ in range(120):
                step_both(rhold, GRIP_OPEN, cell.gl_hold)
            cell.grip_cmd = GRIP_OPEN
        else:
            release(m, d, "right")
            hold(m, d, 0.4, on_frame=on_frame)
        cell.event("S4", "insert done -> S5" if not aborted else "FORCE GUARD -> reject",
                   depth_mm=round(cell.insertion_depth() * 1000, 2), aborted=aborted,
                   seated=res["seated"], peak_wrench_n=res["peak_wrench_n"],
                   cycles=res["cycles"],
                   **({"jaw_mm": round(cell.jaw_mm(), 2)} if cell.jaws else {}))
        cell.qc_jam = aborted

    # ----- S5: retreat right + CAMERA QC verify (oracle kept as cross-check) -----
    if "S5" in steps:
        if cell.jaws:
            # `reach` would drive BOTH arms off their held targets; the left one is
            # the only thing holding the base up now that its weld carries the whole
            # assembly, so the retreat has to be a two-arm servo that keeps the left
            # wrist pinned at gl_hold. The 150 extra cycles let the peg stop ringing
            # before the camera looks at it -- QC measures pixels, not intent.
            servo_both(R_PARK, cell.gl_hold, cell.grip_cmd, seconds=2.2, tol=0.015)
            for _ in range(150):
                step_both(R_PARK, cell.grip_cmd, cell.gl_hold)
        else:
            reach(m, d, {"right": [0.22, 0.36, 0.30]}, seconds=2.2,
                  on_frame=on_frame, tol=0.015)
        import qc as qc_mod
        meas = None
        if getattr(cell, "qc_renderer", None) is not None:
            meas = qc_mod.measure(cell.qc_renderer, d,
                                  unit_z=float(cell.part_pose("base")[2]))
            cam_verdict = qc_mod.verdict(meas)
        # oracle cross-check (sim ground truth; logged for residual tracking)
        depth = cell.insertion_depth()
        tilt = cell.tilt_deg("peg")
        err_xy = cell.align_err_xy()
        oracle_pass = (depth >= 0.015) and (tilt < 6.0) and (err_xy < 0.006) \
            and not getattr(cell, "qc_jam", False)
        if meas is not None:
            cell.qc_pass = (cam_verdict == "ACCEPT") and not getattr(cell, "qc_jam", False)
            cell.qc_meas = meas
            cell.event("S5", "CAMERA QC verify",
                       cam_align_mm=meas["align_err_mm"], cam_depth_mm=meas["depth_mm_est"],
                       cam_peg_present=meas["peg_present"], cam_result=cam_verdict,
                       oracle_depth_mm=depth * 1000, oracle_align_mm=err_xy * 1000,
                       oracle_tilt_deg=tilt,
                       residual_align_mm=(abs(meas["align_err_mm"] - err_xy * 1000)
                                          if meas["align_err_mm"] is not None else None),
                       residual_depth_mm=(abs(meas["depth_mm_est"] - depth * 1000)
                                          if meas["depth_mm_est"] is not None else None))
        else:
            cell.qc_pass = oracle_pass
            cell.event("S5", "QC verify (oracle only — no renderer attached)",
                       depth_mm=depth * 1000, tilt_deg=tilt, err_xy_mm=err_xy * 1000,
                       result="ACCEPT" if oracle_pass else "REJECT")

    # ----- S6: place assembled unit to the accept/reject bin -----
    if "S6" in steps:
        bin_x = -0.24 if getattr(cell, "qc_pass", True) else 0.24
        target_wrist = [bin_x, 0.33, 0.20]
        cell.event("S6", f"place to {'ACCEPT' if bin_x < 0 else 'REJECT'} bin")
        if cell.jaws:
            # These targets are written as LEFT WRIST poses, which silently bakes in
            # the old 85 mm grasp offset. Re-teaching the approach point changes that
            # offset, so on the gripper path the bin targets are expressed where they
            # actually mean something -- on the BASE -- and converted through the
            # measured cell.offl, exactly as S2 does.
            servo_both(R_PARK, np.array([bin_x, 0.41, 0.115]) - cell.offl,
                       cell.grip_cmd, seconds=3.0, tol=0.012)
            servo_both(R_PARK, np.array([bin_x, 0.41, 0.033]) - cell.offl,
                       cell.grip_cmd, seconds=2.0, tol=0.008)
            release(m, d, "left")
            lpark = armL.ee_pos().copy()
            for _ in range(150):
                step_both(R_PARK, cell.grip_cmd, lpark)
        else:
            servo6_one(armL, target_wrist, seconds=3.0)
            servo6_one(armL, [bin_x, 0.33, 0.118], seconds=2.0, tol=0.008)
            release(m, d, "left")
            hold(m, d, 0.5, on_frame=on_frame)
        cell.event("S6", "released on bin",
                   unit_at=list(cell.part_pose("base")), peg_rel_z=float(
                       (cell.part_pose("peg") - cell.part_pose("base"))[2]))

    # ----- S7: retreat home -----
    if "S7" in steps:
        if cell.jaws:
            # Same reason as S5: the right wrist still carries the open jaws and must
            # be commanded, not left to `reach`'s single-arm path, or it drifts down
            # into the bin the left hand has just loaded.
            servo_both(R_PARK, HOME_L, cell.grip_cmd, seconds=2.0, tol=0.015)
            for _ in range(250):
                step_both(R_PARK, cell.grip_cmd, HOME_L)
        else:
            reach(m, d, {"left": [-0.18, 0.36, 0.26]}, seconds=2.0,
                  on_frame=on_frame, tol=0.015)
            hold(m, d, 1.0, on_frame=on_frame)
        cell.event("S7", "cycle complete", cycle_time_s=cell.sim_t())

    return cell.log
