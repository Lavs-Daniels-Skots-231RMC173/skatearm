"""GRAFCET-style soft-PLC sequencer for the SkateArm demonstrator cycle.

Engine: ordered steps; each step runs its ACTION to completion, then waits for
its RECEPTIVITY (a sensor predicate — never a timer, per the task spec) before
the marked transition fires. A guard violation during a guarded action (tau
watchdog) diverts to the reject branch. Every transition is logged with sim
time and telemetry — the log is the seed of the SCADA dashboard.

v1 QC note: the VERIFY step reads part poses directly from the simulator (an
"oracle"); on the real cell this is the camera + metrology station's job.
"""
import json
import time

import mujoco
import numpy as np

from primitives import reach, hold, move_joints, grasp, release, Arm
from insertion import Insertion


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

    # --- sensors / predicates (no timers!) ---
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

    # ----- S0: idle / home -----
    if "S0" in steps:
        cell.event("S0", "cycle start")
        move_joints(m, d, {"a1": 0.3, "a3": 2.2}, seconds=1.8, on_frame=on_frame)
        move_joints(m, d, {"a0": 0.9, "a1": 0.3, "a3": 1.3}, seconds=1.8, on_frame=on_frame)
        # receptivity: parts detected on the table (camera's job on the real cell)
        assert cell.parts_on_table(), "S0->S1 receptivity failed: parts not on table"
        cell.event("S0", "parts detected on table -> S1",
                   base=list(cell.part_pose("base")), peg=list(cell.part_pose("peg")))

    # ----- S1: approach + grasp both (lateral offsets) -----
    if "S1" in steps:
        cell.event("S1", "approach")
        reach(m, d, {"left": [-0.12, 0.36, 0.20], "right": [0.20, 0.44, 0.20]},
              seconds=2.4, on_frame=on_frame, tol=0.012)
        reach(m, d, {"left": [-0.12, 0.36, 0.115], "right": [0.20, 0.44, 0.115]},
              seconds=2.0, on_frame=on_frame, tol=0.010)
        grasp(m, d, "left")
        grasp(m, d, "right")
        hold(m, d, 0.4, on_frame=on_frame)
        armL.lock_orientation()
        armR.lock_orientation()
        # receptivity: both grasps engaged
        assert cell.grasped("left") and cell.grasped("right")
        cell.event("S1", "grasps confirmed -> S2")

    # ----- S2: carry to meet point -----
    if "S2" in steps:
        cell.event("S2", "carry to fixture/staging")
        servo6_both([0.0, 0.33, 0.21], [0.08, 0.41, 0.30], seconds=4.5)
        cell.event("S2", "at meet point -> S3",
                   block_tilt=cell.tilt_deg("base"), peg_tilt=cell.tilt_deg("peg"))

    # ----- S3: align peg over pocket (relative servoing) -----
    if "S3" in steps:
        cell.event("S3", "align", err_xy=cell.align_err_xy())
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
        res = Insertion(m, d, armR, center,
                        hold_arms=[(armL, [0.0, 0.33, 0.21])],
                        on_step=on_frame).run(search=True)
        aborted = res["aborted"] or not res["seated"]
        release(m, d, "right")
        hold(m, d, 0.4, on_frame=on_frame)
        cell.event("S4", "insert done -> S5" if not aborted else "FORCE GUARD -> reject",
                   depth_mm=round(cell.insertion_depth() * 1000, 2), aborted=aborted,
                   seated=res["seated"], peak_wrench_n=res["peak_wrench_n"],
                   cycles=res["cycles"])
        cell.qc_jam = aborted

    # ----- S5: retreat right + CAMERA QC verify (oracle kept as cross-check) -----
    if "S5" in steps:
        reach(m, d, {"right": [0.22, 0.36, 0.30]}, seconds=2.2, on_frame=on_frame, tol=0.015)
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
        servo6_one(armL, target_wrist, seconds=3.0)
        servo6_one(armL, [bin_x, 0.33, 0.118], seconds=2.0, tol=0.008)
        release(m, d, "left")
        hold(m, d, 0.5, on_frame=on_frame)
        cell.event("S6", "released on bin",
                   unit_at=list(cell.part_pose("base")), peg_rel_z=float(
                       (cell.part_pose("peg") - cell.part_pose("base"))[2]))

    # ----- S7: retreat home -----
    if "S7" in steps:
        reach(m, d, {"left": [-0.18, 0.36, 0.26]}, seconds=2.0, on_frame=on_frame, tol=0.015)
        hold(m, d, 1.0, on_frame=on_frame)
        cell.event("S7", "cycle complete", cycle_time_s=cell.sim_t())

    return cell.log
