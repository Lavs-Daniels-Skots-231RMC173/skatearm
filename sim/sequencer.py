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
``make_cell_scene.py --gripper``), BOTH welds are replaced by real
force-servoed grips: each hand closes on its part under force control and
releases by opening, and each hand's receptivity is its own pad force sensor.
The cell is weld-free — ``grasp_left`` and ``grasp_right`` are still declared,
so the same model file still runs the weld path as an A/B control, but the
gripper path activates neither. A model without a ``grip`` actuator runs
exactly the code it always did, so every existing demo, test and log stays on
the weld path.

The second hand is not symmetry for its own sake, and it changes the CYCLE.
The left tool is 60 mm longer than the bare wrist it replaces, and the base is
60 x 40 x 25 mm with an open pocket, so the jaws can only take it across the
40 mm faces with the approach axis pointing straight down (60 mm exceeds the
41.61 mm tip gap measured below, and the 60 x 40 faces are the table and the
pocket mouth).
That pins the left tool pose, and with it pinned the left wrist and the right
hand's pads OVERLAP by 11.4 mm at the best grasp available anywhere on the
part — measured, and invariant to where the two hands meet, because both tools
are then rigidly tied to the base. Two hands cannot hold this base in mid-air
together. So the gripper cycle does what a real cell does when a part is too
small to hand off: the left hand PLACES the base at the assembly station and
lets go (S2), the right inserts into it standing on the table (S3/S4), and the
left comes back for the ASSEMBLED unit and presents it to the QC cameras at
the same fixture pose they were calibrated around (S5). The weld path keeps
its mid-air meet, which is what a weld can do and a gripper cannot.

That detour is also what buys the weld-free path its own camera gate. Between
the right hand backing out and the left hand coming back, the finished unit
stands FREE at the assembly station and no hand is on it -- an instant a welded
cell structurally cannot have, since there the unit leaves a hand only onto a
weld. S5 reads a SECOND CAMERA PAIR at that instant (``qc.STATION_PAIR``:
``qc_top``, which sees both places, plus ``qc_station_side`` aimed at the
station), and that reading is what gates the weld-free cycle. The fixture pair
is still read afterwards at the presented pose and still logged -- occluded, as
published -- so the finding that cost the welds their camera gate stays visible
in the same event stream that now shows it repaid.
"""
import json
import time
from collections import deque

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
# The LEFT tool's pose, and it is FORCED, not chosen. Columns are the wrist
# site's local axes in world: the approach axis (local +y, the direction the
# jaws reach) must point straight down because the base's pocket faces up and
# its 60 x 40 footprint is the table; the closing axis (local x) then has to
# take the part across its 40 mm faces, because 60 mm exceeds the jaws' tip gap;
# which leaves the V-groove running along the base's 60 mm length.
# That tip gap is 41.61 mm MEASURED, and the geometry alone gives 41.37 mm: the
# plate tips are 17.37 mm apart at rest and the jaw's slide range ends at -12 mm.
# The extra 0.24 mm is the stop being soft-limited rather than rigid -- GRIP_OPEN
# parks the jaw 0.12 mm past it, on each of the two -- so 41.61 mm is the
# mechanical maximum and no larger GRIP_OPEN buys any more. Across a 40.0 mm base
# that is 0.80 mm per side, which is what forces the align-then-descend approach
# in centre_pads.
R_DOWN_L = np.array([[0.0, 0.0, 1.0],
                     [-1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0]])
# Pad-centre height above the base's BOTTOM face at the close. The pads are
# 32 mm tall and the side wall they clamp is 20 mm (z 5..25 above the bottom),
# so any centre in 16..21 mm engages the whole wall; 17 mm is the lowest of
# those that still leaves the pad 1 mm clear of the table it is standing on.
LGRIP_REL = 0.017
# Where along the base's 60 mm length the jaws bite. The V's two plate tips are
# the actual contact lines and they sit +-11.3 mm either side of the pad centre,
# so both stay on the wall only for |dx| <= 18.7 mm; and the wrist rides 77 mm
# directly above the pad centre, so dx also has to clear the peg standing proud
# of the pocket when the left hand comes back for the ASSEMBLED unit in S5.
# -18 mm is the compromise: 0.7 mm of tip margin at the end of the wall bought
# with 18 mm of offset from the peg axis (measured clearances in S5's log).
LGRIP_DX = -0.018
# The base is set DOWN at the assembly station rather than held there. 0.0299 is
# where a 45 g base settles on the table (top face z 0.030); the extra 1.6 mm is
# a release height, so the jaws open with the part just above the surface
# instead of driving it into the table through the grip.
TABLE_Z = 0.0299
# Where the assembly station IS, and it is not MEET_BASE. MEET_BASE is a pose in
# mid air 95 mm up, and both arms reach it comfortably; the same xy at TABLE
# height does not work, because the right arm has to fold in to get down there
# and jams its midArm against the robot's own torso. Measured, right arm holding
# the peg, station on the table: x=0.005 leaves 2.24 mm of alignment error with a
# joint pinned at its stop, against a pocket with 1.0 mm of clearance on a D20
# peg; x=0.03..0.11 lands within 0.13..0.41 mm. The LEFT arm is the other half of
# the constraint -- it places the base here and comes back for the assembled unit
# -- and it is happy from 0.005 to 0.05 (place 0.55 mm, re-grip 0.00 mm, 25 deg
# of joint margin) and breaks at 0.07. 0.04 is the overlap. The station is free
# to move because nothing is calibrated to it: the QC cameras are calibrated to
# MEET_BASE, and S5 carries the finished unit back there.
STATION = np.array([0.040, 0.412])
PLACE_BASE = np.array([STATION[0], STATION[1], TABLE_Z + 0.0016])
L_CLEAR = np.array([-0.16, 0.36, 0.26])   # left wrist parked clear for S3/S4
# The gripper-side retune of M2's seat latch. All five are already Insertion
# constructor parameters, so this is a CALL-SITE change, not a change to the
# controller. The shipped defaults (deep_gate 0.040, seat_depth 0.013) cannot
# latch on this path: S3 stages the peg 12 mm above a pocket whose bore is
# 20 mm deep, so the wrist travels 34 mm before the peg even bottoms out.
# seat_hold is the one that matters once the base is PLACED rather than held.
# M2's default latch measures travel after first contact, which assumes the
# fixture gives way — true of a base held on the left arm's position servo,
# false of a base standing on the table. Measured there: no axial reaction at
# all until the peg bottoms out 21.6 mm in, then only 5.65 mm of stack
# compliance left against a 6.0 mm seat_depth, so the latch never fires and the
# controller burns all 2600 cycles regulating force on a finished insert (the
# 2350 wasted cycles are what walk the base 4 mm off station). seat_hold latches
# on the force-and-still criterion instead — the insert IS done.
GRIP_LATCH = dict(f_contact=0.5, seat_depth=0.006, f_target=2.0,
                  deep_gate=0.048, seat_hold=120)


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
        # ...and the LEFT hand's, detected SEPARATELY. Two flags rather than one
        # so a scene with jaws on only one wrist still runs: the unconverted hand
        # falls back to its weld and the cycle is an honest A/B of the two.
        self.gripL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripL")
        self.jawsL = self.gripL >= 0
        self._fL_adr = self._sensor_adr("grip_force_L")
        self._jL_adr = self._sensor_adr("jaw_L")
        # gripper-path state, carried between steps so run_cycle(steps=[...])
        # can still be called in chunks for chunked rendering
        self.lhold = None        # left wrist pose while it holds the base
        self.offl = None         # base_part position minus left wrist position
        self.gl_hold = None      # left wrist target held through S2..S6
        self.grip_cmd = GRIP_OPEN
        self.grip_cmd_L = GRIP_OPEN
        # S5's in-situ reading, taken at the assembly station in the instant no
        # hand is on the part. None on the weld path, where no such instant
        # exists, and None until it is taken -- so "did the station gate run?"
        # is a question about the cell and never about a stale attribute.
        self.qc_station = None
        self.qc_station_verdict = None
        self.qc_gate = None

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

    def grip_force_L(self):
        """Normal force on the LEFT jaw's pad (N). 0.0 if that hand has no jaws.

        A second sensor rather than a sum over both: the two hands hold two
        different parts at two different moments, and a combined reading could
        not tell "the left has the base" from "the right has the peg"."""
        return float(self.d.sensordata[self._fL_adr]) if self._fL_adr >= 0 else 0.0

    def jaw_mm_L(self):
        """Left jaw travel from rest (mm; negative = open). NaN without jaws."""
        return (float(self.d.sensordata[self._jL_adr]) * 1000
                if self._jL_adr >= 0 else float("nan"))

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

        With real jaws a hand reads its OWN pad force sensor — an equality flag
        would only say "commanded", the sensor says "gripped". Each side is
        resolved independently of the other, so a scene with jaws on one wrist
        and a weld on the other still answers honestly for both, and a scene
        with no jaws at all answers exactly as it always did."""
        if side == "right" and self.jaws:
            return self.grip_force() > GRIP_MIN_N
        if side == "left" and self.jawsL:
            return self.grip_force_L() > GRIP_MIN_N
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
        # The LEFT command rides on the cell rather than on the signature. The
        # right one is an argument because S1b/S4 sweep it cycle by cycle in
        # loops that already own it; the left one is set once per phase and then
        # has to survive being handed back out of run_cycle and in again, since
        # the renderer runs the cycle in chunks.
        if cell.gripL >= 0:
            d.ctrl[cell.gripL] = cell.grip_cmd_L
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

    # ----- gripper path: tool geometry, shared by both hands ----------------
    # Both wrists carry the IDENTICAL tool — measured, not assumed: each EE site
    # sits exactly on its wrist body origin, both site frames are right-handed
    # (det +1), and both pad centres sit REACH + PAD_Y along the site's local +y.
    # The `_Mirror__1` in the right chain's body names is a naming convention
    # inherited from the CAD, not a reflected frame, so one pair of helpers
    # serves both hands and simply takes the arm as an argument.
    from make_gripper_cell import REACH, PAD_Y     # M4's jaw geometry
    TOOL = REACH + PAD_Y                           # 60 mm, wrist site -> pad centre

    def pad_centre(arm):
        """World point midway between this hand's two pads — the thing that has
        to land on the part, as opposed to the wrist, which is 60 mm behind it."""
        R = d.site_xmat[arm.site].reshape(3, 3)
        return arm.ee_pos() + R @ np.array([0.0, TOOL, 0.0])

    def tool_tilt(arm):
        """Angle between this hand's approach axis and straight down (deg)."""
        ax = d.site_xmat[arm.site].reshape(3, 3)[:, 1]
        return float(np.degrees(np.arccos(np.clip(-ax[2], -1, 1))))

    def close_jaws(arm, tR, tL, target_n=GRIP_TARGET_N, cycles=400,
                   win=40, f_tol=0.25, f_hold=20):
        """Close one hand on whatever its pads straddle, under force control.

        Creep the command up until the pads report first touch, then regulate
        the pad sensor to ``target_n`` with a rate-limited command — a step
        command bounces the part out of the V instead of seating it in it. One
        loop for both hands: which hand it is decides only which sensor is read
        and which command is written, and BOTH commands are live the whole time
        so the other hand does not drop what it is already holding.

        The regulation loop leaves as soon as the force has actually arrived,
        judged on the MEAN over a trailing window rather than on the
        instantaneous sample — the same reason ``insertion.py``'s seat test is a
        windowed mean. Measured on the right hand closing on the peg: a cylinder
        held in a V has two line contacts, and the pad force rings 10.54 .. 13.37
        N about a 12.0 N target for as long as the loop is willing to run, so an
        instantaneous test never accumulates and the loop always burned its full
        fixed 400 cycles. The mean is right long before that — rolling-40 lands
        within 0.25 N of target at cycle 71 and never leaves it again (worst
        residual over the remaining 329 cycles: 0.196 N). The left hand on the
        base's flat walls does not ring at all, sitting at 12.000 +- 0.001 from
        cycle 82, and passes the same test at cycle 75. At three call sites the
        fixed loop was 9.6 s of a 91 s takt spent regulating a force that had
        stopped moving.
        """
        right = arm is armR
        force = cell.grip_force if right else cell.grip_force_L
        cmd = 0.0
        while force() < 0.05 and cmd < 6.0:
            cmd = min(6.0, cmd + 0.25)
            if right:
                cell.grip_cmd = cmd
            else:
                cell.grip_cmd_L = cmd
            step_both(tR, cell.grip_cmd, tL, n=8)
        f_hist = deque(maxlen=win)
        held = 0
        for _ in range(cycles):
            want = np.clip(cmd + 1.5 * (target_n - force()), 0.0, 60.0)
            cmd = float(np.clip(want, cmd - 0.5, cmd + 0.5))
            if right:
                cell.grip_cmd = cmd
            else:
                cell.grip_cmd_L = cmd
            step_both(tR, cell.grip_cmd, tL)
            f_hist.append(force())
            settled = (len(f_hist) == win
                       and abs(float(np.mean(f_hist)) - target_n) < f_tol)
            held = held + 1 if settled else 0
            if held >= f_hold:
                break
        # hold the orientation actually ACHIEVED, not the ideal one that was
        # commanded into the approach: fighting for the last tenth of a degree
        # against a part now held by friction only walks the part in the V.
        arm.lock_orientation()
        return cmd

    def open_jaws(arm, tR, tL, cycles=600):
        """Let go, and wait on the SENSOR: "released" is the pads reading no
        force, not the command having been written. Returns the cycles taken."""
        if arm is armR:
            cell.grip_cmd = GRIP_OPEN
            force = cell.grip_force
        else:
            cell.grip_cmd_L = GRIP_OPEN
            force = cell.grip_force_L
        for k in range(cycles):
            step_both(tR, cell.grip_cmd, tL)
            if k > 60 and force() <= GRIP_MIN_N:
                return k
        return cycles

    def settle_part(tR, tL, cycles=1200, tol=0.004, stall=200, stall_eps=1e-4):
        """Step both arms on fixed targets until the carried part has stopped
        moving. Every measurement the cycle takes off a held part — the grasp
        offset, the QC presentation — is only worth the speed it was taken at.

        Two ways to be settled. The normal one is: the part has stopped AND the
        left wrist has arrived within ``tol`` of what it was told. The second is
        for targets the arm cannot actually reach — it has stopped, the part has
        stopped, and the residual is no longer coming down. Without that second
        exit the bin placement burned the full 1200 cycles (9.6 s of takt) every
        single cycle: measured, the part was dead still at 0.00002 m/s while the
        wrist sat 7.28 mm short of a commanded pose at the edge of the left arm's
        workspace, and no amount of extra stepping was going to close it.
        """
        best = float("inf")
        best_i = 0
        for i in range(cycles):
            bp = cell.part_pose("base")
            step_both(tR, cell.grip_cmd, tL)
            sp = np.linalg.norm(cell.part_pose("base") - bp) / (4 * m.opt.timestep)
            lerr = float(np.linalg.norm(np.asarray(tL, float) - armL.ee_pos()))
            if lerr < best - stall_eps:
                best, best_i = lerr, i
            if i > 80 and sp < 0.002 and (lerr < tol or i - best_i > stall):
                return i
        return cycles

    def _drive(arm, t, other_t):
        """One cycle with ``arm`` driven to ``t`` and the other hand pinned."""
        if arm is armR:
            step_both(t, cell.grip_cmd, other_t)
        else:
            step_both(other_t, cell.grip_cmd, t)

    def face_down(arm, R_target, other_t, cycles=600):
        """Re-lock this hand's held orientation to ``R_target`` and step until
        the tool actually points there. The V only self-centres on an axis that
        lies along the groove, so no approach starts before it does."""
        arm.lock_orientation()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R_target.flatten())
        arm.q_lock = q
        t = arm.ee_pos().copy()
        for _ in range(cycles):
            _drive(arm, t, other_t)
            if tool_tilt(arm) < 0.4:
                break

    def centre_pads(arm, goal, other_t, cycles=2000, lat_tol=0.0004,
                    dz_max=0.0015):
        """Servo this hand's PAD CENTRE onto ``goal`` (world) while the other
        hand holds ``other_t``. Driving the WRIST to the goal instead would miss
        by the 60 mm length of the tool. Returns the residual (m).

        Lateral first, then down. Commanding the raw 3D error looks right and is
        not: ik_step6 caps the commanded step at 20 mm, so an 87 mm descent
        carrying a 1 mm lateral error spends 99.6% of every step on z and closes
        the lateral error at 0.2 um per cycle -- it never converges, and the
        hand lands wherever the descent happened to drift to. Measured on the
        base pick: 0.8 mm of lateral error at the top became 9.4 mm at the
        bottom. That is survivable on the peg, which sits in the V with 10.8 mm
        of clearance per side, and fatal on the base, which the jaws straddle
        with 0.80 mm per side at their hard stop (41.61 mm of tip gap across a
        40.0 mm part). So: close the lateral error at the height the hand is
        already at, then descend in bounded increments, re-closing it every
        cycle. Height is held -- not crept -- whenever the lateral error is out
        of tolerance, which is what makes the approach converge instead of
        racing the drift down.
        """
        err = goal - pad_centre(arm)
        for _ in range(cycles):
            err = goal - pad_centre(arm)
            cmd = err.copy()
            if np.linalg.norm(err[:2]) > lat_tol:
                cmd[2] = 0.0
            else:
                cmd[2] = float(np.clip(err[2], -dz_max, dz_max))
            _drive(arm, arm.ee_pos() + cmd, other_t)
            if np.linalg.norm(err) < 0.0008 and tool_tilt(arm) < 0.5:
                break
        return float(np.linalg.norm(err))

    # ----- S0: idle / home -----
    if "S0" in steps:
        cell.event("S0", "cycle start")
        # The V's tip gap at rest is 17.4 mm — narrower than the D20 peg and far
        # narrower than the base's 40 mm face. Both jaw pairs have to be
        # commanded open before either arm goes anywhere near a part, or the
        # pads arrive already touching it.
        if cell.jaws:
            d.ctrl[cell.grip] = GRIP_OPEN
        if cell.jawsL:
            d.ctrl[cell.gripL] = GRIP_OPEN
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
            # --- S1a: the LEFT hand takes the base FOR REAL ----------------
            peg0 = cell.part_pose("peg")
            base0 = cell.part_pose("base")
            # Where the left pads have to land, and the wrist that puts them
            # there. LGRIP_DX along the base's 60 mm length, LGRIP_REL up its
            # 20 mm side wall; the wrist then rides TOOL straight above that,
            # because R_DOWN_L points the approach axis at world -z.
            lpad = base0 + np.array([LGRIP_DX, 0.0, LGRIP_REL])
            lgoal = lpad + np.array([0.0, 0.0, TOOL])
            d.ctrl[cell.grip] = GRIP_OPEN
            d.ctrl[cell.gripL] = GRIP_OPEN
            reach(m, d, {"left": list(lgoal + np.array([0.0, 0.0, 0.085])),
                         "right": [0.20, 0.44, 0.24]},
                  seconds=2.4, on_frame=on_frame, tol=0.012, grav_ff=True)
            # both arms are now driven by step_both, which needs a locked
            # orientation on each of them before the first ik_step6
            armL.lock_orientation()
            armR.lock_orientation()
            rt = armR.ee_pos().copy()
            face_down(armL, R_DOWN_L, rt)
            res_l = centre_pads(armL, lpad, rt)
            lt = armL.ee_pos().copy()
            close_jaws(armL, rt, lt)
            cell.event("S1", "left jaws closed on the base",
                       centring_residual_mm=round(res_l * 1000, 3),
                       grip_force_n=cell.grip_force_L(), jaw_mm=cell.jaw_mm_L(),
                       tool_tilt_deg=tool_tilt(armL))
            # Lift clear of the table BEFORE measuring the grasp offset. Every
            # base-frame target downstream is converted through cell.offl, and
            # the one that matters is the IN-FLIGHT offset: the part shifts a
            # little in the V as its weight transfers off the table onto the
            # pads, and a millimetre here is a millimetre at the QC cameras.
            cell.lhold = lt + np.array([0.0, 0.0, 0.060])
            servo_both(rt, cell.lhold, cell.grip_cmd, seconds=2.0, tol=0.006)
            settle_part(rt, cell.lhold)
            cell.lhold = armL.ee_pos().copy()
            cell.offl = cell.part_pose("base") - cell.lhold
            cell.event("S1", "left holds the base",
                       offset_mm=[round(float(v) * 1000, 1) for v in cell.offl],
                       base_tilt_deg=cell.tilt_deg("base"),
                       grip_force_n=cell.grip_force_L())

            # --- S1b: the RIGHT hand grips the peg for real ----------------
            # servo_both, not reach: the base is held by FRICTION now, and
            # reach() would drive the left wrist off its hold target.
            servo_both([peg0[0], peg0[1], peg0[2] + GRIP_REL + TOOL + 0.085],
                       cell.lhold, cell.grip_cmd, seconds=2.6, tol=0.012)
            face_down(armR, R_DOWN, cell.lhold)
            res_r = centre_pads(armR, peg0 + np.array([0.0, 0.0, GRIP_REL]),
                                cell.lhold)
            close_jaws(armR, armR.ee_pos().copy(), cell.lhold)
            cell.event("S1", "right jaws closed on the peg",
                       centring_residual_mm=round(res_r * 1000, 3),
                       grip_force_n=cell.grip_force(), jaw_mm=cell.jaw_mm(),
                       tool_tilt_deg=tool_tilt(armR))
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

    # ----- S2: carry to the assembly station (gripper: PLACE and let go) -----
    if "S2" in steps:
        cell.event("S2", "carry to fixture/staging")
        if cell.jaws:
            # The base is PLACED, not held. Two tools cannot hold this part in
            # mid-air at once: with the left tool pose forced by the geometry,
            # the left wrist and the right hand's pads overlap by 11.4 mm at the
            # best grasp available anywhere on the part, and that number does not
            # improve with where the hands meet because both tools are then tied
            # rigidly to the same base. So the left hand sets it down at the
            # assembly station and gets out of the corridor the right one has to
            # descend through — which is what a real cell does with a part too
            # small to hand off. The station is STATION, chosen so BOTH arms can
            # work at table height (see the constant); the QC pose it is NOT is
            # MEET_BASE, 95 mm up, which S5 carries the assembled unit back to so
            # the cameras keep the calibration they already have.
            gr = np.array([0.08, 0.41, 0.30])
            place_l = PLACE_BASE - cell.offl
            servo_both(gr, place_l, cell.grip_cmd, seconds=4.5, tol=0.006)
            settle_part(gr, place_l)
            cell.event("S2", "base placed at the assembly station",
                       base=list(cell.part_pose("base")),
                       tilt=cell.tilt_deg("base"))
            lpark = armL.ee_pos().copy()
            open_jaws(armL, gr, lpark)
            # Straight UP first. The approach axis is world -z, so the pads only
            # leave the part along +z; any lateral move made before they clear
            # the wall drags the base off the station that was just measured.
            servo_both(gr, lpark + np.array([0.0, 0.0, 0.090]),
                       cell.grip_cmd, seconds=2.0, tol=0.006)
            servo_both(gr, L_CLEAR, cell.grip_cmd, seconds=2.6, tol=0.015)
            cell.gl_hold = armL.ee_pos().copy()   # S3/S4 hold the left arm here
            cell.event("S2", "left released and retracted clear",
                       base=list(cell.part_pose("base")),
                       tilt=cell.tilt_deg("base"), jaw_mm=cell.jaw_mm_L(),
                       grip_force_n=cell.grip_force_L())
        else:
            servo6_both([0.0, 0.33, 0.21], [0.08, 0.41, 0.30], seconds=4.5)
        cell.event("S2",
                   "base standing at the station -> S3" if cell.jaws
                   else "at meet point -> S3",
                   block_tilt=cell.tilt_deg("base"), peg_tilt=cell.tilt_deg("peg"))

    # ----- S3: align peg over pocket (relative servoing) -----
    if "S3" in steps:
        cell.event("S3", "align", err_xy=cell.align_err_xy())
        if cell.jaws:
            # Same relative servo, but the peg is held by friction now, so the
            # convergence test has to HOLD: a single in-tolerance sample can be
            # the peg swinging through zero rather than settled on it.
            #
            # Lateral first, then down, for the reason centre_pads spells out:
            # ik_step6 caps the commanded step at 20 mm, and S3 starts 131 mm
            # above the pocket, so feeding it the raw 3D error spends the whole
            # step budget on z and closes the lateral error at 2 um per cycle.
            # Measured as coded: 78.6 mm of xy error at the start, still 18.6 mm
            # after all 4000 cycles, and S4 was handed an assumed_center 17.2 mm
            # from the base it was inserting into. Hold the height until the peg
            # is over the bore, then come down.
            ok_k = 0
            for _ in range(4000):
                err_xy = (cell.pocket_top() - cell.part_pose("peg"))[:2]
                zerr = (cell.pocket_top()[2] + 0.012) - cell.peg_bottom()[2]
                dz = 0.0 if np.linalg.norm(err_xy) > 0.0035 \
                    else float(np.clip(zerr, -0.0015, 0.0015))
                step_both(armR.ee_pos() + np.array([err_xy[0], err_xy[1], dz]),
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
        if cell.jawsL:
            # Insertion._step() never touches a grip actuator, so whatever is in
            # d.ctrl when it starts is what the left jaws hold for all 2600 cycles.
            # Write the left command explicitly rather than inheriting whatever the
            # last step_both left there: on the place-and-insert path the left hand
            # is OPEN and parked clear, and an inherited closing command would have
            # it squeezing air at 12 N through the whole insert.
            d.ctrl[cell.gripL] = cell.grip_cmd_L
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
        import qc as qc_mod
        if cell.jaws:
            # --- the RIGHT hand backs out of the bore ----------------------
            # Straight UP first, along the axis it came down. Its jaws are open
            # but the pads still straddle the peg, and a lateral move made before
            # they clear it would tip the fresh insert over. `reach` is not an
            # option here either: it would drive BOTH arms off their targets.
            rout = armR.ee_pos() + np.array([0.0, 0.0, 0.085])
            servo_both(rout, cell.gl_hold, cell.grip_cmd, seconds=2.0, tol=0.008)
            servo_both(R_PARK, cell.gl_hold, cell.grip_cmd, seconds=2.4, tol=0.015)

            # --- IN-SITU QC: the instant BOTH hands are clear --------------
            # Right now the finished unit stands FREE at the assembly station.
            # The right jaws opened around it at the end of S4, the right wrist
            # has climbed 85 mm and parked, and the left hand has not started
            # back in. Nothing is touching the part -- an instant the weld cell
            # structurally cannot have, because there the unit is off a hand
            # only while it is on a weld. Losing the welds is what created it.
            # The fixture pair cannot use it: those cameras are aimed at a pose
            # in mid air 95 mm above here, and from the station they read 0 stub
            # px and 46 rim px -- blind, not merely worse. So this gate is the
            # SECOND CAMERA PAIR, aimed at the station: qc_top, which sees both
            # places, plus qc_station_side, the same lens at the same standoff
            # as qc_side and mounted along -y. Same qc.measure(), same masks,
            # same thresholds; only the viewpoint is new.
            if getattr(cell, "qc_renderer", None) is not None:
                st = qc_mod.measure(cell.qc_renderer, d,
                                    unit_z=float(cell.part_pose("base")[2]),
                                    pair=qc_mod.STATION_PAIR)
                cell.qc_station = st
                cell.qc_station_verdict = qc_mod.verdict(st)
                cell.event("S5", "STATION QC verify (in-situ, both hands clear)",
                           cam_align_mm=st["align_err_mm"],
                           cam_depth_mm=st["depth_mm_est"],
                           cam_peg_present=st["peg_present"],
                           cam_result=cell.qc_station_verdict,
                           oracle_depth_mm=cell.insertion_depth() * 1000,
                           oracle_align_mm=cell.align_err_xy() * 1000,
                           oracle_tilt_deg=cell.tilt_deg("peg"),
                           unit_at_station_m=[round(float(v), 5)
                                              for v in cell.part_pose("base")])

            # --- the LEFT hand re-grips the ASSEMBLED unit -----------------
            # It still has to, and the station gate does not change that. The
            # fixture calibration stays where it is -- in a real cell the
            # calibration is the fixed thing, and the numbers in
            # test_manipulation_numbers.py are pinned to it -- so the cell keeps
            # bringing the part back to MEET_BASE for the fixture pair to read.
            # What the station pair adds is a verdict taken where the part
            # actually is; re-teaching the FIXTURE to the table would have
            # thrown away the calibration and the published numbers with it,
            # which is why that is still the wrong way round.
            base1 = cell.part_pose("base")
            lpad = base1 + np.array([LGRIP_DX, 0.0, LGRIP_REL])
            cell.grip_cmd_L = GRIP_OPEN
            servo_both(R_PARK, lpad + np.array([0.0, 0.0, TOOL + 0.085]),
                       cell.grip_cmd, seconds=2.8, tol=0.012)
            face_down(armL, R_DOWN_L, R_PARK)
            res_l = centre_pads(armL, lpad, R_PARK)
            close_jaws(armL, R_PARK, armL.ee_pos().copy())
            cell.event("S5", "left re-gripped the assembled unit",
                       centring_residual_mm=round(res_l * 1000, 3),
                       grip_force_n=cell.grip_force_L(), jaw_mm=cell.jaw_mm_L(),
                       regrip_drift_mm=round(float(np.linalg.norm(
                           cell.part_pose("base") - base1)) * 1000, 2))

            # --- present it at the QC pose, in TWO passes ------------------
            # The offset the unit sits at in the V is not the one S1 measured on
            # the bare base: it is heavier now, and the second close finds a
            # slightly different seat (measured: 0.8 mm in z between the two).
            # So measure the offset actually achieved in flight, move to it, then
            # measure where the base REALLY ended up and correct once. A single
            # pass put the presented base 2.4 mm low, and the QC ROI is only
            # 92 mm tall with a depth estimate calibrated on the unit's z.
            lift = armL.ee_pos() + np.array([0.0, 0.0, 0.060])
            servo_both(R_PARK, lift, cell.grip_cmd, seconds=2.2, tol=0.006)
            settle_part(R_PARK, lift)
            cell.offl = cell.part_pose("base") - armL.ee_pos()
            cell.gl_hold = MEET_BASE - cell.offl
            servo_both(R_PARK, cell.gl_hold, cell.grip_cmd, seconds=3.0, tol=0.006)
            settle_part(R_PARK, cell.gl_hold)
            corr = MEET_BASE - cell.part_pose("base")
            cell.gl_hold = armL.ee_pos() + corr
            servo_both(R_PARK, cell.gl_hold, cell.grip_cmd, seconds=1.8, tol=0.004)
            settle_part(R_PARK, cell.gl_hold)
            cell.offl = cell.part_pose("base") - armL.ee_pos()
            cell.event("S5", "assembled unit presented at the QC pose",
                       base=list(cell.part_pose("base")), want=list(MEET_BASE),
                       correction_mm=[round(float(v) * 1000, 2) for v in corr],
                       residual_mm=round(float(np.linalg.norm(
                           MEET_BASE - cell.part_pose("base"))) * 1000, 2),
                       offset_mm=[round(float(v) * 1000, 1) for v in cell.offl],
                       grip_force_n=cell.grip_force_L())
            # let the peg stop ringing inside the bore before the camera looks
            # at it -- QC measures pixels, not intent.
            for _ in range(150):
                step_both(R_PARK, cell.grip_cmd, cell.gl_hold)
        else:
            reach(m, d, {"right": [0.22, 0.36, 0.30]}, seconds=2.2,
                  on_frame=on_frame, tol=0.015)
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
            # WHICH pair gates. The fixture pair gates wherever it can see the
            # part, and on the weld path that is everywhere -- there is no
            # station reading on that path at all, because the unit is never off
            # a hand there, so this branch is byte-identical to what shipped.
            # On the weld-free path the fixture pair is occluded at this pose by
            # the left tool holding the unit up to it, which is the finding this
            # project published and did not want to lose: so the station reading
            # gates, and the fixture reading is still taken and still logged, as
            # the occluded control that made the case for a second pair.
            gate_verdict, gate = ((cell.qc_station_verdict, "station")
                                  if getattr(cell, "qc_station", None) is not None
                                  else (cam_verdict, "fixture"))
            cell.qc_pass = (gate_verdict == "ACCEPT") and not getattr(cell, "qc_jam", False)
            cell.qc_meas = meas
            cell.qc_gate = gate
            cell.event("S5", "CAMERA QC verify",
                       gate=gate, gate_result=gate_verdict,
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
        # Both bins are served by the LEFT arm, on the gripper path and the weld
        # path alike. That is comfortable for ACCEPT, which sits at x=-0.24 on the
        # left arm's own side (measured: unit released 0.6 mm off the bin centre),
        # and marginal for REJECT at x=+0.24, which is across the whole robot:
        # measured, the left arm gets the unit to x=+0.2050 -- 35.0 mm off centre,
        # with 4.86 deg left on its tightest joint. The unit still lands ON the bin
        # (released at z=0.0399 against a bin top of 0.040; the bin spans x=0.19 to
        # 0.29, so a 60 mm base centred at 0.205 is supported), it is just not
        # centred. Properly the reject bin belongs to the RIGHT arm and wants a
        # handoff after S5; that is a separate piece of work, and this limit is a
        # property of the cell layout rather than of the jaw conversion.
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
            # Down to the same release height the assembly station uses: the jaws
            # open with the unit 1.6 mm above the surface, so it is set down
            # rather than driven into the table through the grip.
            down = np.array([bin_x, 0.41, TABLE_Z + 0.0016]) - cell.offl
            servo_both(R_PARK, down, cell.grip_cmd, seconds=2.2, tol=0.008)
            settle_part(R_PARK, down)
            lpark = armL.ee_pos().copy()
            opened = open_jaws(armL, R_PARK, lpark)
            # And straight UP. The approach axis is world -z, so the pads only
            # leave the part along +z; a lateral retreat started before they
            # clear the wall would drag the unit back off the bin.
            servo_both(R_PARK, lpark + np.array([0.0, 0.0, 0.090]),
                       cell.grip_cmd, seconds=2.2, tol=0.008)
            cell.event("S6", "left jaws opened and retracted",
                       release_cycles=opened, jaw_mm=cell.jaw_mm_L(),
                       grip_force_n=cell.grip_force_L())
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
