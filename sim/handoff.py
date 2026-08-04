#!/usr/bin/env python3
"""Jaw-to-jaw hand-off — one hand takes the part OUT of the other, in mid air.

The cell's own answer to "pass a part between the hands" has always been to put
it down: `sequencer.run_cycle` places the base at the assembly station and picks
it up again (S2/S5), and `benchmark.task_carry` said in as many words that a
true hand-off waits for hardware. Both were true of the WELD cell, where a
"grasp" is an equality constraint and two equalities on one body fight. With
M4's jaws on both wrists neither is true any more: two sets of pads can share
one part, so the transfer is a controls problem, and controls problems belong in
a module with a metric on them.

What makes it work is that the two hands take DIFFERENT lengths of the part.
Each hand's pad plate covers a fixed run of the grip axis, so the giver holds
one end, the receiver comes at the other end from the opposite side, and the
plates pass each other with a stated clearance — ``PAD_GAP`` — that is checked
against the compiled scene rather than argued for. Cross-hand contacts are NOT
excluded in the model (``make_cell_scene._mount_jaws`` excludes only the two
jaws OF ONE hand), so if that clearance were wrong the jaws would collide and
the physics would say so.

The receiving pose is the giving pose rolled half a turn about its own closing
axis: the jaws still close along the same world axis, but the approach axis
points the other way, so the receiver rises from underneath into the gap the
giver left. That is ``R_UP`` below, and it is derived from ``sequencer.R_DOWN``
rather than written out, so the two poses cannot drift apart.

The loops are the sequencer's — face the tool, centre the pads, close on force,
open on force — with one generalisation: the sequencer exits its facing and
centring loops on the angle to STRAIGHT DOWN, which a hand pointing up can never
satisfy. Here the exit is the angle to the wanted approach axis, which is the
same test whenever the wanted axis is down.

Runs on a scene built by ``make_cell_scene.py --gripper`` and takes the
sequencer's own ``Cell``, so no actuator, sensor or body name is spelled twice.
"""
from collections import deque

import mujoco
import numpy as np

from make_gripper_cell import REACH, PAD_Y
from primitives import smoothstep
from sequencer import GRIP_OPEN, GRIP_TARGET_N, GRIP_MIN_N, GRIP_REL, R_DOWN

TOOL = REACH + PAD_Y            # wrist site -> pad centre, along the tool's +y

# The receiver's pose: the giver's, rolled half a turn about the closing axis
# (the tool's local x). Columns of R are [closing, approach, third] in world, so
# negating the last two flips the approach axis and leaves the closing axis put.
R_UP = np.asarray(R_DOWN, float) @ np.diag([1.0, -1.0, -1.0])

PAD_GAP = 0.004         # plate-to-plate clearance the two hands leave each other
STAGE_CLEAR = 0.019     # extra room under the part while the receiver rolls over

# Where the two hands meet, as the part's centre. Not a free choice: a hand-off
# with the part lying along world x is a right-arm workspace boundary (the wrist
# stops 4..111 mm short with a joint already on its limit), and y = 0.44 — the
# row the grasp tasks work on — is out of reach for the right arm at every
# height tried. Standing the part along world z at this point puts both wrists
# on their targets exactly. Measured, not assumed; see test_handoff.py.
MEET = np.array([0.06, 0.40, 0.24])


class HandOff:
    """Move a part from one hand's jaws into the other's, in free space.

    ``giver`` already holds the part (or ``pick()`` makes it so); ``taker`` is
    the empty hand. ``meet`` is where the part's CENTRE ends up. The part is
    held along the world z axis: the giver reaches down onto its top end, the
    taker comes up onto its bottom end.

    ``run()`` returns the numbers that decide whether it worked — how far the
    part fell as it changed hands, how far it slipped when the receiver closed,
    its tilt afterwards, and what each hand had hold of — plus the two
    geometric margins the transfer depends on, measured off the compiled scene.
    """

    def __init__(self, cell, giver, taker, meet, part="peg",
                 target_n=GRIP_TARGET_N, clear=0.090):
        self.cell, self.m, self.d = cell, cell.m, cell.d
        self.giver, self.taker = giver, taker
        self.meet = np.asarray(meet, float)
        self.target_n = target_n
        self.clear = clear
        self.pid = cell.pg if part == "peg" else cell.bp
        self.part = part

        m = self.m
        # the part's half-length along the axis it is held on
        gid = next(g for g in range(m.body_geomadr[self.pid],
                                    m.body_geomadr[self.pid] + m.body_geomnum[self.pid]))
        self.part_half = float(m.geom_size[gid][1])
        # the pad plate's half-length along the groove axis: the plate is a box
        # rotated only ABOUT its local y, so its y half-extent is geom_size[1]
        self.pad_half = float(m.geom_size[self._pads(giver)[0]][1])

        # Where each hand's pads sit on the part. The giver's offset is the
        # sequencer's own release height; the taker's is then forced — as low as
        # it can be while still leaving PAD_GAP between the two plates.
        self.give_rel = GRIP_REL
        self.take_rel = self.give_rel - 2 * self.pad_half - PAD_GAP
        # ...and the taker waits below that until it has rolled its tool over
        self.stage_rel = -(self.part_half + self.pad_half + STAGE_CLEAR)

        self.gt = self.meet + np.array([0.0, 0.0, self.give_rel + TOOL])
        self.tt = self.meet + np.array([0.0, 0.0, self.take_rel - TOOL])
        self.stage = self.meet + np.array([0.0, 0.0, self.stage_rel - TOOL])

        # gravity-only feed-forward on the two arms' hinges (qvel=0 -> no
        # Coriolis), the same cancellation reach(grav_ff=True) applies
        self._gmask = np.zeros(m.nv)
        for arm in (giver, taker):
            for i in arm.vadr:
                if m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                    self._gmask[i] = 1.0
        self._gff = np.zeros(m.nv)

    # ---- the scene, asked rather than assumed --------------------------------
    def _pads(self, arm):
        """The pad-plate geoms of one hand, found through its wrist body."""
        m = self.m
        wrist = m.site_bodyid[arm.site]
        out = []
        for b in range(m.nbody):
            if m.body_parentid[b] != wrist:
                continue
            if not (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("jaw"):
                continue
            out += list(range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]))
        return out

    def _span_z(self, arm):
        """World z range covered by this hand's pads, right now (m)."""
        lo, hi = np.inf, -np.inf
        for g in self._pads(arm):
            c = float(self.d.geom_xpos[g][2])
            R = self.d.geom_xmat[g].reshape(3, 3)
            half = float(np.abs(R[2]) @ self.m.geom_size[g])
            lo, hi = min(lo, c - half), max(hi, c + half)
        return lo, hi

    def _part_span_z(self):
        c = float(self.d.xpos[self.pid][2])
        return c - self.part_half, c + self.part_half

    def margins(self):
        """The two clearances the transfer lives on, measured (mm).

        ``gap`` is plate to plate between the hands; ``give``/``take`` is how
        much of the part each hand's plates actually cover."""
        glo, ghi = self._span_z(self.giver)
        tlo, thi = self._span_z(self.taker)
        plo, phi = self._part_span_z()
        return {"gap_mm": round((glo - thi) * 1000, 2),
                "give_contact_mm": round((min(ghi, phi) - max(glo, plo)) * 1000, 2),
                "take_contact_mm": round((min(thi, phi) - max(tlo, plo)) * 1000, 2)}

    # ---- force / pose helpers ------------------------------------------------
    def _force(self, arm):
        return (self.cell.grip_force() if arm is self.cell.armR
                else self.cell.grip_force_L())

    def _set_cmd(self, arm, cmd):
        if arm is self.cell.armR:
            self.cell.grip_cmd = cmd
        else:
            self.cell.grip_cmd_L = cmd

    def _get_cmd(self, arm):
        return (self.cell.grip_cmd if arm is self.cell.armR
                else self.cell.grip_cmd_L)

    def pad_centre(self, arm):
        R = self.d.site_xmat[arm.site].reshape(3, 3)
        return arm.ee_pos() + R @ np.array([0.0, TOOL, 0.0])

    def axis_err(self, arm, R_target):
        """Angle between this hand's approach axis and the wanted one (deg)."""
        got = self.d.site_xmat[arm.site].reshape(3, 3)[:, 1]
        want = np.asarray(R_target, float)[:, 1]
        return float(np.degrees(np.arccos(np.clip(float(got @ want), -1, 1))))

    def part_tilt(self):
        z = self.d.xmat[self.pid].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(abs(float(z[2])), -1, 1))))

    def part_pos(self):
        return self.d.xpos[self.pid].copy()

    # ---- the driven loops ----------------------------------------------------
    def step(self, t_give, t_take, n=4):
        for arm, t in ((self.giver, t_give), (self.taker, t_take)):
            arm.set_ctrl(arm.ik_step6(np.asarray(t, float))[0])
        self.d.ctrl[self.cell.grip] = self.cell.grip_cmd
        self.d.ctrl[self.cell.gripL] = self.cell.grip_cmd_L
        qv = self.d.qvel.copy()
        self.d.qvel[:] = 0.0
        mujoco.mj_rne(self.m, self.d, 0, self._gff)
        self.d.qvel[:] = qv
        for _ in range(n):
            self.d.qfrc_applied[:] = self._gff * self._gmask
            mujoco.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0.0

    def _drive(self, arm, t, other_t):
        if arm is self.giver:
            self.step(t, other_t)
        else:
            self.step(other_t, t)

    def servo(self, t_give, t_take, seconds, tol=0.008, extra=400):
        """Eased two-arm servo, both hands holding their locked orientation."""
        s = [self.giver.ee_pos().copy(), self.taker.ee_pos().copy()]
        g = [np.asarray(t_give, float), np.asarray(t_take, float)]
        n = max(int(seconds / (self.m.opt.timestep * 4)), 1)
        for i in range(n + extra):
            f = smoothstep(min(1.0, (i + 1) / n))
            self.step(s[0] + (g[0] - s[0]) * f, s[1] + (g[1] - s[1]) * f)
            if i >= n and np.linalg.norm(g[0] - self.giver.ee_pos()) < tol \
                    and np.linalg.norm(g[1] - self.taker.ee_pos()) < tol:
                break

    def face(self, arm, R_target, other_t, cycles=600):
        """Rotate one hand's tool onto the wanted approach axis, in place."""
        arm.lock_orientation()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(R_target, float).flatten())
        arm.q_lock = q
        t = arm.ee_pos().copy()
        for k in range(cycles):
            self._drive(arm, t, other_t)
            if self.axis_err(arm, R_target) < 0.4:
                break
        return k

    def centre(self, arm, goal, R_target, other_t, cycles=2000, lat_tol=0.0004,
               dz_max=0.0015):
        """Bring one hand's pad centre onto a point: sideways first, then in."""
        err = goal - self.pad_centre(arm)
        for _ in range(cycles):
            err = goal - self.pad_centre(arm)
            cmd = err.copy()
            if np.linalg.norm(err[:2]) > lat_tol:
                cmd[2] = 0.0
            else:
                cmd[2] = float(np.clip(err[2], -dz_max, dz_max))
            self._drive(arm, arm.ee_pos() + cmd, other_t)
            if np.linalg.norm(err) < 0.0008 and self.axis_err(arm, R_target) < 0.5:
                break
        return float(np.linalg.norm(err))

    def close(self, arm, t_give, t_take, cycles=400, win=40, f_tol=0.25,
              f_hold=20):
        """Close one hand onto the part under force control."""
        cmd = 0.0
        while self._force(arm) < 0.05 and cmd < 6.0:
            cmd = min(6.0, cmd + 0.25)
            self._set_cmd(arm, cmd)
            self.step(t_give, t_take, n=8)
        hist, held = deque(maxlen=win), 0
        for _ in range(cycles):
            want = np.clip(cmd + 1.5 * (self.target_n - self._force(arm)), 0.0, 60.0)
            cmd = float(np.clip(want, cmd - 0.5, cmd + 0.5))
            self._set_cmd(arm, cmd)
            self.step(t_give, t_take)
            hist.append(self._force(arm))
            settled = (len(hist) == win
                       and abs(float(np.mean(hist)) - self.target_n) < f_tol)
            held = held + 1 if settled else 0
            if held >= f_hold:
                break
        arm.lock_orientation()
        return cmd

    def open(self, arm, t_give, t_take, cycles=600):
        """Open one hand and wait for its pads to actually go quiet."""
        self._set_cmd(arm, GRIP_OPEN)
        for k in range(cycles):
            self.step(t_give, t_take)
            if k > 60 and self._force(arm) <= GRIP_MIN_N:
                return k
        return cycles

    # ---- the two halves of the job ------------------------------------------
    def pick(self, above=0.085, seconds=2.6, park=None, aim=None):
        """The giving hand takes the part off the table. Returns the centring
        residual (m).

        ``aim`` offsets where the giver AIMS relative to where the part really
        is — the per-trial disturbance, the same one ``benchmark.task_carry``
        injects on its grasp targets. It is a real error, not a relabelling:
        the taker is still told to meet the part at ``meet``, so whatever the
        giver picks up crooked, the taker has to cope with.

        The taker holds station at ``park``, which defaults to wherever it
        already is — after ``approach()`` that is out of the giver's way by
        construction, and standing still is one fewer pose to justify. Sending
        it to its staging point this early does NOT work: it then sits over the
        table with its tool still pointing down, and the giver cannot centre.
        """
        # A pick starts with an open hand. Said here rather than left to the
        # caller's staging, because `benchmark.fresh`/`approach` — the two
        # lines every other task starts with — drive the arms with ctrl zeroed
        # and never touch the jaw actuators.
        self._set_cmd(self.giver, GRIP_OPEN)
        self._set_cmd(self.taker, GRIP_OPEN)
        park = np.asarray(park if park is not None else self.taker.ee_pos(),
                          float)
        p0 = self.part_pos() + (np.zeros(3) if aim is None
                                else np.asarray(aim, float))
        top = np.array([p0[0], p0[1], p0[2] + self.give_rel + TOOL + above])
        self.servo(top, park, seconds=seconds, tol=0.012)
        self.face(self.giver, R_DOWN, park)
        res = self.centre(self.giver, p0 + np.array([0.0, 0.0, self.give_rel]),
                          R_DOWN, park)
        self.close(self.giver, self.giver.ee_pos().copy(), park)
        return res

    def transfer(self, carry_s=3.0, stage_s=3.4, back_s=2.4, settle=300):
        """Carry the held part to the meet and pass it. Returns the metrics."""
        park = self.taker.ee_pos().copy()
        self.servo(self.gt, park, seconds=carry_s, tol=0.006)
        carry_err = float(np.linalg.norm(self.gt - self.giver.ee_pos()))
        carried = self.part_pos()
        self.servo(self.gt, self.stage, seconds=stage_s, tol=0.010)

        self.face(self.taker, R_UP, self.gt)
        res = self.centre(self.taker, self.meet + np.array([0.0, 0.0, self.take_rel]),
                          R_UP, self.gt)
        clearances = self.margins()

        before = self.part_pos()
        self.close(self.taker, self.gt, self.taker.ee_pos().copy())
        hold = self.taker.ee_pos().copy()
        slip = float(np.linalg.norm(self.part_pos() - before))
        handed = self.part_pos()

        k = self.open(self.giver, self.giver.ee_pos().copy(), hold)
        out = self.giver.ee_pos() + np.array([0.0, 0.0, self.clear])
        self.servo(out, hold, seconds=back_s, tol=0.010)
        for _ in range(settle):
            self.step(out, hold)

        drop = float(handed[2] - self.part_pos()[2])
        held = bool(drop < 0.010 and self._force(self.taker) > GRIP_MIN_N)
        return dict(clearances,
                    carry_err_mm=round(carry_err * 1000, 2),
                    take_residual_mm=round(res * 1000, 2),
                    slip_mm=round(slip * 1000, 2),
                    drop_mm=round(drop * 1000, 2),
                    tilt_deg=round(self.part_tilt(), 2),
                    carried_mm=round(float(np.linalg.norm(
                        self.part_pos() - carried)) * 1000, 2),
                    hold_force_n=round(self._force(self.taker), 2),
                    release_cycles=int(k),
                    handed=held)

    def run(self, **kw):
        """Pick the part up with one hand and put it in the other."""
        res = self.pick()
        out = self.transfer(**kw)
        out["pick_residual_mm"] = round(res * 1000, 2)
        return out
