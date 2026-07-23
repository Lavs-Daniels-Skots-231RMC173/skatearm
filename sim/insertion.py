"""M2 — force-regulated peg-in-hole insertion (sim-first, contact-force control).

Replaces the M1/v1 open-loop descent ("1.4 mm per cycle until a tau watchdog
trips") with a **hybrid position/force** controller that regulates the axial
contact force instead of pushing until a threshold:

- **Axial admittance on an accumulating z-setpoint.** The commanded wrist height
  integrates the force error: ``z_cmd += clip(kf*(f_ax - f_target), -vz, vz)`` — the
  wrist descends while the measured axial force is below ``f_target`` and eases off
  once it reaches it, so contact force is *regulated* (settles ~f_target), never
  rammed. ``lead_cap`` bounds how far the setpoint may lead the actual wrist so a
  stalled peg cannot wind up a large downward command.
- **Gravity feed-forward** (``mj_rne`` on the hinge joints, qvel zeroed → no
  Coriolis, stable — the same trick ``primitives.reach`` uses) so the arm holds
  its height instead of sagging under its own weight during the slow descent.
- **Spiral hole-search with a pause-and-seat state machine.** When misaligned, the
  wrist xy spirals out from the *assumed* hole centre; the moment the peg starts to
  drop (the wrist descends past ``drop_eps`` over ``drop_win`` cycles) the spiral
  **freezes** so the peg can seat, and if it then stalls without seating the spiral
  **resumes** — this is what recovers an offset the open-loop descent just jams on.
- **Wrench abort** at ``w_abort`` as a safety backstop.

The force source is the M1 wrist wrench (``ee_{side}_force`` rotated into world).
During the force phase the controller uses ONLY the wrench + proprioception (wrist
pose) + the assumed hole centre — never the live peg pose; the simulator's peg pose
is an oracle used only for *scoring* the result in the eval, not for control.

Sim ↔ hardware: the controller is identical on hardware; only the wrench *source*
differs (a real wrist F/T sensor or a joint-torque estimator behind the M1 interface).
"""
from collections import deque

import numpy as np
import mujoco


class Insertion:
    """Force-regulated insertion of the peg held by ``arm`` into a hole whose
    ASSUMED centre is ``center_xy`` (world). Call ``run(search=)``; the peg must
    already be grasped and staged above the hole with orientation locked."""

    def __init__(self, m, d, arm, center_xy,
                 f_contact=1.0, f_target=3.0, w_abort=9.0,
                 vz=3e-4, kf=2.2e-4, lead_cap=0.024,
                 r_max=6e-3, pitch=0.6e-3, search_rate=0.3,
                 deep_gate=0.040, seat_depth=0.013,
                 drop_win=16, drop_eps=1.2e-3, dwell=22,
                 pause_max=70, still_win=40, still_tol=4e-4,
                 substeps=4, max_cycles=2600, hold_arms=None, on_step=None):
        self.m, self.d, self.arm = m, d, arm
        self.center = np.asarray(center_xy, float)[:2]
        self.f_contact, self.f_target, self.w_abort = f_contact, f_target, w_abort
        self.vz, self.kf, self.lead_cap = vz, kf, lead_cap
        self.r_max, self.pitch, self.search_rate = r_max, pitch, search_rate
        self.deep_gate, self.seat_depth = deep_gate, seat_depth
        self.drop_win, self.drop_eps, self.dwell = drop_win, drop_eps, dwell
        self.pause_max, self.still_win, self.still_tol = pause_max, still_win, still_tol
        self.substeps, self.max_cycles = substeps, max_cycles
        # M1 wrist wrench of this arm — fail loud if the F/T sensor is absent
        # (mj_name2id returns -1, and sensor_adr[-1] would silently read the wrong
        # sensor), rather than regulating force on garbage.
        fname = f"ee_{arm.side}_force"
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, fname)
        if sid < 0:
            raise ValueError(f"missing wrist F/T sensor {fname!r} — rebuild the "
                             "model with make_control_model.py (M1)")
        self._f_adr = m.sensor_adr[sid]
        self._site = arm.site
        # hold_arms: other arm(s) to keep servoing to a fixed target every cycle —
        # used in the bimanual sequencer where the left arm must keep holding the
        # base while the right inserts. on_step: per-cycle callback (e.g. render).
        self.hold_arms = list(hold_arms or [])   # [(arm, target_xyz), ...]
        self.on_step = on_step
        # gravity feed-forward over the WORKING arm's hinge dofs plus any held
        # arms'. reach() comps every hinge because it drives both arms; here only
        # these arms are actuated, so comping an idle arm would double up on its
        # own servo hold and drift it.
        self._gff = np.zeros(m.nv)
        self._gmask = np.zeros(m.nv)
        for a in [arm] + [h[0] for h in self.hold_arms]:
            for i in a.vadr:
                if m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                    self._gmask[i] = 1.0

    # --- signals -------------------------------------------------------------
    def _wrench_world(self):
        R = self.d.site_xmat[self._site].reshape(3, 3)
        return R @ np.asarray(self.d.sensordata[self._f_adr:self._f_adr + 3])

    def _grav_ff(self):
        qv = self.d.qvel.copy()
        self.d.qvel[:] = 0.0
        mujoco.mj_rne(self.m, self.d, 0, self._gff)
        self.d.qvel[:] = qv
        return self._gff * self._gmask

    def _step(self, target):
        q, _ = self.arm.ik_step6(target)
        self.arm.set_ctrl(q)
        for a, tgt in self.hold_arms:            # keep the other arm(s) holding (bimanual)
            qh, _ = a.ik_step6(np.asarray(tgt, float))
            a.set_ctrl(qh)
        gff = self._grav_ff()
        for _ in range(self.substeps):
            self.d.qfrc_applied[:] = gff
            mujoco.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0.0
        if self.on_step is not None:
            self.on_step()

    # --- controller ----------------------------------------------------------
    def run(self, search=True, relock=True):
        # relock=True snapshots the current wrist orientation as the hold target
        # (nominal). relock=False keeps the caller's pre-set arm.q_lock, so if that
        # is the UPRIGHT orientation the 6-DoF IK actively LEVELS an initially
        # tilted peg toward vertical while it inserts (theta-misalignment recovery).
        arm = self.arm
        if relock:
            arm.lock_orientation()
        F0 = self._wrench_world()
        z0 = float(arm.ee_pos()[2])
        z_cmd = z0
        theta, r = 0.0, 0.0
        off = np.zeros(2)
        frozen = False
        pause_k = 0
        dwell_k = 0
        contact_z = None
        z_hist = deque(maxlen=max(self.drop_win, self.still_win) + 1)
        peak_w = 0.0
        aborted = False
        seated = False
        cyc = 0
        for cyc in range(self.max_cycles):
            F = self._wrench_world() - F0
            w = float(np.linalg.norm(F))
            peak_w = max(peak_w, w)
            if w > self.w_abort:
                aborted = True
                break
            f_ax = float(-F[2])                    # +ve = base pushing back up on a descent
            z_act = float(arm.ee_pos()[2])

            # first solid contact latches the seat-depth reference AND restarts the
            # drop history, so the search below only ever measures motion that
            # happened AFTER contact (not the fast free-descent that precedes it).
            if contact_z is None and f_ax > self.f_contact:
                contact_z = z_act
                z_hist.clear()
            z_hist.append(z_act)

            # axial admittance on the accumulating z-setpoint
            z_cmd += float(np.clip(self.kf * (f_ax - self.f_target), -self.vz, self.vz))
            z_cmd = max(z_cmd, z_act - self.lead_cap)   # never lead the actual by > lead_cap
            z_cmd = min(z_cmd, z0 + 5e-4)               # and never command upward past the start

            # spiral hole-search with pause-and-seat — ONLY after contact. Before
            # contact the peg drops straight down at the assumed centre; spiralling
            # in free air would just carry it off the hole before it ever arrives.
            if search and contact_z is not None:
                dropping = (len(z_hist) > self.drop_win
                            and (z_hist[-self.drop_win] - z_act) > self.drop_eps)
                if frozen:
                    pause_k += 1
                    still = (len(z_hist) > self.still_win
                             and abs(z_hist[-self.still_win] - z_act) < self.still_tol)
                    if pause_k >= self.pause_max or (still and not dropping):
                        frozen = False           # stalled on the rim → resume the spiral
                        pause_k = 0
                elif dropping:
                    frozen = True                # peg is dropping in → hold xy and let it seat
                    pause_k = 0
                    dwell_k = 0
                # step the spiral only after DWELLing at the current point long
                # enough for the peg to seat if it is over the bore (the drop
                # detector needs ~drop_win cycles of descent to fire) — otherwise
                # the search sweeps the peg past the hole before it can drop in.
                if not frozen:
                    dwell_k += 1
                    if dwell_k >= self.dwell:
                        dwell_k = 0
                        theta += self.search_rate
                        r = min(self.r_max, r + self.pitch)
                        off = r * np.array([np.cos(theta), np.sin(theta)])

            self._step(np.array([self.center[0] + off[0],
                                 self.center[1] + off[1], z_cmd]))

            # seated: descended seat_depth below first contact
            if contact_z is not None and (contact_z - float(arm.ee_pos()[2])) >= self.seat_depth:
                seated = True
                break
            if (z0 - z_act) >= self.deep_gate:   # ran past the working depth → give up
                break
        return {"seated": bool(seated), "aborted": bool(aborted),
                "peak_wrench_n": round(peak_w, 2), "cycles": cyc + 1,
                "descent_mm": round((z0 - float(arm.ee_pos()[2])) * 1000, 2)}
