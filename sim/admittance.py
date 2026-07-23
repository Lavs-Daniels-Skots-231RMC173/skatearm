"""M3 — Cartesian admittance (compliant) arm control (sim-first).

Wraps SkateArm's position servos in a task-space admittance loop at the TCP so
the arm *yields* to an external contact wrench instead of holding rigid or
latching a soft-stop (the cockpit's v1 contact reflex), and unlike M2's insertion
it regulates all three Cartesian axes, not just the insertion axis. This is the
general compliant controller: a tunable virtual mass-spring-damper per axis whose
displacement is added to the nominal TCP target and realised by the existing
6-DoF DLS-IK on the position actuators (qpos is never written directly).

Law (diagonal, per world axis):   Mv·e'' + D·e' + K·e = F_ext
  e     — TCP compliant displacement from the nominal pose x0
  F_ext — external force on the wrist: the M1 wrist wrench, baselined at start so
          the loop responds to the CHANGE (real contact) not the static grasp
          load / sensor bias. Steady state e = F_ext / K: a low-K axis YIELDS
          (e ≈ F/K), a high-K axis HOLDS. D defaults to critical (ζ=1) so the
          yield is smooth and returns without overshoot when the force is removed.

Integrated with a symplectic Euler step (velocity then position) at the 8 ms
control cycle. Commanded TCP = x0 + e, tracked by ik_step6 with orientation held
(translational compliance; rotational compliance is a later extension). Gravity
feed-forward (mj_rne over the arm's hinge dofs, qvel zeroed → no Coriolis) holds
height, exactly as primitives.reach and M2 do.

Wrench sign (measured, sim/_probe_adm): an external +f on the wrist body reads as
−f on ee_{side}_force in world → F_ext = F0 − wrench_world.

Sim ↔ hardware: admittance-on-position-servos ports directly to a real
position-controlled arm (which the Skate is); only the wrench SOURCE differs (a
real wrist F/T sensor or a joint-torque estimator behind the M1 interface).
"""
import numpy as np
import mujoco


def admittance_advance(e, ev, F, K, D, Mv, dt):
    """One symplectic-Euler step of the diagonal admittance ODE
    ``Mv·e'' + D·e' + K·e = F`` (velocity updated first, then position — more
    stable than explicit Euler). Pure and stateless: shared by the single-arm
    ``Admittance`` and the bimanual ``CompliantCarry`` so the control law has one
    definition. e, ev, F, K, D, Mv are per-axis arrays; returns (e, ev)."""
    ev = ev + dt * ((F - D * ev - K * e) / Mv)
    e = e + dt * ev
    return e, ev


class Admittance:
    """Task-space admittance for one ``Arm``. Construct at the pose to hold, then
    call ``step()`` each control cycle (or ``run(seconds)``). ``K``/``D``/``Mv``
    are per-axis (world x, y, z); ``D`` defaults to critical damping. Pass
    ``f_override`` to drive the law from a supplied wrench (unit-testing the
    control law without a physical contact). ``hold_arms``/``on_step`` mirror
    ``Insertion`` so the loop is bimanual- and render-ready."""

    def __init__(self, m, d, arm, x0=None,
                 K=(800., 800., 800.), Mv=(2., 2., 2.), D=None, zeta=1.0,
                 max_offset=0.06, substeps=4, hold_arms=None, on_step=None):
        self.m, self.d, self.arm = m, d, arm
        self.K = np.broadcast_to(np.asarray(K, float), (3,)).copy()
        self.Mv = np.broadcast_to(np.asarray(Mv, float), (3,)).copy()
        self.D = (np.broadcast_to(np.asarray(D, float), (3,)).copy() if D is not None
                  else 2.0 * zeta * np.sqrt(self.K * self.Mv))     # critical by default
        self.max_offset = float(max_offset)
        self.substeps = int(substeps)
        self.dt = m.opt.timestep * self.substeps
        self.x0 = (np.asarray(x0, float).copy() if x0 is not None else arm.ee_pos())
        # M1 wrist wrench — fail loud if the F/T sensor is absent (as in Insertion)
        fname = f"ee_{arm.side}_force"
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, fname)
        if sid < 0:
            raise ValueError(f"missing wrist F/T sensor {fname!r} — rebuild the "
                             "model with make_control_model.py (M1)")
        self._f_adr = m.sensor_adr[sid]
        self._site = arm.site
        self.hold_arms = list(hold_arms or [])
        self.on_step = on_step
        # gravity feed-forward over this arm's (and any held arms') hinge dofs only
        self._gff = np.zeros(m.nv)
        self._gmask = np.zeros(m.nv)
        for a in [arm] + [h[0] for h in self.hold_arms]:
            for i in a.vadr:
                if m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                    self._gmask[i] = 1.0
        arm.lock_orientation()
        self.e = np.zeros(3)
        self.ev = np.zeros(3)
        self.F0 = self._wrench_world()

    # --- signals -----------------------------------------------------------
    def _wrench_world(self):
        R = self.d.site_xmat[self._site].reshape(3, 3)
        return R @ np.asarray(self.d.sensordata[self._f_adr:self._f_adr + 3])

    def ext_wrench(self):
        """External force on the wrist (world frame), baselined and sign-corrected
        so a physical push reads as a force in the SAME direction the arm yields."""
        return self.F0 - self._wrench_world()

    def _grav_ff(self):
        qv = self.d.qvel.copy()
        self.d.qvel[:] = 0.0
        mujoco.mj_rne(self.m, self.d, 0, self._gff)
        self.d.qvel[:] = qv
        return self._gff * self._gmask

    def tcp_offset(self):
        """Actual TCP displacement from the nominal pose x0 (m, world)."""
        return self.arm.ee_pos() - self.x0

    # --- controller --------------------------------------------------------
    def step(self, x0=None, f_override=None):
        """One control cycle: advance the admittance state from the (measured or
        supplied) wrench, command x0 + e through the IK, step physics. Returns the
        commanded compliant offset e (m, world)."""
        if x0 is not None:
            self.x0 = np.asarray(x0, float)
        F = (np.asarray(f_override, float) if f_override is not None
             else self.ext_wrench())
        self.e, self.ev = admittance_advance(self.e, self.ev, F, self.K,
                                             self.D, self.Mv, self.dt)
        self.e = np.clip(self.e, -self.max_offset, self.max_offset)
        q, _ = self.arm.ik_step6(self.x0 + self.e)
        self.arm.set_ctrl(q)
        for a, tgt in self.hold_arms:                 # keep other arm(s) holding
            qh, _ = a.ik_step6(np.asarray(tgt, float))
            a.set_ctrl(qh)
        gff = self._grav_ff()
        for _ in range(self.substeps):
            self.d.qfrc_applied[:] = gff
            mujoco.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0.0
        if self.on_step is not None:
            self.on_step()
        return self.e.copy()

    def run(self, seconds, x0=None, f_override=None, on_step=None):
        """Hold the compliant loop for `seconds` (a fixed nominal pose / wrench)."""
        if on_step is not None:
            self.on_step = on_step
        for _ in range(int(seconds / self.dt)):
            self.step(x0=x0, f_override=f_override)
        return self.e.copy()
