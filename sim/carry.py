"""M3+ — bimanual COMPLIANT carry (object-level Cartesian admittance).

Generalises demo_dual_carry's hand-tuned 1-D load-equaliser
(`corr = 0.011·tanh((fL−fR)·0.4)` on the z-bias) into a principled per-axis
compliant carry: both arms hold a shared object and a SINGLE Cartesian admittance
at the object level yields the whole two-arm assembly to an external disturbance
wrench and returns it — the same yield-at-a-commanded-stiffness law as M3's
single-arm `Admittance`, now regulating the carried object.

Key idea that keeps the closed kinematic loop honest: the object is welded to
BOTH wrists, so the two wrists + object form a loop. The admittance offset e is
added EQUALLY to both wrist targets, i.e. it is a pure *translation* of the rigid
assembly — which the loop permits — so the two compliant commands never fight the
bar constraint (a per-arm offset would).

Driving wrench: the NET external force on the object = the (baselined) sum of the
two M1 wrist wrenches. The internal grasp/squeeze force is equal-and-opposite at
the two wrists and cancels in the sum, leaving just the external + inertial part;
baselining at start removes the static object weight, so the loop responds to the
CHANGE (a real push). Same law (`admittance_advance`), gravity feed-forward and
sign convention (F_ext = F0 − Σ wrench_world) as sim/admittance.py.
"""
import numpy as np
import mujoco

from admittance import admittance_advance


class CompliantCarry:
    """Bimanual compliant carry of one object grasped by ``armL`` and ``armR``.
    Construct AFTER both arms have grasped it; ``offL``/``offR`` are each wrist's
    offset from the object frame x0 (default: captured from the current wrist
    poses, so x0 = the wrist midpoint). Move the carry by passing a new ``x0`` to
    ``step``/``run``; the admittance offset e adds compliance on top. Position-only
    IK per wrist (as demo_dual_carry) — the rigid object fixes orientation."""

    def __init__(self, m, d, armL, armR, offL=None, offR=None, x0=None,
                 K=(600., 600., 600.), Mv=(3., 3., 3.), D=None, zeta=1.0,
                 max_offset=0.06, substeps=4, on_step=None):
        self.m, self.d, self.armL, self.armR = m, d, armL, armR
        self.K = np.broadcast_to(np.asarray(K, float), (3,)).copy()
        self.Mv = np.broadcast_to(np.asarray(Mv, float), (3,)).copy()
        self.D = (np.broadcast_to(np.asarray(D, float), (3,)).copy() if D is not None
                  else 2.0 * zeta * np.sqrt(self.K * self.Mv))
        self.max_offset = float(max_offset)
        self.substeps = int(substeps)
        self.dt = m.opt.timestep * self.substeps
        self.on_step = on_step
        self.x0 = (np.asarray(x0, float).copy() if x0 is not None
                   else 0.5 * (armL.ee_pos() + armR.ee_pos()))
        self.offL = (np.asarray(offL, float) if offL is not None
                     else armL.ee_pos() - self.x0)
        self.offR = (np.asarray(offR, float) if offR is not None
                     else armR.ee_pos() - self.x0)
        self._adr = {}
        for arm in (armL, armR):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"ee_{arm.side}_force")
            if sid < 0:
                raise ValueError(f"missing wrist F/T sensor for {arm.side!r} (M1)")
            self._adr[arm.side] = m.sensor_adr[sid]
        self._gff = np.zeros(m.nv)
        self._gmask = np.zeros(m.nv)
        for arm in (armL, armR):
            for i in arm.vadr:
                if m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                    self._gmask[i] = 1.0
        self.e = np.zeros(3)
        self.ev = np.zeros(3)
        self.F0 = self._net_wrench()

    # --- signals -----------------------------------------------------------
    def _wrench(self, arm):
        R = self.d.site_xmat[arm.site].reshape(3, 3)
        adr = self._adr[arm.side]
        return R @ np.asarray(self.d.sensordata[adr:adr + 3])

    def _net_wrench(self):
        return self._wrench(self.armL) + self._wrench(self.armR)

    def ext_wrench(self):
        """External force on the carried object (world), baselined & sign-fixed."""
        return self.F0 - self._net_wrench()

    def object_pos(self):
        """The carried object's held frame = midpoint of the two wrists (world)."""
        return 0.5 * (self.armL.ee_pos() + self.armR.ee_pos())

    def _grav_ff(self):
        qv = self.d.qvel.copy()
        self.d.qvel[:] = 0.0
        mujoco.mj_rne(self.m, self.d, 0, self._gff)
        self.d.qvel[:] = qv
        return self._gff * self._gmask

    # --- controller --------------------------------------------------------
    def step(self, x0=None, f_override=None):
        """One control cycle: advance the object admittance from the (measured or
        supplied) net wrench, command both wrists to x0 + off + e, step physics.
        Returns the compliant offset e (m, world)."""
        if x0 is not None:
            self.x0 = np.asarray(x0, float)
        F = (np.asarray(f_override, float) if f_override is not None
             else self.ext_wrench())
        self.e, self.ev = admittance_advance(self.e, self.ev, F, self.K,
                                             self.D, self.Mv, self.dt)
        self.e = np.clip(self.e, -self.max_offset, self.max_offset)
        qL, _ = self.armL.ik_step(self.x0 + self.offL + self.e)
        qR, _ = self.armR.ik_step(self.x0 + self.offR + self.e)
        self.armL.set_ctrl(qL)
        self.armR.set_ctrl(qR)
        gff = self._grav_ff()
        for _ in range(self.substeps):
            self.d.qfrc_applied[:] = gff
            mujoco.mj_step(self.m, self.d)
        self.d.qfrc_applied[:] = 0.0
        if self.on_step is not None:
            self.on_step()
        return self.e.copy()

    def run(self, seconds, x0=None, f_override=None):
        for _ in range(int(seconds / self.dt)):
            self.step(x0=x0, f_override=f_override)
        return self.e.copy()


def stage_bar_grasp(model_dir, settle=250, approach_steps=320):
    """Build the dual-carry scene, approach, and grasp the shared bar with BOTH
    wrists (releasing the world pin). Returns (m, d, armL, armR) ready to carry —
    shared by the CI test and any compliant-carry demo. Imports are local so the
    demo module's arg parsing never runs at import time."""
    import demo_dual_carry as DC
    import primitives as P
    scene = DC.build_scene(model_dir)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bar")
    bd0, bdn = m.body_dofadr[bar], m.body_dofnum[bar]
    gmask = np.ones(m.nv); gmask[bd0:bd0 + bdn] = 0.0
    gbuf = np.zeros(m.nv)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    DC.engage(m, d, "wfix")                                # pin the bar during approach
    armL, armR = P.Arm(m, d, "left"), P.Arm(m, d, "right")
    bz = float(d.xpos[bar][2])
    M = np.array([0.0, DC.GY, bz + 0.009])
    P.move_joints(m, d, {"a0": 0.3, "a1": 0.3, "a3": 0.8}, seconds=1.5)
    for _ in range(approach_steps):
        for a, sx in ((armL, -DC.WX), (armR, DC.WX)):
            q, _ = a.ik_step(np.array([M[0] + sx, M[1], M[2]]))
            a.set_ctrl(q)
        qv = d.qvel.copy(); d.qvel[:] = 0.0
        mujoco.mj_rne(m, d, 0, gbuf); d.qvel[:] = qv
        for _ in range(4):
            d.qfrc_applied[:] = gbuf * gmask
            mujoco.mj_step(m, d)
    d.qfrc_applied[:] = 0.0
    DC.engage(m, d, "wL"); DC.engage(m, d, "wR")           # both wrists grasp the bar
    d.eq_active[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "wfix")] = 0
    return m, d, armL, armR
