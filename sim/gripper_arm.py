"""M4 (arm integration) — grasp a part with the wrist-mounted jaws and carry it,
held by FRICTION alone (no weld).

Runs on the opt-in `skt_v3_gripper_cell` model (make_gripper_cell): the right
wrist carries the parallel-jaw gripper. The part is placed in the jaws' grasp
region and pinned to the world only while the jaws close (grasp-force controlled);
the pin is then released and the arm carries the part — if it stays fixed in the
wrist frame through the motion, it is held by the gripper, not a weld. This is the
weld-free replacement for `primitives.grasp()`'s magnetic weld stand-in, on the
actual arm.
"""
import numpy as np
import mujoco

from primitives import Arm, reach
from benchmark import fresh, approach
from eval_insertion import weld_here
from make_gripper_cell import REACH, PAD_Y

LEFT_HOLD = [-0.18, 0.44, 0.20]        # left arm parked out of the way
GRASP_POSE = [0.16, 0.44, 0.16]        # right wrist over the part


def grasp_and_carry(m, f_target=5.0, carry=(-0.05, -0.04, 0.07), on_step=None):
    """Stage the arm, grasp the part to `f_target` N with the jaws, release the
    world-pin, and carry the wrist by `carry` (world m). Returns metrics including
    the part's drift in the wrist frame (small = carried by the grasp, not welded)."""
    d = fresh(m)
    approach(m, d)
    reach(m, d, {"left": LEFT_HOLD, "right": GRASP_POSE},
          seconds=2.4, tol=0.012, grav_ff=True)
    armL, armR = Arm(m, d, "left"), Arm(m, d, "right")
    armL.lock_orientation(); armR.lock_orientation()
    grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "grip")
    fadr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "grip_force")]
    pin = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "pin")
    pbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "part")
    padr = m.jnt_qposadr[m.body_jntadr[pbody]]

    # place the part in the jaws' grasp region and pin it to the world there
    R = d.site_xmat[armR.site].reshape(3, 3)
    center = armR.ee_pos() + R @ np.array([0.0, REACH + PAD_Y, 0.0])
    wq = np.zeros(4); mujoco.mju_mat2Quat(wq, R.flatten())
    d.qpos[padr:padr + 3] = center
    d.qpos[padr + 3:padr + 7] = wq
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    weld_here(m, d, "pin")

    gmask = np.zeros(m.nv)
    for a in (armL, armR):
        for i in a.vadr:
            if m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE:
                gmask[i] = 1.0
    gff = np.zeros(m.nv)
    rt = armR.ee_pos().copy()

    def gf():
        return float(d.sensordata[fadr])

    def hold(tgt, gcmd):
        armR.set_ctrl(armR.ik_step6(np.asarray(tgt, float))[0])
        armL.set_ctrl(armL.ik_step6(np.asarray(LEFT_HOLD, float))[0])
        d.ctrl[grip] = gcmd
        qv = d.qvel.copy(); d.qvel[:] = 0.0
        mujoco.mj_rne(m, d, 0, gff); d.qvel[:] = qv
        for _ in range(4):
            d.qfrc_applied[:] = gff * gmask
            mujoco.mj_step(m, d)
        d.qfrc_applied[:] = 0.0
        if on_step is not None:
            on_step(d)

    cmd = 0.0
    while gf() < 0.3 and cmd < 60.0:                    # close to contact
        cmd += 1.0; hold(rt, cmd)
    for _ in range(220):                               # regulate grasp force to target
        cmd = float(np.clip(cmd + 1.5 * (f_target - gf()), 0.0, 60.0)); hold(rt, cmd)
    grasp_n = gf()
    R0 = d.site_xmat[armR.site].reshape(3, 3)
    rel0 = R0.T @ (d.xpos[pbody] - armR.ee_pos())      # part offset in wrist frame at grasp

    d.eq_active[pin] = 0                               # release the world-pin -> friction only
    for k in range(240):                              # carry: lift + translate the wrist
        s = min(1.0, k / 180.0)
        hold(rt + np.asarray(carry, float) * s, cmd)
    Rf = d.site_xmat[armR.site].reshape(3, 3)
    relf = Rf.T @ (d.xpos[pbody] - armR.ee_pos())
    drift = float(np.linalg.norm(relf - rel0))
    return {"grasp_n": round(grasp_n, 2), "carry_grasp_n": round(gf(), 2),
            "drift_mm": round(drift * 1000, 1), "pin_active": bool(d.eq_active[pin]),
            "carried": bool(drift < 0.02), "wrist": armR.ee_pos().copy(),
            "part": d.xpos[pbody].copy()}
