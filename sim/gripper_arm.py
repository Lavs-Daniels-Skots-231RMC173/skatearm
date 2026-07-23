"""M4 (arm integration) — grasp a part with the wrist-mounted jaws and carry or
PLACE it, held by FRICTION alone (no weld).

Runs on the opt-in `skt_v3_gripper_cell` model (make_gripper_cell): the right
wrist carries the parallel-jaw gripper. The part is placed in the jaws' grasp
region and pinned to the world only while the jaws close (grasp-force controlled);
the pin is then released and the arm carries the part — held by the gripper, not a
weld. `grasp_carry_place` extends this to a full pick-and-place: carry over the
place bin, descend, and open the jaws to release the part onto it. This is the
weld-free replacement for `primitives.grasp()`'s magnetic weld, on the arm.
"""
import numpy as np
import mujoco

from primitives import Arm, reach
from benchmark import fresh, approach
from eval_insertion import weld_here
from make_gripper_cell import REACH, PAD_Y

LEFT_HOLD = [-0.18, 0.44, 0.20]        # left arm parked out of the way
GRASP_POSE = [0.16, 0.44, 0.16]        # right wrist over the part
BIN_XY = np.array([0.108, 0.408])      # place-bin centre (make_gripper_cell)
BIN_TOP = 0.044


def _setup_and_grip(m, f_target, on_step):
    """Stage the arm, place + pin the part in the jaws, and grasp it to `f_target`
    N while holding the arm steady. Returns a context dict with a `hold(target,
    grip_cmd)` step closure and the handles the carry/place phases need."""
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

    rt = armR.ee_pos().copy()
    cmd = 0.0
    while gf() < 0.3 and cmd < 60.0:                    # close to contact
        cmd += 1.0; hold(rt, cmd)
    for _ in range(220):                               # regulate grasp force to target
        cmd = float(np.clip(cmd + 1.5 * (f_target - gf()), 0.0, 60.0)); hold(rt, cmd)
    return dict(d=d, armR=armR, hold=hold, gf=gf, cmd=cmd, pin=pin,
                pbody=pbody, rt=rt, grasp_n=gf())


def grasp_and_carry(m, f_target=5.0, carry=(-0.05, -0.04, 0.07), on_step=None):
    """Grasp the part, release the world-pin, and carry the wrist by `carry` (m).
    Returns the part's drift in the wrist frame (small = carried, not welded)."""
    c = _setup_and_grip(m, f_target, on_step)
    d, armR, hold, pin, pbody, rt = (c["d"], c["armR"], c["hold"], c["pin"],
                                     c["pbody"], c["rt"])
    R0 = d.site_xmat[armR.site].reshape(3, 3)
    rel0 = R0.T @ (d.xpos[pbody] - armR.ee_pos())
    d.eq_active[pin] = 0                               # release the world-pin -> friction only
    for k in range(240):
        s = min(1.0, k / 180.0)
        hold(rt + np.asarray(carry, float) * s, c["cmd"])
    Rf = d.site_xmat[armR.site].reshape(3, 3)
    relf = Rf.T @ (d.xpos[pbody] - armR.ee_pos())
    drift = float(np.linalg.norm(relf - rel0))
    return {"grasp_n": round(c["grasp_n"], 2), "carry_grasp_n": round(c["gf"](), 2),
            "drift_mm": round(drift * 1000, 1), "pin_active": bool(d.eq_active[pin]),
            "carried": bool(drift < 0.02), "wrist": armR.ee_pos().copy(),
            "part": d.xpos[pbody].copy()}


def grasp_carry_place(m, f_target=5.0, lift_z=0.205, place_z=0.132, on_step=None):
    """Full pick-and-place: grasp the part, carry it over the place bin, descend,
    and OPEN the jaws to release it onto the bin — weld-free. Returns metrics
    including whether the part ends released and resting on the bin."""
    c = _setup_and_grip(m, f_target, on_step)
    d, armR, hold, pin, cmd = c["d"], c["armR"], c["hold"], c["pin"], c["cmd"]
    pbody = c["pbody"]
    d.eq_active[pin] = 0                               # friction grasp only
    px, py = float(BIN_XY[0]), float(BIN_XY[1])

    def glide(target, n, gcmd):
        s0 = armR.ee_pos().copy()
        g = np.asarray(target, float)
        for k in range(n):
            s = min(1.0, (k + 1) / n); s = s * s * (3 - 2 * s)
            hold(s0 + (g - s0) * s, gcmd)

    glide([px, py, lift_z], 120, cmd)                 # carry over the bin (lift + across)
    glide([px, py, place_z], 70, cmd)                 # descend onto the bin
    for _ in range(70):                               # OPEN the jaws -> release
        hold([px, py, place_z], -8.0)
    for _ in range(60):                               # retreat
        hold([px, py, lift_z], -8.0)
    part = d.xpos[pbody].copy()
    released = c["gf"]() < 0.5
    on_bin = bool(abs(part[2] - (BIN_TOP + 0.022)) < 0.025
                  and np.linalg.norm(part[:2] - BIN_XY) < 0.03)
    return {"grasp_n": round(c["grasp_n"], 2), "released_grasp_n": round(c["gf"](), 2),
            "part": part, "part_z": round(float(part[2]), 3),
            "released": bool(released), "on_bin": on_bin,
            "placed": bool(on_bin and released)}
