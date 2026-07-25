"""Task-space primitives for SkateArm (v1): closed-loop REACH via damped
least-squares IK on the 8-DoF arm chains, servoed through the position
actuators (never writes qpos directly — physics stays honest).

Motion quality: reach() tracks a *moving* target that eases from the current
EE position to the goal with a smoothstep profile over `seconds`, so servo
commands change gradually (commanding the goal directly produced ~5 m/s EE
whips and ~30 m/s^2 spikes at segment switches — measured).

Controller notes (tried and rejected): integrating the IK update on d.ctrl
winds up (servo whip); a qfrc_bias/kp feedforward includes Coriolis terms =
positive velocity feedback (unstable). Plain P on qpos is stable; the steady-state gravity sag is cancelled by
reach(grav_ff=True): gravity-only feed-forward via mj_rne (qvel=0, so no
Coriolis -> stable, unlike a qfrc_bias feedforward) applied to the hinge
joints through qfrc_applied — measured 30 mm sag -> 1 mm, servo holding offset
0.033 rad -> 0. The feed-forward is cleared on exit so it never leaks into a
later (force-guarded) descent; free bodies (carried parts) still fall.
"""
import mujoco
import numpy as np

ARM_JOINTS = {
    "left":  [f"a{i}_armL_a{8+i}" for i in range(8)],
    "right": [f"a{i}_armR_a{16+i}" for i in range(8)],
}
EE_SITE = {"left": "ee_left", "right": "ee_right"}


def smoothstep(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


class Arm:
    def __init__(self, m, d, side):
        self.m, self.d, self.side = m, d, side
        self.jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS[side]]
        self.qadr = [m.jnt_qposadr[j] for j in self.jids]
        self.vadr = [m.jnt_dofadr[j] for j in self.jids]
        self.lo = np.array([m.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([m.jnt_range[j][1] for j in self.jids])
        self.aids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"pos_{n}") for n in ARM_JOINTS[side]]
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EE_SITE[side])
        self.mirror = 1.0  # mirrored chains take same-sign values (verified Phase 0)
        self.max_step = 0.035  # m of task-space correction per control cycle (8 ms)

    def ee_pos(self):
        return self.d.site_xpos[self.site].copy()

    # joint weights: prefer proximal joints; distal wrist joints have short
    # levers and otherwise get driven straight into their limits
    W = np.diag([1.0, 1.0, 0.8, 1.0, 0.4, 0.25, 0.25, 0.25])
    # comfortable reference posture for the null-space bias
    Q_REF = np.array([0.3, 0.3, 0.0, 0.8, 0.0, 0.2, 0.0, 0.1])

    def ik_step(self, target, gain=0.8, damp=5e-3, posture_gain=0.012):
        """One weighted-DLS step with a null-space pull toward Q_REF.
        Plain DLS from a hanging (outstretched-singular) start dumped huge
        updates into the wrist joints, pinned them at their limits and
        saturated the shoulder — measured, hence the weighting + posture bias."""
        err = target - self.ee_pos()
        # task-space velocity clamp: bound the per-cycle correction so the EE
        # can never be commanded faster than ~0.5 m/s (without this, tracking
        # lag accumulated during the move gets released as one violent
        # catch-up step in the settle phase — measured 3 m/s, a0 at 9 rad/s)
        n = np.linalg.norm(err)
        if n > self.max_step:
            err = err * (self.max_step / n)
        jacp = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, None, self.site)
        J = jacp[:, self.vadr]                      # 3x8
        JW = J @ self.W
        JJt = JW @ J.T + damp * np.eye(3)
        dq = self.W @ J.T @ np.linalg.solve(JJt, err)
        q = np.array([self.d.qpos[a] for a in self.qadr])
        q_cmd = np.array([self.d.ctrl[a] for a in self.aids])
        # Integrate the update on the PREVIOUS COMMAND, not on qpos: the
        # standing ctrl embeds the gravity compensation; rebasing on the
        # sagged physical pose drops that compensation and the arm collapses
        # into the table edge at every segment start (measured). Wind-up is
        # prevented twice: the task-space step is clamped (max_step) and the
        # command may never run further than 0.35 rad from the physical pose
        # (bounded servo torque).
        # null-space posture bias (projected so it doesn't fight the task)
        Jsharp = self.W @ J.T @ np.linalg.inv(JJt)     # weighted pseudo-inverse, 8x3
        Jp = Jsharp @ J                                # J# J, 8x8
        bias = (np.eye(8) - Jp) @ (self.Q_REF * self.mirror - q) * posture_gain
        q_new = np.clip(q + gain * dq + bias, q - 0.35, q + 0.35)
        return np.clip(q_new, self.lo, self.hi), np.linalg.norm(err)

    def lock_orientation(self):
        """Remember the current EE orientation as the hold target for ik_step6."""
        R = self.d.site_xmat[self.site].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        self.q_lock = q

    def ik_step6(self, target, gain=0.8, damp=5e-3, rot_weight=2.0, rot_step=0.10,
                 pos_cap=0.02):
        """6-DOF DLS step: position toward `target` + orientation held at
        q_lock (call lock_orientation() first). Needed for insertion — the
        position-only IK leaves wrist pitch free and the carried peg tilted
        25 deg into the pocket (measured)."""
        err_p = target - self.ee_pos()
        n = np.linalg.norm(err_p)
        if n > pos_cap:
            err_p = err_p * (pos_cap / n)
        # orientation error as angular-velocity vector. KEY TUNING (measured):
        # orientation must DOMINATE (rot_weight 2.0, slow pos_cap) so drift
        # never accumulates — correcting an accumulated 60deg tilt runs the
        # wrist joints into their limits and fails; preventing it keeps them
        # mid-range (1-2 deg tilt over a 16 cm carry).
        R = self.d.site_xmat[self.site].reshape(3, 3)
        q_cur = np.zeros(4)
        mujoco.mju_mat2Quat(q_cur, R.flatten())
        err_r = np.zeros(3)
        mujoco.mju_subQuat(err_r, self.q_lock, q_cur)
        err_r = R @ err_r   # subQuat gives the local-frame velocity; jacr is world-frame
        rn = np.linalg.norm(err_r)
        if rn > rot_step:
            err_r = err_r * (rot_step / rn)
        jacp = np.zeros((3, self.m.nv)); jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
        J = np.vstack([jacp[:, self.vadr], jacr[:, self.vadr]])      # 6x8
        e6 = np.concatenate([err_p, rot_weight * err_r])
        # 6-DOF weighting: the wrist joints ARE the orientation DOFs — they
        # must not be de-weighted here (unlike the position-only W)
        W6 = np.diag([1.0, 1.0, 0.8, 1.0, 0.8, 0.8, 0.8, 0.8])
        JW = J @ W6
        JJt = JW @ J.T + damp * np.eye(6)
        dq = W6 @ J.T @ np.linalg.solve(JJt, e6)
        q = np.array([self.d.qpos[a] for a in self.qadr])
        q_new = np.clip(q + gain * dq, q - 0.35, q + 0.35)
        return np.clip(q_new, self.lo, self.hi), np.linalg.norm(err_p)

    def set_ctrl(self, q):
        for a, v in zip(self.aids, q):
            self.d.ctrl[a] = v


def reach(m, d, targets, seconds=4.0, settle_steps=4, tol=0.01, on_frame=None,
          ease=True, settle_extra=3.0, grav_ff=False):
    """Servo one or both arms to Cartesian targets {'left': xyz, 'right': xyz}.

    With ease=True (default) the commanded target glides from the current EE
    position to the goal on a smoothstep profile over `seconds`, then holds for
    up to `settle_extra` seconds (early-exits once all arms are within `tol`).
    Returns final per-arm position errors (m)."""
    arms = {side: Arm(m, d, side) for side in targets}
    goals = {s: np.asarray(t, float) for s, t in targets.items()}
    starts = {s: a.ee_pos() for s, a in arms.items()}
    dt = m.opt.timestep * settle_steps
    n_move = int(seconds / dt)
    n_settle = int(settle_extra / dt)
    if grav_ff:
        # gravity-only feed-forward (qvel=0 -> no Coriolis, stable): cancel the
        # standing sag on the robot's hinge joints; free bodies (parts) still fall
        _hinge = np.array([m.jnt_type[m.dof_jntid[i]] == mujoco.mjtJoint.mjJNT_HINGE
                           for i in range(m.nv)], float)
        _gff = np.zeros(m.nv)
    for i in range(n_move + n_settle):
        s = smoothstep(i / max(n_move, 1)) if ease else 1.0
        for side, arm in arms.items():
            now_target = starts[side] + (goals[side] - starts[side]) * s
            q, _ = arm.ik_step(now_target)
            arm.set_ctrl(q)
        if grav_ff:
            _qv = d.qvel.copy(); d.qvel[:] = 0.0
            mujoco.mj_rne(m, d, 0, _gff); d.qvel[:] = _qv
        for _ in range(settle_steps):
            if grav_ff:
                d.qfrc_applied[:] = _gff * _hinge
            mujoco.mj_step(m, d)
        if on_frame:
            on_frame(i)
        if i >= n_move and tol and all(
                np.linalg.norm(goals[s] - a.ee_pos()) < tol for s, a in arms.items()):
            break
    if grav_ff:
        d.qfrc_applied[:] = 0.0      # clear the feed-forward — never leak it on
    return {s: float(np.linalg.norm(goals[s] - a.ee_pos())) for s, a in arms.items()}


def move_joints(m, d, arm_pose, seconds=2.0, settle_steps=4, on_frame=None):
    """Joint-space eased move of BOTH arms to a named pose dict like
    {'a0':0.3,'a1':0.3,'a3':0.8}. Used to leave the hanging (singular) rest
    pose before Cartesian IK takes over."""
    arms = {s: Arm(m, d, s) for s in ("left", "right")}
    goal = np.zeros(8)
    for k, v in arm_pose.items():
        goal[int(k[1])] = v
    start = {s: np.array([d.qpos[a] for a in arm.qadr]) for s, arm in arms.items()}
    n = int(seconds / (m.opt.timestep * settle_steps))
    for i in range(n):
        s_ = smoothstep((i + 1) / n)
        for side, arm in arms.items():
            arm.set_ctrl(start[side] + (goal - start[side]) * s_)
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)
        if on_frame:
            on_frame(i)


GRASP_EQ = {"left": "grasp_left", "right": "grasp_right"}


def grasp(m, d, side):
    """Engage the v1 'magnetic' grasp: freeze the part's current pose relative
    to the wrist via the pre-declared weld constraint (real gripper geometry is
    unknown until the hardware arrives — documented stand-in).

    Still the stand-in everywhere except one place: on a scene built by
    `make_cell_scene.py --gripper`, sequencer.py grips the peg with M4's
    actuated jaws (`make_gripper_cell.py`) and never calls this for the RIGHT
    hand. The LEFT hand welds there too, and the default cell welds both.

    Computes the relative pose at engage time and writes it into eq_data, so
    the weld holds the part exactly where it is (no snap)."""
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_EQ[side])
    b1 = m.eq_obj1id[eq]   # wrist body
    b2 = m.eq_obj2id[eq]   # part body
    # relative pose of body2 in body1 frame
    R1 = d.xmat[b1].reshape(3, 3)
    p_rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q1 = d.xquat[b1].copy(); q2 = d.xquat[b2].copy()
    q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, q1)
    q_rel = np.zeros(4); mujoco.mju_mulQuat(q_rel, q1inv, q2)
    m.eq_data[eq, :] = 0
    m.eq_data[eq, 3:6] = p_rel     # relpose position
    m.eq_data[eq, 6:10] = q_rel    # relpose quaternion
    m.eq_data[eq, 10] = 1.0        # torquescale
    d.eq_active[eq] = 1


def release(m, d, side):
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, GRASP_EQ[side])
    d.eq_active[eq] = 0


def hold(m, d, seconds=1.0, settle_steps=4, on_frame=None):
    """Keep current actuator targets; just run physics (used for video tails)."""
    n = int(seconds / (m.opt.timestep * settle_steps))
    for i in range(n):
        for _ in range(settle_steps):
            mujoco.mj_step(m, d)
        if on_frame:
            on_frame(i)
