"""Optional mink-backed IK for the Skate arms — an opt-in, drop-in alternative
to :meth:`ArmKinematics.ik_step` that adds PROACTIVE self-collision avoidance.

Same one-glide-step-per-tick contract as the pure-numpy DLS
(``step(arm, q26, target, target_R, q_ref) -> (new_q26, pos_err_m)``), but the
Cartesian drag is solved by mink's MuJoCo differential-IK QP with a
collision-avoidance limit: the arm glides to a safe standoff from the torso /
legs / other arm instead of the reactive capsule-guard reverting a step (which
can stall). Measured on skt_v3: driving the hand 22 cm into the torso ends at a
+13 mm body standoff instead of -90 mm of penetration.

Fully optional. Requires ``mink`` + ``qpsolvers[daqp]`` and a MuJoCo collision
model carrying ``ee_left`` / ``ee_right`` sites (``skt_v3_collision.xml``). The
heavy imports live inside the constructor, and the caller (the bridge) wraps
every call in try/except that falls back to the numpy DLS — so a missing
dependency or a transient solver failure can never break drag-IK. Off by
default; the server enables it with ``--ik mink`` (or ``SKATE_IK=mink``).
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = {"left": list(range(8, 15)), "right": list(range(16, 23))}
_EE = {"left": "ee_left", "right": "ee_right"}

# Only the DISTAL arm links get collision-avoidance vs the body: the shoulder /
# upperArm sit right next to the torso by construction, so including them would
# fight a permanent (false) proximity constraint. Names follow the skt_v3 model
# (the right arm is the "_Mirror__1" chain).
_DISTAL = {
    "left":  ["lowArm_1", "wrist_a0_1", "wrist_a1_1", "wrist_a2_1", "wrist_a3_1"],
    "right": ["lowArm_Mirror__1", "wrist_a0_Mirror__1", "wrist_a1_Mirror__1",
              "wrist_a2_Mirror__1", "wrist_a3_Mirror__1"],
}
_TORSO = ["Skate_body", "hip_1", "upperLeg_1", "hip_Mirror__1",
          "upperLeg_Mirror__1", "neck_1", "head_1"]
# proximal links of the OTHER arm that a distal arm should also dodge
_OTHER_PROX = {"left": ["upperArm_Mirror__1", "midArm_Mirror__1"],
               "right": ["upperArm_1", "midArm_1"]}


def _other(arm):
    return "right" if arm == "left" else "left"


class MinkIK:
    """mink differential-IK with self-collision avoidance for both Skate arms.

    One instance holds the shared MuJoCo model and the model-only limits;
    :meth:`step` solves ONE arm per call, freezing every other joint, so it
    slots straight into the bridge's per-arm IK loop as a solver swap.
    """

    def __init__(self, collision_xml, vel=3.0, min_dist=0.02,
                 detect=0.10, w_ori=0.6, posture_cost=1e-3):
        import mujoco          # optional deps imported lazily: they only cost
        import mink            # anything when the mink backend is actually on
        self._mj, self._mink = mujoco, mink
        self.model = mujoco.MjModel.from_xml_path(str(collision_xml))
        self.w_ori = float(w_ori)
        self._alljoints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                           for i in range(self.model.njnt)]
        for s in _EE.values():
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s) < 0:
                raise ValueError(f"collision model has no site '{s}'")
        prim = {}
        for i in range(self.model.ngeom):
            if int(self.model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(self.model.geom_bodyid[i]))
                prim.setdefault(b, []).append(i)
        self._cfg = mink.Configuration(self.model)
        self._conf_lim = mink.ConfigurationLimit(self.model)
        self._post = mink.PostureTask(self.model, cost=float(posture_cost))
        self._vlim, self._coll = {}, {}
        for arm, idx in ARM_JOINTS.items():
            active = {self._alljoints[i] for i in idx}
            self._vlim[arm] = mink.VelocityLimit(
                self.model, {j: (vel if j in active else 0.0)
                             for j in self._alljoints})
            distal = [g for b in _DISTAL[arm] for g in prim.get(b, [])]
            avoid = [g for b in (_TORSO + _DISTAL[_other(arm)] + _OTHER_PROX[arm])
                     for g in prim.get(b, [])]
            self._coll[arm] = mink.CollisionAvoidanceLimit(
                self.model, [(distal, avoid)],
                minimum_distance_from_collisions=float(min_dist),
                collision_detection_distance=float(detect),
                bound_relaxation=-1e-3)

    def step(self, arm, q26, target, target_R=None, q_ref=None, dt=0.02):
        """One glide step of ``arm`` toward world ``target`` (m). An optional
        orientation ``target_R`` (3x3) makes it a full 6-DoF pose solve. Returns
        ``(new_q26, pos_err_m_before_step)`` — the same contract the bridge
        expects from :meth:`ArmKinematics.ik_step`, so it is a drop-in swap.
        Only ``arm``'s joints move (every other joint is velocity-pinned). May
        raise on solver failure; the caller falls back to the numpy DLS.
        """
        mink = self._mink
        cfg = self._cfg
        cfg.update(np.asarray(q26, dtype=float))
        site = _EE[arm]
        Tcur = cfg.get_transform_frame_to_world(site, "site")
        perr = float(np.linalg.norm(Tcur.translation() - np.asarray(target, dtype=float)))
        if target_R is None:
            R, ocost = Tcur.rotation(), 0.0          # free wrist (3-DoF position)
        else:
            R, ocost = mink.SO3.from_matrix(np.asarray(target_R, dtype=float)), self.w_ori
        task = mink.FrameTask(site, "site", position_cost=1.0,
                              orientation_cost=ocost, lm_damping=1.0)
        task.set_target(mink.SE3.from_rotation_and_translation(
            R, np.asarray(target, dtype=float)))
        self._post.set_target(np.asarray(q_ref if q_ref is not None else q26, dtype=float))
        limits = [self._conf_lim, self._vlim[arm], self._coll[arm]]
        v = mink.solve_ik(cfg, [task, self._post], dt, solver="daqp",
                          damping=1e-3, limits=limits, safety_break=False)
        cfg.integrate_inplace(v, dt)
        out = np.asarray(q26, dtype=float).copy()
        newq = np.asarray(cfg.q, dtype=float)
        for i in ARM_JOINTS[arm]:
            out[i] = newq[i]
        return out, perr
