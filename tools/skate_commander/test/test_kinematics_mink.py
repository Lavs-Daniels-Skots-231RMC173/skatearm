"""Optional mink IK backend: converges on a reachable target, moves ONLY the
target arm, holds orientation, and — its whole reason to exist — keeps the arm
off the body via proactive collision avoidance. Model-gated and skipped when
mink/mujoco aren't installed, so CI without the optional dep / model stays green.

    SKT_DIR=.../skt_v3 python3 test_kinematics_mink.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
COLL = SKT / "skt_v3_collision.xml"
_KM = Path(__file__).resolve().parents[1] / "skate_commander" / "kinematics_mink.py"


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def _setup():
    """(MinkIK-instance, module, mujoco-model, mujoco-data) or None to skip."""
    try:
        import mujoco  # noqa: F401
        import mink    # noqa: F401
    except ImportError as e:
        _skip(f"optional dep missing: {e}"); return None
    if not COLL.exists():
        _skip("no collision model (set SKT_DIR)"); return None
    import mujoco
    spec = importlib.util.spec_from_file_location("kinematics_mink", _KM)
    km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
    ik = km.MinkIK(str(COLL))
    m = mujoco.MjModel.from_xml_path(str(COLL))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    return ik, km, m, d


def _ee(m, d, name, q):
    import mujoco
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    return d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)].copy()


def test_mink_reaches_and_freezes_other_joints():
    s = _setup()
    if s is None:
        return
    ik, km, m, d = s
    q0 = np.array(d.qpos, float)
    tgt = _ee(m, d, "ee_left", q0) + np.array([-0.06, 0.03, 0.05])   # reachable
    q = q0.copy()
    for _ in range(250):
        q, err = ik.step("left", q, tgt)
    non_left = [i for i in range(26) if i not in range(8, 15)]
    assert np.linalg.norm(_ee(m, d, "ee_left", q) - tgt) < 2e-3       # < 2 mm
    assert np.max(np.abs((q - q0)[non_left])) < 1e-9                  # only the left arm moved


def test_mink_avoids_torso_penetration():
    s = _setup()
    if s is None:
        return
    ik, km, m, d = s
    import mujoco
    prim = {}
    for i in range(m.ngeom):
        if int(m.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i]))
            prim.setdefault(b, []).append(i)
    distal = [g for bb in km._DISTAL["left"] for g in prim.get(bb, [])]
    torso = [g for bb in km._TORSO for g in prim.get(bb, [])]
    q0 = np.array(d.qpos, float)
    tgt = _ee(m, d, "ee_left", q0) + np.array([0.22, 0.0, 0.12])      # deep INTO the torso
    q = q0.copy()
    for _ in range(350):
        q, err = ik.step("left", q, tgt)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    gap = min(mujoco.mj_geomDistance(m, d, g1, g2, 2.0, None)
              for g1 in distal for g2 in torso)
    assert gap > -0.005          # mink holds a standoff instead of penetrating


def test_mink_holds_orientation():
    s = _setup()
    if s is None:
        return
    ik, km, m, d = s
    import mujoco
    q0 = np.array(d.qpos, float)
    p_hold = _ee(m, d, "ee_left", q0)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_left")
    d.qpos[:] = q0; mujoco.mj_forward(m, d)
    Rcur = d.site_xmat[sid].reshape(3, 3).copy()
    th = np.radians(25)
    Rx = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    Rtgt = Rx @ Rcur
    q = q0.copy()
    for _ in range(250):
        q, err = ik.step("left", q, p_hold, target_R=Rtgt)
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    Rend = d.site_xmat[sid].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rtgt @ Rend.T) - 1) / 2, -1, 1)))
    assert np.linalg.norm(_ee(m, d, "ee_left", q) - p_hold) < 3e-3    # position held
    assert ang < 1.0                                                 # orientation reached


if __name__ == "__main__":
    for fn in (test_mink_reaches_and_freezes_other_joints,
               test_mink_avoids_torso_penetration,
               test_mink_holds_orientation):
        fn(); print("ok:", fn.__name__)
