"""FK must match MuJoCo exactly; DLS IK must converge inside joint limits.

Needs mujoco + the official clone:
    SKT_DIR=.../skt_v3 SKATE_MJCF=.../skt_v3_control.xml python3 test_kinematics.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_commander.kinematics import (ArmKinematics, R_to_rpy,  # noqa: E402
                                        reach_map, rot_error, rpy_to_R)
from skate_commander.urdf import parse_urdf           # noqa: E402

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
MJCF = os.environ.get("SKATE_MJCF", str(SKT / "skt_v3_control.xml"))



def _skip(msg):
    """Real pytest.skip under pytest; clean print when run as a standalone script."""
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def _clamped_random(model, rng, scale=0.5):
    q = rng.uniform(-scale, scale, 26)
    for j in model["joints"]:
        if j["index"] is not None and j["lower"] is not None:
            q[j["index"]] = np.clip(q[j["index"]], j["lower"], j["upper"])
    return q


def test_fk_matches_mujoco_and_ik_converges():
    try:
        import mujoco
    except ImportError:
        _skip("mujoco not installed"); return
    if not Path(MJCF).exists():
        _skip("no control model"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    mm = mujoco.MjModel.from_xml_path(MJCF)
    dd = mujoco.MjData(mm)
    rng = np.random.default_rng(3)

    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        ee_link = next(j for j in model["joints"]
                       if j["index"] == kin.idx[-1])["child"]
        bid = mm.body(ee_link).id
        worst = 0.0
        for _ in range(5):
            q = _clamped_random(model, rng)
            dd.qpos[:26] = q
            mujoco.mj_forward(mm, dd)
            worst = max(worst, float(np.linalg.norm(dd.xpos[bid] - kin.fk(q))))
        assert worst < 1e-6, f"{arm}: FK diverges from MuJoCo ({worst} m)"

        ok = 0
        for _ in range(8):
            target = kin.fk(_clamped_random(model, rng, 0.8))
            q = np.zeros(26)
            err = 1.0
            for _ in range(300):
                q, err = kin.ik_step(q, target)
                if err < 0.005:
                    break
            ok += err < 0.005
            for k, i in enumerate(kin.idx):
                assert kin.lo[k] - 1e-9 <= q[i] <= kin.hi[k] + 1e-9
        assert ok >= 7, f"{arm}: IK converged only {ok}/8"
        print(f"PASS {arm}: FK exact, IK {ok}/8")


def test_tool_offset_tracks_mujoco():
    """FK with a TCP offset must equal MuJoCo xpos + xmat @ tool, and the
    IK must drive the OFFSET point (not the wrist) onto the target."""
    try:
        import mujoco
    except ImportError:
        _skip("mujoco not installed"); return
    if not Path(MJCF).exists():
        _skip("no control model"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    mm = mujoco.MjModel.from_xml_path(MJCF)
    dd = mujoco.MjData(mm)
    rng = np.random.default_rng(7)
    tool = np.array([0.02, -0.015, 0.12])     # an off-axis 12 cm "tool"

    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        kin.tool = tool
        ee_link = next(j for j in model["joints"]
                       if j["index"] == kin.idx[-1])["child"]
        bid = mm.body(ee_link).id
        worst = 0.0
        for _ in range(5):
            q = _clamped_random(model, rng)
            dd.qpos[:26] = q
            mujoco.mj_forward(mm, dd)
            want = dd.xpos[bid] + dd.xmat[bid].reshape(3, 3) @ tool
            worst = max(worst, float(np.linalg.norm(want - kin.fk(q))))
        assert worst < 1e-6, f"{arm}: tool FK diverges ({worst} m)"

        target = kin.fk(_clamped_random(model, rng, 0.7))
        q = np.zeros(26)
        err = 1.0
        for _ in range(300):
            q, err = kin.ik_step(q, target)
            if err < 0.005:
                break
        assert err < 0.005, f"{arm}: tool IK did not converge ({err} m)"
        print(f"PASS {arm}: tool-offset FK exact ({worst:.2e} m), IK converges")


def test_posture_hold_no_winding():
    """Jogging the TCP out and back must return (almost) the same joint
    pose. Without the null-space posture anchor the 4 redundant DoF drift
    and the arm slowly winds itself up (the v0.5.0 'выкручивает' bug)."""
    if not Path(MJCF).exists():
        _skip("no control model"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    import numpy as np

    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        q0 = np.zeros(26)
        q0[11] = q0[19] = np.radians(90)            # elbows-bent home
        p0 = kin.fk(q0)

        def roundtrip(q_ref):
            q = q0.copy()
            for tgt in (p0 + [0, 0.15, 0], p0 + [0, 0, 0.12], p0):
                for _ in range(250):
                    q, err = kin.ik_step(q, tgt, q_ref=q_ref)
                    if err < 1e-3:
                        break
            return float(np.max(np.abs(q - q0)))

        anchored = roundtrip(q0)
        free = roundtrip(None)
        assert anchored < 0.06, \
            f"{arm}: wound up {np.degrees(anchored):.1f} deg with anchor"
        print(f"PASS {arm}: out-and-back posture drift "
              f"{np.degrees(anchored):.2f} deg (anchored) vs "
              f"{np.degrees(free):.1f} deg (free)")


def test_fast_jacobian_and_reach_map():
    """Geometric fast Jacobian == numeric central-diff one, and reach_map
    returns a sane dexterity cloud. Pure numpy — no mujoco needed."""
    if not (SKT / "skt_v3.urdf").exists():
        _skip("no URDF"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    kin = ArmKinematics(model, "right")
    base = np.zeros(26)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = base.copy(); q[kin.idx] = rng.uniform(kin.lo, kin.hi)
        p_f, J_f = kin._fk_jac_fast(q)
        assert np.max(np.abs(p_f - kin.fk(q))) < 1e-9, "fast FK must match"
        assert np.max(np.abs(J_f - kin.jacobian(q))) < 1e-6, "fast Jacobian must match numeric"
        assert abs(kin.manipulability_fast(q) - kin.manipulability(q)) < 1e-6
    pts = reach_map(kin, base, n=800, seed=1)
    assert len(pts) == 800
    m = np.array([p[3] for p in pts])
    assert 0.0 <= m.min() and m.max() <= 1.0 and m.mean() > 0.05, "manip in [0,1], non-degenerate"
    print(f"PASS fast-Jacobian == numeric (~1e-6); reach_map {len(pts)} pts, "
          f"manip mean {m.mean():.2f}")


def test_fk_pose_matches_mujoco():
    """The 6-DoF FK orientation (fk_pose R) must equal MuJoCo's body xmat —
    the same gold standard the position FK is already held to."""
    try:
        import mujoco
    except ImportError:
        _skip("mujoco not installed"); return
    if not Path(MJCF).exists():
        _skip("no control model"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    mm = mujoco.MjModel.from_xml_path(MJCF)
    dd = mujoco.MjData(mm)
    rng = np.random.default_rng(5)
    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        ee_link = next(j for j in model["joints"]
                       if j["index"] == kin.idx[-1])["child"]
        bid = mm.body(ee_link).id
        worst = 0.0
        for _ in range(5):
            q = _clamped_random(model, rng)
            dd.qpos[:26] = q
            mujoco.mj_forward(mm, dd)
            p, R = kin.fk_pose(q)
            worst = max(worst, float(np.max(np.abs(dd.xmat[bid].reshape(3, 3) - R))))
        assert worst < 1e-6, f"{arm}: fk_pose orientation diverges from MuJoCo ({worst})"
        print(f"PASS {arm}: fk_pose R matches MuJoCo ({worst:.1e})")


def test_fk_jac6_linear_matches_numeric():
    """The 6x7 geometric Jacobian's linear rows equal the numeric one, its
    angular rows are finite, and fk_pose's R matches fk_jac6's. Pure numpy."""
    if not (SKT / "skt_v3.urdf").exists():
        _skip("no URDF"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    kin = ArmKinematics(model, "right")
    rng = np.random.default_rng(2)
    base = np.zeros(26)
    for _ in range(15):
        q = base.copy(); q[kin.idx] = rng.uniform(kin.lo, kin.hi)
        p, R, J6 = kin.fk_jac6(q)
        assert np.max(np.abs(p - kin.fk(q))) < 1e-9
        assert np.max(np.abs(J6[:3] - kin.jacobian(q))) < 1e-6, "linear rows must match numeric"
        assert J6.shape == (6, len(kin.idx)) and np.all(np.isfinite(J6))
        _, RR = kin.fk_pose(q); assert np.max(np.abs(RR - R)) < 1e-12
    print("PASS fk_jac6: linear==numeric (~1e-6), angular finite, fk_pose R consistent")


def test_ik_6dof_pose_converges():
    """6-DoF IK nulls a nearby pose error — position AND orientation together —
    the way the drag-gizmo servos from the live wrist pose. (Cold-start GLOBAL
    6-DoF IK is a separate, known-hard problem with local minima; the cockpit
    never needs it — the gizmo always starts at the current EE pose.)"""
    if not (SKT / "skt_v3.urdf").exists():
        _skip("no URDF"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    rng = np.random.default_rng(11)
    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        ok = 0
        for _ in range(12):
            qg = _clamped_random(model, rng, 0.8)          # a reachable goal
            tp, tR = kin.fk_pose(qg)
            q = qg.copy()                                   # start near it (teleop reality)
            q[kin.idx] = np.clip(qg[kin.idx] + rng.uniform(-0.6, 0.6, len(kin.idx)),
                                 kin.lo, kin.hi)
            for _ in range(400):
                q, _ = kin.ik_step(q, tp, target_R=tR)
            pc, Rc = kin.fk_pose(q)
            pe = float(np.linalg.norm(tp - pc))
            oe = float(np.degrees(np.linalg.norm(rot_error(Rc, tR))))
            ok += pe < 0.01 and oe < 3.0
            for k, i in enumerate(kin.idx):
                assert kin.lo[k] - 1e-9 <= q[i] <= kin.hi[k] + 1e-9
        assert ok == 12, f"{arm}: 6-DoF pose servo converged only {ok}/12"
        print(f"PASS {arm}: 6-DoF pose servo {ok}/12 (pos < 10 mm, ori < 3 deg)")


def test_ik_6dof_bounded_near_singularity():
    """A far / ill-posed 6-DoF demand must stay finite with joint steps
    clipped by DLS — no NaNs, no runaway (the whole point of damping)."""
    if not (SKT / "skt_v3.urdf").exists():
        _skip("no URDF"); return
    model = parse_urdf(SKT / "skt_v3.urdf")
    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        _, tR = kin.fk_pose(np.zeros(26))
        far = kin.fk(np.zeros(26)) + np.array([1.0, 0.0, 0.0])   # 1 m out of reach
        q = np.zeros(26); prev = q.copy(); maxdq = 0.0
        for _ in range(300):
            q, _ = kin.ik_step(q, far, target_R=tR, dq_max=0.06)
            assert np.all(np.isfinite(q)), f"{arm}: non-finite q near singularity"
            maxdq = max(maxdq, float(np.max(np.abs(q - prev)))); prev = q.copy()
        assert maxdq <= 0.06 + 1e-9, f"{arm}: joint step exceeded dq_max ({maxdq})"
        for k, i in enumerate(kin.idx):
            assert kin.lo[k] - 1e-9 <= q[i] <= kin.hi[k] + 1e-9
        print(f"PASS {arm}: 6-DoF bounded near singularity (max|dq|={maxdq:.3f})")


def test_rot_error_matches_axis_angle():
    """rot_error recovers a known rotation's axis*angle, is zero for identical
    frames, and stays finite in the near-pi branch."""
    from skate_commander.kinematics import _axis_rot
    I = np.eye(3)
    assert np.linalg.norm(rot_error(I, I)) < 1e-12
    for ax in ([0, 0, 1.0], [0, 1.0, 0], [1.0, 0, 0], [1, 1, 1.0]):
        for ang in (0.3, 1.2, 3.0):
            R = _axis_rot(ax, ang)
            e = rot_error(I, R)                       # I -> R  ==  axis*ang
            assert abs(np.linalg.norm(e) - ang) < 1e-6, (ax, ang)
            axn = np.asarray(ax, float); axn /= np.linalg.norm(axn)
            assert np.linalg.norm(e / np.linalg.norm(e) - axn) < 1e-6
    e = rot_error(I, _axis_rot([0, 0, 1.0], np.pi - 1e-8))
    assert np.all(np.isfinite(e)) and abs(np.linalg.norm(e) - np.pi) < 1e-2
    print("PASS rot_error: axis-angle exact + near-pi finite")


def test_rpy_round_trip_and_gimbal_lock():
    """rpy_to_R / R_to_rpy are the one URDF fixed-axis convention in the tree.

    Three places now write an orientation down -- every joint frame parsed out
    of the URDF, the viewer's rotate gizmo, and the orientation a program hands
    to moveto -- and they agree only because they share this pair. So the
    composition is pinned against world-axis rotations built independently of
    it, rather than against a second copy of the same three matrices.

    Straight down is not an exotic pose in this cell; it is how the left tool
    approaches the base. It is also where an arcsin-based inverse reads a
    matrix entry an ulp from unity and throws away most of its digits, so the
    poles are pinned rather than assumed."""
    from skate_commander.kinematics import _axis_rot
    rng = np.random.default_rng(17)

    for _ in range(50):                     # fixed-axis XYZ == intrinsic ZYX
        r, p, y = rng.uniform(-np.pi, np.pi, 3)
        want = (_axis_rot([0, 0, 1.0], y) @ _axis_rot([0, 1.0, 0], p)
                @ _axis_rot([1.0, 0, 0], r))
        R = rpy_to_R(r, p, y)
        assert np.max(np.abs(R - want)) < 1e-12, "not fixed-axis roll->pitch->yaw"
        assert np.max(np.abs(R @ R.T - np.eye(3))) < 1e-12, "not orthonormal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-12, "not a rotation"

    worst = 0.0
    for _ in range(500):
        r = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        y = rng.uniform(-np.pi, np.pi)
        got = np.asarray(R_to_rpy(rpy_to_R(r, p, y)), float)
        worst = max(worst, float(np.max(np.abs(got - [r, p, y]))))
    assert worst < 1e-9, f"rpy round trip drifts {worst} rad"

    offs = [10.0 ** -k for k in range(15)] + [0.0]   # creep onto the pole
    for sgn in (1.0, -1.0):                 # straight down and straight up
        for off in offs:
            for _ in range(6):
                r, y = rng.uniform(-np.pi, np.pi, 2)
                R = rpy_to_R(r, sgn * (np.pi / 2 - off), y)
                roll, pitch, yaw = R_to_rpy(R)
                assert abs(pitch) <= np.pi / 2 + 1e-12, "pitch left its branch"
                back = float(np.linalg.norm(
                    rot_error(rpy_to_R(roll, pitch, yaw), R)))
                assert back < 1e-8, \
                    f"a near-vertical wrist lost {back} rad of round trip at {off}"
        roll, pitch, yaw = R_to_rpy(rpy_to_R(0.4, sgn * np.pi / 2, -1.1))
        assert roll == 0.0, "at the pole roll has to fold into yaw"
        assert abs(abs(pitch) - np.pi / 2) < 1e-12
    print(f"PASS rpy_to_R/R_to_rpy: fixed-axis ZYX, round trip {worst:.1e} rad, "
          "poles fold roll into yaw")


if __name__ == "__main__":
    test_fast_jacobian_and_reach_map()
    test_fk_matches_mujoco_and_ik_converges()
    test_tool_offset_tracks_mujoco()
    test_posture_hold_no_winding()
    test_fk_pose_matches_mujoco()
    test_fk_jac6_linear_matches_numeric()
    test_ik_6dof_pose_converges()
    test_ik_6dof_bounded_near_singularity()
    test_rot_error_matches_axis_angle()
    test_rpy_round_trip_and_gimbal_lock()
