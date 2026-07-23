"""M4 (scoped) — actuated parallel-jaw gripper: grasp-force control + friction
hold (CI).

Two behaviours prove the gripper mechanism on the opt-in `skt_v3_gripper` scene:

1. GRASP-FORCE CONTROL — `close_to_force` regulates the MEASURED grasp force (the
   pad touch sensor) to a commanded target, and is monotone in the target.
2. FRICTION HOLD — with the world-pin released, the part is held by FRICTION alone
   (no weld); it does not fall and the grasp force persists.

Self-contained scene (no robot meshes), so it never touches the arm
models/tests. Headless; SKIPs cleanly without mujoco.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_M = None
_SKIP = None


def _model():
    global _M, _SKIP
    if _M is not None or _SKIP is not None:
        return _M
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _SKIP = "mujoco not available"; return None
    from make_gripper_scene import make
    skt = os.environ.get("SKT_DIR") or tempfile.gettempdir()   # never write into the repo
    _M = mujoco.MjModel.from_xml_path(make(skt))
    return _M


def _grasp(f_target):
    import mujoco
    from gripper import Gripper
    m = _model()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    g = Gripper(m, d)
    z0 = g.peg_pos()[2]
    g.open(cycles=25)
    meas = g.close_to_force(f_target)
    g.set_pin(False)                                  # release the pin -> friction only
    for _ in range(250):
        mujoco.mj_step(m, d)
    return g, z0, meas


def test_grasp_force_control_tracks_target():
    if _model() is None:
        print(f"SKIP: {_SKIP}"); return
    _, _, m2 = _grasp(2.0)
    _, _, m4 = _grasp(4.0)
    assert abs(m2 - 2.0) < 0.6, m2                     # measured grasp reaches 2 N
    assert abs(m4 - 4.0) < 0.6, m4                     # ... and 4 N
    assert m4 > m2 + 1.0, (m2, m4)                     # monotone in the target
    print(f"PASS grasp-force control: 2N->{m2:.2f}, 4N->{m4:.2f} N")


def test_friction_hold_without_weld():
    if _model() is None:
        print(f"SKIP: {_SKIP}"); return
    g, z0, meas = _grasp(3.0)
    assert g.holds(z0), g.peg_pos()                    # held by friction after the pin releases
    assert abs(g.peg_pos()[2] - z0) < 0.01, g.peg_pos()   # barely moved
    assert g.grasp_force() > 1.0, g.grasp_force()      # grasp force persists (no weld)
    print(f"PASS friction hold: grasp {meas:.2f} N, peg dz "
          f"{(g.peg_pos()[2]-z0)*1000:+.1f} mm, hold {g.grasp_force():.2f} N")


if __name__ == "__main__":
    test_grasp_force_control_tracks_target()
    test_friction_hold_without_weld()
    print("GRIPPER (M4 scoped) TEST DONE")
