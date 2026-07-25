"""M4 (scoped) — actuated parallel-jaw gripper: grasp-force control, friction
hold, and the grasp-slip curve (CI).

Behaviours proven on the opt-in `skt_v3_gripper` scene (a rectangular part):

1. GRASP-FORCE CONTROL — `close_to_force` regulates the MEASURED grasp force (the
   pad touch sensor) to a commanded target, and is monotone in the target.
2. FRICTION HOLD — with the world-pin released, the part is held by FRICTION alone
   (no weld); it does not fall and the grasp force persists.
3. GRASP-SLIP — a firmer grasp holds a larger downward payload before the part
   slips out (flat pad-on-face contact makes the hold scale with grasp force).

Self-contained scene (no robot meshes), so it never touches the arm
models/tests. Headless; SKIPs cleanly without mujoco.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

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


def _grasp(f_target, settle=250):
    import mujoco
    from gripper import Gripper
    m = _model()
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    g = Gripper(m, d)
    z0 = g.part_pos()[2]
    g.open(cycles=25)
    meas = g.close_to_force(f_target)
    g.set_pin(False)                                  # release the pin -> friction only
    for _ in range(settle):
        mujoco.mj_step(m, d)
    return g, z0, meas


def test_grasp_force_control_tracks_target():
    if _model() is None:
        pytest.skip(_SKIP)
    _, _, m2 = _grasp(2.0)
    _, _, m4 = _grasp(4.0)
    assert abs(m2 - 2.0) < 0.6, m2                     # measured grasp reaches 2 N
    assert abs(m4 - 4.0) < 0.6, m4                     # ... and 4 N
    assert m4 > m2 + 1.0, (m2, m4)                     # monotone in the target
    print(f"PASS grasp-force control: 2N->{m2:.2f}, 4N->{m4:.2f} N")


def test_friction_hold_without_weld():
    if _model() is None:
        pytest.skip(_SKIP)
    g, z0, meas = _grasp(3.0)
    assert g.holds(z0), g.part_pos()                   # held by friction after the pin releases
    assert abs(g.part_pos()[2] - z0) < 0.01, g.part_pos()   # barely moved
    assert g.grasp_force() > 1.0, g.grasp_force()      # grasp force persists (no weld)
    print(f"PASS friction hold: grasp {meas:.2f} N, part dz "
          f"{(g.part_pos()[2]-z0)*1000:+.1f} mm, hold {g.grasp_force():.2f} N")


def test_grasp_slip_grows_with_force():
    if _model() is None:
        pytest.skip(_SKIP)
    def slip_at(ft):
        g, _, _ = _grasp(ft, settle=60)
        return g.slip_payload(g.part_pos()[2])
    s2, s5 = slip_at(2.0), slip_at(5.0)
    assert s2 and s5, (s2, s5)                          # both actually slipped (a real measurement)
    assert 1.0 < s2 < 20.0 and 1.0 < s5 < 20.0, (s2, s5)
    assert s5 > s2 + 0.5, (s2, s5)                      # a firmer grasp holds a larger payload
    print(f"PASS grasp-slip: 2N grasp slips at {s2} N, 5N grasp at {s5} N")


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
