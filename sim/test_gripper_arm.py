"""M4 (arm integration) — weld-free grasp + carry + place on the arm (CI).

The right wrist's parallel-jaw gripper (make_gripper_cell) grasps a part to a
target force and the arm moves it — with the world-pin released the part is held
by FRICTION, not a weld:

1. GRASP + CARRY — the part stays fixed in the wrist frame through a ~10 cm move
   (small drift = carried by the gripper, not welded).
2. GRASP + CARRY + PLACE — a full pick-and-place: carry the part over the place
   bin, descend, and OPEN the jaws to release it onto the bin.

Runs on the OPT-IN `skt_v3_gripper_cell` model (a separate file), so the default
cell model and every other test are untouched. Headless; needs mujoco + the
collision MJCF (SKT_DIR). SKIPs cleanly without them. Model cached across cases.
"""
import os
import sys
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
    skt = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
    if not (skt / "skt_v3_collision.xml").exists():
        _SKIP = "collision model not available"; return None
    import mujoco
    from make_gripper_cell import make
    _M = mujoco.MjModel.from_xml_path(make(str(skt)))
    return _M


def test_grasp_and_carry_weld_free():
    m = _model()
    if m is None:
        pytest.skip(_SKIP)
    from gripper_arm import grasp_and_carry
    r = grasp_and_carry(m)
    assert r["grasp_n"] > 3.0, r                       # gripped near the 5 N target
    assert not r["pin_active"], r                      # the world-pin is OFF during the carry
    assert r["carry_grasp_n"] > 1.0, r                 # grasp force persists through the carry
    assert r["drift_mm"] < 20.0, r                     # part stays fixed in the wrist frame ...
    assert r["carried"], r                             # ... carried by the gripper, not a weld
    print(f"PASS grasp+carry (weld-free): grasp {r['grasp_n']} N, "
          f"drift {r['drift_mm']} mm, carry-grasp {r['carry_grasp_n']} N")


def test_grasp_carry_place_weld_free():
    m = _model()
    if m is None:
        pytest.skip(_SKIP)
    from gripper_arm import grasp_carry_place
    r = grasp_carry_place(m)
    assert r["grasp_n"] > 3.0, r                       # gripped near the target
    assert r["released"], r                            # jaws opened -> grasp released
    assert r["on_bin"], r                              # part resting on the place bin
    assert r["placed"], r                              # released AND on the bin = placed
    print(f"PASS grasp+carry+place (weld-free): grasp {r['grasp_n']} N, "
          f"part_z {r['part_z']} m, placed {r['placed']}")


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
