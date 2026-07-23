"""M4 (arm integration) — weld-free grasp + carry on the arm (CI).

The right wrist's parallel-jaw gripper (make_gripper_cell) grasps a part to a
target force and the arm carries it: with the world-pin released the part must
stay fixed in the wrist frame through the motion — i.e. held by the gripper's
FRICTION, not a weld. This is the weld-free replacement for the magnetic-weld
grasp stand-in, on the actual arm.

Runs on the OPT-IN `skt_v3_gripper_cell` model (a separate file), so the default
cell model and every other test are untouched. Headless; needs mujoco + the
collision MJCF (SKT_DIR). SKIPs cleanly without them. One staged run, cached.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_R = None
_SKIP = None


def _result():
    global _R, _SKIP
    if _R is not None or _SKIP is not None:
        return _R
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _SKIP = "mujoco not available"; return None
    skt = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
    if not (skt / "skt_v3_collision.xml").exists():
        _SKIP = "collision model not available"; return None
    import mujoco
    from make_gripper_cell import make
    from gripper_arm import grasp_and_carry
    m = mujoco.MjModel.from_xml_path(make(str(skt)))
    _R = grasp_and_carry(m)
    return _R


def test_grasp_and_carry_weld_free():
    r = _result()
    if r is None:
        print(f"SKIP: {_SKIP}"); return
    assert r["grasp_n"] > 3.0, r                       # gripped near the 5 N target
    assert not r["pin_active"], r                      # the world-pin is OFF during the carry
    assert r["carry_grasp_n"] > 1.0, r                 # grasp force persists through the carry
    assert r["drift_mm"] < 20.0, r                     # part stays fixed in the wrist frame ...
    assert r["carried"], r                             # ... carried by the gripper, not a weld
    print(f"PASS arm grasp+carry (weld-free): grasp {r['grasp_n']} N, "
          f"drift {r['drift_mm']} mm, carry-grasp {r['carry_grasp_n']} N")


if __name__ == "__main__":
    test_grasp_and_carry_weld_free()
    print("GRIPPER-ARM (M4) TEST DONE")
