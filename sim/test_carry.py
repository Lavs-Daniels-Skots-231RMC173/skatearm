"""M3+ — bimanual compliant carry (CI).

Both arms grasp a shared bar; the object-level Cartesian admittance
(`carry.CompliantCarry`) is the principled generalisation of demo_dual_carry's
1-D tanh load-equaliser. One behaviour proves it earns its keep: under a real
external disturbance mid-carry the whole two-arm assembly YIELDS along the pushed
axis and RETURNS to the nominal pose on release, without dropping the bar.

Headless; needs mujoco + the collision MJCF (SKT_DIR -> skt_v3). SKIPs cleanly
without them. Staging (approach + bimanual grasp) is cached for the case.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_C = None
_SKIP = None


def _carry():
    global _C, _SKIP
    if _C is not None or _SKIP is not None:
        return _C
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _SKIP = "mujoco not available"; return None
    skt = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
    if not (skt / "skt_v3_collision.xml").exists():
        _SKIP = "collision model not available"; return None
    from carry import stage_bar_grasp
    _C = stage_bar_grasp(str(skt))
    return _C


def test_compliant_carry_yields_and_holds():
    st = _carry()
    if st is None:
        pytest.skip(_SKIP)
    import mujoco
    from carry import CompliantCarry
    m, d, armL, armR = st
    bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bar")
    cc = CompliantCarry(m, d, armL, armR, K=(600., 600., 600.), Mv=(3., 3., 3.), zeta=1.0)
    obj0 = cc.object_pos().copy()
    for _ in range(40):                                   # baseline settle
        cc.step()
    for _ in range(160):                                  # push +10 N along y on the bar
        d.xfrc_applied[bar, :3] = [0.0, 10.0, 0.0]
        cc.step()
    d.xfrc_applied[bar, :] = 0.0
    wl = cc.ext_wrench()
    pushed = (cc.object_pos() - obj0) * 1000.0
    assert wl[1] > 6.0, wl                                # net external wrench was read
    assert pushed[1] > 8.0, pushed                        # the assembly yielded along +y
    assert pushed[1] > abs(pushed[0]), pushed             # predominantly the pushed axis
    assert d.xpos[bar][2] > 0.14, d.xpos[bar]             # bar NOT dropped under the push
    for _ in range(260):                                  # release and return
        cc.step()
    back = (cc.object_pos() - obj0) * 1000.0
    assert np.linalg.norm(back) < 5.0, back               # returned to the nominal carry pose
    assert d.xpos[bar][2] > 0.14, d.xpos[bar]             # still held
    print(f"PASS compliant-carry: yielded {np.round(pushed,1)} mm, "
          f"returned {np.round(back,1)} mm, bar held (z={d.xpos[bar][2]:.3f})")


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
