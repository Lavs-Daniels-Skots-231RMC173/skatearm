"""Gravity feed-forward in reach() cancels the position-servo sag — and is
cleared on exit so it never leaks into later physics. Headless; needs mujoco +
the control MJCF (set SKT_DIR to your skt_v3 folder).

    SKT_DIR=.../skt_v3 python3 sim/test_gravity_ff.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load():
    try:
        import mujoco
    except ImportError:
        return None, None
    xml = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3")) / "skt_v3_control.xml"
    if not xml.exists():
        return None, None
    return mujoco, mujoco.MjModel.from_xml_path(str(xml))


def _reach_sag(grav_ff):
    import primitives as P
    mujoco, m = _load()
    if m is None:
        return None
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    P.move_joints(m, d, {"a0": 0.3, "a1": 0.3, "a3": 0.8}, seconds=1.5)
    arms = {s: P.Arm(m, d, s) for s in ("left", "right")}
    tgt = {s: arms[s].ee_pos() + np.array([0.0, 0.10, 0.06]) for s in arms}
    err = P.reach(m, d, tgt, seconds=3.0, tol=0.0, settle_extra=3.0, grav_ff=grav_ff)
    leaked = float(np.max(np.abs(d.qfrc_applied)))
    return max(err.values()), leaked


def test_gravity_ff_cancels_sag():
    off, on = _reach_sag(False), _reach_sag(True)
    if off is None or on is None:
        pytest.skip("mujoco / control model not available")
    err_off, _ = off
    err_on, leaked_on = on
    assert err_off > 0.015, f"expected a visible sag without ff ({err_off*1000:.1f} mm)"
    assert err_on < 0.005, f"ff should cancel the sag ({err_on*1000:.1f} mm)"
    assert err_on < err_off / 3, "ff should shrink the reach error several-fold"
    assert leaked_on < 1e-9, "grav_ff must be cleared from qfrc_applied on exit"
    print(f"PASS gravity-ff: sag {err_off*1000:.1f} mm -> {err_on*1000:.1f} mm; "
          "qfrc_applied cleared on exit")


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
