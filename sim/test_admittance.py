"""M3 — Cartesian admittance / compliant control (CI).

Two behaviours that prove the arm YIELDS at a commanded stiffness rather than
holding rigid or latching a stop:

1. COMMANDED STIFFNESS — driven by a supplied constant wrench (the control law in
   isolation, no physical contact), the TCP settles to e = F/K per axis: a low-K
   axis yields ~F/K, a high-K axis barely moves, and the spring returns e to ~0
   when the wrench is removed. This is the "moves along the compliant axis AT the
   commanded stiffness" check — deterministic and tight.
2. PUSH-AND-YIELD — a REAL external force on the wrist, read through the M1 wrist
   sensor: the TCP physically yields along the pushed axis and returns toward the
   nominal pose when released (sensor-in-the-loop, end-to-end).

Headless; needs mujoco + the cell MJCF (SKT_DIR -> skt_v3). SKIPs cleanly without
them. The arm returns to its held pose after each case, so one staging is shared.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ARM = None
_SKIP = None


def _armed():
    """Stage the right arm once (approach + reach, gravity-ff); cache it."""
    global _ARM, _SKIP
    if _ARM is not None or _SKIP is not None:
        return _ARM
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _SKIP = "mujoco not available"; return None
    xml = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3")) / "skt_v3_cell.xml"
    if not xml.exists():
        _SKIP = "cell model not available"; return None
    from benchmark import load_cell
    from eval_admittance import stage_arm
    m = load_cell(str(xml.parent))
    d, arm = stage_arm(m)
    _ARM = (m, d, arm)
    return _ARM


def test_commanded_stiffness():
    """e = F/K per axis: compliant x/z yield, stiff y holds, spring returns."""
    st = _armed()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    from admittance import Admittance
    m, d, arm = st
    K = [400.0, 1600.0, 400.0]
    ad = Admittance(m, d, arm, K=K, zeta=1.0)
    ad.run(1.0, f_override=[8.0, 8.0, 0.0])
    e = ad.e * 1000.0                                   # mm
    # per-axis F/K: x 8/400=20 mm, y 8/1600=5 mm, z 0
    assert 16.0 < e[0] < 24.0, e
    assert 3.5 < e[1] < 6.5, e
    assert abs(e[2]) < 1.5, e
    assert e[0] > 2.5 * e[1], e                          # compliant axis yields >> stiff
    for i, fi in enumerate((8.0, 8.0, 0.0)):            # e·K reproduces the wrench
        assert abs(ad.e[i] * K[i] - fi) < 1.5, (i, ad.e[i] * K[i], fi)
    ad.run(0.8, f_override=[0.0, 0.0, 0.0])             # release -> spring returns
    assert np.linalg.norm(ad.e) * 1000.0 < 1.5, ad.e
    print(f"PASS commanded-stiffness: e={np.round(e,1)} mm (F/K=[20,5,0]), returns")


def test_push_and_yield():
    """A real force on the wrist (read through the M1 sensor) yields the TCP along
    the pushed axis and returns to nominal on release."""
    st = _armed()
    if st is None:
        print(f"SKIP: {_SKIP}"); return
    import mujoco  # noqa: F401
    from admittance import Admittance
    m, d, arm = st
    bid = m.site_bodyid[arm.site]
    ad = Admittance(m, d, arm, K=[500.0, 500.0, 500.0], zeta=1.0)
    for _ in range(40):
        ad.step()
    for _ in range(150):                                # push +8 N along y
        d.xfrc_applied[bid, :3] = [0.0, 8.0, 0.0]
        ad.step()
    d.xfrc_applied[bid, :] = 0.0
    wl = ad.ext_wrench()
    pushed = ad.tcp_offset() * 1000.0
    assert wl[1] > 5.0, wl                               # sensor read the external push
    assert pushed[1] > 8.0, pushed                       # TCP yielded along +y
    assert pushed[1] > abs(pushed[0]), pushed            # predominantly the pushed axis
    for _ in range(220):                                # release and return
        ad.step()
    back = ad.tcp_offset() * 1000.0
    assert np.linalg.norm(back) < 4.0, back              # returned to nominal
    print(f"PASS push-and-yield: yielded {np.round(pushed,1)} mm, "
          f"returned to {np.round(back,1)} mm")


if __name__ == "__main__":
    test_commanded_stiffness()
    test_push_and_yield()
    print("ADMITTANCE (M3) TEST DONE")
