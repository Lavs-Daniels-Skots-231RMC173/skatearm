"""Compliant contact mode (M3 admittance, in the cockpit): instead of latching a
soft-stop, the arm YIELDS its TCP along the admittance law driven by the
estimated contact force, and RETURNS when it is released. The default mode stays
"stop" (the existing soft-stop latch — the e2e twin in test_guard.py exercises
that path unchanged).

Pure — no sim endpoint: a synthetic per-arm force drives the bridge's compliant
response so the law is checked deterministically (e -> F/K), plus a regression
guard that "stop" still latches.

    SKT_DIR=.../skt_v3 python -m pytest test/test_compliant.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_commander.bridge import RobotBridge   # noqa: E402

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))


def test_default_is_stop_and_still_latches():
    """Default contact mode is 'stop', and the soft-stop latch is unchanged."""
    br = RobotBridge()
    br.targ = np.zeros(26)
    br.estop = False
    assert br.contact_mode == "stop"
    br._trip_contact()
    assert br.contact_tripped
    br.close()


def _kin_bridge():
    from skate_commander.urdf import parse_urdf
    from skate_commander.kinematics import ArmKinematics
    urdf = SKT / "skt_v3.urdf"
    if not urdf.exists():
        return None
    model = parse_urdf(urdf)
    kin = {a: ArmKinematics(model, a) for a in ("left", "right")}
    br = RobotBridge(kin=kin)
    br.targ = np.zeros(26)
    br.estop = False
    return br


def test_compliant_yields_to_F_over_K_and_returns():
    br = _kin_bridge()
    if br is None:
        print("SKIP: no skt_v3.urdf (set SKT_DIR)"); return
    assert br.set_contact_mode("compliant") == "compliant"
    arm = "right"
    base = br.kin[arm].fk(br.targ).copy()
    F = np.array([6.0, 0.0, 0.0])
    dt = 1.0 / 60.0
    rest = {"left": {"f": [0, 0, 0]}, "right": {"f": [0, 0, 0]}}
    push = {"left": {"f": [0, 0, 0]}, "right": {"f": list(F)}}
    br._compliant_yield(rest, dt)                       # baseline F0 at rest (no move)
    for _ in range(500):
        br._compliant_yield(push, dt)
    e = br._adm_e[arm]
    assert np.linalg.norm(e - F / br.compliant_K) < 1e-3, e         # law: e -> F/K
    yielded = np.asarray(br.ik_targets[arm]) - base
    assert np.linalg.norm(yielded - F / br.compliant_K) < 3e-3, yielded  # TCP target moved by e
    assert not br.contact_tripped                        # never latched in compliant mode
    for _ in range(900):                                 # release -> spring returns
        br._compliant_yield(rest, dt)
    assert np.linalg.norm(br._adm_e[arm]) < 1e-3, br._adm_e[arm]
    br.close()


if __name__ == "__main__":
    test_default_is_stop_and_still_latches()
    test_compliant_yields_to_F_over_K_and_returns()
    print("COMPLIANT-MODE TEST DONE")
