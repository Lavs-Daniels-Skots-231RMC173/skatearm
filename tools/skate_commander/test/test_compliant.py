"""Compliant contact mode (M3 admittance, in the cockpit): instead of latching a
soft-stop, the arm YIELDS its TCP along the admittance law driven by the
estimated contact force, and RETURNS when it is released. The default mode stays
"stop" (the existing soft-stop latch — the e2e twin in test_guard.py exercises
that path unchanged).

Pure — no sim endpoint: a synthetic per-arm force drives the bridge's compliant
response so the law is checked deterministically (e -> F/K), plus a regression
guard that "stop" still latches.

The second half covers the cockpit's toggle: the "contact_mode" command routed
through the real server dispatcher, the snapshot field the UI paints from, and
the two things a mode switch must never do -- clear a latched stop, or move the
arm on the way out of a yield.

    SKT_DIR=.../skt_v3 python -m pytest test/test_compliant.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

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
        pytest.skip("no skt_v3.urdf (set SKT_DIR)")
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


def _cmd(br, cmd):
    """Route a raw command through the server dispatcher -- exactly what the
    cockpit's WebSocket delivers when the toggle is clicked."""
    from skate_commander.server import handle_command
    handle_command(br, cmd)


def test_ws_command_switches_the_mode_and_publishes_it():
    """The cockpit toggle is two halves: a command in, and a snapshot field the
    UI paints from. Pin both -- the chip and the segmented control read
    ``contact.mode`` and nothing else."""
    pytest.importorskip("fastapi")
    br = RobotBridge()
    br.targ = np.zeros(26)
    assert br.snapshot()["contact"]["mode"] == "stop"       # default, unchanged
    _cmd(br, {"type": "contact_mode", "mode": "compliant"})
    assert br.contact_mode == "compliant"
    assert br.snapshot()["contact"]["mode"] == "compliant"
    _cmd(br, {"type": "contact_mode", "mode": "stop"})
    assert br.snapshot()["contact"]["mode"] == "stop"
    br.close()


@pytest.mark.parametrize("cmd", [
    {"type": "contact_mode", "mode": "off"},
    {"type": "contact_mode", "mode": ""},
    {"type": "contact_mode", "mode": None},
    {"type": "contact_mode"},                        # no mode at all
    {"type": "contact_mode", "mode": ["compliant"]},
])
def test_ws_command_refuses_a_junk_mode(cmd):
    """A mode that SUSPENDS the soft-stop latch may only be entered by name: a
    malformed command must not change the mode in either direction, and must
    not raise (one bad frame would take the socket down with it)."""
    pytest.importorskip("fastapi")
    br = RobotBridge()
    br.targ = np.zeros(26)
    br.set_contact_mode("compliant")
    _cmd(br, cmd)
    assert br.contact_mode == "compliant"
    br.set_contact_mode("stop")
    _cmd(br, cmd)
    assert br.contact_mode == "stop"
    br.close()


def test_going_compliant_does_not_clear_a_latched_stop():
    """The toggle is not a hidden Resume. A stop already latched stays latched
    until the operator clears it -- the same rule OBSERVE follows when it lands
    every transition in E-STOP."""
    pytest.importorskip("fastapi")
    br = RobotBridge()
    br.targ = np.zeros(26)
    br.estop = False
    br._trip_contact()
    _cmd(br, {"type": "contact_mode", "mode": "compliant"})
    assert br.contact_tripped                    # still latched, still dampened
    br.clear_contact()                           # only the explicit action clears it
    assert not br.contact_tripped
    br.close()


def test_leaving_compliant_does_not_snap_the_arm_back():
    """Switching back to "stop" while the arm is yielded must not MOVE it. The
    admittance state is dropped, but the pose it produced is where the arm
    already is -- unwinding the yield would command a jump back into whatever
    the arm is leaning on."""
    br = _kin_bridge()
    if br is None:
        pytest.skip("no skt_v3.urdf (set SKT_DIR)")
    arm, dt = "right", 1.0 / 60.0
    br.set_contact_mode("compliant")
    rest = {"left": {"f": [0, 0, 0]}, "right": {"f": [0, 0, 0]}}
    push = {"left": {"f": [0, 0, 0]}, "right": {"f": [6.0, 0.0, 0.0]}}
    br._compliant_yield(rest, dt)
    for _ in range(500):
        br._compliant_yield(push, dt)
    yielded = np.asarray(br.ik_targets[arm], dtype=float).copy()
    assert np.linalg.norm(br._adm_e[arm]) > 5e-3, "the arm never yielded"
    br.set_contact_mode("stop")
    assert np.allclose(np.asarray(br.ik_targets[arm], dtype=float), yielded)
    assert not np.any(br._adm_e[arm])            # state dropped, not unwound
    br.close()


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
