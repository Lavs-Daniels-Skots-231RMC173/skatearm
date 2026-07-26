"""The cockpit's TCP wrench: which backend feeds it, and can the fallback fly.

Two halves, because M1 asks that either can back the interface:

* the ESTIMATOR (bridge._tcp_force) -- recover an applied end-effector force
  from the joint torques via F = (J·Jᵀ)⁻¹·J·tau over the 3×N position
  Jacobian: pure linear algebra plus a real-arm Jacobian check;
* the SELECTION (bridge._tcp_wrench) -- a measured wrist F/T reading wins
  wherever one exists, per arm, and says so in ``src``; the estimate is what
  fills in for hardware with no cell.

    SKT_DIR=.../skt_v3 python -m pytest test/test_force.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_commander.kinematics import ArmKinematics   # noqa: E402
from skate_commander.urdf import parse_urdf             # noqa: E402

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))


def _estimate(J, tau_arm):
    """Exactly the bridge's estimator: F = (J·Jᵀ)⁻¹·J·tau."""
    JJt = J @ J.T + 1e-9 * np.eye(3)
    return np.linalg.solve(JJt, J @ tau_arm)


def test_recovers_applied_force_synthetic():
    """For any full-rank 3×N Jacobian, tau = Jᵀ·F must invert back to F."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        J = rng.standard_normal((3, 7))          # full row rank almost surely
        F = rng.standard_normal(3) * 10.0
        tau = J.T @ F                            # joint torques an external F produces
        assert np.allclose(_estimate(J, tau), F, atol=1e-6)


def test_ignores_nullspace_torque():
    """Torque in the Jacobian null space (dofs that don't move the TCP) must
    contribute ~zero estimated TCP force."""
    J = np.zeros((3, 7))
    J[0, 0] = J[1, 1] = J[2, 2] = 1.0            # only the first 3 dofs move the TCP
    tau = np.array([0, 0, 0, 7, -4, 2, 9], float)
    assert np.allclose(_estimate(J, tau), [0, 0, 0], atol=1e-9)


def test_real_arm_jacobian_recovery():
    """With the actual SkateArm position Jacobian (3×N), the estimator recovers
    an applied TCP force at a well-conditioned pose."""
    urdf = SKT / "skt_v3.urdf"
    if not urdf.exists():
        pytest.skip("no skt_v3.urdf (set SKT_DIR)")
    model = parse_urdf(urdf)
    rng = np.random.default_rng(1)
    for arm in ("left", "right"):
        kin = ArmKinematics(model, arm)
        q = np.zeros(26)
        q[np.asarray(kin.idx, dtype=int)] = np.linspace(0.2, 0.8, len(kin.idx))
        _, J = kin._fk_jac_fast(q)
        assert J.shape == (3, len(kin.idx))
        if np.linalg.svd(J, compute_uv=False)[-1] < 1e-3:
            continue                             # singular here; skip this arm/pose
        F = rng.standard_normal(3) * 5.0
        tau = J.T @ F
        assert np.allclose(_estimate(J, tau), F, atol=1e-6)


# ── backend selection: measured wrench vs joint-torque estimate ──────────────

class _State:
    """The slice of TelemetryState the wrench path actually touches."""

    def __init__(self, q=None, tau=None, wrenches=None):
        self._q, self._tau, self._w = q, tau, wrenches

    def dof_pos(self):
        return self._q

    def dof_torque(self):
        return self._tau

    def wrenches(self):
        return self._w


def _bridge():
    """A RobotBridge with only ``kin`` populated -- the wrench path needs
    nothing else, and building the full bridge would drag in a live link."""
    urdf = SKT / "skt_v3.urdf"
    if not urdf.exists():
        pytest.skip("no skt_v3.urdf (set SKT_DIR)")
    from skate_commander.bridge import RobotBridge

    model = parse_urdf(urdf)
    b = RobotBridge.__new__(RobotBridge)
    b.kin = {"left": ArmKinematics(model, "left"),
             "right": ArmKinematics(model, "right")}
    return b


def _pose(b):
    q = np.zeros(26)
    for kin in b.kin.values():
        q[np.asarray(kin.idx, dtype=int)] = np.linspace(0.2, 0.8, len(kin.idx))
    return q


MEAS = {"left": {"f": [1.5, -2.0, 9.5], "m": [0.1, 0.0, -0.42]},
        "right": {"f": [0.0, 0.0, 0.4], "m": [0.0, 0.01, 0.0]}}


def test_estimate_is_the_fallback_when_no_sensor():
    """No wrist F/T on the wire (i.e. the real Skate) — the overlay still gets
    a force, tagged as an estimate, with no moment to give."""
    b = _bridge()
    q = _pose(b)
    st = _State(q, np.zeros(26), None)
    out = b._tcp_wrench(st)
    for arm, kin in b.kin.items():
        assert out[arm]["src"] == "estimate"
        assert out[arm]["m"] is None          # position Jacobian: no moments
        assert len(out[arm]["f"]) == 3 and len(out[arm]["p"]) == 3
        _p, J = kin._fk_jac_fast(q)
        F = np.random.default_rng(2).standard_normal(3) * 4.0
        tau = np.zeros(26)
        tau[np.asarray(kin.idx, dtype=int)] = J.T @ F
        got = b._tcp_wrench(_State(q, tau, None))[arm]
        assert np.allclose(got["f"], F, atol=1e-3), "estimate path broke"
        assert got["mag"] == round(float(np.linalg.norm(F)), 2)


def test_measured_wrench_wins():
    """A wrist F/T reading replaces the estimate outright — value, magnitude,
    moment and source — while the TCP anchor stays the kinematic one, so the
    arrow does not jump when the backend changes under it."""
    b = _bridge()
    q = _pose(b)
    tau = np.full(26, 3.0)                    # a NON-zero estimate to displace
    est = b._tcp_wrench(_State(q, tau, None))
    out = b._tcp_wrench(_State(q, tau, MEAS))
    for arm in b.kin:
        assert out[arm]["src"] == "sensor"
        assert out[arm]["f"] == MEAS[arm]["f"]
        assert out[arm]["m"] == MEAS[arm]["m"]
        assert out[arm]["mag"] == round(float(np.linalg.norm(MEAS[arm]["f"])), 2)
        assert out[arm]["p"] == est[arm]["p"], "TCP anchor moved with the source"
    assert est["left"]["f"] != out["left"]["f"], "test proved nothing"


def test_selection_is_per_arm():
    """One cell, one bare wrist is a real configuration (and a half-valid
    packet drops the bad arm). Each arm reports its own best source rather than
    the whole overlay falling back together."""
    b = _bridge()
    q = _pose(b)
    out = b._tcp_wrench(_State(q, np.full(26, 3.0), {"left": MEAS["left"]}))
    assert out["left"]["src"] == "sensor" and out["left"]["m"] is not None
    assert out["right"]["src"] == "estimate" and out["right"]["m"] is None


def test_no_pose_means_no_overlay():
    """Without joint positions there is no TCP to anchor an arrow to, so the
    overlay reports nothing — even though a wrench did arrive."""
    b = _bridge()
    assert b._tcp_wrench(_State(None, np.zeros(26), MEAS)) is None
    assert b._tcp_wrench(_State(_pose(b), None, MEAS)) is None
