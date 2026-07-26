"""The wrist wrench a client receives IS the wrist sensor's reading.

M1 built a true 6-axis wrist F/T sensor into the control MJCF and
``sim/test_ft_sensor.py`` calibrated it against known loads. This pins the
other half: that the number travelling over the UDP wire as telemetry id 6 is
that same calibrated reading -- same frame, same sign, same bits -- and not a
lookalike computed somewhere along the way.

Two independent assertions, because either alone is weak:

* EXACT -- the streamed floats equal ``R @ d.sensor(...).data`` recomputed from
  the endpoint's own unstepped MjData, to the bit. No tolerance to hide a
  frame slip or a lost rotation.
* PHYSICAL -- the streamed delta under a known applied load passes the same
  < 0.05 N calibration bound ``sim/test_ft_sensor.py`` asserts on the sensor
  directly. This is what pins the SIGN end to end: the sensor reports the
  reaction the wrist carries, and the cockpit's joint-torque estimator returns
  that same convention, so the two backends are interchangeable and neither
  needs a negation. Get this wrong and the force arrow flips the moment a real
  sensor appears -- which is exactly the bug a smoke test that only checks
  "the field is present" would ship.

Needs mujoco + the control-ready MJCF (skips cleanly if either is missing):

    SKT_DIR=.../skt_v3 python3 tools/skate_ros2/test/test_wrist_wrench.py
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_ros2.protocol import WRENCH_ID, SkateLink        # noqa: E402

HANDS = {"left": "wrist_a3_1", "right": "wrist_a3_Mirror__1"}


def _find_model():
    """$SKATE_MJCF wins; otherwise the repo-wide $SKT_DIR convention."""
    skt = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
    p = os.environ.get("SKATE_MJCF") or str(skt / "skt_v3_control.xml")
    return p if Path(p).exists() else None


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _endpoint(**kw):
    """A sim endpoint that is NOT running -- the test drives it step by step,
    so every packet can be compared against a known, frozen MjData."""
    try:
        import mujoco
    except ImportError:
        pytest.skip("mujoco not installed")
    model = _find_model()
    if model is None:
        pytest.skip("no control model (set $SKT_DIR or $SKATE_MJCF)")

    from skate_ros2.sim_endpoint import SkateSimEndpoint

    ep = SkateSimEndpoint(model, port=_free_port(), bind="127.0.0.1",
                          verbose=False, **kw)
    return mujoco, ep


def _attach(ep):
    """A client whose heartbeat has been seen, so telemetry will be sent."""
    link = SkateLink("127.0.0.1", ep.port)
    link.poll()                    # first poll heartbeats
    assert ep.pump_network() >= 1, "endpoint never saw the client heartbeat"
    return link


def _settle(mujoco, ep, nmax=6000, tol=1e-5):
    for k in range(nmax):
        mujoco.mj_step(ep.m, ep.d)
        if k > 50 and float(np.max(np.abs(ep.d.qvel))) < tol:
            return k
    return nmax


def _reset(ep):
    ep.d.qpos[:] = 0
    ep.d.qvel[:] = 0
    ep.d.ctrl[:] = 0
    ep.d.xfrc_applied[:] = 0


def _round_trip(ep, link, timeout=1.0):
    """One telemetry burst -> the client's decoded wrench (arm -> f/m)."""
    link.state.wrist_wrench = None
    ep.send_telemetry()
    deadline = time.monotonic() + timeout
    while link.state.wrist_wrench is None and time.monotonic() < deadline:
        link.poll()
        time.sleep(0.002)
    assert link.state.wrist_wrench is not None, "no id-6 wrench arrived"
    assert link.decode_errors == 0
    return link.state.wrenches()


def test_streamed_wrench_is_the_sensor_reading():
    mujoco, ep = _endpoint()
    try:
        if not ep.wrist_sites:
            pytest.skip("this control model carries no wrist F/T sensors")
        link = _attach(ep)

        _reset(ep)
        _settle(mujoco, ep)
        base = _round_trip(ep, link)

        for arm, site in ep.wrist_sites.items():
            # recomputed here, independently, in sim/test_ft_sensor.py's frame
            R = np.array(ep.d.site(site).xmat).reshape(3, 3)
            f = R @ np.array(ep.d.sensor(site + "_force").data)
            m = R @ np.array(ep.d.sensor(site + "_torque").data)
            assert base[arm]["f"] == [float(v) for v in f], \
                f"{arm}: streamed force is not the sensor reading"
            assert base[arm]["m"] == [float(v) for v in m], \
                f"{arm}: streamed moment is not the sensor reading"

        # the no-load reading is the hand's own weight, NOT a tared zero --
        # that is what a real F/T cell reads before you zero it, and the
        # overlay's 0.5 N noise floor is what hides it.
        assert 0.2 < np.linalg.norm(base["left"]["f"]) < 1.0

        # ... and under known loads the WIRE value passes M1's own bound.
        for arm, body in HANDS.items():
            if arm not in ep.wrist_sites:
                continue
            hid = ep.m.body(body).id
            f0 = np.array(base[arm]["f"])
            for Fw in ([0, 0, -10.0], [10.0, 0, 0], [0, 8.0, 0], [5.0, -5.0, -5.0]):
                _reset(ep)
                ep.d.xfrc_applied[hid, :3] = Fw
                _settle(mujoco, ep)
                delta = np.array(_round_trip(ep, link)[arm]["f"]) - f0
                err = float(np.linalg.norm(delta + np.array(Fw)))
                assert err < 0.05, (
                    f"{arm}: applied {Fw} N, streamed delta {np.round(delta, 3)}"
                    f", err {err:.3f} N -- frame or sign lost on the wire")
        link.close()
    finally:
        ep.close()


def test_wrench_survives_the_live_loop():
    """Over the real run loop the wrench keeps arriving, keeps advancing, and
    does not disturb the firmware ids it shares the socket with."""
    _mujoco, ep = _endpoint(telemetry_hz=50.0)
    try:
        if not ep.wrist_sites:
            pytest.skip("this control model carries no wrist F/T sensors")
        th = threading.Thread(target=ep.run, kwargs={"duration": 2.0},
                              daemon=True)
        th.start()

        link = SkateLink("127.0.0.1", ep.port)
        deadline = time.monotonic() + 3.0
        while link.state.wrist_wrench is None and time.monotonic() < deadline:
            link.poll()
            time.sleep(0.01)
        assert link.state.wrist_wrench is not None, "no wrench over the live wire"
        t0 = link.state.wrist_wrench["t"]

        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            link.poll()
            time.sleep(0.01)

        w = link.state.wrenches()
        assert sorted(w) == ["left", "right"]
        assert link.state.wrist_wrench["t"] > t0, "wrench stamp is frozen"
        assert link.state.dof_pos() is not None, "id 6 starved the state_est stream"
        assert link.decode_errors == 0 and ep.decode_errors == 0

        link.close()
        th.join(timeout=6)
    finally:
        ep.close()


def test_absent_sensors_are_not_a_fault():
    """A control MJCF built before M1 has no F/T sensors. That robot is still a
    valid robot: discovery is tolerant, id 6 is simply never sent, and the rest
    of the telemetry is unaffected. (The real Skate is exactly this case.)"""
    mujoco, ep = _endpoint()
    try:
        assert ep.wrist_sites == {"left": "ee_left", "right": "ee_right"}, \
            "wrist F/T discovery missed the M1 sensors in the control model"

        ep.wrist_sites = {}                      # a model without them
        assert ep._wrist_wrench() is None

        link = _attach(ep)
        _settle(mujoco, ep, nmax=200)
        ep.send_telemetry()
        deadline = time.monotonic() + 0.5
        while link.state.dof_pos() is None and time.monotonic() < deadline:
            link.poll()
            time.sleep(0.005)
        assert link.state.dof_pos() is not None, "ordinary telemetry stopped"
        assert link.state.wrist_wrench is None, "sent id 6 with no sensors"
        assert link.state.wrenches() is None
        link.close()
    finally:
        ep.close()


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
