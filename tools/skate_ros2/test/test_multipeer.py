"""Multi-peer: one sim endpoint, a commander AND an observer, over real UDP.

The sim streams telemetry to every recently-heard client (so a cockpit can
watch the twin while the ROS 2 driver commands it), and its deadman watchdog
follows the COMMAND stream — an observer's heartbeats must never keep a dead
commander's last command alive.

Needs mujoco + the control-ready MJCF (skips cleanly if either is missing):
    SKATE_MJCF=/path/to/skt_v3_control.xml python3 test/test_multipeer.py
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

from skate_ros2.protocol import SkateLink       # noqa: E402


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


def test_observer_sees_commanders_motion():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed")
    model = _find_model()
    if model is None:
        pytest.skip("no control model (set $SKT_DIR or $SKATE_MJCF)")

    from skate_ros2.sim_endpoint import SkateSimEndpoint

    port = _free_port()
    ep = SkateSimEndpoint(model, port=port, telemetry_hz=50.0,
                          bind="127.0.0.1", realtime=True, verbose=False)
    th = threading.Thread(target=ep.run, kwargs={"duration": 14.0},
                          daemon=True)
    th.start()

    commander = SkateLink("127.0.0.1", port)
    observer = SkateLink("127.0.0.1", port)

    # telemetry must reach BOTH peers
    deadline = time.monotonic() + 3.0
    while ((not commander.connected or not observer.connected)
           and time.monotonic() < deadline):
        commander.poll()
        observer.poll()
        time.sleep(0.02)
    assert commander.connected, "commander got no telemetry"
    assert observer.connected, "observer got no telemetry (multi-peer broken)"

    start = np.array(commander.state.dof_pos())

    # commander raises the left elbow; the observer only heartbeats/polls
    targ = start.copy()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        t = time.monotonic() - t0
        s = min(t / 1.5, 1.0)
        s = s * s * (3 - 2 * s)                     # smooth ramp
        targ[11] = (1 - s) * start[11] + s * 1.2    # left elbow
        commander.send_command(targ, deadman=(1, 1, 1))
        commander.poll()
        observer.poll()
        time.sleep(1.0 / 60.0)

    # the OBSERVER's telemetry shows the motion the commander drove
    obs = np.array(observer.state.dof_pos())
    assert abs(obs[11] - 1.2) < 0.05, f"observer stale: elbow at {obs[11]:.3f}"
    assert not ep.dampened, "sim dampened while the commander streamed"

    # commander dies; the observer keeps heartbeating — the sim must STILL
    # dampen (deadman follows the command stream, not any packet)
    for _ in range(40):
        observer.poll()
        time.sleep(1.0 / 60.0)
    assert ep.dampened, "observer heartbeats kept a dead commander alive"

    commander.close()
    observer.close()
    th.join(timeout=16)
    ep.close()


def test_unkeyed_peer_gets_nothing_and_commands_nothing():
    """With $SKATE_AUTH set on both ends, a client without the key is not a
    peer at all: it receives no telemetry and cannot drive the arms.

    The red-check is built in -- the keyed client runs the SAME motion at the
    end, so a no-op authentication (one that accepted everyone) would let the
    unkeyed client move the elbow and fail the first half, while an
    over-zealous one that rejected everyone would fail the second.
    """
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed")
    model = _find_model()
    if model is None:
        pytest.skip("no control model (set $SKT_DIR or $SKATE_MJCF)")

    from skate_ros2.sim_endpoint import SkateSimEndpoint

    key = b"a-shared-secret-for-this-test-only"
    port = _free_port()
    ep = SkateSimEndpoint(model, port=port, telemetry_hz=50.0,
                          bind="127.0.0.1", realtime=True, verbose=False,
                          key=key)
    th = threading.Thread(target=ep.run, kwargs={"duration": 14.0}, daemon=True)
    th.start()

    good = SkateLink("127.0.0.1", port, key=key)
    bad = SkateLink("127.0.0.1", port)          # same wire, no key

    # Wait for state_est specifically: it is the one telemetry object big enough
    # to be FRAGMENTED, so this also proves signed fragments reassemble over
    # real UDP, not just in the unit test's memory.
    deadline = time.monotonic() + 4.0
    while good.state.dof_pos() is None and time.monotonic() < deadline:
        good.poll()
        bad.poll()
        time.sleep(0.02)
    assert good.connected, "keyed client got no telemetry"
    assert good.state.dof_pos() is not None, "signed fragments never reassembled"
    assert good.auth_errors == 0, "keyed client rejected its own endpoint"

    start = np.array(good.state.dof_pos())

    # the UNKEYED client tries to drive the elbow for 2 s
    targ = start.copy()
    targ[11] = 1.2
    t0 = time.monotonic()
    while time.monotonic() - t0 < 2.0:
        bad.send_command(targ, deadman=(1, 1, 1))
        bad.poll()
        good.poll()
        time.sleep(1.0 / 60.0)

    assert bad.state.n_packets == 0, "unkeyed client received telemetry"
    assert ep.dampened, "unkeyed client took the robot out of dampen"
    assert ep.n_cmds == 0, "unkeyed command was accepted"
    assert ep.auth_errors > 0, "endpoint never counted the refusals"
    assert [a[1] for a in ep.peers] == [good._sock.getsockname()[1]], \
        "an unauthenticated sender was registered as a peer"
    now = np.array(good.state.dof_pos())
    assert abs(now[11] - start[11]) < 0.05, "the arm moved for an unkeyed client"

    # ...and the SAME motion from the keyed client does work
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        t = time.monotonic() - t0
        s = min(t / 1.5, 1.0)
        s = s * s * (3 - 2 * s)
        targ[11] = (1 - s) * start[11] + s * 1.2
        good.send_command(targ, deadman=(1, 1, 1))
        good.poll()
        time.sleep(1.0 / 60.0)
    assert abs(good.state.dof_pos()[11] - 1.2) < 0.05, "keyed client blocked too"

    good.close()
    bad.close()
    th.join(timeout=16)
    ep.close()



def test_reported_q_snaps_limit_epsilon():
    """Telemetry snaps eps-outside-limit positions onto the limit (MoveIt's
    start-state bounds check refuses raw soft-constraint epsilon)."""
    try:
        import mujoco  # noqa: F401
    except ImportError:
        pytest.skip("mujoco not installed")
    model = _find_model()
    if model is None:
        pytest.skip("no control model (set $SKT_DIR or $SKATE_MJCF)")

    from skate_ros2.sim_endpoint import SkateSimEndpoint

    ep = SkateSimEndpoint(model, port=_free_port(), bind="127.0.0.1",
                          verbose=False)
    lo11 = ep.lo[11]
    ep.d.qpos[11] = lo11 - 5e-7          # soft-constraint epsilon at the stop
    assert ep._reported_q()[11] == lo11, "epsilon violation not snapped"
    ep.d.qpos[11] = lo11 - 0.02          # a REAL violation must stay visible
    assert abs(ep._reported_q()[11] - (lo11 - 0.02)) < 1e-12
    ep.close()




if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
