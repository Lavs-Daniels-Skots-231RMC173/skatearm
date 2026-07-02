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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skate_ros2.protocol import SkateLink       # noqa: E402


def _find_model():
    p = os.environ.get("SKATE_MJCF")
    if p and Path(p).exists():
        return p
    return None


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
        print("SKIP: mujoco not installed")
        return
    model = _find_model()
    if model is None:
        print("SKIP: set $SKATE_MJCF to skt_v3_control.xml")
        return

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


if __name__ == "__main__":
    test_observer_sees_commanders_motion()
    print("OK")
