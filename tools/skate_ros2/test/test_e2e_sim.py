"""End-to-end: SkateLink client <-> MuJoCo sim endpoint over real UDP.

Needs mujoco + the control-ready MJCF (skips cleanly if either is missing):
    SKATE_MJCF=/path/to/skt_v3_control.xml python3 test/test_e2e_sim.py
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

from skate_ros2 import names                    # noqa: E402
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


def test_wave_over_the_wire():
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
    th = threading.Thread(target=ep.run, kwargs={"duration": 12.0},
                          daemon=True)
    th.start()

    link = SkateLink("127.0.0.1", port)
    deadline = time.monotonic() + 3.0
    while not link.connected and time.monotonic() < deadline:
        link.poll()
        time.sleep(0.02)
    assert link.connected, "no telemetry from sim endpoint"

    start_pose = np.array(link.state.dof_pos())
    n_state0 = link.state.n_packets

    # stream a smooth elbow raise for 4 s at 60 Hz
    targ = start_pose.copy()
    t0 = time.monotonic()
    sent = 0
    while True:
        t = time.monotonic() - t0
        if t > 4.0:
            break
        s = min(t / 2.0, 1.0)                       # 2 s smooth ramp
        s = s * s * (3 - 2 * s)
        targ[11] = (1 - s) * start_pose[11] + s * 1.5   # left elbow
        targ[19] = (1 - s) * start_pose[19] + s * 1.5   # right elbow
        link.send_command(targ, deadman=(1, 1, 1))
        link.poll()
        sent += 1
        time.sleep(1.0 / 60.0)

    # let it settle, keep deadman engaged
    for _ in range(30):
        link.send_command(targ, deadman=(1, 1, 1))
        link.poll()
        time.sleep(1.0 / 60.0)

    pos = np.array(link.state.dof_pos())
    err11 = abs(pos[11] - 1.5)
    err19 = abs(pos[19] - 1.5)
    pkts = link.state.n_packets - n_state0
    rate = pkts / (time.monotonic() - t0)
    print(f"sent {sent} cmds; telemetry {pkts} pkts (~{rate:.0f}/s incl. all "
          f"ids); elbow errors L={err11:.4f} R={err19:.4f} rad")
    assert err11 < 0.05 and err19 < 0.05, "sim arm did not track command"
    assert rate > 60, "telemetry too slow (expect ~50 Hz x 4 ids)"

    # deadman: go fully silent (poll() would heartbeat, and per the official
    # docs ANY packet resets the firmware watchdog) — robot must dampen ~0.3 s
    time.sleep(0.5)
    assert ep.dampened, "endpoint must dampen after 0.3 s of silence"

    link.close()
    th.join(timeout=15)
    ep.close()


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
