"""Full stack: WebSocket client -> FastAPI server -> RobotBridge -> UDP ->
MuJoCo sim endpoint. The browser is the only thing missing."""

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))


SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
MODEL = os.environ.get("SKATE_MJCF", str(SKT / "skt_v3_control.xml"))


def test_ws_jog_roundtrip():
    model_xml, skt_dir = MODEL, str(SKT)
    if not Path(model_xml).exists():
        pytest.skip("no control model")
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    from skate_commander.server import build_app
    from skate_ros2.sim_endpoint import SkateSimEndpoint

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    ep = SkateSimEndpoint(model_xml, port=port, bind="127.0.0.1", verbose=False)
    th = threading.Thread(target=ep.run, kwargs={"duration": 40.0}, daemon=True)
    th.start()

    app = build_app(skt_dir, sim_port=port)
    with TestClient(app) as client:
        # REST surface
        m = client.get("/api/model").json()
        assert len(m["joint_names"]) == 26
        mesh = m["mesh_files"][0]
        assert client.get(f"/meshes/{mesh}").status_code == 200
        assert client.get("/meshes/../skt_v3.urdf").status_code in (404, 422)
        assert client.get("/meshes/nope.stl").status_code == 404

        with client.websocket_connect("/ws") as ws:
            def pump(seconds, msg=None):
                end = time.monotonic() + seconds
                last = None
                first = True
                while time.monotonic() < end:
                    if msg and first:
                        ws.send_text(json.dumps(msg)); first = False
                    last = json.loads(ws.receive_text())
                return last

            st = pump(1.0)
            assert st["connected"] and st["armed"] and st["estop"]

            st = pump(0.5, {"type": "resume"})
            assert not st["estop"] and st["live"]

            q0 = st["q"][11]
            pump(1.2, {"type": "jog_start", "idx": 11, "dir": 1})
            st = pump(0.8, {"type": "jog_stop", "idx": 11})
            assert st["q"][11] > q0 + 0.2, "elbow must move under ws jog"

            st = pump(0.3, {"type": "estop"})
            assert st["estop"] and not st["live"]
    ep.close()
    th.join(timeout=5)


if __name__ == "__main__":                 # direct run = pytest run, so a
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))   # skip reads as "s"
