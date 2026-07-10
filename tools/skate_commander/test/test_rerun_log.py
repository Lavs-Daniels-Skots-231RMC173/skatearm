"""Optional rerun.io telemetry logger: builds against the collision model and
logs a real frame (meshed robot + TCP + IK target + obstacles + time-series)
without error. Model-gated and skipped when rerun/mujoco aren't installed, so
CI without the optional dep stays green.

    SKT_DIR=.../skt_v3 python3 test_rerun_log.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
COLL = SKT / "skt_v3_collision.xml"
_PKG = Path(__file__).resolve().parents[1] / "skate_commander"


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _PKG / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rerun_logger_logs_a_frame():
    try:
        import mujoco  # noqa: F401
        import rerun   # noqa: F401
    except ImportError as e:
        _skip(f"optional dep missing: {e}"); return
    if not COLL.exists():
        _skip("no collision model (set SKT_DIR)"); return

    rl = _load("rerun_log")
    model = _load("urdf").parse_urdf(SKT / "skt_v3.urdf")
    kin = {a: _load("kinematics").ArmKinematics(model, a) for a in ("left", "right")}

    class StubBridge:
        pass
    br = StubBridge()
    br.kin = kin
    br.ik_targets = {"left": [-0.1, 0.05, -0.1], "right": None}

    logger = rl.RerunLogger(str(COLL), spawn=False, every=1)
    q = np.zeros(26); q[11] = 0.4; q[19] = -0.3
    snap = {"q": q.tolist(), "dq": (np.ones(26) * 0.1).tolist(),
            "ik": {"left": 0.012, "right": None},
            "ik_ori": {"left": 1.5, "right": None},
            "manip": {"left": 0.45, "right": 0.4},
            "obstacles": [{"type": "box", "p": [0.3, 0, 0.2], "s": [0.05, 0.05, 0.05]},
                          {"type": "cyl", "p": [0.2, 0.1, 0.1], "s": [0.04, 0.1]}]}
    logger._log(br, snap)          # real logging path — surfaces any error
    logger.log(br, {"q": None})    # public no-op path must not raise
    assert logger.model.nbody == 28
    logger.close()


if __name__ == "__main__":
    test_rerun_logger_logs_a_frame()
    print("PASS test_rerun_logger_logs_a_frame")
