"""Smoke tests for the bimanual benchmark harness: it runs a trial of each
task and returns a well-formed report. Needs mujoco + the built cell scene
(skt_v3_cell.xml); skips cleanly otherwise (like the other model-gated tests).

    SKT_DIR=.../skt_v3 python test_benchmark.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def _ready():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        _skip("mujoco not installed"); return False
    if not (SKT / "skt_v3_cell.xml").exists():
        _skip("no cell scene (run make_cell_scene.py)"); return False
    return True


def test_benchmark_reach_runs():
    if not _ready():
        return
    import benchmark
    report = benchmark.run(str(SKT), ["reach"], trials=1, seed=0)
    assert report["reach"]["trials"], "no reach trial recorded"
    r = report["reach"]["trials"][0]
    assert r["max_err_mm"] < 20.0 and isinstance(r["success"], bool)
    assert "/" in report["reach"]["summary"]["success_rate"]
    print("PASS benchmark reach smoke:", report["reach"]["summary"])


def test_benchmark_all_tasks_smoke():
    if not _ready():
        return
    import benchmark
    report = benchmark.run(str(SKT), ["reach", "carry", "insert", "insert_m2"], trials=1, seed=1)
    for t in ("reach", "carry", "insert", "insert_m2"):
        assert t in report and report[t]["trials"], f"{t} produced no trial"
        assert "success_rate" in report[t]["summary"]
    print("PASS benchmark all-tasks smoke (reach/carry/insert/insert_m2 each ran)")


if __name__ == "__main__":
    test_benchmark_reach_runs()
    test_benchmark_all_tasks_smoke()
