"""Smoke tests for the bimanual benchmark harness: it runs a trial of each
task and returns a well-formed report. Needs mujoco + the built cell scene
(skt_v3_cell.xml); skips cleanly otherwise (like the other model-gated tests).
The `handoff` task also needs the jaws scene, which `benchmark.load_cell`
builds from the cell scene on demand -- so nothing extra is gated on here.

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
    report = benchmark.run(str(SKT), ["reach", "carry", "handoff", "insert", "insert_m2"], trials=1, seed=1)
    for t in ("reach", "carry", "handoff", "insert", "insert_m2"):
        assert t in report and report[t]["trials"], f"{t} produced no trial"
        assert "success_rate" in report[t]["summary"]
        assert report[t]["scene"] in benchmark.SCENES, t
    assert report["meta"]["tasks"] == ["reach", "carry", "handoff", "insert", "insert_m2"]
    print("PASS benchmark all-tasks smoke (reach/carry/handoff/insert/insert_m2 each ran)")


def test_benchmark_handoff_actually_changes_hands():
    """The one task that is not a co-carry: the peg must end up in the OTHER
    hand. Asserted on the report's own fields rather than re-measuring, because
    those fields are what the committed numbers and the prose are made of."""
    if not _ready():
        return
    import benchmark
    report = benchmark.run(str(SKT), ["handoff"], trials=1, seed=0)
    assert report["handoff"]["scene"] == "cell_gripper"     # jaws, never welds
    r = report["handoff"]["trials"][0]
    assert r["handed"], "peg did not survive the giver letting go"
    assert r["gap_mm"] > 0, "the two pad plates touched"
    assert r["peg_tilt_deg"] < 20.0 and r["drop_mm"] < 10.0
    assert r["success"] is True
    print("PASS benchmark handoff smoke:", report["handoff"]["summary"])


def test_benchmark_task_numbers_do_not_depend_on_the_other_tasks():
    """A task measured alone must measure the same thing it measures in a full
    run -- otherwise `--tasks insert` cannot be compared against the committed
    suite, which is the entire point of a benchmark.

    This is a regression test with a specific regression behind it: while every
    task shared one generator, adding `handoff` between `carry` and `insert`
    silently moved the published numbers of both inserts. `benchmark.task_rng`
    keys the stream on the task instead. Wall-clock is excluded because it is
    the machine's, not the run's."""
    if not _ready():
        return
    import benchmark
    alone = benchmark.run(str(SKT), ["insert"], trials=1, seed=0)
    beside = benchmark.run(str(SKT), ["carry", "insert"], trials=1, seed=0)

    def rows(rep):
        return [{k: v for k, v in r.items() if k != "wall_s"}
                for r in rep["insert"]["trials"]]

    assert rows(alone) == rows(beside), "insert changed because carry ran first"
    print("PASS benchmark per-task streams are independent of task order")


if __name__ == "__main__":
    test_benchmark_reach_runs()
    test_benchmark_all_tasks_smoke()
    test_benchmark_handoff_actually_changes_hands()
    test_benchmark_task_numbers_do_not_depend_on_the_other_tasks()
