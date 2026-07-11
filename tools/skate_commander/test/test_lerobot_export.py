"""LeRobot v3.0 exporter + EpisodeRecorder: records synthetic teleop poses and
writes a dataset with the right structure. Gated on pandas/pyarrow so CI without
the optional deps stays green (the writer needs them; the cockpit imports lazily).
"""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

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


def _have_deps():
    return all(importlib.util.find_spec(m) for m in ("pandas", "pyarrow"))


def test_recorder_export_structure():
    if not _have_deps():
        _skip("no pandas/pyarrow")
        return
    le = _load("lerobot_export")
    rec = le.EpisodeRecorder(fps=30)
    rng = np.random.default_rng(0)
    for e in range(3):
        rec.start()
        pose = rng.standard_normal(26).astype("float32")
        for _ in range(6 + e):
            pose = pose + 0.01 * rng.standard_normal(26).astype("float32")
            rec.observe(pose, 1.0 / 30)
        rec.stop()
    assert rec.status()["episodes"] == 3

    root = Path(tempfile.mkdtemp()) / "ds"
    try:
        rec.export(root, task="reach test")
        for rel in ["meta/info.json", "meta/stats.json", "meta/tasks.parquet",
                    "data/chunk-000/file-000.parquet",
                    "meta/episodes/chunk-000/file-000.parquet"]:
            assert (root / rel).exists(), f"missing {rel}"

        import pyarrow.parquet as pq
        info = json.loads((root / "meta/info.json").read_text())
        assert info["codebase_version"] == "v3.0"
        assert info["total_episodes"] == 3
        assert info["features"]["observation.state"]["shape"] == [14]
        assert info["features"]["action"]["shape"] == [14]

        tbl = pq.read_table(root / "data/chunk-000/file-000.parquet")
        assert {"observation.state", "action", "timestamp", "frame_index",
                "episode_index", "index", "task_index"}.issubset(set(tbl.schema.names))
        # next-pose action drops one frame per episode: (6-1)+(7-1)+(8-1) = 18
        assert tbl.num_rows == 18
    finally:
        shutil.rmtree(root.parent, ignore_errors=True)


def test_empty_export_raises():
    if not _have_deps():
        _skip("no pandas/pyarrow")
        return
    le = _load("lerobot_export")
    rec = le.EpisodeRecorder(fps=30)
    raised = False
    try:
        rec.export(Path(tempfile.mkdtemp()) / "x")
    except ValueError:
        raised = True
    assert raised, "export with no episodes should raise ValueError"
