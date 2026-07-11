"""Optional LeRobotDataset v3.0 export for the Skate Commander cockpit (opt-in).

Records dense teach-in / teleop episodes (the bimanual arm pose per tick) and
writes them as a **LeRobotDataset v3.0** — the Hugging Face / LeRobot standard —
so cockpit demos become training-ready data for ACT / Diffusion Policy / pi0.

Dependency-light on purpose: the writer needs only numpy + pandas + pyarrow (all
ship cp313-win wheels), NOT torch / lerobot / ffmpeg, so it runs on the plain
Windows cockpit. The heavy `pandas` / `pyarrow` imports live inside the writer,
so importing this module never fails and recording never touches them — a missing
dep only surfaces (with a clear message) if you actually export. The produced
dataset loads with the real `LeRobotDataset(root=...)` on any Linux/HF machine.

State/action are joint positions (ALOHA convention): observation.state[t] is the
arm pose at frame t, action[t] is the next commanded pose (pose[t+1]).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

CODEBASE_VERSION = "v3.0"

# The 14 bimanual arm joints inside the skt_v3 26-DoF qpos (left 8-14, right 16-22).
ARM_IDX = list(range(8, 15)) + list(range(16, 23))
ARM_NAMES = [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]


def _stats(arr):
    """Per-column stats for a [N, D] array (D>=1). ``count`` is length-1, the
    rest are length-D — matching lerobot's compute_stats layout."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
        "q01": np.quantile(a, 0.01, axis=0).tolist(),
        "q10": np.quantile(a, 0.10, axis=0).tolist(),
        "q50": np.quantile(a, 0.50, axis=0).tolist(),
        "q90": np.quantile(a, 0.90, axis=0).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).tolist(),
    }


def write_lerobot_dataset(root, episodes, fps, state_names, action_names,
                          robot_type="skt_v3"):
    """Write a LeRobotDataset v3.0 (state+action only, no video).

    episodes: list of {"state": [T,Ds], "action": [T,Da], "task": str}
    Single data shard + single episodes shard (fine for teach-in logs).
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _fsl(arr, dim):  # [N,dim] float32 -> fixed_size_list<float>[dim]
        flat = pa.array(np.ascontiguousarray(arr, dtype=np.float32).reshape(-1))
        return pa.FixedSizeListArray.from_arrays(flat, dim)

    root = pathlib.Path(root)
    if not episodes:
        raise ValueError("no episodes to export")
    ds_dim, da_dim = len(state_names), len(action_names)

    tasks = []
    for ep in episodes:
        if ep["task"] not in tasks:
            tasks.append(ep["task"])
    task_index = {t: i for i, t in enumerate(tasks)}

    S, A, TS, FI, EI, IDX, TI, ep_meta, g = [], [], [], [], [], [], [], [], 0
    for ei, ep in enumerate(episodes):
        s = np.asarray(ep["state"], dtype=np.float32)
        a = np.asarray(ep["action"], dtype=np.float32)
        if s.ndim != 2 or a.ndim != 2 or s.shape[1] != ds_dim or a.shape[1] != da_dim \
                or len(s) != len(a):
            raise ValueError(f"episode {ei}: bad shapes state={s.shape} action={a.shape}")
        n = len(s)
        ti, start = task_index[ep["task"]], g
        S.append(s); A.append(a)
        TS.append((np.arange(n) / float(fps)).astype(np.float32))
        FI.append(np.arange(n, dtype=np.int64))
        EI.append(np.full(n, ei, dtype=np.int64))
        IDX.append(np.arange(start, start + n, dtype=np.int64))
        TI.append(np.full(n, ti, dtype=np.int64))
        g += n
        ep_meta.append({"episode_index": ei, "tasks": [ep["task"]], "length": n,
                        "from": start, "to": g, "state": s, "action": a})
    S, A = np.concatenate(S), np.concatenate(A)
    TS, FI = np.concatenate(TS), np.concatenate(FI)
    EI, IDX, TI = np.concatenate(EI), np.concatenate(IDX), np.concatenate(TI)
    total = int(len(S))

    tbl = pa.table({
        "observation.state": _fsl(S, ds_dim), "action": _fsl(A, da_dim),
        "timestamp": pa.array(TS, type=pa.float32()),
        "frame_index": pa.array(FI), "episode_index": pa.array(EI),
        "index": pa.array(IDX), "task_index": pa.array(TI),
    })
    (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, root / "data/chunk-000/file-000.parquet")

    (root / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks).to_parquet(
        root / "meta/tasks.parquet")

    scal = {"timestamp": TS.astype(np.float64)[:, None], "frame_index": FI[:, None],
            "episode_index": EI[:, None], "index": IDX[:, None], "task_index": TI[:, None]}
    stats = {"observation.state": _stats(S), "action": _stats(A)}
    for k, v in scal.items():
        stats[k] = _stats(v)
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))

    STAT_KEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]
    feats = ["observation.state", "action", "timestamp", "frame_index",
             "episode_index", "index", "task_index"]
    cols = {"episode_index": [], "tasks": [], "length": [],
            "data/chunk_index": [], "data/file_index": [],
            "dataset_from_index": [], "dataset_to_index": []}
    for f in feats:
        for sk in STAT_KEYS:
            cols[f"stats/{f}/{sk}"] = []
    for m in ep_meta:
        cols["episode_index"].append(m["episode_index"]); cols["tasks"].append(m["tasks"])
        cols["length"].append(m["length"])
        cols["data/chunk_index"].append(0); cols["data/file_index"].append(0)
        cols["dataset_from_index"].append(m["from"]); cols["dataset_to_index"].append(m["to"])
        sl = slice(m["from"], m["to"])
        per = {"observation.state": _stats(m["state"]), "action": _stats(m["action"]),
               "timestamp": _stats(TS[sl][:, None].astype(np.float64)),
               "frame_index": _stats(FI[sl][:, None]), "episode_index": _stats(EI[sl][:, None]),
               "index": _stats(IDX[sl][:, None]), "task_index": _stats(TI[sl][:, None])}
        for f in feats:
            for sk in STAT_KEYS:
                cols[f"stats/{f}/{sk}"].append(per[f][sk])
    (root / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), root / "meta/episodes/chunk-000/file-000.parquet")

    def fdef(dtype, shape, names):
        return {"dtype": dtype, "shape": list(shape), "names": names}
    features = {
        "observation.state": fdef("float32", [ds_dim], list(state_names)),
        "action": fdef("float32", [da_dim], list(action_names)),
        "timestamp": fdef("float32", [1], None), "frame_index": fdef("int64", [1], None),
        "episode_index": fdef("int64", [1], None), "index": fdef("int64", [1], None),
        "task_index": fdef("int64", [1], None),
    }
    info = {
        "codebase_version": CODEBASE_VERSION, "robot_type": robot_type,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
        "fps": int(fps), "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": None, "features": features,
    }
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    return root


class EpisodeRecorder:
    """Dense teach-in / teleop recorder. Hooked into ``bridge.tick`` alongside the
    keypose PoseRecorder: while active it samples the commanded arm pose at ``fps``;
    each REC start/stop is one episode. :meth:`export` writes all episodes as a
    LeRobotDataset v3.0. Cheap (one small array per frame); off unless recording."""

    def __init__(self, fps=30, arm_idx=None, joint_names=None):
        self.fps = int(fps)
        self.arm_idx = list(ARM_IDX if arm_idx is None else arm_idx)
        self.joint_names = list(joint_names or ARM_NAMES)
        self.episodes = []            # list of [T, 14] float32 arm-pose arrays
        self._buf = None
        self._acc = 0.0
        self.active = False

    def start(self):
        self._buf, self._acc, self.active = [], 0.0, True

    def observe(self, targ, dt):
        if not self.active or targ is None:
            return
        self._acc += float(dt)
        if self._acc + 1e-9 >= 1.0 / self.fps:
            self._acc -= 1.0 / self.fps
            self._buf.append(np.asarray(targ, dtype=np.float32)[self.arm_idx].copy())

    def stop(self):
        self.active = False
        if self._buf is not None and len(self._buf) >= 3:
            self.episodes.append(np.stack(self._buf))
        self._buf = None

    def clear(self):
        self.episodes = []

    def status(self):
        return {"active": self.active, "episodes": len(self.episodes),
                "frames": int(sum(len(e) for e in self.episodes)),
                "recording_frames": (len(self._buf) if self._buf is not None else 0)}

    def export(self, root, task="teleop"):
        if not self.episodes:
            raise ValueError("no recorded episodes - hit REC, drive the arms, stop, then export")
        eps = [{"state": p[:-1], "action": p[1:], "task": task} for p in self.episodes]
        return write_lerobot_dataset(root, eps, self.fps, self.joint_names,
                                     self.joint_names, robot_type="skt_v3")
