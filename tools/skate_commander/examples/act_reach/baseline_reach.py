"""No-vision baseline for the ACT reach eval.

Drives both arms to the dataset's MEAN pose (ignoring the camera) and measures
reach error on the SAME fixed eval targets ``rollout_act.py`` uses
(``numpy.random.default_rng(123)`` + the same rejection sampler). This is the
error floor a policy with *zero* target information achieves — the control the
ACT rollout is compared against: if the ACT policy did nothing but replay an
average trajectory, it would score like this.

    MUJOCO_GL=osmesa python baseline_reach.py [N_EPISODES]      # default 24

Writes ``baseline_metrics.json`` (same schema as rollout_act.py's
``rollout_metrics.json``) into ``ACT_TMP``. Env overrides: ``SKT_DIR`` (model
dir), ``REACH_DATASET`` (dataset dir), ``ACT_TMP`` (output dir).
"""
import os
import sys
import json
import glob

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import pyarrow.parquet as pq

_SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SC)
from skate_commander import camera as cammod

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
OUT = os.environ.get("ACT_TMP", os.path.dirname(os.path.abspath(__file__)))
MD = os.environ.get("SKT_DIR", os.path.join(_SC, "skate_teleop", "skt_v3"))
DATASET = os.environ.get("REACH_DATASET", os.path.join(_SC, "lerobot_datasets", "reach_act"))
ARM_L, ARM_R = list(range(8, 15)), list(range(16, 23))
ARM_IDX = ARM_L + ARM_R

m = mujoco.MjModel.from_xml_path(cammod.build_scene_xml(MD))
d = mujoco.MjData(m)
jid = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
dof = {q: int(m.jnt_dofadr[jid[q]]) for q in ARM_IDX}
rng_ = {q: m.jnt_range[jid[q]] for q in ARM_IDX}

# mean commanded arm pose over the whole dataset -> the best a no-vision policy
# can do is drive to this fixed pose regardless of where the (random) target is.
pf = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))[0]
MEAN = np.asarray(pq.read_table(pf).column("action").to_pylist(), dtype=float).mean(0)


def ik_step(site, target, qidx, iters=400):
    sid = m.site(site).id
    didx = [dof[q] for q in qidx]
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = np.asarray(target, float) - d.site_xpos[sid]
        if np.linalg.norm(err) < 6e-4:
            break
        jp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, None, sid)
        J = jp[:, didx]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        for k, q in enumerate(qidx):
            lo, hi = rng_[q]
            d.qpos[q] = np.clip(d.qpos[q] + 0.6 * dq[k], lo, hi)
    mujoco.mj_forward(m, d)


def resid(site, target, qidx):
    d.qpos[:] = 0
    ik_step(site, target, qidx)
    return float(np.linalg.norm(np.asarray(target, float) - d.site_xpos[m.site(site).id]))


rng = np.random.default_rng(123)   # identical target stream to rollout_act.py


def sample_targets():
    tR = tL = None
    for _ in range(60):
        tR = np.array([rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        tL = np.array([-rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        if resid("ee_right", tR, ARM_R) < 0.012 and resid("ee_left", tL, ARM_L) < 0.012:
            return tR, tL
    return tR, tL


R, L = [], []
for ep in range(N):
    tR, tL = sample_targets()
    d.qpos[:] = 0
    d.qpos[ARM_IDX] = MEAN          # no vision: go to the average pose, target ignored
    mujoco.mj_forward(m, d)
    R.append(float(np.linalg.norm(d.site("ee_right").xpos - tR)))
    L.append(float(np.linalg.norm(d.site("ee_left").xpos - tL)))

mR, mL = float(np.mean(R)), float(np.mean(L))
worst = [max(a, b) for a, b in zip(R, L)]
succ = float(np.mean([1.0 if w < 0.08 else 0.0 for w in worst]))
open(os.path.join(OUT, "baseline_metrics.json"), "w").write(json.dumps(
    {"n": N, "meanR": mR, "meanL": mL, "median_worsthand": float(np.median(worst)),
     "success_8cm": succ, "R": R, "L": L}, indent=2))
print("BASELINE  mean R %.3f  L %.3f m   success@8cm %.0f%%   (n=%d)" % (mR, mL, succ * 100, N))
