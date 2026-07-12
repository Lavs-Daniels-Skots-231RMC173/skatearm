"""State-only learned baseline for the ACT reach eval.

A behaviour-cloning control that answers "does the ~5 cm reach come from the
*camera*, or just from the model class / the closed loop?". Trains a small MLP
that maps joint state -> next commanded pose (NO image) on the same 40-episode
dataset, then rolls it out closed-loop on the SAME fixed eval targets
rollout_act.py uses (numpy.random.default_rng(123) + the same rejection
sampler), applied kinematically over the same 55 steps. With no target
information in its input, the best a learned policy can do is regress to the
average reach behaviour -- so this is a *learned* error floor, a stronger
control than the fixed mean-pose baseline_reach.py.

    MUJOCO_GL=osmesa python state_baseline_reach.py [N_EPISODES]   # default 24

Writes state_baseline_metrics.json (schema matches rollout_metrics.json) into
ACT_TMP. Env overrides: SKT_DIR, REACH_DATASET, ACT_TMP, STEPS.
"""
import os, sys, json, glob
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco
import pyarrow.parquet as pq
_SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SC)
from skate_commander import camera as cammod

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
STEPS = int(os.environ.get("STEPS", "55"))
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

# ---- train the state-only MLP:  joint state (14) -> next commanded pose (14) ----
fs = sorted(glob.glob(os.path.join(DATASET, "data", "**", "*.parquet"), recursive=True))
X = np.concatenate([np.asarray(pq.read_table(f).column("observation.state").to_pylist(), dtype=np.float32) for f in fs])
Y = np.concatenate([np.asarray(pq.read_table(f).column("action").to_pylist(), dtype=np.float32) for f in fs])
torch.manual_seed(0)
np.random.seed(0)
xmu, xsd = X.mean(0), X.std(0) + 1e-6
ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
Xn = torch.tensor((X - xmu) / xsd)
Yn = torch.tensor((Y - ymu) / ysd)
net = torch.nn.Sequential(torch.nn.Linear(14, 128), torch.nn.ReLU(),
                          torch.nn.Linear(128, 128), torch.nn.ReLU(),
                          torch.nn.Linear(128, 14))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
lossf = torch.nn.MSELoss()
n = Xn.shape[0]
for ep in range(400):
    perm = torch.randperm(n)
    for i in range(0, n, 256):
        b = perm[i:i + 256]
        opt.zero_grad()
        l = lossf(net(Xn[b]), Yn[b])
        l.backward()
        opt.step()
net.eval()
xmu_t, xsd_t = torch.tensor(xmu), torch.tensor(xsd)
ymu_t, ysd_t = torch.tensor(ymu), torch.tensor(ysd)


def predict(state):
    with torch.no_grad():
        xn = (torch.tensor(state, dtype=torch.float32) - xmu_t) / xsd_t
        yn = net(xn.unsqueeze(0)).squeeze(0)
        return (yn * ysd_t + ymu_t).numpy()


print("trained state-only MLP  final-batch loss %.4f  (n=%d frames)" % (float(l.detach()), n), flush=True)

R, L = [], []
for ep in range(N):
    tR, tL = sample_targets()
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    for t in range(STEPS):
        state = np.asarray(d.qpos[ARM_IDX], dtype=np.float32)
        d.qpos[ARM_IDX] = predict(state)
        mujoco.mj_forward(m, d)
    R.append(float(np.linalg.norm(d.site("ee_right").xpos - tR)))
    L.append(float(np.linalg.norm(d.site("ee_left").xpos - tL)))

mR, mL = float(np.mean(R)), float(np.mean(L))
worst = [max(a, b) for a, b in zip(R, L)]
succ = float(np.mean([1.0 if w < 0.08 else 0.0 for w in worst]))
open(os.path.join(OUT, "state_baseline_metrics.json"), "w").write(json.dumps(
    {"kind": "state_only_bc", "n": N, "meanR": mR, "meanL": mL,
     "median_worsthand": float(np.median(worst)), "success_8cm": succ, "R": R, "L": L}, indent=2))
print("STATE-ONLY  mean R %.3f  L %.3f m   success@8cm %.0f%%   (n=%d)" % (mR, mL, succ * 100, N))
