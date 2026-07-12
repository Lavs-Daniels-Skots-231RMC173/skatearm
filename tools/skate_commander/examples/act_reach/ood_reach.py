"""Out-of-distribution reach eval: roll the trained ACT policy out on targets
BEYOND the training box (further forward / further out, near the reach limit) to
see whether it interpolates only inside its training reach volume or extrapolates.
Same closed-loop kinematic rollout as rollout_act.py; writes ood_metrics.json.

    MUJOCO_GL=osmesa python ood_reach.py <checkpoint> [N]   [MODE=ood|indist]
"""
import os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco
_SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SC)
from skate_commander import camera as cammod
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors

CKPT = sys.argv[1]
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 24
MODE = os.environ.get("MODE", "ood")
MD = os.environ.get("SKT_DIR", os.path.join(_SC, "skate_teleop", "skt_v3"))
OUT = os.environ.get("ACT_TMP", os.path.dirname(os.path.abspath(__file__)))
IMG, STEPS = 256, 55
CAM = dict(az=270, el=-11, dist=1.4, look=(0.0, 0.08, 0.0))
ORANGE, BLUE = (1.0, 0.55, 0.10, 1.0), (0.15, 0.5, 1.0, 1.0)
ARM_L, ARM_R = list(range(8, 15)), list(range(16, 23))
ARM_IDX = ARM_L + ARM_R

# training box (rollout_act.py) vs OOD box (shifted further forward/out, at reach edge)
BOX = {"indist": dict(x=(0.09, 0.32), y=(0.26, 0.42), z=(-0.13, 0.09)),
       "ood":    dict(x=(0.30, 0.44), y=(0.42, 0.54), z=(-0.04, 0.14))}[MODE]

m = mujoco.MjModel.from_xml_path(cammod.build_scene_xml(MD))
d = mujoco.MjData(m)
GRAY = np.array([0.75, 0.77, 0.80, 1.0])
fl = int(m.geom("floor").id)
for i in range(m.ngeom):
    if i != fl:
        m.geom_matid[i] = -1
        m.geom_rgba[i] = GRAY
for nm in ("target_geom", "target2_geom", "table_top"):
    try:
        m.geom_rgba[m.geom(nm).id, 3] = 0.0
    except Exception:
        pass
jid = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
dof = {q: int(m.jnt_dofadr[jid[q]]) for q in ARM_IDX}
rng_ = {q: m.jnt_range[jid[q]] for q in ARM_IDX}
renderer = mujoco.Renderer(m, IMG, IMG)
CAMERA = mujoco.MjvCamera()
CAMERA.type = mujoco.mjtCamera.mjCAMERA_FREE
CAMERA.lookat[:] = CAM["look"]
CAMERA.distance, CAMERA.azimuth, CAMERA.elevation = CAM["dist"], CAM["az"], CAM["el"]


def render_frame(tR, tL):
    renderer.update_scene(d, camera=CAMERA)
    for pos, rgba in ((tR, ORANGE), (tL, BLUE)):
        s = renderer.scene
        g = s.geoms[s.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.06, 0, 0.]),
                            np.asarray(pos, float), np.eye(3).flatten(), np.asarray(rgba, np.float32))
        g.emission = 0.35
        s.ngeom += 1
    return np.ascontiguousarray(renderer.render())


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


rng = np.random.default_rng(123)


def sample_targets():
    """Return (tR, tL, max_ik_resid). Prefer a pair both hands can reach to <2cm;
    if none in 120 tries, return the most-reachable pair seen so the metric is
    never silently contaminated by an unreachable target (its residual is logged)."""
    best = None
    for _ in range(120):
        tR = np.array([rng.uniform(*BOX["x"]), rng.uniform(*BOX["y"]), rng.uniform(*BOX["z"])])
        tL = np.array([-rng.uniform(*BOX["x"]), rng.uniform(*BOX["y"]), rng.uniform(*BOX["z"])])
        mr = max(resid("ee_right", tR, ARM_R), resid("ee_left", tL, ARM_L))
        if best is None or mr < best[0]:
            best = (mr, tR, tL)
        if mr < 0.02:
            return tR, tL, mr
    return best[1], best[2], best[0]


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ACTPolicy.from_pretrained(CKPT)
policy.to(device).eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                     preprocessor_overrides={"device_processor": {"device": device}})
TASK = "reach the orange (right hand) and blue (left hand) targets"

R, L, IKR = [], [], []
for ep in range(NEP):
    tR, tL, ikr = sample_targets()
    IKR.append(ikr)
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    policy.reset()
    for t in range(STEPS):
        img = render_frame(tR, tL)
        state = np.asarray(d.qpos[ARM_IDX], dtype=np.float32)
        obs = {"observation.images.front": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255),
               "observation.state": torch.from_numpy(state).unsqueeze(0), "task": TASK}
        with torch.no_grad():
            action = post(policy.select_action(pre(obs)))
        d.qpos[ARM_IDX] = action.squeeze(0).detach().cpu().numpy()
        mujoco.mj_forward(m, d)
    R.append(float(np.linalg.norm(d.site("ee_right").xpos - tR)))
    L.append(float(np.linalg.norm(d.site("ee_left").xpos - tL)))

mR, mL = float(np.mean(R)), float(np.mean(L))
worst = [max(a, b) for a, b in zip(R, L)]
succ = float(np.mean([1.0 if w < 0.08 else 0.0 for w in worst]))
reach = float(np.mean([1.0 if r < 0.02 else 0.0 for r in IKR]))
open(os.path.join(OUT, "ood_metrics.json"), "w").write(json.dumps(
    {"mode": MODE, "n": NEP, "meanR": mR, "meanL": mL, "median_worsthand": float(np.median(worst)),
     "success_8cm": succ, "mean_ik_resid": float(np.mean(IKR)), "frac_reachable_2cm": reach,
     "R": R, "L": L, "ik_resid": IKR}, indent=2))
print("%-7s mean R %.3f  L %.3f m   success@8cm %.0f%%   targets-reachable<2cm %.0f%%   (n=%d)"
      % (MODE.upper(), mR, mL, succ * 100, reach * 100, NEP))
