"""Closed-loop rollout of a trained ACT policy in the MuJoCo sim: reset to home,
show random reachable orange/blue targets, let the policy drive the arms from the
rendered image + joint state, and record a GIF + final reach error."""
import os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco
from PIL import Image
_SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tools/skate_commander
sys.path.insert(0, _SC)
from skate_commander import camera as cammod
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors

CKPT = sys.argv[1]
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 55
MD = os.environ.get("SKT_DIR", os.path.join(_SC, "skate_teleop", "skt_v3"))
OUT = os.environ.get("ACT_TMP", os.path.dirname(os.path.abspath(__file__)))
IMG = 256
CAM = dict(az=270, el=-11, dist=1.4, look=(0.0, 0.08, 0.0))
ORANGE, BLUE = (1.0, 0.55, 0.10, 1.0), (0.15, 0.5, 1.0, 1.0)
ARM_L, ARM_R = list(range(8, 15)), list(range(16, 23))
ARM_IDX = ARM_L + ARM_R

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
    tR = tL = None
    for _ in range(60):
        tR = np.array([rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        tL = np.array([-rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        if resid("ee_right", tR, ARM_R) < 0.012 and resid("ee_left", tL, ARM_L) < 0.012:
            return tR, tL
    return tR, tL


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ACTPolicy.from_pretrained(CKPT)
policy.to(device)
policy.eval()
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": device}})
TASK = "reach the orange (right hand) and blue (left hand) targets"
print("loaded policy from", CKPT, "device", device, flush=True)

all_frames, results = [], []
for ep in range(NEP):
    tR, tL = sample_targets()
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    policy.reset()
    keep = ep < 4
    frames = []
    for t in range(STEPS):
        img = render_frame(tR, tL)
        if keep:
            frames.append(img)
        state = np.asarray(d.qpos[ARM_IDX], dtype=np.float32)
        obs = {
            "observation.images.front": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255),
            "observation.state": torch.from_numpy(state).unsqueeze(0), "task": TASK,
        }
        with torch.no_grad():
            action = post(policy.select_action(pre(obs)))
        d.qpos[ARM_IDX] = action.squeeze(0).detach().cpu().numpy()
        mujoco.mj_forward(m, d)
    if keep:
        frames.append(render_frame(tR, tL))
        all_frames.append(frames)
    eR = float(np.linalg.norm(d.site("ee_right").xpos - tR))
    eL = float(np.linalg.norm(d.site("ee_left").xpos - tL))
    results.append((eR, eL))
    print("ep%d  final EE err  R=%.3f  L=%.3f m" % (ep, eR, eL), flush=True)

n = min(4, len(all_frames))
T = min(len(f) for f in all_frames[:n])
CELL = 220
gif = []
for t in range(T):
    grid = Image.new("RGB", (2 * CELL, 2 * CELL), (18, 18, 22))
    for k in range(n):
        grid.paste(Image.fromarray(all_frames[k][t]).resize((CELL, CELL)), ((k % 2) * CELL, (k // 2) * CELL))
    gif.append(grid)
gif[0].save(OUT + "/rollout.gif", save_all=True, append_images=gif[1:], duration=60, loop=0)
mR = float(np.mean([r[0] for r in results]))
mL = float(np.mean([r[1] for r in results]))
print("MEAN final EE err  R=%.3f  L=%.3f m  over %d eps" % (mR, mL, NEP), flush=True)
import json
allc = [max(r[0], r[1]) for r in results]
succ = float(np.mean([1.0 if c < 0.08 else 0.0 for c in allc]))
open(OUT + "/rollout_metrics.json", "w").write(json.dumps(
    {"n": len(results), "meanR": mR, "meanL": mL,
     "median_worsthand": float(np.median(allc)), "success_8cm": succ,
     "R": [r[0] for r in results], "L": [r[1] for r in results]}, indent=2))
print("success@8cm %.2f  median-worst %.3f" % (succ, float(np.median(allc))), flush=True)
print("WROTE rollout.gif + rollout_metrics.json", flush=True)
