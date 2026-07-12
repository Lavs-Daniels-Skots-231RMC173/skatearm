"""Fully dynamic rollout of the trained ACT policy.

Instead of teleporting the arm to each predicted pose (the kinematic rollout in
rollout_act.py), this commands the pose through the model's torque-limited
position servos and integrates the full rigid-body dynamics (mj_step) under
gravity until the step settles, then lets the policy observe the *actual*
achieved state and predict again. Same scene and the same 24 in-distribution
targets rollout_act.py uses -- an honest dynamic-vs-kinematic comparison.

Contacts are disabled in the control scene (the raw converted meshes self-jam at
the shoulder mounts; a free-space reach needs no contact), so this adds gravity,
inertia, actuator torque limits and servo tracking -- not self-collision.

    MUJOCO_GL=osmesa python dynamic_reach.py <checkpoint> [N]   # default 24

Writes dynamic_metrics.json and dynamic_rollout.gif into ACT_TMP.
Env: SKT_DIR, ACT_TMP, STEPS, KMAX, SETTLE.
"""
import os, sys, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, torch, mujoco
from PIL import Image
_SC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _SC)
from skate_commander import camera as cammod
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies import make_pre_post_processors

CKPT = sys.argv[1]
NEP = int(sys.argv[2]) if len(sys.argv) > 2 else 24
STEPS = int(os.environ.get("STEPS", "55"))
KMAX = int(os.environ.get("KMAX", "120"))          # max mj_step substeps per policy step
SETTLE = float(os.environ.get("SETTLE", "0.05"))   # rad/s: arm settled below this
GIF_SUB = int(os.environ.get("GIF_SUB", "12"))     # capture a frame every N substeps (motion)
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


rng = np.random.default_rng(123)   # identical target stream to rollout_act.py


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
policy.to(device).eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                     preprocessor_overrides={"device_processor": {"device": device}})
TASK = "reach the orange (right hand) and blue (left hand) targets"
print("dynamic rollout  device", device, "KMAX", KMAX, "settle", SETTLE, flush=True)

R, L, track, ksteps = [], [], [], []
all_frames = []
diverged = 0
for ep in range(NEP):
    tR, tL = sample_targets()
    d.qpos[:] = 0
    d.qvel[:] = 0
    d.ctrl[:] = 0
    mujoco.mj_forward(m, d)
    policy.reset()
    keep = ep < 4
    frames = []
    for t in range(STEPS):
        img = render_frame(tR, tL)
        if keep and t == 0:
            frames.append(img)
        state = np.asarray(d.qpos[ARM_IDX], dtype=np.float32)
        obs = {"observation.images.front": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255),
               "observation.state": torch.from_numpy(state).unsqueeze(0), "task": TASK}
        with torch.no_grad():
            action = post(policy.select_action(pre(obs))).squeeze(0).detach().cpu().numpy()
        d.ctrl[ARM_IDX] = action           # command arms; other actuators hold home (ctrl=0)
        for k in range(1, KMAX + 1):
            mujoco.mj_step(m, d)
            if keep and k % GIF_SUB == 0:
                frames.append(render_frame(tR, tL))
            if k >= 15 and np.max(np.abs(d.qvel[ARM_IDX])) < SETTLE:
                break
        ksteps.append(k)
        track.append(float(np.linalg.norm(d.qpos[ARM_IDX] - action)))
    if keep:
        frames.append(render_frame(tR, tL))
        all_frames.append(frames)
    if not np.all(np.isfinite(d.qpos)):
        diverged += 1
        continue
    R.append(float(np.linalg.norm(d.site("ee_right").xpos - tR)))
    L.append(float(np.linalg.norm(d.site("ee_left").xpos - tL)))

mR, mL = float(np.mean(R)), float(np.mean(L))
worst = [max(a, b) for a, b in zip(R, L)]
succ = float(np.mean([1.0 if w < 0.08 else 0.0 for w in worst]))
open(os.path.join(OUT, "dynamic_metrics.json"), "w").write(json.dumps(
    {"kind": "dynamic_mjstep", "n": len(R), "meanR": mR, "meanL": mL,
     "median_worsthand": float(np.median(worst)), "success_8cm": succ,
     "mean_track_rad": float(np.mean(track)), "mean_substeps": float(np.mean(ksteps)),
     "diverged": diverged, "R": R, "L": L}, indent=2))
print("DYNAMIC  mean R %.3f  L %.3f m   success@8cm %.0f%%   track %.4f rad   ~%.0f substeps/step   diverged %d   (n=%d)"
      % (mR, mL, succ * 100, float(np.mean(track)), float(np.mean(ksteps)), diverged, len(R)))

if all_frames:
    n = min(4, len(all_frames))
    T = min(len(f) for f in all_frames[:n])
    stepf = max(1, T // 55)
    idxs = list(range(0, T, stepf))
    CELL = 176
    gif = []
    for ti in idxs:
        grid = Image.new("RGB", (2 * CELL, 2 * CELL), (18, 18, 22))
        for kk in range(n):
            grid.paste(Image.fromarray(all_frames[kk][ti]).resize((CELL, CELL)), ((kk % 2) * CELL, (kk // 2) * CELL))
        gif.append(grid.convert("P", palette=Image.ADAPTIVE, colors=128))
    gif[0].save(os.path.join(OUT, "dynamic_rollout.gif"), save_all=True, append_images=gif[1:], duration=70, loop=0, optimize=True)
    print("WROTE dynamic_rollout.gif  frames", len(gif), flush=True)
