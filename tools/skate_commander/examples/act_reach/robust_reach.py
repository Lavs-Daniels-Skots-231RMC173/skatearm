"""Domain-randomization robustness eval for the trained ACT reach policy.

The headline reach eval renders one fixed, clean visual condition. Real cameras
are mis-calibrated and lit differently -- the usual reason a sim policy fails to
transfer. This rolls the SAME checkpoint on the SAME 24 in-distribution targets
under perturbed *nuisance* conditions while keeping the task cues (orange/blue
targets) fixed, and reports how much the reach degrades. Three conditions:

  clean : the exact headline render (sanity -- should reproduce ~5 cm)
  cam   : camera extrinsics jittered (az/el/distance/look) -- calibration error
  dr    : cam + lighting (light + headlight) + robot/floor appearance

It measures visual robustness -- a sim-only proxy for sim-to-real, not transfer.

    MUJOCO_GL=osmesa python robust_reach.py <checkpoint> [N]   # default 24

Writes robust_metrics.json (+ robust_conditions.png) into ACT_TMP.
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
MD = os.environ.get("SKT_DIR", os.path.join(_SC, "skate_teleop", "skt_v3"))
OUT = os.environ.get("ACT_TMP", os.path.dirname(os.path.abspath(__file__)))
IMG = 256
CAM0 = dict(az=270.0, el=-11.0, dist=1.4, look=(0.0, 0.08, 0.0))
ORANGE, BLUE = (1.0, 0.55, 0.10, 1.0), (0.15, 0.5, 1.0, 1.0)
ARM_L, ARM_R = list(range(8, 15)), list(range(16, 23))
ARM_IDX = ARM_L + ARM_R

m = mujoco.MjModel.from_xml_path(cammod.build_scene_xml(MD))
d = mujoco.MjData(m)
fl = int(m.geom("floor").id)
HIDE = set()
for nm in ("target_geom", "target2_geom", "table_top"):
    try:
        HIDE.add(int(m.geom(nm).id))
    except Exception:
        pass
robot_geoms = [i for i in range(m.ngeom) if i != fl and i not in HIDE]
for i in robot_geoms:
    m.geom_matid[i] = -1
for gid in HIDE:
    m.geom_rgba[gid, 3] = 0.0
GRAY0 = np.array([0.75, 0.77, 0.80, 1.0])
FLOOR0 = m.geom_rgba[fl].copy()
LDIFF0 = m.light_diffuse.copy()
LDIR0 = m.light_dir.copy()
HLAMB0 = np.array(m.vis.headlight.ambient)
HLDIFF0 = np.array(m.vis.headlight.diffuse)
jid = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
dof = {q: int(m.jnt_dofadr[jid[q]]) for q in ARM_IDX}
rng_ = {q: m.jnt_range[jid[q]] for q in ARM_IDX}
renderer = mujoco.Renderer(m, IMG, IMG)
CAMERA = mujoco.MjvCamera()
CAMERA.type = mujoco.mjtCamera.mjCAMERA_FREE


def set_condition(mode, prng):
    """Apply one episode's visual condition. Nuisance vars only; targets fixed."""
    az, el, dist = CAM0["az"], CAM0["el"], CAM0["dist"]
    look = list(CAM0["look"])
    gray, floor = GRAY0.copy(), FLOOR0.copy()
    ldiff, ldir = LDIFF0.copy(), LDIR0.copy()
    hlamb, hldiff = HLAMB0.copy(), HLDIFF0.copy()
    if mode in ("cam", "dr"):
        az += prng.uniform(-12, 12)
        el += prng.uniform(-6, 6)
        dist *= prng.uniform(0.92, 1.08)
        look = [look[k] + prng.uniform(-0.03, 0.03) for k in range(3)]
    if mode == "dr":
        ldiff = np.clip(ldiff * prng.uniform(0.7, 1.15, ldiff.shape), 0, 1)
        ldir = ldir + prng.uniform(-0.3, 0.3, ldir.shape)
        hlamb = np.clip(hlamb * prng.uniform(0.8, 1.2), 0, 1)
        hldiff = np.clip(hldiff * prng.uniform(0.8, 1.2), 0, 1)
        gray = np.clip(gray * np.append(prng.uniform(0.82, 1.12) * prng.uniform(0.95, 1.05, 3), 1.0), 0, 1)
        floor = np.clip(floor * np.append(prng.uniform(0.6, 1.4, 3), 1.0), 0, 1)
    CAMERA.lookat[:] = look
    CAMERA.distance, CAMERA.azimuth, CAMERA.elevation = dist, az, el
    for i in robot_geoms:
        m.geom_rgba[i] = gray
    m.geom_rgba[fl] = floor
    m.light_diffuse[:] = ldiff
    m.light_dir[:] = ldir
    m.vis.headlight.ambient[:] = hlamb
    m.vis.headlight.diffuse[:] = hldiff


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


TARGETS = [sample_targets() for _ in range(NEP)]   # sampled once, shared across conditions

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = ACTPolicy.from_pretrained(CKPT)
policy.to(device).eval()
pre, post = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=CKPT,
                                     preprocessor_overrides={"device_processor": {"device": device}})
TASK = "reach the orange (right hand) and blue (left hand) targets"
print("robustness eval  device", device, "  N", NEP, flush=True)


def run_mode(mode, grab=None):
    R, L = [], []
    for ep in range(NEP):
        prng = np.random.default_rng(3000 + ep)
        set_condition(mode, prng)
        tR, tL = TARGETS[ep]
        d.qpos[:] = 0
        mujoco.mj_forward(m, d)
        policy.reset()
        for t in range(STEPS):
            img = render_frame(tR, tL)
            if grab is not None and ep == 0 and t == 0:
                grab.append(img)
            state = np.asarray(d.qpos[ARM_IDX], dtype=np.float32)
            obs = {"observation.images.front": torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255),
                   "observation.state": torch.from_numpy(state).unsqueeze(0), "task": TASK}
            with torch.no_grad():
                action = post(policy.select_action(pre(obs))).squeeze(0).detach().cpu().numpy()
            d.qpos[ARM_IDX] = action
            mujoco.mj_forward(m, d)
        R.append(float(np.linalg.norm(d.site("ee_right").xpos - tR)))
        L.append(float(np.linalg.norm(d.site("ee_left").xpos - tL)))
    worst = [max(a, b) for a, b in zip(R, L)]
    return {"mode": mode, "n": NEP, "meanR": float(np.mean(R)), "meanL": float(np.mean(L)),
            "median_worsthand": float(np.median(worst)),
            "success_8cm": float(np.mean([1.0 if w < 0.08 else 0.0 for w in worst])), "R": R, "L": L}


grabs, res = {}, {}
for mode in ("clean", "cam", "dr"):
    g = []
    res[mode] = run_mode(mode, grab=g)
    grabs[mode] = g[0] if g else None
    r = res[mode]
    print("%-6s mean R %.3f  L %.3f m   success@8cm %.0f%%   (n=%d)"
          % (mode.upper(), r["meanR"], r["meanL"], r["success_8cm"] * 100, NEP), flush=True)

open(os.path.join(OUT, "robust_metrics.json"), "w").write(json.dumps(res, indent=2))

panels = [grabs.get(k) for k in ("clean", "cam", "dr")]
if all(p is not None for p in panels):
    from PIL import ImageDraw
    W = 256
    strip = Image.new("RGB", (3 * W, W + 22), (245, 247, 250))
    dd = ImageDraw.Draw(strip)
    for i, (img, lab) in enumerate(zip(panels, ["clean", "camera jitter", "full DR"])):
        strip.paste(Image.fromarray(img), (i * W, 22))
        dd.text((i * W + 6, 6), lab, fill=(30, 40, 55))
    strip.save(os.path.join(OUT, "robust_conditions.png"))
    print("WROTE robust_conditions.png", flush=True)
