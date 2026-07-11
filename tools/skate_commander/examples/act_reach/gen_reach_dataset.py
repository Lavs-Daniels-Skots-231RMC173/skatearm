"""Generate a bimanual-reach LeRobotDataset v3.0 (WITH video) for ACT.

Fixed 3/4-front camera (option B). Each episode: sample two reachable targets
(orange=right, blue=left), glide both hands home->target in a straight Cartesian
line via DLS-IK, recording per frame: the rendered camera image (with the two
target handles drawn), the 14-DoF arm pose (observation.state) and the next
commanded pose (action, ALOHA convention). Written via the real lerobot 0.6.0.
"""
import os, sys, shutil, pathlib
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
from PIL import Image, ImageDraw

_SC = pathlib.Path(__file__).resolve().parents[2]   # tools/skate_commander
sys.path.insert(0, str(_SC))
from skate_commander import camera as cammod
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Paths are configurable via env vars; defaults are repo-relative.
MD = os.environ.get("SKT_DIR", str(_SC / "skate_teleop" / "skt_v3"))
OUT_ROOT = pathlib.Path(os.environ.get("REACH_DATASET", str(_SC / "lerobot_datasets" / "reach_act")))
TMP = os.environ.get("ACT_TMP", str(pathlib.Path(__file__).resolve().parent))
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 40
IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 256
FPS = 30
T = 48                       # frames per glide (state/action pairs = T-1)
SEED = 7
CAM = dict(az=270, el=-11, dist=1.4, look=(0.0, 0.08, 0.0))
ORANGE = (1.0, 0.55, 0.10, 1.0)
BLUE = (0.15, 0.5, 1.0, 1.0)
ARM_L, ARM_R = list(range(8, 15)), list(range(16, 23))
ARM_IDX = ARM_L + ARM_R
JOINT_NAMES = [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]


m = mujoco.MjModel.from_xml_path(cammod.build_scene_xml(MD))
d = mujoco.MjData(m)
GRAY = np.array([0.75, 0.77, 0.80, 1.0])
try:
    _floor = int(m.geom("floor").id)
except Exception:
    _floor = -1
for i in range(m.ngeom):
    if i != _floor:
        m.geom_matid[i] = -1
        m.geom_rgba[i] = GRAY
for nm in ("target_geom", "target2_geom", "table_top"):
    try:
        m.geom_rgba[m.geom(nm).id, 3] = 0.0
    except Exception:
        pass
jid_of = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
dof_of = {q: int(m.jnt_dofadr[jid_of[q]]) for q in ARM_IDX}
rng_of = {q: m.jnt_range[jid_of[q]] for q in ARM_IDX}
renderer = mujoco.Renderer(m, IMG, IMG)


def cam_obj():
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = CAM["look"]
    c.distance, c.azimuth, c.elevation = CAM["dist"], CAM["az"], CAM["el"]
    return c


CAMERA = cam_obj()


def ik_step(site, target, qidx, iters=80, step=0.6, damp=1e-3):
    """Warm-started DLS-IK (does NOT reset qpos) so an episode glides smoothly."""
    sid = m.site(site).id
    didx = [dof_of[q] for q in qidx]
    for _ in range(iters):
        mujoco.mj_forward(m, d)
        err = np.asarray(target, float) - d.site_xpos[sid]
        if np.linalg.norm(err) < 6e-4:
            break
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        J = jacp[:, didx]
        dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(3), err)
        for k, q in enumerate(qidx):
            lo, hi = rng_of[q]
            d.qpos[q] = np.clip(d.qpos[q] + step * dq[k], lo, hi)


def render_frame(tR, tL):
    renderer.update_scene(d, camera=CAMERA)
    for pos, rgba in ((tR, ORANGE), (tL, BLUE)):
        s = renderer.scene
        if s.ngeom < s.maxgeom:
            g = s.geoms[s.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.06, 0, 0.]), np.asarray(pos, float),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            g.emission = 0.35
            s.ngeom += 1
    return np.ascontiguousarray(renderer.render())


def smootherstep(a):
    return a * a * a * (a * (a * 6 - 15) + 10)


# home EE positions (qpos = 0)
d.qpos[:] = 0
mujoco.mj_forward(m, d)
eeR_home = d.site("ee_right").xpos.copy()
eeL_home = d.site("ee_left").xpos.copy()
rng = np.random.default_rng(SEED)


def _resid(site, target, qidx):
    d.qpos[:] = 0
    ik_step(site, target, qidx, iters=400)
    return float(np.linalg.norm(np.asarray(target, float) - d.site_xpos[m.site(site).id]))


def sample_targets():
    """Wide/varied forward targets (full workspace), reachability-rejected so
    both hands land on the marker (<1.2 cm)."""
    tR = tL = None
    for _ in range(60):
        tR = np.array([rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        tL = np.array([-rng.uniform(0.09, 0.32), rng.uniform(0.26, 0.42), rng.uniform(-0.13, 0.09)])
        if _resid("ee_right", tR, ARM_R) < 0.012 and _resid("ee_left", tL, ARM_L) < 0.012:
            return tR, tL
    return tR, tL


TASK = "reach the orange (right hand) and blue (left hand) targets"
if OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)
features = {
    "observation.images.front": {"dtype": "video", "shape": (IMG, IMG, 3),
                                  "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
    "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
}
ds = LeRobotDataset.create(repo_id="skate/reach_act", fps=FPS, features=features,
                           root=OUT_ROOT, robot_type="skt_v3", use_videos=True)

contact = []
max_res = 0.0
for ei in range(N_EP):
    tR, tL = sample_targets()
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    poses, imgs = [], []
    for t in range(T):
        a = smootherstep(t / (T - 1))
        curR = eeR_home + a * (tR - eeR_home)
        curL = eeL_home + a * (tL - eeL_home)
        ik_step("ee_right", curR, ARM_R)
        ik_step("ee_left", curL, ARM_L)
        poses.append(d.qpos[ARM_IDX].copy())
        imgs.append(render_frame(tR, tL))
    rR = np.linalg.norm(tR - d.site("ee_right").xpos)
    rL = np.linalg.norm(tL - d.site("ee_left").xpos)
    max_res = max(max_res, rR, rL)
    for t in range(T - 1):
        ds.add_frame({"observation.images.front": imgs[t],
                      "observation.state": poses[t].astype(np.float32),
                      "action": poses[t + 1].astype(np.float32), "task": TASK})
    ds.save_episode()
    if ei < 3:
        contact += [imgs[0], imgs[T // 2], imgs[T - 2]]
    print("episode %d/%d frames %d  final-resid R=%.3f L=%.3f" % (ei + 1, N_EP, T - 1, rR, rL), flush=True)

ds.finalize()
print("FINAL max reach residual over all episodes: %.3f m" % max_res, flush=True)

if contact:
    cols = 3
    rows = len(contact) // cols
    g = Image.new("RGB", (cols * IMG, rows * IMG), (20, 20, 24))
    for i, im in enumerate(contact):
        g.paste(Image.fromarray(im), ((i % cols) * IMG, (i // cols) * IMG))
    g.save(TMP + "/reach_contact.png")
    print("WROTE reach_contact.png", flush=True)
print("DONE root=%s episodes=%d" % (OUT_ROOT, N_EP), flush=True)
