"""Skate Commander server — FastAPI + WebSocket bridge to a Skate (sim/real).

    python3 -m skate_commander.server --model-dir /path/to/skate_teleop/skt_v3 \
        [--real-host r.local] [--spawn-sim /path/to/skt_v3_control.xml] \
        [--port 8088]

Serves the single-page UI, the URDF model tree, the STL meshes (from YOUR
local Rbotic/skate_teleop clone — meshes are not redistributed), and a
WebSocket carrying telemetry down (~20 Hz) and commands up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, JSONResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from .bridge import RobotBridge
from .kinematics import ArmKinematics, reach_map
from .origin import origin_allowed
from .program import PoseRecorder, ProgramRunner
from .urdf import joint_limits, parse_urdf
from . import camera, detect, grasp, ibvs, nl, vision

from skate_ros2 import names  # noqa: E402  (path set up by .bridge)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
SEQ_DIR = Path(__file__).resolve().parents[1] / "sequences"
PROG_DIR = Path(__file__).resolve().parents[1] / "programs"
LEROBOT_DIR = Path(__file__).resolve().parents[1] / "lerobot_datasets"
TOOLS_FILE = Path(__file__).resolve().parents[1] / "tcp_tools.json"


def _seq_name_ok(name):
    return (isinstance(name, str) and 0 < len(name) <= 40
            and all(c.isalnum() or c in "-_" for c in name))

TX_HZ = 60.0          # bridge tick / command rate
WS_HZ = 20.0          # UI telemetry rate

# manual UI inputs interrupt a running program (same rule as the sequencer)
_MANUAL = {"jog_start", "jog_step", "set_joint", "ik_target", "cart_step",
           "wp_goto", "wp_play", "home", "carry_grab", "carry_step",
           "carry_release"}


def compute_mirror_map(kin):
    """Derive the left<->right mirror convention numerically from FK.

    Returns ``(signs[8], axis)``: flipping world coordinate ``axis`` maps
    the left wrist onto the right at the neutral pose, and ``signs[k]``
    makes arm-slot ``k`` of one arm reproduce the mirrored motion of the
    other. Measured from the model, not guessed from the URDF convention.
    """
    q0 = np.zeros(names.N_JOINTS)
    pL, pR = kin["left"].fk(q0), kin["right"].fk(q0)
    axis = int(np.argmax(np.abs(pL - pR)))

    def mirr(v):
        v = v.copy()
        v[axis] = -v[axis]
        return v

    signs = np.ones(8)
    for k in range(7):                      # slot 7 = gripper, sign +1
        qa = q0.copy()
        qa[8 + k] = 0.3
        dl = kin["left"].fk(qa) - pL
        qb = q0.copy()
        qb[16 + k] = 0.3
        dr = kin["right"].fk(qb) - pR
        signs[k] = (1.0 if np.linalg.norm(dr - mirr(dl))
                    <= np.linalg.norm(dr + mirr(dl)) else -1.0)
    return signs, axis


def _load_tools():
    tools = {"flange": [0.0, 0.0, 0.0]}
    if TOOLS_FILE.exists():
        try:
            data = json.loads(TOOLS_FILE.read_text())
            for k, v in data.items():
                if (_seq_name_ok(k) and isinstance(v, list) and len(v) == 3
                        and all(abs(float(x)) <= 500.0 for x in v)):
                    tools[k] = [float(x) for x in v]
        except Exception as e:
            print(f"[commander] WARNING: bad {TOOLS_FILE.name}: {e}")
    return tools


def _obb_hit(c, M, half, o):
    """True if a robot link — taken as the ORIENTED BOX ``(c, M, half)`` — overlaps
    the axis-aligned virtual obstacle ``o`` ({type:'box'|'cyl', p:[x,y,z], s:[...]}).

    ``M`` is the link's world rotation as MuJoCo's flat row-major 9-vector; its
    COLUMNS are the box axes. The test is **exact**, by the separating-axis
    theorem: two convex boxes are disjoint iff one of fifteen axes separates
    them — the three world axes, the three link axes, and their nine cross
    products. Nothing is sampled and nothing is wrapped in a ball, so a link
    reserves its own volume and not a sphere around it.

    A vertical cylinder is tested against its own enclosing box: that can only
    over-block, never under-block, and it is still far tighter than a bounding
    ball. (The cockpit only ever spawns boxes; ``cyl`` arrives through the API.)
    """
    import math
    p, s = o.get("p"), o.get("s")
    if not p or not s:
        return False
    if o.get("type") == "cyl":                      # s = [radius, half-height]
        if len(s) < 2:
            return False
        eb = (float(s[0]), float(s[0]), float(s[1]))
    else:                                           # s = [hx, hy, hz]
        if len(s) < 3:
            return False
        eb = (float(s[0]), float(s[1]), float(s[2]))
    ea = (float(half[0]), float(half[1]), float(half[2]))
    d = (c[0] - float(p[0]), c[1] - float(p[1]), c[2] - float(p[2]))

    # broad phase: the link's circumscribed sphere against the obstacle box.
    # Cheap, and it is the branch almost every (link, obstacle) pair takes.
    rr = math.sqrt(ea[0] * ea[0] + ea[1] * ea[1] + ea[2] * ea[2])
    gx = max(abs(d[0]) - eb[0], 0.0)
    gy = max(abs(d[1]) - eb[1], 0.0)
    gz = max(abs(d[2]) - eb[2], 0.0)
    if gx * gx + gy * gy + gz * gz >= rr * rr:
        return False

    # R[i][j] = (link axis i) . (world axis j) — the transpose of M
    R = [[M[0], M[3], M[6]],
         [M[1], M[4], M[7]],
         [M[2], M[5], M[8]]]
    # the epsilon keeps a link axis that is exactly parallel to a world axis
    # from producing a zero-length cross product and a false separation
    A = [[abs(R[i][j]) + 1e-9 for j in range(3)] for i in range(3)]
    t = [R[i][0] * d[0] + R[i][1] * d[1] + R[i][2] * d[2] for i in range(3)]

    for i in range(3):                              # the link's own axes
        if abs(t[i]) > ea[i] + eb[0] * A[i][0] + eb[1] * A[i][1] + eb[2] * A[i][2]:
            return False
    for j in range(3):                              # the world axes
        if abs(d[j]) > eb[j] + ea[0] * A[0][j] + ea[1] * A[1][j] + ea[2] * A[2][j]:
            return False
    for i in range(3):                              # the nine cross products
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            ra = ea[i1] * A[i2][j] + ea[i2] * A[i1][j]
            rb = eb[j1] * A[i][j2] + eb[j2] * A[i][j1]
            if abs(t[i2] * R[i1][j] - t[i1] * R[i2][j]) > ra + rb:
                return False
    return True


def make_collision_guard(collision_xml, get_obstacles=None):
    """Self-collision predicate over the SkateArm primitive-collision model.

    ``make_collision_model.py`` fits one capsule (or a sphere, where the link is
    near-isotropic) to each link's compiled AABB — 22 capsules and 5 spheres on
    skt_v3, and no boxes since that rework; ``--boxes`` keeps the old v0.4 shape
    for comparison. Self-collision is therefore MuJoCo-exact on those primitives.
    The *keep-out* half of the guard reasons over a different volume on purpose —
    see ``keepout`` below.

    The physics collision model excludes pairs that touch at the neutral pose
    (hanging hands sit right next to the hips!) so the robot doesn't jam in
    sim. A *predictive* guard must still see those pairs — so we build a
    guard-specific variant: distant excludes (kinematically > 3 hops apart,
    e.g. hand<->thigh) are re-enabled, and instead of excluding them we
    tolerate exactly the contact depth they have at the neutral pose.
    q26 -> True if the pose penetrates >2 mm (or >4 mm past baseline).
    """
    import xml.etree.ElementTree as ET

    import mujoco
    collision_xml = Path(collision_xml)
    m0 = mujoco.MjModel.from_xml_path(str(collision_xml))

    def chain(b):
        out = []
        while b != 0:
            out.append(b)
            b = m0.body_parentid[b]
        out.append(0)
        return out

    def hops(a, b):                       # kinematic distance between bodies
        ca, cb = chain(a), chain(b)
        return min(i + cb.index(x) for i, x in enumerate(ca) if x in cb)

    tree = ET.parse(collision_xml)
    cel = tree.getroot().find("contact")
    removed = 0
    if cel is not None:
        for ex in list(cel.findall("exclude")):
            try:
                b1 = m0.body(ex.get("body1")).id
                b2 = m0.body(ex.get("body2")).id
            except Exception:
                continue
            if hops(b1, b2) > 3:          # distant pair: the guard must see it
                cel.remove(ex)
                removed += 1
    guard_xml = collision_xml.parent / ".skate_guard.xml"
    tree.write(str(guard_xml))
    gm = mujoco.MjModel.from_xml_path(str(guard_xml))
    gd = mujoco.MjData(gm)

    # contacts already present at the neutral pose are tolerated up to their
    # baseline depth — anything deeper (or any NEW pair) is a violation
    gd.qpos[:] = 0
    mujoco.mj_forward(gm, gd)
    base = {}
    for i in range(gd.ncon):
        c = gd.contact[i]
        key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
        base[key] = min(base.get(key, 0.0), float(c.dist))
    print(f"[commander] guard model: {removed} distant pairs re-enabled, "
          f"{len(base)} neutral-pose contacts baselined")

    def guard(q):
        gd.qpos[:26] = q
        mujoco.mj_forward(gm, gd)
        for i in range(gd.ncon):
            c = gd.contact[i]
            key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
            thr = base[key] - 0.004 if key in base else -0.002
            if float(c.dist) < thr:
                return True
        if get_obstacles is not None:                      # virtual user obstacles
            obs = get_obstacles()
            if obs:
                for gi, ctr, half in keepout:
                    M, px = gd.geom_xmat[gi], gd.geom_xpos[gi]
                    c = (px[0] + M[0] * ctr[0] + M[1] * ctr[1] + M[2] * ctr[2],
                         px[1] + M[3] * ctr[0] + M[4] * ctr[1] + M[5] * ctr[2],
                         px[2] + M[6] * ctr[0] + M[7] * ctr[1] + M[8] * ctr[2])
                    if any(_obb_hit(c, M, half, o) for o in obs):
                        return True
        return False

    geom_ids = [i for i in range(gm.ngeom) if int(gm.geom_bodyid[i]) != 0]

    # Keep-out volumes: ONE oriented box per link — the geom's **compiled AABB**,
    # which is MuJoCo's own bound on that mesh. Two things follow, and the old
    # per-geom bounding sphere had neither. It provably covers the link, so the
    # test cannot be blind to a link entering a keep-out box; and it is the link
    # and nothing more, so the planner stops detouring around phantom volume.
    # The visual mesh carries the real link shape (the collision primitives are
    # deliberately SHRUNK so the robot does not jam against itself in sim, which
    # is the right volume for self-collision and the wrong one for a keep-out);
    # a body without a mesh falls back to its primitives.
    _by_body = {}
    for i in geom_ids:
        _by_body.setdefault(int(gm.geom_bodyid[i]), []).append(i)
    keepout = []
    for _b, _gids in _by_body.items():
        _meshes = [i for i in _gids if int(gm.geom_type[i]) == 7]
        for i in (_meshes or _gids):
            ab = gm.geom_aabb[i]
            keepout.append((i, (float(ab[0]), float(ab[1]), float(ab[2])),
                            (float(ab[3]), float(ab[4]), float(ab[5]))))

    def collision_view(q):
        """World-space poses of the guard model's geoms at pose ``q`` (26-vec).
        Each geom is flagged ``h`` when it is in a *violating* contact. Drives
        the cockpit's collision-mesh overlay (the capsule/box model the guard
        actually reasons over) so you can see why a move is blocked."""
        import numpy as _np
        gd.qpos[:26] = q
        mujoco.mj_forward(gm, gd)
        hit = set()
        for i in range(gd.ncon):
            c = gd.contact[i]
            key = (min(c.geom1, c.geom2), max(c.geom1, c.geom2))
            thr = base[key] - 0.004 if key in base else -0.002
            if float(c.dist) < thr:
                hit.add(int(c.geom1)); hit.add(int(c.geom2))
        out = []
        for i in geom_ids:
            m = gd.geom_xmat[i].reshape(3, 3)
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0:
                s = 0.5 / _np.sqrt(tr + 1.0)
                qw = 0.25 / s; qx = (m[2, 1] - m[1, 2]) * s
                qy = (m[0, 2] - m[2, 0]) * s; qz = (m[1, 0] - m[0, 1]) * s
            else:
                k = int(_np.argmax(_np.diag(m)))
                if k == 0:
                    s = 2.0 * _np.sqrt(max(1e-9, 1 + m[0, 0] - m[1, 1] - m[2, 2]))
                    qw = (m[2, 1] - m[1, 2]) / s; qx = 0.25 * s
                    qy = (m[0, 1] + m[1, 0]) / s; qz = (m[0, 2] + m[2, 0]) / s
                elif k == 1:
                    s = 2.0 * _np.sqrt(max(1e-9, 1 + m[1, 1] - m[0, 0] - m[2, 2]))
                    qw = (m[0, 2] - m[2, 0]) / s; qx = (m[0, 1] + m[1, 0]) / s
                    qy = 0.25 * s; qz = (m[1, 2] + m[2, 1]) / s
                else:
                    s = 2.0 * _np.sqrt(max(1e-9, 1 + m[2, 2] - m[0, 0] - m[1, 1]))
                    qw = (m[1, 0] - m[0, 1]) / s; qx = (m[0, 2] + m[2, 0]) / s
                    qy = (m[1, 2] + m[2, 1]) / s; qz = 0.25 * s
            out.append({
                "t": int(gm.geom_type[i]),
                "s": [round(float(x), 4) for x in gm.geom_size[i]],
                "p": [round(float(x), 4) for x in gd.geom_xpos[i]],
                "q": [round(float(qw), 4), round(float(qx), 4),
                      round(float(qy), 4), round(float(qz), 4)],
                "h": (i in hit),
            })
        return out

    guard.collision_view = collision_view
    return guard


def build_app(model_dir, real_host="r.local", sim_port=2000,
              sim_host="127.0.0.1", collision_model=None, ik="dls", rerun=False):
    model_dir = Path(model_dir)
    urdf_path = model_dir / "skt_v3.urdf"
    mesh_dir = model_dir / "skt_v3_meshes" / "scaled_stl_files"
    if not urdf_path.exists():
        raise FileNotFoundError(f"{urdf_path} not found — point --model-dir "
                                "at the skt_v3 folder of your skate_teleop clone")
    model = parse_urdf(urdf_path)
    # resolve every mesh against both known layouts of the official clone
    mesh_dirs = [model_dir / "skt_v3_meshes" / "scaled_stl_files",
                 model_dir / "skt_v3_meshes"]
    mesh_paths = {}
    for name in model["mesh_files"]:
        for d in mesh_dirs:
            if (d / name).exists():
                mesh_paths[name] = d / name
                break
    missing = set(model["mesh_files"]) - set(mesh_paths)
    if missing:
        print(f"[commander] WARNING: {len(missing)} URDF meshes not found "
              f"under {model_dir} — the viewer will fall back to a stick "
              f"figure for those links: {sorted(missing)[:3]}...")

    kin = {arm: ArmKinematics(model, arm) for arm in ("left", "right")}
    bridge = RobotBridge(real_host=real_host, sim_port=sim_port,
                         sim_host=sim_host,
                         limits=joint_limits(model), kin=kin)
    bridge.mirror_signs, bridge.mirror_axis = compute_mirror_map(kin)
    print(f"[commander] mirror map: axis={'xyz'[bridge.mirror_axis]} "
          f"signs={[int(s) for s in bridge.mirror_signs]}")
    if collision_model:
        bridge.guard = make_collision_guard(collision_model, get_obstacles=lambda: bridge.obstacles)
        print("[commander] collision guard ON — self-colliding targets are "
              "rejected before they reach the robot")
    if str(ik).lower() == "mink" and collision_model:
        try:
            from .kinematics_mink import MinkIK
            bridge.mink = MinkIK(collision_model)
            print("[commander] IK backend: mink (MuJoCo QP + proactive "
                  "self-collision avoidance) — DLS is the automatic fallback")
        except Exception as e:
            print(f"[commander] mink IK unavailable ({type(e).__name__}: {e}); "
                  "falling back to the numpy DLS")
    rerun_logger = None
    if rerun and collision_model:
        try:
            from .rerun_log import RerunLogger
            rerun_logger = RerunLogger(collision_model)
            print("[commander] rerun logging ON — streaming the twin (meshed robot "
                  "+ time-series) to a rerun viewer")
        except Exception as e:
            print(f"[commander] rerun unavailable ({type(e).__name__}: {e}); "
                  "telemetry logging OFF")
    runner = ProgramRunner(bridge)
    bridge.recorder = PoseRecorder()      # teach-in: observed in tick()
    try:
        from .lerobot_export import EpisodeRecorder
        bridge.ds_rec = EpisodeRecorder(fps=30)   # dense LeRobot capture (opt-in export)
    except Exception as e:
        print(f"[commander] LeRobot export unavailable ({type(e).__name__}: {e})")
        bridge.ds_rec = None
    tools = _load_tools()

    def save_tools():
        TOOLS_FILE.write_text(json.dumps(tools, indent=1))

    app = FastAPI(title="Skate Commander")
    app.state.bridge = bridge
    app.state.runner = runner
    app.state.tools = tools
    app.state.clients = 0
    app.state.reachmap = {}                 # cached dexterity clouds per arm
    app.state.rerun = rerun_logger          # optional rerun.io telemetry (opt-in)

    def _cam_qpos():
        try:
            q = bridge.link.state.dof_pos()
        except Exception:
            q = None
        if q is None and bridge.targ is not None:
            q = list(bridge.targ)
        return q

    try:
        scene_path = camera.build_scene_xml(model_dir)
        app.state.camera = camera.CameraStreamer(scene_path, _cam_qpos)
    except Exception as exc:                       # camera is optional
        print(f"[commander] camera disabled: {exc}")
        app.state.camera = None

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/model")
    async def api_model():
        return JSONResponse(model)

    @app.get("/api/reachmap")
    def api_reachmap(arm: str = "right", n: int = 3000):
        """Manipulability heat-map: a dexterity point cloud over the arm's
        reachable workspace — [[x, y, z, manip], ...], manip in [0, 1] (0 =
        singular, 1 = isotropic). Computed once per arm and cached. Sync `def`
        so the ~1.5 s sample runs in FastAPI's threadpool, not the event loop."""
        n = max(200, min(int(n), 6000))
        key = f"{arm}:{n}"
        if arm in kin and key not in app.state.reachmap:
            app.state.reachmap[key] = reach_map(
                kin[arm], np.array(names.DEFAULT_POSE, dtype=float), n=n)
        return JSONResponse({"arm": arm, "points": app.state.reachmap.get(key, [])})

    @app.get("/api/reachable")
    def api_reachable(arm: str = "right", x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Reachability preview: is world point (x, y, z) [m] inside ``arm``'s
        reachable workspace? Nearest-sample distance against the cached dexterity
        reach-map (robust to IK local minima). Sync def -> FastAPI threadpool."""
        if arm not in kin:
            return JSONResponse({"reachable": False, "dist_mm": None})
        key = f"{arm}:3000"
        if key not in app.state.reachmap:
            app.state.reachmap[key] = reach_map(
                kin[arm], np.array(names.DEFAULT_POSE, dtype=float), n=3000)
        pts = app.state.reachmap.get(key, [])
        if not pts:
            return JSONResponse({"reachable": False, "dist_mm": None})
        arr = np.asarray(pts, dtype=float)[:, :3]
        d = float(np.min(np.linalg.norm(arr - np.array([x, y, z], dtype=float), axis=1)))
        return JSONResponse({"reachable": bool(d < 0.045), "dist_mm": round(d * 1000.0, 1)})

    @app.get("/api/pointcloud")
    def api_pointcloud(stride: int = 5):
        """Work-camera depth back-projected to a coloured world point cloud
        ([[x, y, z, r, g, b], ...]) — what the camera sees, in 3D. Serviced on
        the camera's GL render thread (sync def -> FastAPI threadpool)."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"error": "no camera"})
        res = cam.cloud(stride=max(2, min(int(stride), 12)))
        return JSONResponse({"points": res} if isinstance(res, list) else res)

    @app.get("/api/grasp")
    def api_grasp(stride: int = 4):
        """Synthesise a top-down grasp from the work-camera point cloud: remove
        the table, cluster what's left, and fit a parallel-jaw grasp to the
        object's OWN geometry — centre, a measured grasp height, footprint, yaw
        and a gripper-width feasibility check. Returns the grasp dict for the
        twin overlay; it moves nothing (use /api/smart_pick to execute). Sync
        def -> FastAPI threadpool (GL render + numpy)."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "no camera"})
        cloud = cam.cloud(stride=max(2, min(int(stride), 10)))
        if not isinstance(cloud, list):
            return JSONResponse({"found": False,
                                 "error": cloud.get("error", "no cloud")})
        g = grasp.plan_grasp(cloud, max_width=camera.MAX_GRIPPER_WIDTH,
                             workspace=camera.WORKSPACE_AABB)
        return JSONResponse(g)

    @app.get("/api/grasps")
    def api_grasps(stride: int = 4):
        """ALL graspable objects on the work surface, each with a synthesised
        grasp and a detector label (built-in colour/shape; optional YOLO when
        opted in via SKATE_YOLO). Moves nothing -- the twin overlay + object
        selector read this. Sync def -> threadpool (GL render + numpy)."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "no camera",
                                 "objects": []})
        sc = cam.scene(stride=max(2, min(int(stride), 10)))
        if "error" in sc:
            return JSONResponse({"found": False, "error": sc["error"],
                                 "objects": []})
        res = grasp.plan_grasps(sc["cloud"], max_width=camera.MAX_GRIPPER_WIDTH,
                                workspace=camera.WORKSPACE_AABB)
        if res.get("found"):
            detect.detect(res["objects"], rgb_image=sc.get("rgb"),
                          cam=sc.get("cam"))
        return JSONResponse(res)

    @app.get("/api/preview")
    def api_preview(action: str = "home", i: int = -1):
        """Preview a commanded motion WITHOUT executing — returns the target
        joint pose + TCP(s) so the cockpit can draw a GHOST ROBOT and require
        approval before the move runs (MoveIt-Pro-style 'show intent first')."""
        base = (bridge.targ if bridge.targ is not None
                else np.array(names.DEFAULT_POSE, dtype=float))
        goal = np.array(base, dtype=float)
        if action == "home":
            goal[8:] = np.array(names.DEFAULT_POSE, dtype=float)[8:]   # arms+head home
        elif action == "waypoint":
            wps = getattr(bridge, "waypoints", [])
            if not (0 <= i < len(wps)):
                return JSONResponse({"error": "bad waypoint"})
            goal = np.array(wps[i], dtype=float)
        else:
            return JSONResponse({"error": "unknown action"})
        tcp = {}
        for arm, k in bridge.kin.items():
            try:
                tcp[arm] = [round(float(v), 5) for v in k.fk(goal)]
            except Exception:
                pass
        route = {}                       # planned collision-free route's TCP trail per arm
        try:
            path = bridge.plan_path(goal)
            if path and len(path) >= 2:
                for arm, k in bridge.kin.items():
                    try:
                        route[arm] = [[round(float(v), 5) for v in k.fk(np.asarray(q, dtype=float))]
                                      for q in path]
                    except Exception:
                        pass
        except Exception:
            pass
        return JSONResponse({"action": action,
                             "q": [round(float(v), 6) for v in goal],
                             "tcp": tcp, "route": route})

    @app.get("/api/sequences")
    async def api_sequences():
        if not SEQ_DIR.exists():
            return JSONResponse([])
        return JSONResponse(sorted(p.stem for p in SEQ_DIR.glob("*.json")))

    @app.get("/api/tools")
    async def api_tools():
        return JSONResponse(tools)

    @app.get("/api/programs")
    async def api_programs():
        if not PROG_DIR.exists():
            return JSONResponse([])
        return JSONResponse(sorted(p.stem for p in PROG_DIR.glob("*.py")))

    @app.get("/api/programs/{name}")
    async def api_program(name: str):
        path = PROG_DIR / f"{name}.py"
        if not _seq_name_ok(name) or not path.exists():
            return JSONResponse({"error": "unknown program"}, status_code=404)
        return PlainTextResponse(path.read_text(encoding="utf-8"))

    @app.get("/api/recording")
    async def api_recording():
        return PlainTextResponse(bridge.recorder.result)

    @app.get("/api/lerobot/status")
    async def api_lerobot_status():
        r = bridge.ds_rec
        return JSONResponse(r.status() if r is not None
                            else {"episodes": 0, "frames": 0, "active": False})

    @app.post("/api/lerobot/export")
    async def api_lerobot_export(req: Request):
        r = bridge.ds_rec
        if r is None:
            return JSONResponse({"error": "LeRobot export unavailable"}, status_code=503)
        try:
            data = await req.json()
        except Exception:
            data = {}
        task = ((data or {}).get("task") or "teleop").strip() or "teleop"
        name = ((data or {}).get("name") or "skate_teleop").strip()
        if not _seq_name_ok(name):
            return JSONResponse({"error": "bad dataset name"}, status_code=400)
        root = LEROBOT_DIR / name
        try:
            import shutil
            if root.exists():
                shutil.rmtree(root)
            r.export(root, task=task)
            st = r.status()
            return JSONResponse({"ok": True, "root": str(root), "task": task,
                                 "episodes": st["episodes"], "frames": st["frames"]})
        except Exception as e:
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)

    @app.post("/api/nl")
    async def api_nl(req: Request):
        """Natural language -> rbt program. Returns code for the editor only;
        it never moves the robot (the editor still runs through the safe bridge)."""
        try:
            data = await req.json()
        except Exception:
            data = {}
        return JSONResponse(nl.generate((data or {}).get("text", "")))

    @app.get("/api/cameras")
    async def api_cameras():
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"cameras": [], "current": None})
        return JSONResponse({"cameras": cam.cams, "current": cam.cam})

    @app.get("/camstream")
    async def camstream(cam: str = None):
        streamer = app.state.camera
        if streamer is None:
            return JSONResponse({"error": "camera unavailable"}, status_code=503)
        if cam:
            streamer.set_cam(cam)

        async def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                yield boundary + streamer.jpeg() + b"\r\n"
                await asyncio.sleep(1 / 15)

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/detect")
    async def api_detect():
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "camera unavailable"})
        return JSONResponse(cam.detect())

    @app.post("/api/pick")
    async def api_pick():
        """Detect the workspace target and run a pick program (right arm) through
        the same guarded runner. Moves nothing unless armed + resumed."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "camera unavailable"})
        det = cam.detect()
        if not det.get("found"):
            return JSONResponse({"found": False,
                                 "error": det.get("error", "no target seen")})
        x, y, z = det["world_mm"]
        code = (
            f"# pick: magenta target detected at ({x:.0f}, {y:.0f}, {z:.0f}) mm\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {z + 90:.0f})\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {z + 12:.0f})\n"
            f"rbt.gripper('right', 0)\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {z + 130:.0f})\n"
        )
        ok = runner.run(code)
        return JSONResponse({"found": True, "ran": bool(ok),
                             "world_mm": det["world_mm"],
                             "pixel": det.get("pixel"), "code": code})

    @app.post("/api/servo_pick")
    async def api_servo_pick():
        """Closed-loop image-based visual servo pick (right arm). Detects the
        static target once, then drives the gripper's image feature onto it
        while descending — robust to camera-calibration error, unlike the
        one-shot /api/pick. Moves nothing unless armed + resumed."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "camera unavailable"})
        if runner.running:
            return JSONResponse({"found": False, "error": "a program is running"})
        kr = bridge.kin.get("right")
        if bridge.targ is None or bridge.estop or kr is None:
            return JSONResponse({"found": False,
                                 "error": "right arm not armed / resumed"})
        obs = cam.detect(ee_world=list(kr.fk(bridge.targ)))
        if not obs.get("found") or "cam" not in obs:
            return JSONResponse({"found": False,
                                 "error": obs.get("error", "no target seen")})
        s_cube = np.asarray(obs["pixel"], float)
        C = obs["cam"]
        f0, cx0, cy0 = vision.intrinsics(C["fovy"], C["W"], C["H"])
        pos0 = np.asarray(C["pos"], float)
        mat0 = np.asarray(C["mat"], float).reshape(3, 3)
        servo = ibvs.ImageServo(pos0, mat0, C["fovy"], C["W"], C["H"],
                                gain=0.7, max_step=0.03)
        grasp_z = camera.GRASP_Z
        z = grasp_z + 0.08
        err_px = 999.0
        for _ in range(20):                     # image servo, descending
            ee = kr.fk(bridge.targ)
            ee_pix = np.asarray(
                vision.project(ee, pos0, mat0, f0, cx0, cy0)[:2], float)
            dxy, err_px = servo.step(s_cube, ee_pix, ee)
            z = max(grasp_z + 0.03, z - 0.008)
            bridge.set_ik_target(
                "right", [ee[0] + dxy[0], ee[1] + dxy[1], z], auto=True)
            for _ in range(18):                 # let the tick loop track it
                await asyncio.sleep(1 / 60)
            if err_px < 3.0 and z <= grasp_z + 0.035:
                break
        ee = kr.fk(bridge.targ)
        x, y, gz = ee[0] * 1000, ee[1] * 1000, grasp_z * 1000
        code = (
            f"# servo-pick: IBVS-aligned over target (image err {err_px:.1f}px)\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {gz + 12:.0f})\n"
            f"rbt.gripper('right', 0)\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {gz + 130:.0f})\n"
        )
        ok = runner.run(code)
        return JSONResponse({"found": True, "ran": bool(ok),
                             "image_err_px": round(float(err_px), 1),
                             "world_mm": [round(x, 1), round(y, 1)]})

    @app.post("/api/smart_pick")
    def api_smart_pick(stride: int = 4, target: str = None):
        """Cloud grasp synthesis -> pick (right arm) through the guarded runner.
        Plans on the point cloud (table removed, object clustered), then picks
        at the MEASURED grasp centre/height — no colour or fixed-height prior.
        Refuses an object too wide for the jaws. Moves nothing unless armed +
        resumed. Sync def -> threadpool (GL render + numpy)."""
        cam = app.state.camera
        if cam is None:
            return JSONResponse({"found": False, "error": "camera unavailable"})
        if runner.running:
            return JSONResponse({"found": False, "error": "a program is running"})
        sc = cam.scene(stride=max(2, min(int(stride), 10)))
        if "error" in sc:
            return JSONResponse({"found": False, "error": sc["error"]})
        res = grasp.plan_grasps(sc["cloud"], max_width=camera.MAX_GRIPPER_WIDTH,
                                workspace=camera.WORKSPACE_AABB)
        if not res.get("found"):
            return JSONResponse({"found": False,
                                 "error": res.get("reason", "no object")})
        detect.detect(res["objects"], rgb_image=sc.get("rgb"), cam=sc.get("cam"))
        g = detect.pick_target(res["objects"], target)
        if g is None:
            return JSONResponse({"found": True, "ran": False,
                                 "error": f"no object matches {target!r}",
                                 "objects": [o["label"] for o in res["objects"]]})
        if not g.get("feasible"):
            return JSONResponse({"found": True, "ran": False, "feasible": False,
                                 "label": g.get("label"),
                                 "error": f"{g.get('label', 'object')} "
                                 f"{g['width_mm']:.0f}mm wider than the "
                                 f"{g['max_width_mm']:.0f}mm gripper",
                                 "width_mm": g["width_mm"]})
        x, y, gz = g["center_mm"]
        code = (
            f"# smart-pick: {g.get('label', 'object')} grasp at "
            f"({x:.0f}, {y:.0f}, {gz:.0f}) mm - "
            f"obj {g['length_mm']:.0f}x{g['width_mm']:.0f} mm, yaw {g['yaw_deg']:.0f} deg\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {gz + 90:.0f})\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {gz + 12:.0f})\n"
            f"rbt.gripper('right', 0)\n"
            f"rbt.moveto('right', {x:.0f}, {y:.0f}, {gz + 130:.0f})\n"
        )
        ok = runner.run(code)
        return JSONResponse({"found": True, "ran": bool(ok), "feasible": True,
                             "label": g.get("label"), "id": g.get("id"),
                             "center_mm": g["center_mm"], "yaw_deg": g["yaw_deg"],
                             "width_mm": g["width_mm"], "length_mm": g["length_mm"],
                             "n": g.get("n"), "code": code})

    @app.get("/meshes/{name}")
    async def mesh(name: str):
        path = mesh_paths.get(name)           # whitelist, no traversal
        if path is None:
            return JSONResponse({"error": "unknown mesh"}, status_code=404)
        return FileResponse(path, media_type="application/octet-stream")

    def _ws_origin_ok(sock):
        """Cross-site WebSocket guard — see origin.origin_allowed for the rule
        and why a same-host comparison alone does not stop DNS rebinding."""
        return origin_allowed(sock.headers.get("origin"), sock.headers.get("host"))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        if not _ws_origin_ok(sock):
            await sock.close(code=1008)        # cross-site origin — refuse
            return
        await sock.accept()
        app.state.clients += 1
        try:
            while True:
                try:
                    raw = await asyncio.wait_for(sock.receive_text(),
                                                 timeout=1.0 / WS_HZ)
                except asyncio.TimeoutError:
                    raw = None
                pong = None
                if raw is not None:                # one bad command must not
                    try:                           # drop the whole connection
                        cmd = json.loads(raw)
                        if isinstance(cmd, dict) and cmd.get("type") == "ping":
                            pong = cmd.get("t")    # echo the client clock -> real RTT
                        else:
                            handle_command(bridge, cmd, runner=runner,
                                           tools=tools, save_tools=save_tools)
                    except Exception as e:
                        print(f"[commander] ignored bad command: {e}")
                snap = bridge.snapshot(ui_attached=app.state.clients > 0)
                snap["prog"] = runner.snapshot()
                snap["prog"]["rec"] = bridge.recorder.snapshot()
                if pong is not None:
                    snap["pong"] = pong
                await sock.send_text(json.dumps(snap))
        except WebSocketDisconnect:
            pass
        finally:
            app.state.clients -= 1
            if app.state.clients <= 0:
                bridge.jog_stop()             # nobody at the controls

    @app.on_event("startup")
    async def start_tick():
        async def loop():
            dt = 1.0 / TX_HZ
            while True:
                snap = bridge.tick(dt, ui_attached=app.state.clients > 0)
                if app.state.rerun is not None:
                    app.state.rerun.log(bridge, snap)
                await asyncio.sleep(dt)
        app.state.tick_task = asyncio.create_task(loop())

    @app.on_event("shutdown")
    async def stop_tick():
        app.state.tick_task.cancel()
        if getattr(app.state, "rerun", None) is not None:
            app.state.rerun.close()
        bridge.close()
        if getattr(app.state, "camera", None):
            app.state.camera.close()

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def handle_command(bridge: RobotBridge, cmd: dict, runner=None, tools=None,
                   save_tools=None):
    t = cmd.get("type")
    if runner is not None and runner.running and t in _MANUAL:
        runner.stop("manual input")        # the human at the panel wins
    if t == "cart_step":
        delta = cmd.get("delta", ())
        if isinstance(delta, (list, tuple)) and len(delta) == 3:
            d = [max(-0.2, min(0.2, float(x))) for x in delta]
            bridge.cart_step(cmd.get("arm"), d)
    elif t == "carry_grab":
        bridge.carry_grab()
    elif t == "carry_step":
        delta = cmd.get("delta", ())
        if isinstance(delta, (list, tuple)) and len(delta) == 3:
            d = [max(-0.2, min(0.2, float(x))) for x in delta]
            bridge.carry_step(d)
    elif t == "carry_release":
        bridge.carry_release()
    elif t == "mirror":
        bridge.mirror = bool(cmd.get("on"))
        if not bridge.mirror:
            bridge.jog_stop()              # don't leave a ghost jog running
    elif t == "tool_set":
        if tools and cmd.get("name") in tools:
            off_m = [v / 1000.0 for v in tools[cmd["name"]]]
            bridge.set_tool(cmd.get("arm"), cmd["name"], off_m)
    elif t == "tool_def":
        name, xyz = cmd.get("name", ""), cmd.get("xyz_mm", ())
        if (tools is not None and _seq_name_ok(name) and name != "flange"
                and isinstance(xyz, (list, tuple)) and len(xyz) == 3):
            try:
                vals = [float(x) for x in xyz]
            except (TypeError, ValueError):
                return
            if all(abs(v) <= 500.0 for v in vals):
                tools[name] = vals
                if save_tools:
                    save_tools()
                for arm, nm in list(bridge.tool_names.items()):
                    if nm == name:         # live-update an attached tool
                        bridge.set_tool(arm, name, [v / 1000 for v in vals])
    elif t == "tool_del":
        name = cmd.get("name", "")
        if tools and name in tools and name != "flange":
            del tools[name]
            if save_tools:
                save_tools()
            for arm, nm in list(bridge.tool_names.items()):
                if nm == name:
                    bridge.set_tool(arm, "flange", [0.0, 0.0, 0.0])
    elif t == "prog_run":
        if runner:
            runner.run(cmd.get("code"))
    elif t == "prog_step":
        if runner:
            runner.step(cmd.get("code"))
    elif t == "prog_stop":
        if runner:
            runner.stop()
    elif t == "prog_save":
        name = cmd.get("name", "")
        if _seq_name_ok(name) and isinstance(cmd.get("code"), str):
            PROG_DIR.mkdir(exist_ok=True)
            (PROG_DIR / f"{name}.py").write_text(cmd["code"],
                                                 encoding="utf-8")
    elif t == "rec_start":
        if bridge.recorder is not None:
            bridge.recorder.start(bridge.targ)
        if bridge.ds_rec is not None:
            bridge.ds_rec.start()
    elif t == "rec_stop":
        if bridge.recorder is not None:
            bridge.recorder.stop()
        if bridge.ds_rec is not None:
            bridge.ds_rec.stop()
    elif t == "lerobot_clear":
        if bridge.ds_rec is not None:
            bridge.ds_rec.clear()
    elif t == "jog_start":
        bridge.jog_start(int(cmd["idx"]), int(cmd["dir"]))
    elif t == "jog_stop":
        bridge.jog_stop(cmd.get("idx"))
    elif t == "speed":
        bridge.set_speed(cmd.get("scale"))
    elif t == "pause":
        bridge.set_paused(cmd.get("on"))
    elif t == "step":
        bridge.step(cmd.get("n", 1))
    elif t == "collision":
        bridge.set_show_collision(cmd.get("on"))
    elif t == "force":
        bridge.set_show_force(cmd.get("on"))
    elif t == "tune":
        bridge.set_tuning(**(cmd.get("params") or {}))
    elif t == "obstacle_add":
        bridge.add_obstacle(cmd.get("shape", "box"), cmd.get("p"), cmd.get("s"))
    elif t == "obstacle_clear":
        bridge.clear_obstacles()
    elif t == "obstacle_del":
        bridge.delete_obstacle(cmd.get("id"))
    elif t == "obstacle_move":
        bridge.update_obstacle(cmd.get("id"), cmd.get("p"))
    elif t == "obstacle_resize":
        bridge.resize_obstacle(cmd.get("id"), cmd.get("s"))
    elif t == "jog_step":
        bridge.jog_step(int(cmd["idx"]), float(cmd["delta"]))
    elif t == "set_joint":
        bridge.set_joint(int(cmd["idx"]), float(cmd["value"]))
    elif t == "ik_target":
        bridge.set_ik_target(cmd.get("arm"), cmd.get("pos", ()),
                             auto=bool(cmd.get("auto")), rot=cmd.get("rot"))
    elif t == "ik_clear":
        bridge.clear_ik_target(cmd.get("arm"))
    elif t == "wp_add":
        bridge.wp_add()
    elif t == "wp_delete":
        bridge.wp_delete(int(cmd.get("idx", -1)))
    elif t == "wp_clear":
        bridge.wp_clear()
    elif t == "wp_goto":
        bridge.wp_goto(int(cmd.get("idx", -1)))
    elif t == "wp_play":
        bridge.wp_play(loop=bool(cmd.get("loop", False)))
    elif t == "wp_stop":
        bridge.seq_stop()
    elif t == "wp_save":
        name = cmd.get("name", "")
        if _seq_name_ok(name) and bridge.waypoints:
            SEQ_DIR.mkdir(exist_ok=True)
            (SEQ_DIR / f"{name}.json").write_text(json.dumps(
                {"names": bridge.wp_names,
                 "q": [w.tolist() for w in bridge.waypoints]}))
    elif t == "wp_load":
        name = cmd.get("name", "")
        path = SEQ_DIR / f"{name}.json"
        if _seq_name_ok(name) and path.exists():
            import numpy as np
            data = json.loads(path.read_text())
            bridge.wp_clear()
            bridge.waypoints = [np.asarray(q, dtype=float)
                                for q in data["q"]]
            bridge.wp_names = list(data["names"])
    elif t == "estop":
        bridge.trigger_estop()
    elif t == "resume":
        bridge.resume()
    elif t == "home":
        bridge.home()
    elif t == "reset_contact":
        bridge.clear_contact()
    elif t == "contact_mode":
        # the contact RESPONSE, not its sensitivity: "stop" latches the soft
        # stop (default), "compliant" yields the TCP instead. The bridge is the
        # validator -- anything else leaves the mode in effect untouched.
        bridge.set_contact_mode(cmd.get("mode"))
    elif t == "set_mode":
        bridge.set_mode(cmd["mode"])
    elif t == "observe":
        bridge.set_observe(bool(cmd.get("on", not bridge.observe)))


SIM_DIR = Path(__file__).resolve().parents[3] / "sim"
SKT_TELEOP_URL = "https://github.com/Rbotic/skate_teleop.git"


def _find_model_dir():
    """Locate the skt_v3 model folder: $SKT_DIR, then common clone locations
    near the current dir / repo / home. Returns a Path or None."""
    import os
    cands = []
    if os.environ.get("SKT_DIR"):
        cands.append(Path(os.environ["SKT_DIR"]))
    here = Path.cwd()
    root = Path(__file__).resolve().parents[3]
    for base in (here, here.parent, here.parent.parent, root, root.parent,
                 Path.home()):
        cands += [base / "skt_v3", base / "skate_teleop" / "skt_v3"]
    for c in cands:
        if (c / "skt_v3.urdf").exists() or (c / "skt_v3_converted.xml").exists():
            return c
    return None


def _ensure_collision_model(model_dir):
    """Make sure skt_v3_collision.xml exists — generating the control then the
    collision model from the official converted model on first run. The
    collision model is used both as the sim-endpoint model and the guard model."""
    md = Path(model_dir)
    collision = md / "skt_v3_collision.xml"
    if collision.exists():
        return collision
    if not (md / "skt_v3_control.xml").exists():
        if not (md / "skt_v3_converted.xml").exists():
            raise FileNotFoundError(
                f"{md} has no skt_v3_converted.xml — point --model-dir at the "
                "skt_v3 folder of your Rbotic/skate_teleop clone")
        print("[commander] one-time setup: building control model…")
        subprocess.run([sys.executable, str(SIM_DIR / "make_control_model.py"),
                        str(md)], check=True)
    print("[commander] one-time setup: building collision model…")
    subprocess.run([sys.executable, str(SIM_DIR / "make_collision_model.py"),
                    str(md)], check=True)
    if not collision.exists():
        raise RuntimeError(f"collision model generation did not produce {collision}")
    return collision


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Skate Commander — web cockpit for the Skate twin/robot")
    ap.add_argument("--model-dir", default=None,
                    help="skt_v3 folder of your Rbotic/skate_teleop clone "
                         "(auto-detected from $SKT_DIR or a nearby clone if omitted)")
    ap.add_argument("--real", action="store_true",
                    help="drive a real Skate instead of the local sim endpoint")
    ap.add_argument("--real-host", default="r.local")
    ap.add_argument("--sim-host", default="127.0.0.1",
                    help="advanced: attach to a sim endpoint on another host "
                         "(e.g. a WSL IP running the ROS 2 stack) instead of "
                         "127.0.0.1; a non-local host disables --spawn-sim")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open the cockpit in a browser on startup")
    ap.add_argument("--spawn-sim", metavar="MJCF_XML", default=None,
                    help="advanced: explicit sim-endpoint model "
                         "(default: the auto-generated skt_v3_collision.xml)")
    ap.add_argument("--collision-model", metavar="COLLISION_XML", default=None,
                    help="advanced: explicit collision-guard model "
                         "(default: the auto-generated skt_v3_collision.xml)")
    ap.add_argument("--ik", choices=("dls", "mink"), default=None,
                    help="drag-IK backend: 'dls' (default, pure-numpy DLS) or "
                         "'mink' (MuJoCo QP with proactive self-collision "
                         "avoidance; needs the mink package + a collision model, "
                         "auto-falls back to dls). Env: SKATE_IK.")
    ap.add_argument("--rerun", action="store_true", default=None,
                    help="stream the live twin (meshed robot + joint/IK time-series) "
                         "to a rerun.io viewer for debugging; needs rerun-sdk. "
                         "Env: SKATE_RERUN=1.")
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir) if args.model_dir else _find_model_dir()
    if not model_dir or not Path(model_dir).exists():
        sys.exit("[commander] couldn't find the skt_v3 model.\n"
                 f"  clone it once:  git clone {SKT_TELEOP_URL}\n"
                 "  then re-run from that folder, or pass --model-dir <…>/skt_v3")

    sim_mode = not args.real
    collision_xml = None
    if (sim_mode and not args.spawn_sim) or args.collision_model is None:
        try:
            collision_xml = _ensure_collision_model(model_dir)
        except Exception as e:
            if sim_mode and not args.spawn_sim:
                sys.exit(f"[commander] setup failed (the sim needs the model): {e}")
            print(f"[commander] WARNING: collision guard unavailable: {e}")

    local_sim = args.sim_host in ("127.0.0.1", "localhost")
    spawn = args.spawn_sim or (str(collision_xml)
                               if sim_mode and local_sim else None)
    guard = args.collision_model or (str(collision_xml) if collision_xml else None)

    sim_proc = None
    if spawn:
        sim_proc = subprocess.Popen(
            [sys.executable, "-m", "skate_ros2.sim_endpoint",
             "--model", spawn, "--quiet"],
            cwd=str(Path(__file__).resolve().parents[2] / "skate_ros2"))
        print(f"[commander] sim endpoint spawned (pid {sim_proc.pid})")

    import os
    import uvicorn
    ik = (args.ik or os.environ.get("SKATE_IK", "dls")).lower()
    use_rerun = bool(args.rerun) or os.environ.get("SKATE_RERUN", "") not in ("", "0", "off", "no", "false")
    app = build_app(str(model_dir), real_host=args.real_host,
                    sim_host=args.sim_host, collision_model=guard,
                    ik=("mink" if ik == "mink" else "dls"), rerun=use_rerun)
    url = f"http://{args.host}:{args.port}"
    mode = "REAL" if args.real else "SIM"
    print(f"[commander] {url}  ({mode} — starts dampened, press RESUME in the UI)")
    if not args.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.3, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if sim_proc:
            sim_proc.terminate()


if __name__ == "__main__":
    main()
