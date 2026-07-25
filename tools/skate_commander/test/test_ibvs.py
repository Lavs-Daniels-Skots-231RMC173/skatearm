"""IBVS robustness: with a deliberately mis-calibrated BELIEVED camera model,
closed-loop image servoing still drives the EE onto the target, while one-shot
open-loop back-projection (same wrong model) misses badly.

Needs mujoco + Pillow + the official clone. Drives the real bridge + the
CameraStreamer against the skate_ros2 MuJoCo sim endpoint:

    SKT_DIR=.../skt_v3 SKATE_MJCF=.../skt_v3_control.xml python3 test_ibvs.py
"""
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))

SKT = Path(os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
MJCF = os.environ.get("SKATE_MJCF", str(SKT / "skt_v3_control.xml"))
DT = 1 / 60.0
CUBE_XY = np.array([0.13, 0.35])          # true target in the generated scene



def _skip(msg):
    """Real pytest.skip under pytest; clean print when run as a standalone script."""
    import sys
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"SKIP: {msg}")


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


_GL = None
_PROBE = ("import mujoco; "
          "mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 8, 8)")


def _gl_ok():
    """Can MuJoCo open a render context here? Both tests draw frames, so on a
    headless box without a GL backend they must skip rather than explode.

    The probe runs in a child process on purpose: a context that fails to come
    up leaves MuJoCo's GL state poisoned, and the *second* attempt in the same
    interpreter aborts the process outright instead of raising. One child, one
    verdict, cached — the parent never touches a broken backend.
    """
    global _GL
    if _GL is None:
        _GL = subprocess.run([sys.executable, "-c", _PROBE],
                             capture_output=True).returncode == 0
    return _GL


NO_GL = ("no MuJoCo GL backend — set MUJOCO_GL=osmesa (apt install libosmesa6) "
         "or MUJOCO_GL=egl, or run on a machine with a display")


def test_ibvs_robust_to_miscalibration():
    try:
        import mujoco  # noqa: F401
        import PIL      # noqa: F401
    except ImportError:
        _skip("mujoco / Pillow not installed"); return
    if not Path(MJCF).exists():
        _skip("no control model"); return
    if not _gl_ok():
        _skip(NO_GL); return

    from skate_commander import camera, vision
    from skate_commander.bridge import RobotBridge
    from skate_commander.ibvs import ImageServo
    from skate_commander.kinematics import ArmKinematics
    from skate_commander.urdf import joint_limits, parse_urdf
    from skate_ros2.sim_endpoint import SkateSimEndpoint

    model = parse_urdf(SKT / "skt_v3.urdf")
    kin = {a: ArmKinematics(model, a) for a in ("left", "right")}
    port = _free_port()
    ep = SkateSimEndpoint(MJCF, port=port, bind="127.0.0.1", verbose=False)
    threading.Thread(target=ep.run, kwargs={"duration": 60.0},
                     daemon=True).start()
    br = RobotBridge(sim_host="127.0.0.1", sim_port=port,
                     limits=joint_limits(model), kin=kin)
    scene = camera.build_scene_xml(str(SKT))
    cam = camera.CameraStreamer(scene,
                                lambda: (None if br.targ is None else list(br.targ)))

    def spin_until(pred, timeout):
        """Tick the bridge until pred() holds. True if it did, False on timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            br.tick(DT, ui_attached=True); time.sleep(DT)
            if pred():
                return True
        return False

    def track(secs=0.5):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            br.tick(DT, ui_attached=True); time.sleep(DT)
            if br.ik_targets.get("right") is None:
                break

    # The bridge arms from telemetry and only leaves E-STOP once the link is
    # fresh; under a software renderer the camera thread pushes that past a
    # second. Wait for the condition, never for a fixed interval: resume() is
    # a silent no-op while the link is stale, set_ik_target() is then ignored,
    # and every assertion below would pass judgement on an arm that never moved.
    assert spin_until(lambda: br.targ is not None and br.link.connected, 10.0), \
        (f"bridge never came up (armed={br.targ is not None}, "
         f"connected={br.link.connected})")
    assert br.resume(), "bridge refused to leave E-STOP"
    kr = kin["right"]

    # detect the static target ONCE while the arm is clear of it
    obs = cam.detect(ee_world=list(kr.fk(br.targ)))
    assert obs.get("found") and "cam" in obs, obs
    s_cube = np.array(obs["pixel"])
    C = obs["cam"]
    f0, cx0, cy0 = vision.intrinsics(C["fovy"], C["W"], C["H"])
    pos0 = np.array(C["pos"]); mat0 = np.array(C["mat"]).reshape(3, 3)

    def ee_pixel():
        return np.array(vision.project(kr.fk(br.targ), pos0, mat0,
                                       f0, cx0, cy0)[:2])

    approach_z = camera.GRASP_Z + 0.08
    br.set_ik_target("right", [0.10, 0.32, approach_z], auto=True)
    track(3.0)

    # deliberately MIS-CALIBRATED believed model (3 cm offset, 12% wrong fovy)
    bad_pos = pos0 + np.array([0.03, 0.0, 0.0])
    bad_fovy = C["fovy"] * 1.12
    servo = ImageServo(bad_pos, mat0, bad_fovy, C["W"], C["H"],
                       gain=0.7, max_step=0.03)

    # open-loop baseline: back-project the cube pixel through the bad model
    fb, cxb, cyb = vision.intrinsics(bad_fovy, C["W"], C["H"])
    P_ol = vision.backproject(s_cube[0], s_cube[1], camera.GRASP_Z,
                              bad_pos, mat0, fb, cxb, cyb)
    open_err = float(np.linalg.norm(P_ol[:2] - CUBE_XY))

    # IBVS closed loop: drive the gripper pixel onto s_cube while descending
    z = approach_z
    err_px = 999.0
    it = 0
    for it in range(45):
        dxy, err_px = servo.step(s_cube, ee_pixel(), kr.fk(br.targ))
        z = max(camera.GRASP_Z + 0.03, z - 0.006)
        ee = kr.fk(br.targ)
        br.set_ik_target("right", [ee[0] + dxy[0], ee[1] + dxy[1], z],
                         auto=True)
        track(0.5)
        if err_px < 3.0 and z <= camera.GRASP_Z + 0.035:
            break
    ibvs_err = float(np.linalg.norm(kr.fk(br.targ)[:2] - CUBE_XY))

    assert open_err > 0.020, f"perturbation too small ({open_err*1000:.1f}mm)"
    assert ibvs_err < 0.012, f"IBVS did not converge ({ibvs_err*1000:.1f}mm)"
    assert ibvs_err < open_err / 2.5, "IBVS not clearly better than open-loop"
    assert err_px < 6.0, f"image error not nulled ({err_px:.1f}px)"
    print(f"PASS IBVS: open-loop miss {open_err*1000:.1f} mm vs closed-loop "
          f"{ibvs_err*1000:.1f} mm ({open_err/ibvs_err:.1f}x), image err "
          f"{err_px:.1f}px in {it+1} iters")
    cam.close(); br.close(); ep.close()


def test_depth_cloud_backprojection():
    """The work-camera depth back-projects to a world cloud whose magenta points
    cluster at the cube's true position (~mm) — validates vision.depth_cloud."""
    try:
        import mujoco
    except ImportError:
        _skip("mujoco not installed"); return
    if not Path(MJCF).exists():
        _skip("no control model"); return
    if not _gl_ok():
        _skip(NO_GL); return
    from skate_commander import camera, vision
    scene = camera.build_scene_xml(SKT)
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    W, H = 320, 240
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera="cam_work"); rgb = r.render()
    r.enable_depth_rendering(); r.update_scene(d, camera="cam_work"); depth = r.render()
    r.disable_depth_rendering()
    cid = m.camera("cam_work").id
    cpos = np.asarray(d.cam_xpos[cid]); cmat = np.asarray(d.cam_xmat[cid]).reshape(3, 3)
    fovy = float(m.cam_fovy[cid])
    cloud = vision.depth_cloud(depth, rgb, cpos, cmat, fovy, stride=2, zmax=0.75)
    assert len(cloud) > 200, "cloud should have points"
    C = (cloud[:, 3:] * 255).astype(int)
    mag = ((C[:, 0] > 110) & (C[:, 2] > 100)
           & (C[:, 0] - C[:, 1] > 60) & (C[:, 2] - C[:, 1] > 40))
    assert mag.sum() >= 20, f"too few magenta points ({int(mag.sum())})"
    xy = cloud[mag, :2].mean(0)
    assert np.linalg.norm(xy - CUBE_XY) < 0.01, \
        f"magenta cluster {xy} off the true cube {CUBE_XY}"
    print(f"PASS depth cloud: {len(cloud)} pts, magenta -> "
          f"{[round(float(v), 3) for v in cloud[mag, :3].mean(0)]} (cube {list(CUBE_XY)})")


if __name__ == "__main__":
    test_ibvs_robust_to_miscalibration()
    test_depth_cloud_backprojection()
