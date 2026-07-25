#!/usr/bin/env python3
"""Phase 1 demo: bimanual PICK & PLACE in the work-cell scene.

Sequence (all under physics):
  1. fold elbows + raise   — collision-aware route over the table edge
  2. hover + descend       — IK to just above the base part / peg
  3. GRASP (both hands)    — v1 'magnetic' weld stand-in engages
  4. lift & carry          — parts leave the table, carried to chest height
  5. lower & RELEASE       — parts placed back on the table
  6. retreat & hold        — clip ends at rest

The grasp is a weld constraint engaged at the part's current pose (no snap) —
a documented stand-in until the real Skate gripper geometry is known.

M4's actuated jaws are the sim replacement, and `sequencer.py` drives the RIGHT
hand of the full S0-S7 cycle with them on a scene built by
`make_cell_scene.py --gripper`. This Phase 1 demo deliberately stays on the weld
(both hands): it runs on the default cell model, which is unchanged.

Usage:
    python make_control_model.py /path/to/skate_teleop/skt_v3
    python make_collision_model.py /path/to/skate_teleop/skt_v3
    python make_cell_scene.py /path/to/skate_teleop/skt_v3
    python demo_cell_pick.py --model /path/to/skate_teleop/skt_v3 --out demo.mp4
"""
import argparse
import os
import sys

import imageio
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import reach, hold, move_joints, grasp, release  # noqa: E402

BASE_XY = (-0.12, 0.44)   # parts moved closer to center for the assembly demo
PEG_XY = (0.12, 0.44)
HOVER_Z, GRASP_Z, CARRY_Z = 0.20, 0.115, 0.28
TOTAL_S = 26.0  # nominal timeline for the camera ease


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--out", default="cell_pick_demo.mp4", help=".mp4 or .gif")
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--size", default=None, help="WxH")
    args = ap.parse_args()

    is_mp4 = args.out.lower().endswith(".mp4")
    fps = args.fps or (25 if is_mp4 else 12)
    size = args.size or ("960x720" if is_mp4 else "480x360")
    w, h = (int(x) for x in size.lower().split("x"))

    m = mujoco.MjModel.from_xml_path(os.path.join(args.model, "skt_v3_cell.xml"))
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)

    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0.30, 0.15]
    cam.distance = 1.9
    cam.elevation = -20

    if is_mp4:
        writer = imageio.get_writer(args.out, fps=fps, codec="libx264",
                                    quality=7, pixelformat="yuv420p")
    else:
        writer = imageio.get_writer(args.out, fps=fps, loop=0)

    state = {"n": 0, "t": 0.0}

    def on_frame(_):
        state["n"] += 1
        if state["n"] % 5:
            return
        state["t"] += 5 * 0.008
        s = state["t"] / TOTAL_S
        # Skate camera convention: azimuth ~240..285 faces the robot's FRONT
        cam.azimuth = 240 + 45 * (0.5 - 0.5 * np.cos(np.pi * min(s, 1.0)))
        r.update_scene(d, camera=cam)
        writer.append_data(r.render())

    move_joints(m, d, {"a1": 0.3, "a3": 2.2}, seconds=2.0, on_frame=on_frame)
    move_joints(m, d, {"a0": 0.9, "a1": 0.3, "a3": 1.3}, seconds=2.0, on_frame=on_frame)
    reach(m, d, {"left": [*BASE_XY, HOVER_Z], "right": [*PEG_XY, HOVER_Z]},
          seconds=2.5, on_frame=on_frame, tol=0.012)
    reach(m, d, {"left": [*BASE_XY, GRASP_Z], "right": [*PEG_XY, GRASP_Z]},
          seconds=2.0, on_frame=on_frame, tol=0.010)
    grasp(m, d, "left")
    grasp(m, d, "right")
    hold(m, d, 0.5, on_frame=on_frame)
    reach(m, d, {"left": [-0.15, 0.40, CARRY_Z], "right": [0.15, 0.40, CARRY_Z]},
          seconds=2.8, on_frame=on_frame, tol=0.012)
    hold(m, d, 0.8, on_frame=on_frame)
    reach(m, d, {"left": [*BASE_XY, GRASP_Z], "right": [*PEG_XY, GRASP_Z]},
          seconds=2.8, on_frame=on_frame, tol=0.010)
    release(m, d, "left")
    release(m, d, "right")
    hold(m, d, 0.6, on_frame=on_frame)
    reach(m, d, {"left": [-0.18, 0.40, 0.26], "right": [0.18, 0.40, 0.26]},
          seconds=2.2, on_frame=on_frame, tol=0.012)
    hold(m, d, 1.0, on_frame=on_frame)

    writer.close()
    # verify the parts actually travelled and returned
    for name in ("base_part", "peg"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"{name} final z: {d.xpos[b][2]:.3f} (on table ≈ 0.03–0.05)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
