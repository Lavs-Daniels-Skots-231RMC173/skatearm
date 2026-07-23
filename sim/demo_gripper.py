"""Render the M4 (scoped) gripper: force-controlled close, friction hold with no
weld, then release. Writes docs/img/gripper_grasp.gif.

    python demo_gripper.py <model_dir>
"""
import os
import sys

import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gripper_scene import make
from gripper import Gripper

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\dino3\skate_teleop\skt_v3"
GIF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "img", "gripper_grasp.gif")

m = mujoco.MjModel.from_xml_path(make(MODEL_DIR))
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
g = Gripper(m, d)
ren = mujoco.Renderer(m, 420, 560)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.27]
cam.distance, cam.azimuth, cam.elevation = 0.34, 88, -10
frames = []


def cap():
    ren.update_scene(d, cam)
    frames.append(ren.render().copy())


def run(cycles, ctrl, every=3):
    for i in range(cycles):
        d.ctrl[g.aid] = ctrl
        for _ in range(4):
            mujoco.mj_step(m, d)
        if i % every == 0:
            cap()


run(14, -8.0)                                   # jaws open, peg pinned
cmd = 0.0
for i in range(210):                            # force-controlled close to ~4 N
    cmd = (cmd + 1.0 if (g.grasp_force() < 0.3 and cmd < 60)
           else float(np.clip(cmd + 1.5 * (4.0 - g.grasp_force()), 0, 60)))
    d.ctrl[g.aid] = cmd
    for _ in range(4):
        mujoco.mj_step(m, d)
    if i % 3 == 0:
        cap()
g.set_pin(False)                                # release the world-pin
run(45, cmd)                                    # held by friction alone
run(55, -8.0)                                   # open -> peg drops

os.makedirs(os.path.dirname(GIF), exist_ok=True)
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=55, loop=0)
print(f"wrote {GIF} ({len(frames)} frames), peg z={g.peg_pos()[2]:.3f}")
