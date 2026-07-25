"""Render the M3 push-and-yield: the arm holds a pose under Cartesian admittance;
an external force pushes the wrist, the TCP YIELDS along the force and RETURNS to
the nominal pose when released. Writes docs/img/push_and_yield.gif.

    python demo_admittance.py <model_dir>
"""
import os
import sys

import numpy as np
import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import load_cell
from eval_admittance import stage_arm
from admittance import Admittance

MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.environ.get("SKT_DIR", "/tmp/skate_teleop/skt_v3"))
GIF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "img", "push_and_yield.gif")

m = load_cell(MODEL_DIR)
d, arm = stage_arm(m)
wb = m.site_bodyid[arm.site]
ad = Admittance(m, d, arm, K=(300., 300., 300.), zeta=1.0)   # softer for a visible yield
ren = mujoco.Renderer(m, 384, 512)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.14, 0.41, 0.16]
cam.distance, cam.azimuth, cam.elevation = 0.82, 90, -8
frames = []


def cap():
    ren.update_scene(d, cam)
    frames.append(ren.render().copy())


def phase(cycles, fext, every=3):
    f = np.asarray(fext, float)
    for i in range(cycles):
        d.xfrc_applied[wb, :3] = f
        ad.step()
        if i % every == 0:
            cap()
    d.xfrc_applied[wb, :] = 0.0


phase(22, [0, 0, 0])          # hold nominal
phase(55, [16, 0, 0])         # push +x -> yield (swings right)
phase(70, [0, 0, 0])          # release -> return
phase(55, [-16, 0, 0])        # push -x -> yield (swings left)
phase(80, [0, 0, 0])          # release -> return

os.makedirs(os.path.dirname(GIF), exist_ok=True)
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=55, loop=0)
print(f"wrote {GIF} ({len(frames)} frames)")
