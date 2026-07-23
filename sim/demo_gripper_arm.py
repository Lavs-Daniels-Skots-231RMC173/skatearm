"""Render the M4 arm integration: the wrist gripper grasps a part (grasp-force
controlled), the arm carries it over the place bin and opens the jaws to release
it — a full weld-free pick-and-place. Writes docs/img/gripper_arm_place.gif.

    python demo_gripper_arm.py <model_dir>
"""
import os
import sys

import mujoco
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gripper_cell import make
from gripper_arm import grasp_carry_place

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\dino3\skate_teleop\skt_v3"
GIF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "img", "gripper_arm_place.gif")

m = mujoco.MjModel.from_xml_path(make(MODEL_DIR))
ren = mujoco.Renderer(m, 372, 496)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.13, 0.43, 0.12]
cam.distance, cam.azimuth, cam.elevation = 0.58, 34, -13
frames = []
n = [0]


def on_step(d):
    n[0] += 1
    if n[0] % 6 == 0:
        ren.update_scene(d, cam)
        frames.append(ren.render().copy())


r = grasp_carry_place(m, on_step=on_step)
os.makedirs(os.path.dirname(GIF), exist_ok=True)
imgs = [Image.fromarray(f) for f in frames]
imgs[0].save(GIF, save_all=True, append_images=imgs[1:], duration=55, loop=0)
print(f"wrote {GIF} ({len(frames)} frames); grasp {r['grasp_n']} N, "
      f"placed {r['placed']}, part_z {r['part_z']}")
