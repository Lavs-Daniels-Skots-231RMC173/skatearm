"""M4 (scoped) eval — actuated parallel-jaw gripper: grasp-force control + a
friction hold with no weld.

For each target grasp force: open the jaws, close-to-force (regulating the
measured pad force), release the world-pin, and report whether the part is held
by friction alone and by how much it moved. The measured grasp force tracks the
target; the light peg is held with margin across the stable range. Ejection above
~5 N (a rigid cylinder squeezed between rigid pads) is the grasp-slip / contact
frontier deferred to full M4.

    python eval_gripper.py --model /path/to/skt_v3
"""
import argparse
import os
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gripper_scene import make          # noqa: E402
from gripper import Gripper                   # noqa: E402


def sweep(model_dir, targets=(2.0, 3.0, 4.0, 5.0)):
    m = mujoco.MjModel.from_xml_path(make(model_dir))
    print("grasp-force control + friction hold (no weld):\n")
    for ft in targets:
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        g = Gripper(m, d)
        z0 = g.peg_pos()[2]
        g.open(cycles=25)
        meas = g.close_to_force(ft)
        g.set_pin(False)                              # friction grasp only
        for _ in range(250):
            mujoco.mj_step(m, d)
        dz = (g.peg_pos()[2] - z0) * 1000
        print(f"  target {ft:4.1f} N  ->  measured grasp {meas:4.2f} N   "
              f"held={g.holds(z0)!s:5}  peg dz {dz:+5.1f} mm   hold {g.grasp_force():4.2f} N")


def main():
    ap = argparse.ArgumentParser(description="M4 (scoped) gripper eval")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    args = ap.parse_args()
    sweep(args.model)


if __name__ == "__main__":
    main()
