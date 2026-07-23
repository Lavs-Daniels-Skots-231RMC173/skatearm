"""M4 (scoped) eval — actuated parallel-jaw gripper: grasp-force control, a
friction hold with no weld, and the grasp-slip curve.

  * force + hold — for each target grasp force: open, close-to-force (regulating
    the measured pad force), release the world-pin, report whether the part is
    held by friction alone and by how much it moved.
  * slip curve — for each grasp force, ramp a downward payload on the held part
    until it slips out; the slip payload (holding capacity) grows with the grasp
    force, the classic grasp-slip relationship.

    python eval_gripper.py --model /path/to/skt_v3 [--mode hold|slip|both]
"""
import argparse
import os
import sys

import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_gripper_scene import make          # noqa: E402
from gripper import Gripper                   # noqa: E402


def _grasp(m, ft, settle=80):
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    g = Gripper(m, d)
    z0 = g.part_pos()[2]
    g.open(cycles=25)
    meas = g.close_to_force(ft)
    g.set_pin(False)                                  # friction grasp only
    for _ in range(settle):
        mujoco.mj_step(m, d)
    return g, z0, meas


def hold(m, targets=(2.0, 3.0, 4.0, 5.0)):
    print("grasp-force control + friction hold (no weld):\n")
    for ft in targets:
        g, z0, meas = _grasp(m, ft, settle=250)
        dz = (g.part_pos()[2] - z0) * 1000
        print(f"  target {ft:4.1f} N  ->  measured grasp {meas:4.2f} N   "
              f"held={g.holds(z0)!s:5}  part dz {dz:+5.1f} mm   hold {g.grasp_force():4.2f} N")


def slip(m, targets=(2.0, 3.0, 4.0, 5.0)):
    print("grasp-slip curve — downward payload until the part slips out:\n")
    for ft in targets:
        g, z0, meas = _grasp(m, ft, settle=60)
        z_grip = g.part_pos()[2]
        payload = g.slip_payload(z_grip)
        tag = f"slips at {payload:5.2f} N payload" if payload else "held > 30 N"
        print(f"  grasp {meas:4.2f} N  ->  {tag}")


def main():
    ap = argparse.ArgumentParser(description="M4 (scoped) gripper eval")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--mode", default="both", choices=["hold", "slip", "both"])
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(make(args.model))
    if args.mode in ("hold", "both"):
        hold(m); print()
    if args.mode in ("slip", "both"):
        slip(m)


if __name__ == "__main__":
    main()
