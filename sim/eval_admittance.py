"""M3 eval — Cartesian admittance: the commanded-stiffness curve and a
push-and-yield, the compliant-control analogue of M2's misalignment curve.

Stages the right arm at a held pose, then:
  * commanded-stiffness curve — drive the admittance from a SUPPLIED wrench (no
    physical contact, so nothing but the control law sets the motion) and show the
    TCP yield e settles to F/K across a stiffness sweep (e·K ≈ F), returning to 0
    when the wrench is removed. This is "moves along the compliant axis AT the
    commanded stiffness", measured.
  * push-and-yield — apply a REAL external force to the wrist body, read it back
    through the M1 wrist sensor, and show the TCP physically yields along the
    pushed axis and returns toward the nominal pose on release (sensor-in-the-loop).

    python eval_admittance.py --model /path/to/skt_v3 [--mode curve|push|both]
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import Arm, reach          # noqa: E402
from benchmark import load_cell, fresh, approach  # noqa: E402
from admittance import Admittance          # noqa: E402

POSE = {"left": [-0.20, 0.42, 0.16], "right": [0.16, 0.40, 0.17]}


def stage_arm(m, side="right"):
    """approach + reach to the held pose (gravity-ff), return (d, Arm)."""
    d = fresh(m)
    approach(m, d)
    reach(m, d, POSE, seconds=2.4, tol=0.012, grav_ff=True)
    return d, Arm(m, d, side)


def stiffness_curve(m, Ks=(200, 400, 800, 1600, 3200), f=8.0):
    d, arm = stage_arm(m)
    print(f"staged. commanded-stiffness curve (supplied wrench F={f:.0f} N along "
          f"+x; steady-state e should equal F/K):\n")
    for K in Ks:
        ad = Admittance(m, d, arm, K=[K, K, K], zeta=1.0)
        ad.run(1.0, f_override=[f, 0.0, 0.0])
        e = float(ad.e[0])
        ad.run(0.8, f_override=[0.0, 0.0, 0.0])          # release
        print(f"  K={K:>5} N/m  ->  yield {e*1000:5.1f} mm  (F/K {f/K*1000:5.1f} mm)"
              f"   e·K {e*K:4.1f} N   after-release {ad.e[0]*1000:+.1f} mm")
    print("\n  per-axis (stiff y, compliant x/z; K=[400,1600,400], F=[8,8,0]):")
    ad = Admittance(m, d, arm, K=[400, 1600, 400], zeta=1.0)
    ad.run(1.0, f_override=[8.0, 8.0, 0.0])
    print(f"    e = {np.round(ad.e*1000, 1)} mm  (F/K = [20.0, 5.0, 0.0])")


def push_and_yield(m, f=8.0, axis=1):
    d, arm = stage_arm(m)
    bid = m.site_bodyid[arm.site]
    ad = Admittance(m, d, arm, K=[500, 500, 500], zeta=1.0)
    for _ in range(40):                                  # baseline settle
        ad.step()
    fx = np.zeros(3); fx[axis] = f
    for _ in range(150):                                 # push
        d.xfrc_applied[bid, :3] = fx
        ad.step()
    d.xfrc_applied[bid, :] = 0.0
    pushed, wl = ad.tcp_offset() * 1000, ad.ext_wrench()
    for _ in range(220):                                 # release and return
        ad.step()
    back = ad.tcp_offset() * 1000
    ax = "xyz"[axis]
    print(f"push-and-yield (real force on the wrist, read back through the M1 "
          f"sensor; +{f:.0f} N along {ax}):\n")
    print(f"  measured external wrench   {np.round(wl, 2)} N")
    print(f"  TCP yielded                {np.round(pushed, 1)} mm")
    print(f"  after release              {np.round(back, 1)} mm (returns to nominal)")


def main():
    ap = argparse.ArgumentParser(description="M3 Cartesian admittance eval")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--mode", default="both", choices=["curve", "push", "both"])
    args = ap.parse_args()
    m = load_cell(args.model)
    if args.mode in ("curve", "both"):
        stiffness_curve(m)
        print()
    if args.mode in ("push", "both"):
        push_and_yield(m)


if __name__ == "__main__":
    main()
