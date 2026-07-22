"""M2 eval — misalignment-tolerance sweep for the force-regulated insertion.

Stages the peg over the pocket ONCE, clamps the base into a rigid fixture (the
``fixture_base`` weld — so the sweep is deterministic with no base recoil), then
snapshots the sim state and, for each xy misalignment (the offset between the
controller's ASSUMED hole centre and the true hole) restores the snapshot and runs
the :class:`insertion.Insertion` controller. The controller only ever sees the
wrench + wrist proprioception + assumed centre; the peg pose is used ONLY here, for
scoring the result (the "oracle").

The headline metric is the **misalignment-tolerance curve**: insertion success vs
initial xy error, with the spiral hole-search ON, and a search-OFF baseline that
shows the open-loop descent just jams. This is the result that turns "assembles when
perfectly aligned" into "assembles under realistic misalignment via contact force".

    python eval_insertion.py --model /path/to/skt_v3 --offsets 0,2,4,6,8 --dirs 6 \
        --no-search-baseline
"""
import argparse
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import reach, hold, grasp, Arm  # noqa: E402
from benchmark import load_cell, fresh, approach, body_id, servo6_both  # noqa: E402
from insertion import Insertion  # noqa: E402

POCKET_UP = 0.027        # pocket opening above the base body origin (from make_cell_scene)
PEG_HALF = 0.020         # peg half-height (peg bottom = peg origin - PEG_HALF)
SEAT_MIN = 0.010         # oracle: >= this insertion depth counts as seated
XY_MAX = 0.006           # oracle: peg centre within this of the hole counts as in-pocket


def weld_here(m, d, name):
    """Activate a weld holding its two bodies at their CURRENT relative pose
    (generalised primitives.grasp: also welds a body to the world)."""
    eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    R1 = d.xmat[b1].reshape(3, 3)
    p_rel = R1.T @ (d.xpos[b2] - d.xpos[b1])
    q1, q2 = d.xquat[b1].copy(), d.xquat[b2].copy()
    q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, q1)
    q_rel = np.zeros(4); mujoco.mju_mulQuat(q_rel, q1inv, q2)
    m.eq_data[eq, :] = 0
    m.eq_data[eq, 3:6] = p_rel
    m.eq_data[eq, 6:10] = q_rel
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def snapshot(d):
    return {"qpos": d.qpos.copy(), "qvel": d.qvel.copy(), "act": d.act.copy(),
            "ctrl": d.ctrl.copy(), "eqa": d.eq_active.copy()}


def restore(m, d, s):
    d.qpos[:] = s["qpos"]; d.qvel[:] = s["qvel"]; d.act[:] = s["act"]
    d.ctrl[:] = s["ctrl"]; d.eq_active[:] = s["eqa"]
    d.qfrc_applied[:] = 0
    mujoco.mj_forward(m, d)


def stage(m):
    """Bimanual stage: grasp the parts, carry to the meet point, align the peg a
    few mm above the pocket, then clamp the base into the rigid fixture. Returns
    (d, armR, W0, bp, pg) where W0 is the wrist xy that centres the peg on the hole."""
    d = fresh(m)
    approach(m, d)
    armL, armR = Arm(m, d, "left"), Arm(m, d, "right")
    reach(m, d, {"left": [-0.18, 0.44, 0.20], "right": [0.18, 0.44, 0.20]},
          seconds=2.4, tol=0.012, grav_ff=True)
    reach(m, d, {"left": [-0.18, 0.44, 0.115], "right": [0.18, 0.44, 0.115]},
          seconds=2.0, tol=0.010, grav_ff=True)
    grasp(m, d, "left"); grasp(m, d, "right"); hold(m, d, 0.4)
    armL.lock_orientation(); armR.lock_orientation()
    MEET_L = [-0.053, 0.41, 0.21]
    servo6_both(m, d, armL, armR, MEET_L, [0.053, 0.41, 0.30], seconds=4.5)
    bp, pg = body_id(m, "base_part"), body_id(m, "peg")

    def pocket_top():
        return d.xpos[bp] + np.array([0, 0, POCKET_UP])

    def peg_bottom():
        return d.xpos[pg] + np.array([0, 0, -PEG_HALF])

    for _ in range(500):                                 # align the peg ~12mm above the opening
        exy = (pocket_top() - d.xpos[pg])[:2]
        zerr = (pocket_top()[2] + 0.012) - peg_bottom()[2]
        q, _ = armR.ik_step6(armR.ee_pos() + np.array([exy[0], exy[1], zerr])); armR.set_ctrl(q)
        qL, _ = armL.ik_step6(np.asarray(MEET_L)); armL.set_ctrl(qL)
        for _ in range(4):
            mujoco.mj_step(m, d)
        if np.linalg.norm(exy) < 0.0035 and abs(zerr) < 0.006:
            break
    weld_here(m, d, "fixture_base")
    hold(m, d, 0.2)
    return d, armR, armR.ee_pos()[:2].copy(), bp, pg


def score(d, bp, pg):
    """Oracle: is the peg actually seated? (peg centre over the hole AND descended
    at least SEAT_MIN into the pocket). Used only to score — never for control."""
    rel = d.xpos[pg] - d.xpos[bp]
    rel_xy = float(np.linalg.norm(rel[:2]))
    depth = float((d.xpos[bp][2] + POCKET_UP) - (d.xpos[pg][2] - PEG_HALF))
    return bool(rel_xy < XY_MAX and depth >= SEAT_MIN), rel_xy, depth


def run_one(m, d, armR, snap, W0, bp, pg, offset_xy, search=True, **params):
    """Restore the staged snapshot and run one insertion at the given xy offset
    between assumed centre and true hole. Returns the controller result with the
    ORACLE seated verdict substituted in."""
    restore(m, d, snap)
    res = Insertion(m, d, armR, W0 + np.asarray(offset_xy, float), **params).run(search=search)
    seated, rel_xy, depth = score(d, bp, pg)
    res["seated"] = seated
    res["rel_xy_mm"] = round(rel_xy * 1000, 1)
    res["depth_mm"] = round(depth * 1000, 1)
    return res


def sweep(model_dir, offsets_mm, dirs, no_search_baseline=False):
    m = load_cell(model_dir)
    d, armR, W0, bp, pg = stage(m)
    snap = snapshot(d)
    angles = [2 * np.pi * k / dirs for k in range(dirs)]
    print(f"staged (W0={W0.round(4)}). misalignment-tolerance sweep, {dirs} dirs/offset:\n")
    modes = [("search", True)] + ([("no-search", False)] if no_search_baseline else [])
    report = {}
    for mode, search in modes:
        print(f"--- {mode} ---")
        report[mode] = {}
        for off_mm in offsets_mm:
            r = off_mm / 1000.0
            ok = 0
            peaks = []
            for a in angles:
                off = [r * np.cos(a), r * np.sin(a)] if off_mm else [0.0, 0.0]
                res = run_one(m, d, armR, snap, W0, bp, pg, off, search=search)
                ok += int(res["seated"])
                peaks.append(res["peak_wrench_n"])
                if off_mm == 0:
                    break                                # a single trial at zero offset
            n = 1 if off_mm == 0 else dirs
            report[mode][off_mm] = (ok, n)
            print(f"  offset {off_mm:>2} mm: {ok}/{n} seated   "
                  f"(peak wrench max {max(peaks):.1f} N)")
        print()
    return report


def main():
    ap = argparse.ArgumentParser(description="M2 misalignment-tolerance sweep")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--offsets", default="0,2,4,6,8", help="xy offsets in mm, comma-sep")
    ap.add_argument("--dirs", type=int, default=6, help="directions per offset")
    ap.add_argument("--no-search-baseline", action="store_true",
                    help="also run with the spiral search OFF")
    args = ap.parse_args()
    offsets = [int(x) for x in args.offsets.split(",") if x.strip() != ""]
    sweep(args.model, offsets, args.dirs, args.no_search_baseline)


if __name__ == "__main__":
    main()
