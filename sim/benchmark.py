#!/usr/bin/env python3
"""Bimanual benchmark suite — repeatable two-arm tasks in the MuJoCo work-cell
with quantitative metrics. Headless (no render), seeded, aggregated over trials.

Every task runs under physics through the same task-space primitives the
cockpit and the work-cell demos use (position + 6-DoF DLS IK on the position
servos — qpos is never written directly). Tasks:

  reach   — both arms servo to random reachable target pairs; metric = final
            EE position error and success within tolerance.
  carry   — the left arm grasps the base part and the right the peg, then both
            carry their objects together (6-DoF, orientation-locked) to shifted
            targets; metric = each object retained (not dropped), carry distance
            and tilt. (A true weld-transfer hand-off waits for the real gripper.)
  insert  — the full bimanual peg-in-hole: lateral-offset grasps, a 6-DoF
            orientation-locked carry to the meet point, relative-servo align,
            then a force-guarded (tau-watchdog) descent; metric = insertion
            depth, peg tilt, in-pocket and abort.
  insert_m2 — as `insert`, but the M2 force-regulated controller (wrist-wrench
            regulation + spiral bore-search, sim/insertion.py) replaces the
            tau-watchdog descent; metric = in-pocket, peak wrench, tilt, abort.

Usage (build the models + cell scene once, then run):
    python make_control_model.py   /path/to/skate_teleop/skt_v3
    python make_collision_model.py /path/to/skate_teleop/skt_v3
    python make_cell_scene.py      /path/to/skate_teleop/skt_v3
    python benchmark.py --model /path/to/skate_teleop/skt_v3 \
        [--trials 5] [--tasks reach,handoff,insert] [--seed 0] [--json out.json]
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import reach, hold, move_joints, grasp, release, Arm  # noqa: E402
from insertion import Insertion  # noqa: E402

TABLE_Z = 0.10          # a peg below this height counts as dropped


def load_cell(model_dir):
    xml = os.path.join(model_dir, "skt_v3_cell.xml")
    if not os.path.exists(xml):
        sys.exit("run make_cell_scene.py first (needs skt_v3_cell.xml)")
    return mujoco.MjModel.from_xml_path(xml)


def fresh(m, settle=500):
    """A fresh data handle with the parts settled on the table."""
    d = mujoco.MjData(m)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    return d


def approach(m, d):
    """Fold the elbows, then raise the arms over the table edge — the
    collision-safe joint-space route the demos use before Cartesian IK (a
    straight path from the hanging rest pose crosses the table front edge)."""
    move_joints(m, d, {"a1": 0.3, "a3": 2.2}, seconds=1.6)
    move_joints(m, d, {"a0": 0.9, "a1": 0.3, "a3": 1.3}, seconds=1.6)


def body_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def tilt_deg(m, d, bid):
    """Angle of a body's local +z axis from world vertical (deg)."""
    zc = d.xmat[bid].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(zc @ np.array([0, 0, 1.0]), -1, 1))))


def summarize(vals):
    a = np.array(vals, float)
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "min": round(float(a.min()), 2), "max": round(float(a.max()), 2)}


def servo6_one(m, d, arm, target, seconds=3.0, tol=0.010, extra=400):
    """Eased single-arm 6-DoF servo (holds orientation via lock_orientation)."""
    s0, g = arm.ee_pos(), np.asarray(target, float)
    n = max(int(seconds / (m.opt.timestep * 4)), 1)
    for i in range(n + extra):
        ss = min(1.0, (i + 1) / n); ss = ss * ss * (3 - 2 * ss)
        q, _ = arm.ik_step6(s0 + (g - s0) * ss)
        arm.set_ctrl(q)
        for _ in range(4):
            mujoco.mj_step(m, d)
        if i >= n and np.linalg.norm(g - arm.ee_pos()) < tol:
            break


def servo6_both(m, d, armL, armR, tL, tR, seconds=4.5, tol=0.010, extra=400):
    """Eased two-arm 6-DoF servo (both arms hold their locked orientation)."""
    sL, sR = armL.ee_pos(), armR.ee_pos()
    gL, gR = np.asarray(tL, float), np.asarray(tR, float)
    n = max(int(seconds / (m.opt.timestep * 4)), 1)
    for i in range(n + extra):
        ss = min(1.0, (i + 1) / n); ss = ss * ss * (3 - 2 * ss)
        qL, _ = armL.ik_step6(sL + (gL - sL) * ss); armL.set_ctrl(qL)
        qR, _ = armR.ik_step6(sR + (gR - sR) * ss); armR.set_ctrl(qR)
        for _ in range(4):
            mujoco.mj_step(m, d)
        if i >= n and np.linalg.norm(gL - armL.ee_pos()) < tol \
                and np.linalg.norm(gR - armR.ee_pos()) < tol:
            break


# ---- task: bimanual reach --------------------------------------------------
def task_reach(m, trials, rng):
    """Both arms servo to random reachable target pairs (left on -x, right on
    +x), chained from the previous pose. Position task (free wrist)."""
    d = fresh(m)
    approach(m, d)
    rows = []
    for _ in range(trials):
        tL = [rng.uniform(-0.20, -0.05), rng.uniform(0.38, 0.46), rng.uniform(0.15, 0.28)]
        tR = [rng.uniform(0.05, 0.20), rng.uniform(0.38, 0.46), rng.uniform(0.15, 0.28)]
        t0 = time.perf_counter()
        err = reach(m, d, {"left": tL, "right": tR}, seconds=3.0, tol=0.010,
                    settle_extra=2.5, grav_ff=True)
        me = max(err.values())
        rows.append({"err_l_mm": round(err["left"] * 1000, 2),
                     "err_r_mm": round(err["right"] * 1000, 2),
                     "max_err_mm": round(me * 1000, 2),
                     "wall_s": round(time.perf_counter() - t0, 2),
                     "success": bool(me < 0.012)})
    return rows


# ---- task: bimanual carry --------------------------------------------------
def task_carry(m, trials, rng):
    """Coordinated two-arm transport: the left arm grasps the base part and the
    right the peg (the proven dual-grasp), then both carry their objects
    together (6-DoF, orientation-locked) to shifted targets. Metric = each
    object retained (not dropped), how far it was carried, and its tilt.
    (A true hand-off waits for the real gripper — every benchmark task runs the
    DEFAULT cell, where both grasps are magnetic weld stand-ins, and passing one
    object between two welds is a hardware-era task. M4's actuated jaws exist,
    but they are wired into sequencer.py's cycle on the right hand and the
    opt-in `--gripper` scene, not into these tasks.)"""
    bp, pg = body_id(m, "base_part"), body_id(m, "peg")
    rows = []
    for _ in range(trials):
        d = fresh(m)
        approach(m, d)
        armL, armR = Arm(m, d, "left"), Arm(m, d, "right")
        jx = rng.uniform(-0.004, 0.004)
        base0, peg0 = d.xpos[bp].copy(), d.xpos[pg].copy()
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.20], "right": [0.18 + jx, 0.44, 0.20]},
              seconds=2.4, tol=0.012, grav_ff=True)
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.115], "right": [0.18 + jx, 0.44, 0.115]},
              seconds=2.0, tol=0.010, grav_ff=True)
        grasp(m, d, "left"); grasp(m, d, "right"); hold(m, d, 0.4)
        armL.lock_orientation(); armR.lock_orientation()
        servo6_both(m, d, armL, armR, [-0.10, 0.42, 0.26], [0.10, 0.42, 0.26], seconds=4.0)
        base_ret = bool(d.xpos[bp][2] > base0[2] + 0.02)   # lifted clear, not dropped
        peg_ret = bool(d.xpos[pg][2] > peg0[2] + 0.02)
        rows.append({"base_z_mm": round(float(d.xpos[bp][2]) * 1000, 1),
                     "peg_z_mm": round(float(d.xpos[pg][2]) * 1000, 1),
                     "base_carried_mm": round(float(np.linalg.norm(d.xpos[bp] - base0)) * 1000, 1),
                     "peg_carried_mm": round(float(np.linalg.norm(d.xpos[pg] - peg0)) * 1000, 1),
                     "peg_tilt_deg": round(tilt_deg(m, d, pg), 1),
                     "retained": bool(base_ret and peg_ret),
                     "success": bool(base_ret and peg_ret and tilt_deg(m, d, pg) < 20)})
    return rows


# ---- task: peg-in-hole insert ---------------------------------------------
def task_insert(m, trials, rng):
    """Full bimanual assembly: lateral-offset grasps, orientation-locked carry,
    relative-servo align, force-guarded descent (tau watchdog on the right)."""
    bp, pg = body_id(m, "base_part"), body_id(m, "peg")
    tau_ids = [m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR,
               f"tau_a{k}_armR_a{16 + k}")] for k in range(8)]
    DEPTH_TARGET, TAU_LIMIT = 0.018, 25.0
    rows = []
    for _ in range(trials):
        d = fresh(m)
        approach(m, d)
        armL, armR = Arm(m, d, "left"), Arm(m, d, "right")
        jx = rng.uniform(-0.004, 0.004)
        t0 = time.perf_counter()
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.20], "right": [0.18 + jx, 0.44, 0.20]},
              seconds=2.4, tol=0.012, grav_ff=True)
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.115], "right": [0.18 + jx, 0.44, 0.115]},
              seconds=2.0, tol=0.010, grav_ff=True)
        grasp(m, d, "left"); grasp(m, d, "right"); hold(m, d, 0.4)
        armL.lock_orientation(); armR.lock_orientation()
        MEET_L = [-0.053, 0.41, 0.21]
        servo6_both(m, d, armL, armR, MEET_L, [0.053, 0.41, 0.30], seconds=4.5)

        def pocket_top():
            return d.xpos[bp] + np.array([0, 0, 0.027])

        def peg_bottom():
            return d.xpos[pg] + np.array([0, 0, -0.020])

        for _ in range(400):                                    # align over pocket
            exy = (pocket_top() - d.xpos[pg])[:2]
            zerr = (pocket_top()[2] + 0.030) - peg_bottom()[2]
            q, _ = armR.ik_step6(armR.ee_pos() + np.array([exy[0], exy[1], zerr])); armR.set_ctrl(q)
            qL, _ = armL.ik_step6(np.asarray(MEET_L)); armL.set_ctrl(qL)
            for _ in range(4):
                mujoco.mj_step(m, d)
            if np.linalg.norm(exy) < 0.0035 and abs(zerr) < 0.007:
                break
        tau0, aborted = sum(abs(d.sensordata[a]) for a in tau_ids), False
        for _ in range(1500):                                   # force-guarded insert
            exy = (pocket_top() - d.xpos[pg])[:2]
            if pocket_top()[2] - peg_bottom()[2] >= DEPTH_TARGET:
                break
            q, _ = armR.ik_step6(armR.ee_pos() + np.array([exy[0] * 0.8, exy[1] * 0.8, -0.0014]))
            armR.set_ctrl(q)
            qL, _ = armL.ik_step6(np.asarray(MEET_L)); armL.set_ctrl(qL)
            for _ in range(4):
                mujoco.mj_step(m, d)
            if sum(abs(d.sensordata[a]) for a in tau_ids) > tau0 + TAU_LIMIT:
                aborted = True
                break
        depth = pocket_top()[2] - peg_bottom()[2]
        rel = d.xpos[pg] - d.xpos[bp]
        in_pocket = bool(np.linalg.norm(rel[:2]) < 0.006 and 0.015 < rel[2] < 0.03)
        rows.append({"depth_mm": round(float(depth) * 1000, 1),
                     "peg_tilt_deg": round(tilt_deg(m, d, pg), 1),
                     "wall_s": round(time.perf_counter() - t0, 1),
                     "aborted": bool(aborted), "in_pocket": in_pocket,
                     "success": bool(depth >= 0.017 and not aborted and in_pocket)})
    return rows


# ---- task: force-regulated peg-in-hole (M2) --------------------------------
def task_insert_m2(m, trials, rng):
    """Force-regulated peg-in-hole (M2 `Insertion`): the same bimanual staging as
    `insert`, but the right arm regulates the wrist contact force and spiral-searches
    the bore instead of the open-loop tau-watchdog descent. A small random xy
    misalignment (assumed centre vs true bore, <=4 mm) is injected each trial to
    exercise the search; the left arm keeps holding the base at the meet point."""
    bp, pg = body_id(m, "base_part"), body_id(m, "peg")
    MEET_L = [-0.053, 0.41, 0.21]
    rows = []
    for _ in range(trials):
        d = fresh(m)
        approach(m, d)
        armL, armR = Arm(m, d, "left"), Arm(m, d, "right")
        jx = rng.uniform(-0.004, 0.004)
        t0 = time.perf_counter()
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.20], "right": [0.18 + jx, 0.44, 0.20]},
              seconds=2.4, tol=0.012, grav_ff=True)
        reach(m, d, {"left": [-0.18 + jx, 0.44, 0.115], "right": [0.18 + jx, 0.44, 0.115]},
              seconds=2.0, tol=0.010, grav_ff=True)
        grasp(m, d, "left"); grasp(m, d, "right"); hold(m, d, 0.4)
        armL.lock_orientation(); armR.lock_orientation()
        servo6_both(m, d, armL, armR, MEET_L, [0.053, 0.41, 0.30], seconds=4.5)

        def pocket_top():
            return d.xpos[bp] + np.array([0, 0, 0.027])

        def peg_bottom():
            return d.xpos[pg] + np.array([0, 0, -0.020])

        for _ in range(400):                                # align ~12mm above the opening
            exy = (pocket_top() - d.xpos[pg])[:2]
            zerr = (pocket_top()[2] + 0.012) - peg_bottom()[2]
            q, _ = armR.ik_step6(armR.ee_pos() + np.array([exy[0], exy[1], zerr])); armR.set_ctrl(q)
            qL, _ = armL.ik_step6(np.asarray(MEET_L)); armL.set_ctrl(qL)
            for _ in range(4):
                mujoco.mj_step(m, d)
            if np.linalg.norm(exy) < 0.0035 and abs(zerr) < 0.007:
                break
        # residual misalignment the controller must absorb (the align above already
        # gets within ~3.5mm; inject a small extra so trials aren't all identical)
        ang, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.0025)
        center = armR.ee_pos()[:2] + np.array([rr * np.cos(ang), rr * np.sin(ang)])
        res = Insertion(m, d, armR, center, hold_arms=[(armL, MEET_L)]).run(search=True)
        # score on the oracle peg-in-base pose (rel is invariant to base recoil under
        # the left arm), not the controller's proprioceptive seated flag
        rel = d.xpos[pg] - d.xpos[bp]
        in_pocket = bool(np.linalg.norm(rel[:2]) < 0.006 and 0.015 < rel[2] < 0.035)
        rows.append({"offset_mm": round(rr * 1000, 1),
                     "peg_rel_z_mm": round(float(rel[2]) * 1000, 1),
                     "peg_tilt_deg": round(tilt_deg(m, d, pg), 1),
                     "peak_wrench_n": res["peak_wrench_n"],
                     "wall_s": round(time.perf_counter() - t0, 1),
                     "aborted": bool(res["aborted"]), "in_pocket": in_pocket,
                     "success": bool(in_pocket and not res["aborted"])})
    return rows


TASKS = {"reach": task_reach, "carry": task_carry, "insert": task_insert,
         "insert_m2": task_insert_m2}


def run(model_dir, tasks, trials, seed):
    """Run the named tasks and return a {task: {trials, summary}} report."""
    m = load_cell(model_dir)
    rng = np.random.default_rng(seed)
    report = {}
    for name in tasks:
        if name not in TASKS:
            sys.exit(f"unknown task: {name} (choose from {', '.join(TASKS)})")
        rows = TASKS[name](m, trials, rng)
        succ = sum(r["success"] for r in rows)
        num_keys = [k for k, v in rows[0].items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
        summary = {k: summarize([r[k] for r in rows]) for k in num_keys}
        summary["success_rate"] = f"{succ}/{len(rows)}"
        report[name] = {"trials": rows, "summary": summary}
    return report


def main():
    ap = argparse.ArgumentParser(description="SkateArm bimanual benchmark suite")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--tasks", default="reach,carry,insert,insert_m2",
                    help=f"comma-separated; choose from {', '.join(TASKS)}")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    report = run(args.model, tasks, args.trials, args.seed)
    for name in tasks:
        print(f"\n=== {name} ({args.trials} trials) ===")
        for r in report[name]["trials"]:
            print("  ", r)
        print("  SUMMARY:", report[name]["summary"])
    if args.json:
        import json
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
