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
        --no-search-baseline --json sim/eval_data/insertion.json

``--json`` writes the same numbers as a committed artefact, so the figures quoted
in ``docs/MANIPULATION.md`` have raw data behind them (``sim/test_manipulation_numbers.py``
pins the prose to that file in CI).
"""
import argparse
import json
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


def _peg_tilt(d, pg):
    """Angle of the peg's long axis (local +z) from world vertical, degrees."""
    z = d.xmat[pg].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))


def run_one_theta(m, d, armR, snap, W0, bp, pg, theta_deg, axis, search=True, **params):
    """Inject an initial peg tilt (~theta_deg about `axis`), then insert with the
    hold orientation kept UPRIGHT (relock=False) so the 6-DoF IK LEVELS the peg
    while it seats. Reports the injected tilt, the final tilt and the oracle seat."""
    restore(m, d, snap)
    armR.lock_orientation()
    q_up = np.array(armR.q_lock, float)
    qa = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qa, np.asarray(axis, float), np.radians(theta_deg))
    q_tilt = np.zeros(4)
    mujoco.mju_mulQuat(q_tilt, qa, q_up)
    armR.q_lock = q_tilt
    ee = armR.ee_pos().copy()
    for _ in range(90):                          # tilt the staged peg to ~theta
        armR.set_ctrl(armR.ik_step6(ee)[0])
        for _ in range(4):
            mujoco.mj_step(m, d)
    tilt0 = _peg_tilt(d, pg)
    armR.q_lock = q_up                           # target upright -> the IK levels it
    res = Insertion(m, d, armR, W0, **params).run(search=search, relock=False)
    seated, rel_xy, depth = score(d, bp, pg)
    res.update(theta_cmd_deg=theta_deg, tilt0_deg=round(tilt0, 1),
               tiltf_deg=round(_peg_tilt(d, pg), 1), seated=seated,
               rel_xy_mm=round(rel_xy * 1000, 1), depth_mm=round(depth * 1000, 1))
    return res


def theta_sweep(model_dir, thetas_deg, dirs=2):
    m = load_cell(model_dir)
    d, armR, W0, bp, pg = stage(m)
    snap = snapshot(d)
    axes = [[np.cos(a), np.sin(a), 0.0]
            for a in [2 * np.pi * k / dirs for k in range(dirs)]]
    print(f"staged. peg-tilt (theta) tolerance sweep, {dirs} tilt-axes/theta:\n")
    rows = []
    for th in thetas_deg:
        ok, t0s, tfs = 0, [], []
        for ax in axes:
            r = run_one_theta(m, d, armR, snap, W0, bp, pg, th, ax)
            ok += int(r["seated"]); t0s.append(r["tilt0_deg"]); tfs.append(r["tiltf_deg"])
            if th == 0:
                break
        n = 1 if th == 0 else dirs
        rows.append({"theta_cmd_deg": th, "tilt_injected_deg": round(max(t0s), 1),
                     "seated": ok, "trials": n,
                     "tilt_final_max_deg": round(max(tfs), 1)})
        print(f"  cmd {th:>2} deg -> injected tilt {max(t0s):4.1f} deg: {ok}/{n} seated, "
              f"levelled to <= {max(tfs):.1f} deg")
    return {"eval": "insertion-theta", "milestone": "M2",
            "source": "sim/eval_insertion.py --theta",
            "tilt_axes_per_theta": dirs, "theta_sweep": rows}


def sweep(model_dir, offsets_mm, dirs, no_search_baseline=False):
    m = load_cell(model_dir)
    d, armR, W0, bp, pg = stage(m)
    snap = snapshot(d)
    angles = [2 * np.pi * k / dirs for k in range(dirs)]
    print(f"staged (W0={W0.round(4)}). misalignment-tolerance sweep, {dirs} dirs/offset:\n")
    modes = [("search", True)] + ([("no-search", False)] if no_search_baseline else [])
    report = {"eval": "insertion", "milestone": "M2",
              "source": "sim/eval_insertion.py", "dirs_per_offset": dirs,
              "offsets_mm": list(offsets_mm), "w0_xy_m": [round(v, 4) for v in W0.tolist()],
              "oracle": {"seat_min_mm": SEAT_MIN * 1000, "xy_max_mm": XY_MAX * 1000},
              "modes": {}}
    for mode, search in modes:
        print(f"--- {mode} ---")
        rows = []
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
            rows.append({"offset_mm": off_mm, "seated": ok, "trials": n,
                         "peak_wrench_max_n": round(float(max(peaks)), 1)})
            print(f"  offset {off_mm:>2} mm: {ok}/{n} seated   "
                  f"(peak wrench max {max(peaks):.1f} N)")
        report["modes"][mode] = rows
        print()
    return report


def main():
    ap = argparse.ArgumentParser(description="M2 misalignment-tolerance sweep")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--offsets", default="0,2,4,6,8", help="xy offsets in mm, comma-sep")
    ap.add_argument("--dirs", type=int, default=6, help="directions per offset")
    ap.add_argument("--no-search-baseline", action="store_true",
                    help="also run with the spiral search OFF")
    ap.add_argument("--theta", default=None,
                    help="run the peg-tilt tolerance sweep instead (degrees, e.g. 0,3,6,9,12)")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the results as a JSON artefact")
    args = ap.parse_args()
    if args.theta is not None:
        thetas = [int(x) for x in args.theta.split(",") if x.strip() != ""]
        out = theta_sweep(args.model, thetas, dirs=max(2, args.dirs // 3))
    else:
        offsets = [int(x) for x in args.offsets.split(",") if x.strip() != ""]
        out = sweep(args.model, offsets, args.dirs, args.no_search_baseline)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
