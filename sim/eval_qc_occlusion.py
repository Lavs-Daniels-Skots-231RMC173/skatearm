"""M4 eval — what converting the last weld to jaws cost the camera QC gate,
counted in the pixels the QC pipeline actually analyses.

M4 replaced both welds with force-servoed jaws. The left tool's approach is then
FORCED top-down (the pocket faces up, and the base's 60 mm length exceeds the
41.61 mm tip gap), which parks the left wrist between ``qc_top`` and the unit at
exactly the pose the cameras were calibrated around. This measures that instead
of asserting it, on the two cells side by side:

  * WELD — ``skt_v3_cell.xml``, the default cell every other demo, log and
    benchmark uses;
  * JAWS — ``skt_v3_cell_gripper.xml``, the weld-free cell.

Both run the SAME ``sequencer.run_cycle``, the same ``sim/qc.py`` with its masks
and its 300 px inspection ROI untouched, and a renderer of the same size — so a
difference between the two columns is the CELL and not the measurement. The
counts come off the mask arrays ``qc.measure()`` handed back to S5, i.e. the
frame the cell's own verdict was taken on, not a re-render of it afterwards.
The unit's world pose at that instant is recorded for both paths, because "same
probe, different cell" is only an honest claim if the subject was in the same
place. It is not in EXACTLY the same place -- two different cells settle the
part slightly differently -- so the offset is reported in mm and, more usefully,
in top-camera pixels: the reader can check for themselves that it is a small
fraction of the inspection window and cannot account for the peg loss.

The four mask counts are named for the array each comes from, because they are
not the same kind of mask (see ``qc.masks`` in the artefact). In particular the
side-view count is qc.py's exposed-stub band -- broad yellow-or-orange in the
70 rows above the block's top edge, the pixels ``depth_mm_est`` is derived from
-- and NOT the strict ``_yellow`` mask the top view uses.

    MUJOCO_GL=egl python eval_qc_occlusion.py --model /path/to/skt_v3
                                              [--json sim/eval_data/qc_occlusion.json]

Two cameras are rendered, so this needs a GL backend and the opt-in gripper
scene (``sim/make_cell_scene.py --gripper``). The artefact it writes needs
neither: ``sim/test_manipulation_numbers.py`` pins the published figures to the
JSON alone and stays in the hardware-free CI job.

On ``cycle_time_s``: a camera REJECT routes S6 to the reject bin, which the left
arm has to reach for across the whole robot, so the jaws column here runs LONGER
than the 75.8 s takt quoted elsewhere. That figure is the oracle-gated cycle
``sim/test_cell_gripper.py`` runs — ACCEPT, near bin, no renderer attached. Same
cycle, different S6 branch, and which branch it takes is precisely what this
eval measures.
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequencer import Cell, run_cycle   # noqa: E402
import qc as qc_mod                     # noqa: E402

# The resolution qc.py's mm-per-px was calibrated at; at 720 rows the centred
# 300 px ROI is the z 0.084 .. 0.176 inspection window sequencer.py's MEET_BASE
# note quotes.
QC_W, QC_H = 960, 720
SETTLE = 500
SCENES = (("weld", "skt_v3_cell.xml"), ("jaws", "skt_v3_cell_gripper.xml"))
UPTO_QC = ["S0", "S1", "S2", "S3", "S4", "S5"]
AFTER_QC = ["S6", "S7"]


def roi_px():
    """The inspection window ``qc.measure()`` really analyses, read off its own
    signature — this eval cannot claim a 300 px window while qc.py uses another."""
    return 2 * inspect.signature(qc_mod.measure).parameters["roi_half"].default


def thresholds():
    """``qc.verdict``'s own accept limits, likewise read off the signature rather
    than restated. The guard test runs in the hardware-free CI job and cannot
    import qc.py (it needs MuJoCo), so recording them here is what lets that test
    re-derive a verdict instead of trusting the ``verdict`` field — and lets it
    notice if the gate is ever quietly widened."""
    p = inspect.signature(qc_mod.verdict).parameters
    return {k: float(p[k].default) for k in ("depth_min", "align_max", "tilt_max")}


def run_path(model_dir, xml, tag):
    """One cell, one full cycle, the QC frame its own S5 was judged on."""
    path = os.path.join(model_dir, xml)
    if not os.path.exists(path):
        raise SystemExit(f"no {xml} at {model_dir} "
                         f"(run sim/make_cell_scene.py {model_dir} --gripper first)")
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    for _ in range(SETTLE):
        mujoco.mj_step(m, d)

    cell = Cell(m, d, qc_renderer=mujoco.Renderer(m, QC_H, QC_W))
    cell.t0 = d.time
    print(f"\n{tag.upper():<5} {xml}   jaws right={cell.jaws} left={cell.jawsL}")

    # Stop at the camera verdict so the unit can be located while it is still
    # standing where the cameras saw it; S6 carries it to a bin.
    run_cycle(cell, steps=UPTO_QC)
    meas = getattr(cell, "qc_meas", None)
    if meas is None:
        raise SystemExit(f"{tag}: S5 took no camera measurement")
    ev = [e for e in cell.log if e["msg"] == "CAMERA QC verify"][-1]
    px = {k: int(v.sum()) for k, v in meas["_masks"].items()}
    unit = [round(float(v), 5) for v in cell.part_pose("base")]
    run_cycle(cell, steps=AFTER_QC)

    print(f"      qc_top   peg  {px['top_peg']:6d} px   rim {px['top_blk']:6d} px")
    print(f"      qc_side  stub {px['side_peg']:6d} px   rim {px['side_blk']:6d} px")
    print(f"      camera   peg_present {ev['cam_peg_present']}  align {ev['cam_align_mm']}"
          f"  depth {ev['cam_depth_mm']}  -> {ev['cam_result']}")
    print(f"      oracle   depth {ev['oracle_depth_mm']:.4f}  align {ev['oracle_align_mm']:.4f}"
          f"  tilt {ev['oracle_tilt_deg']:.4f}")

    return {
        "scene": xml,
        "jaws_right": bool(cell.jaws),
        "jaws_left": bool(cell.jawsL),
        "px": {"top_peg": px["top_peg"], "top_rim": px["top_blk"],
               "side_stub_band": px["side_peg"], "side_rim": px["side_blk"]},
        "camera": {"peg_present": bool(ev["cam_peg_present"]),
                   "align_err_mm": ev["cam_align_mm"],
                   "depth_mm_est": ev["cam_depth_mm"],
                   "verdict": ev["cam_result"]},
        "oracle": {"depth_mm": round(ev["oracle_depth_mm"], 4),
                   "align_mm": round(ev["oracle_align_mm"], 4),
                   "tilt_deg": round(ev["oracle_tilt_deg"], 4)},
        "mm_per_px": {k: round(float(v), 5) for k, v in meas["_mpp"].items()},
        "unit_at_qc_m": unit,
        "cycle_time_s": round(cell.log[-1]["cycle_time_s"], 3),
    }


def main():
    ap = argparse.ArgumentParser(
        description="M4 eval: the QC inspection window, weld cell vs weld-free cell")
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="also write the results as a JSON artefact")
    args = ap.parse_args()

    print(f"QC inspection window at the verify pose — {QC_W}x{QC_H}, "
          f"{roi_px()} px centred ROI, qc.py masks unchanged")
    paths = {tag: run_path(args.model, xml, tag) for tag, xml in SCENES}
    w, j = paths["weld"], paths["jaws"]
    dxyz = (np.array(j["unit_at_qc_m"]) - np.array(w["unit_at_qc_m"])) * 1000.0
    delta_mm = float(np.linalg.norm(dxyz))

    out = {
        "eval": "qc_occlusion", "milestone": "M4",
        "source": "sim/eval_qc_occlusion.py",
        "qc": {"render_px": [QC_W, QC_H], "roi_px": roi_px(),
               "accept_thresholds": thresholds(),
               "masks": {
                   "top_peg": "qc._yellow(top) & ROI — strict lit peg-top yellow",
                   "top_rim": "qc._cyan(top) & ROI — pocket block, incl. the rim ring",
                   "side_stub_band": "qc._peg_colors(side) & stub band — broad "
                                     "yellow|orange in the 70 rows above the block's "
                                     "top edge, the pixels depth_mm_est comes from",
                   "side_rim": "qc._cyan(side) & ROI",
               },
               "measured_by": "sequencer S5 — qc.measure() masks, not a re-render"},
        "paths": paths,
        "summary": {
            "top_peg_px": {"weld": w["px"]["top_peg"], "jaws": j["px"]["top_peg"]},
            "top_rim_px": {"weld": w["px"]["top_rim"], "jaws": j["px"]["top_rim"]},
            "top_rim_loss_pct": round(
                100.0 * (1.0 - j["px"]["top_rim"] / w["px"]["top_rim"]), 1),
            "verdict": {"weld": w["camera"]["verdict"], "jaws": j["camera"]["verdict"]},
            "unit_pose_delta_mm": round(float(delta_mm), 2),
            "unit_pose_delta_xyz_mm": [round(float(v), 2) for v in dxyz],
            # The same offset in the units the claim is made in: the subject moved
            # this far inside a roi_px window, while the peg count went to zero.
            "unit_pose_delta_top_px": round(delta_mm / w["mm_per_px"]["top"], 1),
        },
    }

    s = out["summary"]
    print(f"\nqc_top peg px  {s['top_peg_px']['weld']} -> {s['top_peg_px']['jaws']}   "
          f"rim px {s['top_rim_px']['weld']} -> {s['top_rim_px']['jaws']} "
          f"({s['top_rim_loss_pct']} % of the rim gone)")
    print(f"verdict        {s['verdict']['weld']} -> {s['verdict']['jaws']}, on a unit "
          f"presented {s['unit_pose_delta_mm']} mm apart between the two cells "
          f"({s['unit_pose_delta_top_px']} px of a {roi_px()} px window)")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
