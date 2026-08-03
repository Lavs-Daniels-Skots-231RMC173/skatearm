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
rows above the block's top edge, the pixels ``depth_mm_est`` is derived from --
and NOT the strict ``_yellow`` mask the top view uses. That band was a flat row
count once; it is derived from the peg's own length now, so the artefact carries
the derivation and the height it lands on rather than a number to be trusted.

WHAT THE OCCLUSION COST, AND WHAT PAID IT BACK. The blindness above is a finding
about the cell, not a verdict on it: those two cameras are aimed at a fixture
pose the weld-free cycle has no reason to visit, and the repair named in the
README was in-situ optics -- "a second camera pair, not a re-calibration". The
weld-free path has them now. Its S5 reads ``qc.STATION_PAIR`` at the assembly
station in the instant BOTH hands are off the part, which is an instant only a
cell with no welds left in it can have, and THAT reading is what gates the
cycle. So each column below reports up to three things: the FIXTURE reading,
unchanged, still the occlusion measurement and still the reason a second pair
exists; the STATION reading, on the weld-free path only (``null`` on the weld
path, where the unit is never out of a hand to be looked at); and which of the
two the cell's verdict actually came from.

    MUJOCO_GL=egl python eval_qc_occlusion.py --model /path/to/skt_v3
                                              [--json sim/eval_data/qc_occlusion.json]

Up to three cameras are rendered, so this needs a GL backend and the opt-in
gripper scene (``sim/make_cell_scene.py --gripper``). The artefact it writes
needs neither: ``sim/test_manipulation_numbers.py`` pins the published figures
to the JSON alone and stays in the hardware-free CI job.

On ``cycle_time_s``: S6 carries an ACCEPT to the near bin and a REJECT across
the whole robot to the far one, so the cycle time is a readout of the gate's
verdict. Ungated, the weld-free column took the reject detour and ran long.
Gated at the station it lands on the same 75.84 s takt
``sim/test_cell_gripper.py`` measures with no renderer attached at all, off the
oracle -- two independent gates, one cycle, one number to three decimals. That
takt was the last figure in these docs with no committed artefact behind it.
"""
import argparse
import inspect
import json
import os
import re
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequencer import Cell, run_cycle, STATION, TABLE_Z   # noqa: E402
import make_cell_scene as mcs           # noqa: E402
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


def pairs():
    """Both camera pairs exactly as qc.py declares them, field names included --
    taken off the namedtuple rather than transcribed, so a pair that grows a
    field turns up in the artefact instead of being quietly dropped from it."""
    return {p.name: {k: (float(v) if isinstance(v, float) else v)
                     for k, v in p._asdict().items() if k != "name"}
            for p in (qc_mod.FIXTURE_PAIR, qc_mod.STATION_PAIR)}


def station_camera():
    """The station camera as it is really written into the scene, parsed back out
    of ``make_cell_scene.SCENE_HEAD`` -- plus the three constants that aim it,
    checked against the cell they claim to describe.

    make_cell_scene.py must not import sequencer.py: the sequencer needs MuJoCo
    and the builder is imported by the hardware-free CI job, so the station's
    position, the block's top height and the camera standoff are restated there.
    Restated is fine. Unchecked is not, and this is the one place in the tree
    where checking is free -- MuJoCo is loaded, ``sequencer.STATION`` and
    ``TABLE_Z`` are imported, ``qc.SIDE_CAM_X`` is imported. The parsed camera
    line then goes into the artefact, so the hardware-free guard can parse
    SCENE_HEAD for itself and compare against it without importing either the
    sequencer or a model. Drift in either file fails CI on one side or the other.
    """
    assert tuple(round(float(v), 6) for v in STATION) == mcs.STATION_XY, \
        (tuple(STATION), mcs.STATION_XY)
    assert mcs.STATION_CAM_STANDOFF == qc_mod.SIDE_CAM_X, \
        (mcs.STATION_CAM_STANDOFF, qc_mod.SIDE_CAM_X)
    assert mcs.STATION_BLK_TOP > TABLE_Z, (mcs.STATION_BLK_TOP, TABLE_Z)

    line = re.search(r'<camera name="%s".*?/>' % qc_mod.STATION_PAIR.side,
                     mcs.SCENE_HEAD).group(0)
    pos = [float(v) for v in re.search(r'pos="([^"]+)"', line).group(1).split()]
    # the camera is on the station's own x, one standoff back along -y, at the
    # height of the block's top face -- restated as a claim, checked as arithmetic
    assert pos[0] == mcs.STATION_XY[0], (pos, mcs.STATION_XY)
    assert round(mcs.STATION_XY[1] - pos[1], 6) == mcs.STATION_CAM_STANDOFF
    assert pos[2] == mcs.STATION_BLK_TOP, (pos, mcs.STATION_BLK_TOP)
    return {
        "scene_head_line": line,
        "pos_m": pos,
        "fovy_deg": float(re.search(r'fovy="([^"]+)"', line).group(1)),
        "aimed_at": {"station_xy_m": list(mcs.STATION_XY),
                     "blk_top_z_m": mcs.STATION_BLK_TOP,
                     "standoff_m": mcs.STATION_CAM_STANDOFF},
        "sequencer": {"station_xy_m": [round(float(v), 6) for v in STATION],
                      "table_z_m": float(TABLE_Z),
                      "base_height_m": round(mcs.STATION_BLK_TOP - TABLE_Z, 6)},
    }


def blk_top_z(m, unit_pose):
    """Where the unit's top face really is, given the body pose the cell logged:
    the base body's z plus the highest point of any geom hanging off it, read
    from the compiled model. ``make_cell_scene.STATION_BLK_TOP`` is what this is
    supposed to be; the eval reports the difference in mm rather than asserting
    a tolerance, because the honest bound is "the camera is aimed at the top of
    the block" and a reader can see for themselves how close that is."""
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_part")
    tops = [float(m.geom_pos[g][2] + m.geom_size[g][2])
            for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    return float(unit_pose[2]) + max(tops)


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
    station = _station_reading(m, cell)
    run_cycle(cell, steps=AFTER_QC)

    print(f"      qc_top   peg  {px['top_peg']:6d} px   rim {px['top_blk']:6d} px")
    print(f"      qc_side  stub {px['side_peg']:6d} px   rim {px['side_blk']:6d} px")
    print(f"      camera   peg_present {ev['cam_peg_present']}  align {ev['cam_align_mm']}"
          f"  depth {ev['cam_depth_mm']}  -> {ev['cam_result']}  [fixture pair]")
    print(f"      oracle   depth {ev['oracle_depth_mm']:.4f}  align {ev['oracle_align_mm']:.4f}"
          f"  tilt {ev['oracle_tilt_deg']:.4f}")
    if station is None:
        print("      station  none -- the unit is never out of a hand on this path")
    else:
        s, spx = station["camera"], station["px"]
        print(f"      station  peg {spx['top_peg']:6d} px   rim {spx['top_rim']:6d} px   "
              f"stub {spx['side_stub_band']:6d} px")
        print(f"               peg_present {s['peg_present']}  align {s['align_err_mm']}"
              f"  depth {s['depth_mm_est']}  -> {s['verdict']}  [station pair]")
    print(f"      GATE     {ev['gate']} -> {ev['gate_result']}   qc_pass {cell.qc_pass}")

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
        "stub_band_px": int(meas["_band_px"]),
        "pair": meas["_pair"].name,
        "unit_at_qc_m": unit,
        # the in-situ reading, or None and the reason there cannot be one
        "station": station,
        "station_none_because": None if station is not None else
        "the weld path never has the unit out of a hand: it leaves one grip only "
        "onto a weld, so no instant exists in which to inspect it in place",
        # which reading the cell's own verdict came from, and what that verdict was
        "gate": {"pair": ev["gate"], "verdict": ev["gate_result"],
                 "qc_pass": bool(cell.qc_pass)},
        "cycle_time_s": round(cell.log[-1]["cycle_time_s"], 3),
    }


def _station_reading(m, cell):
    """S5's in-situ measurement, if this cell took one.

    Same shape as the fixture column beside it, deliberately: both came out of
    the same ``qc.measure()`` with the same masks and the same thresholds, and
    the artefact should make that comparable at a glance rather than make the
    reader take it on the prose. ``blk_top_aim_err_mm`` is the one extra field --
    how far the camera's aim height sits from where the unit's top face actually
    ended up, measured off the compiled model, since that constant is the one
    thing about this pair that had to be restated instead of imported."""
    st = getattr(cell, "qc_station", None)
    if st is None:
        return None
    ev = [e for e in cell.log if e["msg"].startswith("STATION QC verify")][-1]
    px = {k: int(v.sum()) for k, v in st["_masks"].items()}
    at = ev["unit_at_station_m"]
    return {
        "pair": st["_pair"].name,
        "when": ev["msg"],
        "px": {"top_peg": px["top_peg"], "top_rim": px["top_blk"],
               "side_stub_band": px["side_peg"], "side_rim": px["side_blk"]},
        "camera": {"peg_present": bool(ev["cam_peg_present"]),
                   "align_err_mm": ev["cam_align_mm"],
                   "depth_mm_est": ev["cam_depth_mm"],
                   "verdict": ev["cam_result"]},
        "oracle": {"depth_mm": round(ev["oracle_depth_mm"], 4),
                   "align_mm": round(ev["oracle_align_mm"], 4),
                   "tilt_deg": round(ev["oracle_tilt_deg"], 4)},
        "mm_per_px": {k: round(float(v), 5) for k, v in st["_mpp"].items()},
        "stub_band_px": int(st["_band_px"]),
        "unit_at_station_m": at,
        "blk_top_aim_err_mm": round(
            (mcs.STATION_BLK_TOP - blk_top_z(m, at)) * 1000.0, 3),
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

    # which path can have an in-situ reading at all is a property of the cell, not
    # of this script, so state it as one rather than let a None slip through quiet
    assert j["station"] is not None, "the weld-free path took no station reading"
    assert w["station"] is None, "the weld path cannot have an unhanded instant"
    # one lens at one standoff on every lateral view here, so one derived band
    band_px = {w["stub_band_px"], j["stub_band_px"], j["station"]["stub_band_px"]}
    assert len(band_px) == 1, band_px

    out = {
        "eval": "qc_occlusion", "milestone": "M4",
        "source": "sim/eval_qc_occlusion.py",
        "qc": {"render_px": [QC_W, QC_H], "roi_px": roi_px(),
               "accept_thresholds": thresholds(),
               # the two pairs, and the scene geometry that aims the second one
               "pairs": pairs(),
               "station_camera": station_camera(),
               "stub_band": {
                   "peg_len_mm": float(qc_mod.PEG_LEN_MM),
                   "px": band_px.pop(),
                   "derivation": "ceil(peg_len_mm / mm_per_px.side) — qc.stub_band_px(); "
                                 "an exposed stub cannot be taller than the peg. It was "
                                 "a flat 70 px, which capped depth_mm_est at 16.58 mm "
                                 "and put a dead band above the 15.0 mm reject line",
               },
               "masks": {
                   "top_peg": "qc._yellow(top) & ROI — strict lit peg-top yellow",
                   "top_rim": "qc._cyan(top) & ROI — pocket block, incl. the rim ring",
                   "side_stub_band": "qc._peg_colors(side) & stub band — broad "
                                     "yellow|orange in the stub_band.px rows above the "
                                     "block's top edge, the pixels depth_mm_est comes from",
                   "side_rim": "qc._cyan(side) & ROI",
               },
               "measured_by": "sequencer S5 — qc.measure() masks, not a re-render"},
        "paths": paths,
        "summary": {
            "top_peg_px": {"weld": w["px"]["top_peg"], "jaws": j["px"]["top_peg"]},
            "top_rim_px": {"weld": w["px"]["top_rim"], "jaws": j["px"]["top_rim"]},
            "top_rim_loss_pct": round(
                100.0 * (1.0 - j["px"]["top_rim"] / w["px"]["top_rim"]), 1),
            # the FIXTURE pair's verdicts -- the occlusion finding, unchanged
            "verdict": {"weld": w["camera"]["verdict"], "jaws": j["camera"]["verdict"]},
            # and what the cell actually decided on, once the weld-free path had a
            # pair it could see the part with
            "gate": {"weld": w["gate"]["pair"], "jaws": j["gate"]["pair"]},
            "gate_verdict": {"weld": w["gate"]["verdict"], "jaws": j["gate"]["verdict"]},
            "station_verdict": {"weld": None,
                                "jaws": j["station"]["camera"]["verdict"]},
            # S6 carries an ACCEPT to the near bin and a REJECT to the far one, so
            # the takt is downstream of the gate and belongs beside it
            "cycle_time_s": {"weld": w["cycle_time_s"], "jaws": j["cycle_time_s"]},
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
