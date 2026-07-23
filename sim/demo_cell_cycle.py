#!/usr/bin/env python3
"""Phase 1: the FULL AUTOMATIC CYCLE under the GRAFCET sequencer, rendered
with an HMI-style overlay (current step + live metrics).

The sequencer (sequencer.py) runs the demonstrator cycle S0..S7 with
sensor-based receptivities (no timers): parts check -> approach + grasp ->
orientation-locked carry -> align -> force-guarded insert -> QC verify ->
place to ACCEPT/REJECT bin -> retreat. Every transition is logged; the log
(see logs/cycle_001.json) is the seed of the SCADA dashboard.

Measured reference cycle: 42.4 s — inside the spec's 60 s takt target.

Usage:
    python make_control_model.py /path/to/skate_teleop/skt_v3
    python make_collision_model.py /path/to/skate_teleop/skt_v3
    python make_cell_scene.py /path/to/skate_teleop/skt_v3
    python demo_cell_cycle.py --model /path/to/skate_teleop/skt_v3 \
        --out cycle.mp4 --log cycle.json
"""
import argparse
import json
import os
import sys

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequencer import Cell, run_cycle  # noqa: E402

STEP_TITLES = {
    "S0": "S0  HOME / PARTS CHECK", "S1": "S1  APPROACH + GRASP",
    "S2": "S2  CARRY TO FIXTURE", "S3": "S3  ALIGN PEG / POCKET",
    "S4": "S4  INSERT (FORCE-REGULATED)", "S5": "S5  QC VERIFY",
    "S6": "S6  PLACE TO BIN", "S7": "S7  CYCLE COMPLETE",
}
TOTAL_S = 43.0


def load_fonts():
    base = "/usr/share/fonts/truetype/dejavu/"
    try:
        return (ImageFont.truetype(base + "DejaVuSerif-Bold.ttf", 20),
                ImageFont.truetype(base + "DejaVuSansMono.ttf", 14))
    except Exception:
        f = ImageFont.load_default()
        return f, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--out", default="cell_cycle_demo.mp4", help=".mp4 or .gif")
    ap.add_argument("--log", default="cycle_log.json")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--size", default="960x720", help="WxH")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))

    m = mujoco.MjModel.from_xml_path(os.path.join(args.model, "skt_v3_cell.xml"))
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)

    FONT, FONT2 = load_fonts()
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0.30, 0.15]
    cam.distance = 1.85
    cam.elevation = -18

    if args.out.lower().endswith(".mp4"):
        writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                                    quality=7, pixelformat="yuv420p")
    else:
        writer = imageio.get_writer(args.out, fps=args.fps, loop=0)

    state = {"n": 0, "t": 0.0}
    holder = {}

    def metric_line(cell):
        step = cell.log[-1]["step"] if cell.log else "S0"
        if step == "S3":
            return f"align err: {cell.align_err_xy()*1000:5.1f} mm"
        if step == "S4":
            return (f"depth: {max(0, cell.insertion_depth())*1000:5.1f} / 18.0 mm"
                    f"   [force-regulated]")
        if step == "S5":
            for e in reversed(cell.log):
                if e["step"] == "S5" and "result" in e:
                    return (f"depth {e['depth_mm']:.1f} mm  tilt {e['tilt_deg']:.1f} deg"
                            f"  ->  {e['result']}")
            return "measuring..."
        if step in ("S6", "S7"):
            return f"QC: {'ACCEPT' if getattr(holder['cell'], 'qc_pass', True) else 'REJECT'}"
        return f"t = {state['t']:.1f} s"

    def on_frame(_=None):
        state["n"] += 1
        if state["n"] % 5:
            return
        state["t"] += 5 * 0.008
        s = state["t"] / TOTAL_S
        cam.azimuth = 235 + 50 * (0.5 - 0.5 * np.cos(np.pi * min(s, 1.0)))
        r.update_scene(d, camera=cam)
        im = Image.fromarray(r.render())
        dr = ImageDraw.Draw(im, "RGBA")
        cell = holder["cell"]
        step = cell.log[-1]["step"] if cell.log else "S0"
        sc = w / 640.0
        dr.rectangle([8 * sc, 8 * sc, 412 * sc, 66 * sc], fill=(10, 14, 24, 190))
        dr.text((18 * sc, 12 * sc), STEP_TITLES.get(step, step), font=FONT,
                fill=(120, 220, 255))
        dr.text((18 * sc, 40 * sc), metric_line(cell), font=FONT2, fill=(230, 230, 230))
        dr.text((18 * sc, h - 22 * sc), "SkateArm  |  GRAFCET cycle  |  sim",
                font=FONT2, fill=(160, 160, 160))
        writer.append_data(np.asarray(im))

    cell = Cell(m, d, on_frame=on_frame)
    cell.t0 = d.time
    holder["cell"] = cell
    run_cycle(cell)
    writer.close()
    json.dump(cell.log, open(args.log, "w"), indent=1)
    print(f"saved {args.out} and {args.log}")
    print("cycle time: %.1f s (takt target 60 s)" % cell.log[-1]["cycle_time_s"])


if __name__ == "__main__":
    main()
