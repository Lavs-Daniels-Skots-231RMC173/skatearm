#!/usr/bin/env python3
"""Phase 1: the FULL AUTOMATIC CYCLE under the GRAFCET sequencer, rendered
with an HMI-style overlay (current step + live metrics).

The sequencer (sequencer.py) runs the demonstrator cycle S0..S7 with
sensor-based receptivities (no timers): parts check -> approach + grasp ->
orientation-locked carry -> align -> force-guarded insert -> QC verify ->
place to ACCEPT/REJECT bin -> retreat. Every transition is logged; the log
(see logs/cycle_001.json) is the seed of the SCADA dashboard.

Measured reference cycle: 42.6 s — inside the spec's 60 s takt target
(``logs/cycle_001.json``, regenerated with ``--no-render`` after M2's
force-regulated insert replaced the Phase-1 tau watchdog in S4).

``--gripper`` runs the same GRAFCET on the WELD-FREE cell instead
(``skt_v3_cell_gripper.xml``, written by ``make_cell_scene.py --gripper``):
both hands hold their parts with actuated jaws and pad friction, no `weld`
equality anywhere. It is a different cycle, not a re-skin of this one — jaws
cannot hold the base in mid-air while the peg goes in, so the left hand sets
the base down on the table, opens, and re-grips the assembled unit afterwards
to carry it past the QC cameras in their calibrated pose. Measured 75.8 s
against the weld path's 42.6 s; ``sim/test_cell_gripper.py`` itemises where the
extra 33 s goes.

Usage:
    python make_control_model.py /path/to/skate_teleop/skt_v3
    python make_collision_model.py /path/to/skate_teleop/skt_v3
    python make_cell_scene.py /path/to/skate_teleop/skt_v3            # weld cell
    python make_cell_scene.py /path/to/skate_teleop/skt_v3 --gripper  # jaw cell
    python demo_cell_cycle.py --model /path/to/skate_teleop/skt_v3 \
        --out cycle.mp4 --log cycle.json [--gripper]

``--no-render`` runs the same cycle without a GL context and writes only the
log — the way ``logs/cycle_001.json`` is regenerated on a headless machine.
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sequencer import Cell, run_cycle  # noqa: E402

STEP_TITLES = {
    "S0": "S0  HOME / PARTS CHECK", "S1": "S1  APPROACH + GRASP",
    "S2": "S2  CARRY TO FIXTURE", "S3": "S3  ALIGN PEG / POCKET",
    "S4": "S4  INSERT (FORCE-REGULATED)", "S5": "S5  QC VERIFY",
    "S6": "S6  PLACE TO BIN", "S7": "S7  CYCLE COMPLETE",
}
# The jaw cycle does different work in S1/S2/S5 and the overlay should say so:
# S2 is where the base is set DOWN (a weld carries it to a mid-air meet), and
# S5 has to pick the finished unit back up before it can be inspected.
GRIPPER_TITLES = dict(STEP_TITLES, **{
    "S1": "S1  GRASP BOTH (JAWS)",
    "S2": "S2  PLACE BASE / RELEASE",
    "S5": "S5  RE-GRIP + QC VERIFY",
})
TOTAL_S = 43.0            # measured weld cycle 42.6 s
GRIPPER_TOTAL_S = 76.0    # measured jaw cycle 75.8 s


def load_fonts():
    from PIL import ImageFont
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
    ap.add_argument("--no-render", action="store_true",
                    help="run the cycle headless (no GL context) and write only --log")
    ap.add_argument("--gripper", action="store_true",
                    help="run the WELD-FREE cell (skt_v3_cell_gripper.xml): both "
                         "hands hold their parts with jaws and pad friction")
    args = ap.parse_args()
    w, h = (int(x) for x in args.size.lower().split("x"))

    xml = "skt_v3_cell_gripper.xml" if args.gripper else "skt_v3_cell.xml"
    path = os.path.join(args.model, xml)
    if not os.path.exists(path):
        sys.exit(f"{path} not found — run make_cell_scene.py "
                 f"{args.model}{' --gripper' if args.gripper else ''} first")
    titles = GRIPPER_TITLES if args.gripper else STEP_TITLES
    total_s = GRIPPER_TOTAL_S if args.gripper else TOTAL_S
    takt = 85.0 if args.gripper else 60.0

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)

    if args.no_render:                       # same cycle, no renderer, no imageio/PIL
        cell = Cell(m, d)
        cell.t0 = d.time
        run_cycle(cell)
        json.dump(cell.log, open(args.log, "w"), indent=1)
        print(f"saved {args.log}")
        print("cycle time: %.1f s (takt target %.0f s)"
              % (cell.log[-1]["cycle_time_s"], takt))
        return

    import imageio
    from PIL import Image, ImageDraw
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
        s = state["t"] / total_s
        cam.azimuth = 235 + 50 * (0.5 - 0.5 * np.cos(np.pi * min(s, 1.0)))
        r.update_scene(d, camera=cam)
        im = Image.fromarray(r.render())
        dr = ImageDraw.Draw(im, "RGBA")
        cell = holder["cell"]
        step = cell.log[-1]["step"] if cell.log else "S0"
        sc = w / 640.0
        dr.rectangle([8 * sc, 8 * sc, 412 * sc, 66 * sc], fill=(10, 14, 24, 190))
        dr.text((18 * sc, 12 * sc), titles.get(step, step), font=FONT,
                fill=(120, 220, 255))
        dr.text((18 * sc, 40 * sc), metric_line(cell), font=FONT2, fill=(230, 230, 230))
        dr.text((18 * sc, h - 22 * sc),
                "SkateArm  |  GRAFCET cycle  |  "
                + ("weld-free jaws, both hands  |  sim" if args.gripper else "sim"),
                font=FONT2, fill=(160, 160, 160))
        writer.append_data(np.asarray(im))

    cell = Cell(m, d, on_frame=on_frame)
    cell.t0 = d.time
    holder["cell"] = cell
    run_cycle(cell)
    writer.close()
    json.dump(cell.log, open(args.log, "w"), indent=1)
    print(f"saved {args.out} and {args.log}")
    print("cycle time: %.1f s (takt target %.0f s)"
          % (cell.log[-1]["cycle_time_s"], takt))


if __name__ == "__main__":
    main()
