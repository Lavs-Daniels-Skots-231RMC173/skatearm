"""Generate skt_v3_cell.xml: the SkateArm work-cell scene on top of the
collision model — work table, base part (60x40x25 mm), peg (D20x40),
accept/reject bins.

Parts are free bodies with real masses (PETG-ish): base ~45 g, peg ~12 g.
The base part's bore is either the v1 SQUARE blind pocket (22x22 mm, default) or,
with --round-bore, the spec's ROUND chamfered bore: a faceted-cylinder wall ring
(bore radius ~10.4 mm -> ~0.4 mm clearance on the D20 peg, H9-ish) with a wider
lead-in mouth. Same peg either way; the round bore is OPT-IN so the default (and
CI) stay on the square v1 stand-in.

    python make_cell_scene.py /path/to/skate_teleop/skt_v3 [--round-bore]
"""
import math
import os
import sys

TABLE = {"pos": (0, 0.50, 0.0), "half": (0.45, 0.12, 0.03)}  # top z=0.03, front edge y=0.38

SCENE_HEAD = """
    <geom name="floor" type="plane" pos="0 0 -1.05" size="4 4 0.1" material="grid"/>
    <light pos="1.5 1.5 2" dir="-0.4 -0.4 -1" diffuse="0.6 0.6 0.6"/>
    <light pos="-1 2 1.5" dir="0.3 -0.6 -1" diffuse="0.3 0.3 0.3"/>
    <camera name="qc_top" pos="0 0.41 0.60" zaxis="0 0 1" fovy="42"/>
    <camera name="qc_side" pos="0.32 0.41 0.13" xyaxes="0 1 0 0 0 1" fovy="38"/>
    <geom name="table" type="box" pos="0 0.50 0" size="0.45 0.12 0.03" rgba="0.55 0.42 0.28 1" friction="0.8 0.005 0.0001"/>
    <geom name="bin_accept" type="box" pos="-0.24 0.41 0.035" size="0.05 0.05 0.005" rgba="0.2 0.7 0.3 1"/>
    <geom name="bin_reject" type="box" pos="0.24 0.41 0.035" size="0.05 0.05 0.005" rgba="0.8 0.25 0.2 1"/>
"""

PEG = """
    <body name="peg" pos="0.12 0.44 0.0501">
      <freejoint/>
      <geom name="peg_body" type="cylinder" size="0.010 0.020" rgba="0.9 0.6 0.15 1" density="950" friction="0.9 0.005 0.0001"/>
    </body>
"""

SQUARE_BASE = """    <body name="base_part" pos="-0.12 0.44 0.0301">
      <freejoint/>
      <geom type="box" pos="0 0 0.0025" size="0.030 0.020 0.0025" rgba="0.2 0.55 0.65 1" density="900"/>
      <geom type="box" pos="0 -0.0155 0.015" size="0.030 0.0045 0.010" rgba="0.2 0.55 0.65 1" density="900"/>
      <geom type="box" pos="0 0.0155 0.015" size="0.030 0.0045 0.010" rgba="0.2 0.55 0.65 1" density="900"/>
      <geom type="box" pos="-0.0205 0 0.015" size="0.0095 0.011 0.010" rgba="0.2 0.55 0.65 1" density="900"/>
      <geom type="box" pos="0.0205 0 0.015" size="0.0095 0.011 0.010" rgba="0.2 0.55 0.65 1" density="900"/>
    </body>
"""


def round_base(r_bore=0.0104, depth=0.020, wall_t=0.005, n=20, mouth=0.0018,
               mouth_h=0.004, dens=900, rgba="0.2 0.55 0.65 1"):
    """base_part with a ROUND chamfered bore, approximated by `n` radial wall
    boxes (a faceted cylinder of inner radius `r_bore`, depth `depth`) on a base
    plate, plus a wider ring at the top (`mouth`, `mouth_h`) as a lead-in mouth."""
    G = ['<geom type="box" pos="0 0 0.0025" size="0.030 0.020 0.0025" '
         'rgba="%s" density="%d"/>' % (rgba, dens)]
    h1 = depth - mouth_h
    for k in range(n):
        a = 2 * math.pi * k / n
        deg, cx, cy = math.degrees(a), math.cos(a), math.sin(a)
        hw = r_bore * math.tan(math.pi / n) + 0.0009           # tangential half-width (overlap)
        px, py = (r_bore + wall_t / 2) * cx, (r_bore + wall_t / 2) * cy
        G.append('<geom type="box" pos="%.5f %.5f %.5f" size="%.5f %.5f %.5f" '
                 'euler="0 0 %.3f" rgba="%s" density="%d"/>'
                 % (px, py, 0.005 + h1 / 2, wall_t / 2, hw, h1 / 2, deg, rgba, dens))
        hwm = (r_bore + mouth) * math.tan(math.pi / n) + 0.0009
        pmx, pmy = (r_bore + mouth + wall_t / 2) * cx, (r_bore + mouth + wall_t / 2) * cy
        G.append('<geom type="box" pos="%.5f %.5f %.5f" size="%.5f %.5f %.5f" '
                 'euler="0 0 %.3f" rgba="%s" density="%d"/>'
                 % (pmx, pmy, 0.005 + depth - mouth_h / 2, wall_t / 2, hwm,
                    mouth_h / 2, deg, rgba, dens))
    return ('    <body name="base_part" pos="-0.12 0.44 0.0301">\n      <freejoint/>\n      '
            + "\n      ".join(G) + "\n    </body>\n")


# v1 grasp stand-in: weld constraints, inactive until grasp() engages them at
# runtime. fixture_base: an inactive base<->world weld — the M2 insertion eval
# clamps the base into a RIGID fixture so the misalignment sweep is deterministic.
EQUALITY = """
  <equality>
    <weld name="grasp_left" body1="wrist_a3_1" body2="base_part" active="false" solref="0.005 1"/>
    <weld name="grasp_right" body1="wrist_a3_Mirror__1" body2="peg" active="false" solref="0.005 1"/>
    <weld name="fixture_base" body1="base_part" active="false" solref="0.002 1"/>
  </equality>
"""


def make(model_dir, round_bore=False, out=None):
    src = os.path.join(model_dir, "skt_v3_collision.xml")
    if not os.path.exists(src):
        sys.exit("run make_collision_model.py first")
    xml = open(src).read()
    xml = xml.replace(">", """>
  <visual><global offwidth="1280" offheight="960"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.95" rgb2="0.82 0.84 0.88" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/></asset>""", 1)
    base = round_base() if round_bore else SQUARE_BASE
    xml = xml.replace("<worldbody>", "<worldbody>" + SCENE_HEAD + base + PEG, 1)
    xml = xml.replace("</mujoco>", EQUALITY + "</mujoco>", 1)
    out = os.path.join(model_dir, out or "skt_v3_cell.xml")
    open(out, "w").write(xml)
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(out)
    d = mujoco.MjData(m)
    for _ in range(2000):
        mujoco.mj_step(m, d)
    for name in ("base_part", "peg"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        print(f"{name}: mass {m.body_mass[b]*1000:.0f} g, settled at {d.xpos[b].round(3)}")
    kind = "round chamfered bore" if round_bore else "square v1 pocket"
    print(f"wrote {out} ({kind}); NaN: {np.isnan(d.qpos).any()}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    make(args[0] if args else ".", round_bore="--round-bore" in sys.argv)
