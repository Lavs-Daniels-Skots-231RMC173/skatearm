"""Generate skt_v3_cell.xml: the SkateArm work-cell scene on top of the
collision model — work table, base part (60x40x25 mm), peg (D20x40),
accept/reject bins.

Parts are free bodies with real masses (PETG-ish): base ~45 g, peg ~12 g.
The base part's bore is either the v1 SQUARE blind pocket (22x22 mm, default) or,
with --round-bore, the spec's ROUND chamfered bore: a faceted-cylinder wall ring
(bore radius ~10.4 mm -> ~0.4 mm clearance on the D20 peg, H9-ish) with a wider
lead-in mouth. Same peg either way; the round bore is OPT-IN so the default (and
CI) stay on the square v1 stand-in.

--gripper writes skt_v3_cell_gripper.xml instead: the same cell with M4's
actuated parallel jaws on BOTH wrists, so the whole cycle grips and releases for
real rather than through the `grasp_right` / `grasp_left` welds. Both equalities
are still emitted — the default model shares this builder and needs them — and
the gripper path simply never engages either. Also OPT-IN: passing --gripper
leaves the default model and every existing test untouched, byte for byte.

THREE fixed QC cameras go into both scenes, for the same reason both equalities
do: one builder. `qc_top` + `qc_side` are the fixture pair the camera pipeline
shipped with — the unit is carried up to them and inspected in mid air.
`qc_station_side` looks at the assembly station on the table instead, which is
the only place the weld-free cell ever has the finished unit in the clear, and
the weld path never reads it. Cameras carry no dynamics, so the default model
still steps identically and every number measured from it still reproduces.

    python make_cell_scene.py /path/to/skate_teleop/skt_v3 [--round-bore] [--gripper]
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

TABLE = {"pos": (0, 0.50, 0.0), "half": (0.45, 0.12, 0.03)}  # top z=0.03, front edge y=0.38

# The ASSEMBLY STATION: where the weld-free cycle sets the base down, inserts,
# and leaves the finished unit standing on the table. These three numbers exist
# only to aim `qc_station_side`, and they are restated here rather than
# imported because sequencer.py needs MuJoCo and this builder deliberately does
# not (the hardware-free CI job imports this module). They are not left on
# trust: sim/eval_qc_occlusion.py asserts them against sequencer.STATION and
# sequencer.TABLE_Z and writes both into the artefact, and the hardware-free
# guard parses the camera back out of SCENE_HEAD and checks it against that
# artefact. A drift is a CI failure, not a surprise on the bench.
STATION_XY = (0.040, 0.412)          # == sequencer.STATION
STATION_BLK_TOP = 0.0549             # == sequencer.TABLE_Z + the base's 25 mm height
STATION_CAM_STANDOFF = 0.320         # == qc.SIDE_CAM_X: same lens, same distance

SCENE_HEAD = """
    <geom name="floor" type="plane" pos="0 0 -1.05" size="4 4 0.1" material="grid"/>
    <light pos="1.5 1.5 2" dir="-0.4 -0.4 -1" diffuse="0.6 0.6 0.6"/>
    <light pos="-1 2 1.5" dir="0.3 -0.6 -1" diffuse="0.3 0.3 0.3"/>
    <camera name="qc_top" pos="0 0.41 0.60" zaxis="0 0 1" fovy="42"/>
    <camera name="qc_side" pos="0.32 0.41 0.13" xyaxes="0 1 0 0 0 1" fovy="38"/>
    <camera name="qc_station_side" pos="%.3f %.3f %.4f" xyaxes="1 0 0 0 0 1" fovy="38"/>
    <geom name="table" type="box" pos="0 0.50 0" size="0.45 0.12 0.03" rgba="0.55 0.42 0.28 1" friction="0.8 0.005 0.0001"/>
    <geom name="bin_accept" type="box" pos="-0.24 0.41 0.035" size="0.05 0.05 0.005" rgba="0.2 0.7 0.3 1"/>
    <geom name="bin_reject" type="box" pos="0.24 0.41 0.035" size="0.05 0.05 0.005" rgba="0.8 0.25 0.2 1"/>
""" % (STATION_XY[0], STATION_XY[1] - STATION_CAM_STANDOFF, STATION_BLK_TOP)

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


# --- OPT-IN: M4's parallel jaws on BOTH wrists (--gripper) -----------------
# MuJoCo's default solref (0.02 s time constant) scales constraint stiffness with
# the constraint's effective mass. On a 6 g jaw that makes both the pad contacts
# and the jaw's own end stop behave like ~4 kN/m springs: a 60 N close buries the
# pads 16 mm past the stop and 8 mm inside the peg, and the force loop ends up
# regulating pad-on-pad instead of pad-on-part. Steel pads and a steel stop need
# saying so explicitly. priority=1 makes the pad's stiffness win the pair, so the
# peg keeps the contact parameters every other cell test already sees.
STIFF = "-100000 -300"
JAW_DAMP = "60"            # geared jaw: ~0.1 m/s of closing speed, not 1.2
_C = math.sqrt(0.5)
_T, _W, _L = 0.003, 0.008, 0.016   # pad plate half thickness / half width / half length


def _vee_jaw(name, jnt, sign, reach, open_x, pad_y, mu, tag=None):
    """One jaw: a slide joint along the wrist's local x carrying a 90 deg
    V-groove pad — two plates whose faces are tangent to the D20 peg, so it
    self-centres on four line contacts instead of squirting out of a flat pinch.
    The groove runs along the approach axis (local y) = the peg's axis.

    Travel is asymmetric on purpose: open far enough to straddle a D20 peg (the
    V's tip gap at rest is 17.4 mm), hard stop at +8 mm, because the V seats a
    D20 peg at 5.86 mm and the two plates would start colliding with each other
    at 8.7 mm.
    """
    tag = tag or name[-1]
    b = ET.Element("body", {"name": name, "pos": f"{sign*open_x} {reach} 0"})
    ET.SubElement(b, "joint", {
        "name": jnt, "type": "slide", "axis": "1 0 0", "damping": JAW_DAMP,
        "range": "-0.012 0.008" if sign < 0 else "-0.008 0.012",
        "solreflimit": STIFF})
    ET.SubElement(b, "site", {"name": f"pad{tag}_s", "type": "box",
                              "size": "0.013 0.018 0.015",
                              "pos": f"{0.005 * -sign:.6f} {pad_y} 0"})
    for k, zs in ((1, +1), (2, -1)):
        # face normal n = (c, 0, -zs*s) for the LEFT jaw (mirrored in x on the
        # right); the box's +x maps to n, so euler_y = +45 deg * zs, negated on
        # the mirror.
        ET.SubElement(b, "geom", {
            "name": f"pad{tag}{k}", "type": "box", "size": f"{_T} {_L} {_W}",
            "pos": f"{(-_T * _C + _W * _C) * -sign:.6f} {pad_y} {zs * (_T + _W) * _C:.6f}",
            "euler": f"0 {zs * math.pi / 4 * -sign:.6f} 0",
            "rgba": "0.2 0.6 0.9 1", "friction": mu,
            "contype": "1", "conaffinity": "1",     # the cell pads grip the cell
            "solref": STIFF, "priority": "1"})
    return b


LEFT_WRIST = "wrist_a3_1"


def _mount_jaws(root, wrist_name, names, act, sensors):
    """Bolt one pair of V-groove jaws onto ``wrist_name``.

    ``names`` is (left_jaw, right_jaw, left_joint, right_joint, left_tag,
    right_tag); ``act`` the motor name; ``sensors`` the (touch, jointpos) pair.
    Both hands are the SAME tool — same reach, same travel, same pads — so this
    is one function called twice rather than two near-copies that can drift.
    """
    from make_gripper_cell import REACH, OPEN, PAD_Y, MU

    jl, jr, njl, njr, tl, tr = names
    wrist = next(b for b in root.iter("body") if b.get("name") == wrist_name)
    wrist.append(_vee_jaw(jl, njl, -1, REACH, OPEN, PAD_Y, MU, tag=tl))
    wrist.append(_vee_jaw(jr, njr, +1, REACH, OPEN, PAD_Y, MU, tag=tr))

    # A real gripper's drive prevents the jaws meeting; the sim needs telling.
    con = root.find("contact")
    if con is None:
        con = ET.SubElement(root, "contact")
    ET.SubElement(con, "exclude", {"body1": jl, "body2": jr})

    ET.SubElement(root.find("actuator"), "motor",
                  {"name": act, "joint": njl, "ctrlrange": "-60 60", "gear": "1"})
    ET.SubElement(root.find("equality"), "joint",
                  {"name": f"couple_{act}", "joint1": njr, "joint2": njl,
                   "polycoef": "0 -1 0 0 0", "solref": STIFF})
    sens = root.find("sensor")
    ET.SubElement(sens, "touch", {"name": sensors[0], "site": f"pad{tr}_s"})
    ET.SubElement(sens, "jointpos", {"name": sensors[1], "joint": njl})


def add_gripper(root):
    """Bolt M4's jaws onto BOTH wrists of an already-built cell scene.

    The cell is weld-free: `grasp_left` and `grasp_right` are both still
    DECLARED — so the very same model file can be driven down the weld path as
    an A/B control — but the gripper path activates neither. Each hand reports
    its grasp from its own pad force sensor.

    Converting the left hand is not symmetry for its own sake. The left tool is
    45 mm of jaw plus 15 mm of pad longer than the bare wrist it replaces, and
    the base part is 60 x 40 x 25 mm with an open pocket on top, so the jaws can
    only take it across the 40 mm faces (60 mm exceeds the 41.61 mm tip gap the
    jaws reach at their hard stop, which leaves 0.80 mm per side on 40 mm) with
    the approach axis pointing straight down. That pins the left tool pose, and
    with it pinned the left wrist and the right hand's pads overlap by 11.4 mm
    at the best grasp available anywhere — the two hands cannot hold the base in
    mid-air together, which is why the sequencer PLACES the base and inserts into
    it standing on the table (see sequencer.run_cycle S2/S3).
    """
    from make_gripper_cell import WRIST

    _mount_jaws(root, WRIST,
                ("jawL", "jawR", "jL", "jR", "L", "R"),
                "grip", ("grip_force", "jaw"))
    _mount_jaws(root, LEFT_WRIST,
                ("jawLl", "jawLr", "jLl", "jLr", "Ll", "Lr"),
                "gripL", ("grip_force_L", "jaw_L"))

    # MuJoCo's default impratio=1 makes the friction constraint as soft as the
    # normal one, so a gripped part creeps along the pads even when the friction
    # cone has an order of magnitude in hand. Raising it is the documented fix
    # for grasping; it stiffens the tangential constraint, it does not add grip.
    # cone="elliptic" is the other half and the one that actually mattered: with
    # the default pyramidal cone the Gauss-Seidel solver let the peg creep
    # 12.23 mm along the pads during the S3 alignment, which no amount of
    # noslip_iterations fixed. Elliptic takes that to 0.18 mm (measured).
    # NOTE: this is a whole-model option, so it changes the weld path's contact
    # behaviour in this model too — see sim/README.md.
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
    opt.set("impratio", "10")
    opt.set("cone", "elliptic")
    return root


def make(model_dir, round_bore=False, gripper=False, out=None):
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
    if gripper:
        xml = ET.tostring(add_gripper(ET.fromstring(xml)), encoding="unicode")
    out = os.path.join(model_dir,
                       out or ("skt_v3_cell_gripper.xml" if gripper else "skt_v3_cell.xml"))
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
    kind += ", V-groove jaws on both wrists" if gripper else ""
    print(f"wrote {out} ({kind}); NaN: {np.isnan(d.qpos).any()}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    make(args[0] if args else ".", round_bore="--round-bore" in sys.argv,
         gripper="--gripper" in sys.argv)
