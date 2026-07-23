"""M4 (arm integration) — generate skt_v3_gripcell.xml: the SkateArm collision
model with an actuated parallel-jaw gripper on the RIGHT wrist and a graspable
box part, for a weld-free grasp-and-carry.

OPT-IN and separate from skt_v3_cell.xml, so the default cell model and every
existing test stay untouched (exactly the --round-bore pattern). The jaws are
children of the wrist body, defined in its LOCAL frame: they extend along local
+y (the approach axis — points down at a table-facing pose) and close along local
x. Pads and part share a private contact group so they grip each other but never
the arm. A world-pin holds the part while the jaws close; releasing it leaves the
part held by the gripper's FRICTION alone as the arm carries it.

    python make_gripper_cell.py /path/to/skate_teleop/skt_v3
"""
import os
import sys
import xml.etree.ElementTree as ET

WRIST = "wrist_a3_Mirror__1"          # right wrist
REACH = 0.045                         # jaws extend this far down local +y from the wrist
OPEN = 0.020                          # each jaw's rest offset along local x (±)
PAD_Y = 0.015                         # pad centre beyond the jaw body along local +y
MU = "1.6 0.2 0.02"
PART = (0.010, 0.018, 0.022)          # graspable box half-sizes (x = grip axis)


def _jaw(name, jnt, sign):
    b = ET.Element("body", {"name": name, "pos": f"{sign*OPEN} {REACH} 0"})
    lo, hi = ("-0.006 0.03" if sign < 0 else "-0.03 0.006").split(" ", 1)
    ET.SubElement(b, "joint", {"name": jnt, "type": "slide", "axis": "1 0 0",
                               "range": f"{lo} {hi}", "damping": "4"})
    ET.SubElement(b, "geom", {"name": f"pad{name[-1]}", "type": "box",
                              "size": "0.004 0.016 0.016", "pos": f"0 {PAD_Y} 0",
                              "rgba": "0.2 0.6 0.9 1", "friction": MU,
                              "contype": "2", "conaffinity": "2"})
    ET.SubElement(b, "site", {"name": f"pad{name[-1]}_s", "type": "box",
                              "size": "0.006 0.017 0.017", "pos": f"0 {PAD_Y} 0"})
    return b


SCENE = """
    <geom name="gfloor" type="plane" pos="0 0 -1.05" size="4 4 0.1" rgba="0.85 0.86 0.88 1"
          contype="0" conaffinity="0"/>
    <light pos="0.3 0.3 1.6" dir="-0.3 -0.3 -1" diffuse="0.7 0.7 0.7"/>
    <camera name="grip_cam" pos="0.55 0.44 0.30" xyaxes="0 -1 0 0.4 0 1" fovy="42"/>
"""


def make(model_dir, out_name="skt_v3_gripcell.xml"):
    src = os.path.join(model_dir, "skt_v3_collision.xml")
    if not os.path.exists(src):
        sys.exit("run make_collision_model.py first")
    root = ET.parse(src).getroot()

    wrist = next(b for b in root.iter("body") if b.get("name") == WRIST)
    wrist.append(_jaw("jawL", "jL", -1))
    wrist.append(_jaw("jawR", "jR", +1))

    wb = root.find("worldbody")
    for el in ET.fromstring("<r>" + SCENE + "</r>"):
        wb.append(el)
    part = ET.SubElement(wb, "body", {"name": "part", "pos": "0.16 0.44 0.10"})
    ET.SubElement(part, "freejoint", {})
    ET.SubElement(part, "geom", {"name": "part", "type": "box",
                                 "size": f"{PART[0]} {PART[1]} {PART[2]}",
                                 "rgba": "0.9 0.6 0.15 1", "density": "700",
                                 "friction": MU, "contype": "2", "conaffinity": "2"})

    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    ET.SubElement(act, "motor", {"name": "grip", "joint": "jL",
                                 "ctrlrange": "-60 60", "gear": "1"})

    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "joint", {"name": "couple", "joint1": "jR", "joint2": "jL",
                                "polycoef": "0 -1 0 0 0"})
    ET.SubElement(eq, "weld", {"name": "pin", "body1": "part",
                               "active": "false", "solref": "0.005 1"})

    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "touch", {"name": "grip_force", "site": "padR_s"})
    ET.SubElement(sens, "jointpos", {"name": "jaw", "joint": "jL"})

    out = os.path.join(model_dir, out_name)
    ET.ElementTree(root).write(out)
    import mujoco
    m = mujoco.MjModel.from_xml_path(out)              # compile check
    print(f"wrote {out}: {m.nbody} bodies, {m.nu} actuators, {m.nsensor} sensors")
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else ".")
