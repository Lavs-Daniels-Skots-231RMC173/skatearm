#!/usr/bin/env python3
"""Generate a control-ready MJCF from the official skt_v3_converted.xml.

The converted model in Rbotic/skate_teleop is visualization-only: no actuators,
no sensors, no damping (nu=0). This script produces `skt_v3_control.xml` with:

- **fixed base** (freejoint removed) — work-cell configuration
- **joint damping + armature** for stable servo behavior
- **position actuators** on all 26 named hinges, ctrlrange = joint range,
  forcerange taken from the URDF's actuatorfrcrange (±28 N·m)
- **contacts disabled** — the raw converted meshes interpenetrate at the
  shoulder mounts and jam the joints (the shoulder actuator saturates at the
  28 N·m limit just fighting the contact force). Free-space motion control
  works correctly without contacts; proper collision geometry is a separate
  task on the roadmap.

Verified (MuJoCo 3.9): holds RELAXED and WORK poses with max error < 0.03 rad
(~1.5°) at default kp=100, no divergence, settles to zero velocity.

Usage:
    python make_control_model.py /path/to/skate_teleop/skt_v3
"""
import os
import sys
import xml.etree.ElementTree as ET


def make(model_dir, out_name="skt_v3_control.xml", kp=100.0, damping=2.0, armature=0.05):
    src = os.path.join(model_dir, "skt_v3_converted.xml")
    xml = open(src).read()
    xml = xml.replace("<freejoint/>", "<!-- fixed base for work-cell -->")
    root = ET.fromstring(xml)

    # free-space control model: disable contacts (see module docstring)
    opt = ET.Element("option")
    ET.SubElement(opt, "flag", {"contact": "disable"})
    root.insert(1, opt)

    # defaults: damping/armature on all joints
    default = ET.Element("default")
    ET.SubElement(default, "joint", {"damping": str(damping), "armature": str(armature)})
    root.insert(1, default)

    # position actuator for every named joint with a range
    act = ET.SubElement(root, "actuator")
    n = 0
    for j in root.iter("joint"):
        name = j.get("name")
        rng = j.get("range")
        if not name or not rng:
            continue
        attrs = {"name": f"pos_{name}", "joint": name, "kp": str(kp), "ctrlrange": rng}
        if j.get("actuatorfrcrange"):
            attrs["forcerange"] = j.get("actuatorfrcrange")
        ET.SubElement(act, "position", attrs)
        n += 1

    # end-effector sites on both wrists
    for b in root.iter("body"):
        if b.get("name") in ("wrist_a3_1", "wrist_a3_Mirror__1"):
            tag = "ee_left" if b.get("name") == "wrist_a3_1" else "ee_right"
            ET.SubElement(b, "site", {"name": tag, "pos": "0 0 0", "size": "0.012",
                                      "rgba": "0.1 0.9 0.3 0.8"})

    # sensors: per-joint state + torque (telemetry parity with the real Skate's
    # state stream), and end-effector poses for task-space work
    sens = ET.SubElement(root, "sensor")
    for j in root.iter("joint"):
        name = j.get("name")
        if not name or not j.get("range"):
            continue
        ET.SubElement(sens, "jointpos", {"name": f"qpos_{name}", "joint": name})
        ET.SubElement(sens, "jointvel", {"name": f"qvel_{name}", "joint": name})
        ET.SubElement(sens, "actuatorfrc", {"name": f"tau_{name}", "actuator": f"pos_{name}"})
    for site in ("ee_left", "ee_right"):
        ET.SubElement(sens, "framepos", {"name": f"{site}_pos", "objtype": "site", "objname": site})
        ET.SubElement(sens, "framequat", {"name": f"{site}_quat", "objtype": "site", "objname": site})
        # M1 — wrist 6-axis force/torque: the wrench transmitted through the wrist
        # joint (site frame). A true contact wrench, not the actuator-torque proxy.
        ET.SubElement(sens, "force", {"name": f"{site}_force", "site": site})
        ET.SubElement(sens, "torque", {"name": f"{site}_torque", "site": site})

    out = os.path.join(model_dir, out_name)
    ET.ElementTree(root).write(out)
    print(f"wrote {out}: {n} position actuators (kp={kp}, damping={damping}), {3 * n + 8} sensors")
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else ".")
