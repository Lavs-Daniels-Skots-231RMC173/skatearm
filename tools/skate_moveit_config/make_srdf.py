#!/usr/bin/env python3
"""Generate config/skate.srdf — the MoveIt 2 semantic description for the Skate
(skt_v3) bimanual upper body — straight from the URDF.

Deriving the SRDF (groups + the collision matrix) from the URDF keeps it in
lockstep with the model and with sim/make_collision_model.py's structural
excludes, instead of hand-typing ~200 collision pairs. The planning groups are
serial KDL chains from the fixed base (base_link2) to each wrist tip; the
grippers are single-joint groups. Joint/link names are the exact skt_v3 URDF
names (tools/skate_ros2/skate_ros2/names.py).

Honesty note: this is an authored SRDF, not the MoveIt Setup Assistant's
sampled output — the ACM covers adjacent links plus the known self-overlap
classes (intra-arm non-adjacent, arm<->lower-body). Inter-arm and arm<->torso
pairs stay ACTIVE (that's the bimanual-safety point). Regenerate the fully
sampled ACM with the Setup Assistant on a ROS 2 box before hardware use.

    python make_srdf.py --model /path/to/skate_teleop/skt_v3
"""
import argparse
import os
import xml.etree.ElementTree as ET

ARM = ["shoulder", "upperArm", "midArm", "lowArm",
       "wrist_a0", "wrist_a1", "wrist_a2", "wrist_a3"]
LOWER = ["hip", "upperLeg", "lowerLeg", "wheel"]
LEFT_JOINTS = [f"a{i}_armL_a{8 + i}" for i in range(7)]     # protocol 8..14
RIGHT_JOINTS = [f"a{i}_armR_a{16 + i}" for i in range(7)]   # protocol 16..22


def load_urdf(model_dir):
    """Return (robot_name, joints). The SRDF's <robot name> MUST match the
    URDF's or MoveIt refuses the semantic description."""
    root = ET.parse(os.path.join(model_dir, "skt_v3.urdf")).getroot()
    joints = [(j.get("name"), j.find("parent").get("link"), j.find("child").get("link"))
              for j in root.findall("joint")]
    return root.get("name"), joints


def acm(joints):
    pairs = {}

    def add(a, b, reason):
        pairs.setdefault(tuple(sorted((a, b))), reason)

    for _name, p, c in joints:
        add(p, c, "Adjacent")
    for suf in ("_1", "_Mirror__1"):
        arm = [l + suf for l in ARM]
        for i in range(len(arm)):
            for k in range(i + 2, len(arm)):        # intra-arm, non-adjacent
                add(arm[i], arm[k], "Never")
        for al in arm:
            for lb in LOWER:
                for lsuf in ("_1", "_Mirror__1"):   # arm <-> lower body
                    add(al, lb + lsuf, "Never")
    # Mesh-level contacts in the DOCUMENTED rest pose: with the arms hanging
    # both forearms graze the torso, and MoveIt's start-state collision check
    # refuses to plan FROM home without these two excludes. (The cockpit's
    # own capsule guard keeps arm<->torso pairs ACTIVE — capsules don't graze
    # there; this is purely the URDF-mesh geometry.)
    for suf in ("_1", "_Mirror__1"):
        add("base_link", "lowArm" + suf, "Default")
    return pairs


def _home_state(joints, elbow):
    return "".join(
        f'\n    <joint name="{j}" value="{1.5708 if j == elbow else 0.0}"/>'
        for j in joints)


def build_srdf(robot_name, joints):
    L = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!-- MoveIt 2 semantic description for the R.Botic Skate (skt_v3).',
         '     GENERATED from skt_v3.urdf by make_srdf.py — do not hand-edit;',
         '     re-run the generator, or the MoveIt Setup Assistant, instead.',
         '     NB: robot name matches the URDF (MoveIt requires it). -->',
         f'<robot name="{robot_name}">',
         '  <!-- planning groups: serial KDL chains, fixed base to wrist tip -->',
         '  <group name="left_arm"><chain base_link="base_link2" tip_link="wrist_a2_1"/></group>',
         '  <group name="right_arm"><chain base_link="base_link2" tip_link="wrist_a2_Mirror__1"/></group>',
         '  <group name="both_arms"><group name="left_arm"/><group name="right_arm"/></group>',
         '  <group name="left_gripper"><joint name="a7_armL_a15"/></group>',
         '  <group name="right_gripper"><joint name="a7_armR_a23"/></group>',
         '  <!-- group states (home = elbows bent 90 deg, per names.DEFAULT_POSE) -->',
         f'  <group_state name="home" group="left_arm">{_home_state(LEFT_JOINTS, "a3_armL_a11")}\n  </group_state>',
         f'  <group_state name="home" group="right_arm">{_home_state(RIGHT_JOINTS, "a3_armR_a19")}\n  </group_state>',
         '  <group_state name="open" group="left_gripper"><joint name="a7_armL_a15" value="0.35"/></group_state>',
         '  <group_state name="closed" group="left_gripper"><joint name="a7_armL_a15" value="0.0"/></group_state>',
         '  <group_state name="open" group="right_gripper"><joint name="a7_armR_a23" value="0.35"/></group_state>',
         '  <group_state name="closed" group="right_gripper"><joint name="a7_armR_a23" value="0.0"/></group_state>',
         '  <!-- end effectors -->',
         '  <end_effector name="left_ee" parent_link="wrist_a2_1" group="left_gripper"/>',
         '  <end_effector name="right_ee" parent_link="wrist_a2_Mirror__1" group="right_gripper"/>',
         '  <!-- fixed base (work-cell configuration) -->',
         '  <virtual_joint name="world_joint" type="fixed" parent_frame="world" child_link="base_link2"/>',
         '  <!-- allowed-collision matrix (adjacent + known self-overlap classes) -->']
    pairs = acm(joints)
    for a, b in sorted(pairs):
        L.append(f'  <disable_collisions link1="{a}" link2="{b}" reason="{pairs[(a, b)]}"/>')
    L.append('</robot>')
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to skate_teleop/skt_v3")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config", "skate.srdf"))
    args = ap.parse_args()
    robot_name, joints = load_urdf(args.model)
    srdf = build_srdf(robot_name, joints)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(srdf)
    print(f"wrote {args.out}: robot='{robot_name}', "
          f"{srdf.count('disable_collisions')} ACM pairs "
          f"(groups: left_arm, right_arm, both_arms, left/right_gripper)")


if __name__ == "__main__":
    main()
