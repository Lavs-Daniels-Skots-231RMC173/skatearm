#!/usr/bin/env python3
"""Insert the SkateSystem <ros2_control> block into the skt_v3 URDF.

The 14 structural arm joints (grippers excluded — they get their own
controller when a real gripper interface lands) are derived from
``skate_ros2.names``, the same source the driver and SRDF generator use, so
the three never drift apart.

    python3 make_control_urdf.py /path/to/skt_v3/skt_v3.urdf -o /tmp/skate_control.urdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:  # installed alongside skate_ros2 (same colcon ws) …
    from skate_ros2 import names
except ImportError:  # … or running from the repo checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skate_ros2"))
    from skate_ros2 import names


def arm_joint_names():
    """The 14 structural arm joints in protocol order (no grippers)."""
    idx = list(range(names.LEFT_ARM.start, names.LEFT_GRIPPER)) + \
          list(range(names.RIGHT_ARM.start, names.RIGHT_GRIPPER))
    return [names.JOINT_NAMES[i] for i in idx]


def ros2_control_block(state_topic="/joint_states",
                       command_topic="/skate/joint_position_cmd"):
    """The <ros2_control> XML block for the SkateSystem hardware plugin."""
    lines = [
        '  <ros2_control name="SkateSystem" type="system">',
        '    <hardware>',
        '      <plugin>skate_ros2_control/SkateSystem</plugin>',
        f'      <param name="state_topic">{state_topic}</param>',
        f'      <param name="command_topic">{command_topic}</param>',
        '    </hardware>',
    ]
    for joint in arm_joint_names():
        lines += [
            f'    <joint name="{joint}">',
            '      <command_interface name="position"/>',
            '      <state_interface name="position"/>',
            '      <state_interface name="velocity"/>',
            '    </joint>',
        ]
    lines.append('  </ros2_control>')
    return "\n".join(lines)


def make_control_urdf(urdf_text, state_topic="/joint_states",
                      command_topic="/skate/joint_position_cmd"):
    """Return the URDF with the ros2_control block inserted before </robot>."""
    if "</robot>" not in urdf_text:
        raise ValueError("not a URDF: no closing </robot> tag")
    block = ros2_control_block(state_topic, command_topic)
    return urdf_text.replace("</robot>", block + "\n</robot>")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="skt_v3 URDF + SkateSystem ros2_control block")
    ap.add_argument("urdf", help="path to skt_v3.urdf")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: stdout)")
    ap.add_argument("--state-topic", default="/joint_states")
    ap.add_argument("--command-topic", default="/skate/joint_position_cmd")
    args = ap.parse_args(argv)

    text = make_control_urdf(Path(args.urdf).read_text(encoding="utf-8"),
                             args.state_topic, args.command_topic)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out} ({len(arm_joint_names())} controlled joints)")
    else:
        print(text)


if __name__ == "__main__":
    main()
