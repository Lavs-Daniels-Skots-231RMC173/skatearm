"""make_control_urdf: the generated ros2_control block is sane. ROS-free."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "skate_ros2"))

from make_control_urdf import (arm_joint_names, make_control_urdf,  # noqa: E402
                               ros2_control_block)
from skate_ros2 import names  # noqa: E402

MINI_URDF = "<robot name=\"r04\">\n  <link name=\"base\"/>\n</robot>"


def test_joint_list():
    joints = arm_joint_names()
    assert len(joints) == 14, joints
    # protocol order: 7 left then 7 right, structural only
    assert joints[0] == names.JOINT_NAMES[names.LEFT_ARM.start]
    assert names.JOINT_NAMES[names.LEFT_GRIPPER] not in joints
    assert names.JOINT_NAMES[names.RIGHT_GRIPPER] not in joints
    assert all("armL" in j for j in joints[:7])
    assert all("armR" in j for j in joints[7:])


def test_block_is_valid_xml_with_interfaces():
    root = ET.fromstring(ros2_control_block())
    assert root.tag == "ros2_control" and root.get("type") == "system"
    assert root.find("hardware/plugin").text == "skate_ros2_control/SkateSystem"
    topics = {p.get("name"): p.text for p in root.findall("hardware/param")}
    assert topics == {"state_topic": "/joint_states",
                      "command_topic": "/skate/joint_position_cmd"}
    joints = root.findall("joint")
    assert len(joints) == 14
    for j in joints:
        assert j.find("command_interface").get("name") == "position"
        states = [s.get("name") for s in j.findall("state_interface")]
        assert states == ["position", "velocity"]


def test_insertion_keeps_urdf_parseable():
    out = make_control_urdf(MINI_URDF)
    root = ET.fromstring(out)                     # whole document still XML
    assert root.tag == "robot"
    rc = root.findall("ros2_control")
    assert len(rc) == 1
    assert out.rstrip().endswith("</robot>")
    # custom topics reach the params
    out2 = make_control_urdf(MINI_URDF, "/js", "/cmd")
    assert "<param name=\"state_topic\">/js</param>" in out2
    assert "<param name=\"command_topic\">/cmd</param>" in out2


def test_rejects_non_urdf():
    try:
        make_control_urdf("<xml/>")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a non-URDF document")


if __name__ == "__main__":
    test_joint_list()
    test_block_is_valid_xml_with_interfaces()
    test_insertion_keeps_urdf_parseable()
    test_rejects_non_urdf()
    print("OK")
