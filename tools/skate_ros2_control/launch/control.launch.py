"""Bring up ros2_control for the Skate arms over the skate_ros2 driver.

    ros2 launch skate_ros2_control control.launch.py \
        model_path:=/path/to/skate_teleop/skt_v3 robot_host:=127.0.0.1

Starts: robot_state_publisher (ros2_control-augmented URDF), the
controller_manager with the SkateSystem hardware, the two arm
JointTrajectoryControllers (names match skate_moveit_config), and the
skate_ros2 driver that owns the UDP wire + safety. No
joint_state_broadcaster — the driver already publishes /joint_states.
"""

import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_generator():
    share = Path(get_package_share_directory("skate_ros2_control"))
    spec = importlib.util.spec_from_file_location(
        "make_control_urdf", share / "make_control_urdf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nodes(context, *args):
    model_path = Path(LaunchConfiguration("model_path").perform(context))
    robot_host = LaunchConfiguration("robot_host").perform(context)
    robot_port = int(LaunchConfiguration("robot_port").perform(context))
    with_driver = LaunchConfiguration("driver").perform(context).lower() != "false"

    urdf = model_path / "skt_v3.urdf"
    if not urdf.exists():
        raise RuntimeError(f"{urdf} not found - point model_path at the "
                           "skt_v3 folder of a skate_teleop clone")
    gen = _load_generator()
    robot_description = gen.make_control_urdf(urdf.read_text(encoding="utf-8"))

    share = Path(get_package_share_directory("skate_ros2_control"))
    controllers = str(share / "config" / "controllers.yaml")

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="controller_manager", executable="ros2_control_node",
             parameters=[controllers],
             remappings=[("~/robot_description", "/robot_description")],
             output="screen"),
        Node(package="controller_manager", executable="spawner",
             arguments=["skate_left_arm_controller"], output="screen"),
        Node(package="controller_manager", executable="spawner",
             arguments=["skate_right_arm_controller"], output="screen"),
    ]

    if with_driver:
        nodes.append(
            Node(package="skate_ros2", executable="driver",
                 parameters=[{"robot_host": robot_host,
                              "robot_port": robot_port}],
                 output="screen"))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "model_path",
            description="skt_v3 folder of a Rbotic/skate_teleop clone"),
        DeclareLaunchArgument("robot_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("robot_port", default_value="2000"),
        DeclareLaunchArgument(
            "driver", default_value="true",
            description="also start the skate_ros2 driver (false = reuse one)"),
        OpaqueFunction(function=_nodes),
    ])
