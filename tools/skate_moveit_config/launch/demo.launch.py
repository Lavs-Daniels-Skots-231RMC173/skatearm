"""Bring up MoveIt 2 for the Skate bimanual upper body against the sim (or the
real robot):

    ros2 launch skate_moveit_config demo.launch.py \\
        model_path:=/path/to/skate_teleop/skt_v3 robot_host:=127.0.0.1

Starts robot_state_publisher (URDF), move_group (SRDF + OMPL + the
FollowJointTrajectory controllers), the skate_driver (UDP link) and the
moveit_bridge (turns MoveIt trajectories into skate/joint_position_cmd), plus
RViz with the MotionPlanning panel. The sim endpoint (or the real Skate) must
be reachable at robot_host:2000 — e.g. `python -m skate_ros2.sim_endpoint`.

The URDF lives in your Rbotic/skate_teleop clone (like the sim tools), so it is
passed in via model_path rather than vendored into this package.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _setup(context, *_a, **_k):
    model_path = LaunchConfiguration("model_path").perform(context)
    robot_host = LaunchConfiguration("robot_host").perform(context)
    use_rviz = LaunchConfiguration("rviz").perform(context) == "true"

    pkg = get_package_share_directory("skate_moveit_config")
    with open(os.path.join(model_path, "skt_v3.urdf")) as f:
        urdf_xml = f.read()
    # the skt_v3 URDF names meshes scheme-lessly ("skt_v3_meshes/..."), which the
    # resource retriever tries to resolve as a URL host (10 s DNS hang per mesh)
    # — rewrite to absolute file:// URIs under model_path so they load locally
    urdf_xml = urdf_xml.replace('filename="skt_v3_meshes/',
                                'filename="file://' + model_path + '/skt_v3_meshes/')
    robot_description = {"robot_description": urdf_xml}
    with open(os.path.join(pkg, "config", "skate.srdf")) as f:
        robot_description_semantic = {"robot_description_semantic": f.read()}
    # group solvers must sit under robot_description_kinematics — passing the
    # yaml top-level (left_arm:/right_arm:) raw leaves move_group/RViz with
    # "No kinematics plugins defined" and a non-draggable goal marker
    kinematics = {"robot_description_kinematics":
                  _yaml(os.path.join(pkg, "config", "kinematics.yaml"))}
    controllers = _yaml(os.path.join(pkg, "config", "moveit_controllers.yaml"))
    joint_limits = {"robot_description_planning":
                    _yaml(os.path.join(pkg, "config", "joint_limits.yaml"))}
    # ROS 2 Jazzy planning-pipeline format: plugins + request/response adapters
    # are LISTS (Humble used a singular plugin + a space-joined adapter string).
    ompl = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": {
            "planning_plugins": ["ompl_interface/OMPLPlanner"],
            # NB: an epsilon-out-of-bounds START state aborts planning here
            # (CheckStartStateBounds) and Jazzy's fix_start_state only
            # normalizes continuous joints — it cannot clamp a revolute
            # limit violation. The sim endpoint therefore snaps reported
            # joint positions onto their limits within 1e-5 rad; on real
            # hardware the driver is the place for the same snap.
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
        },
    }
    ompl["ompl"].update(_yaml(os.path.join(pkg, "config", "ompl_planning.yaml")))

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             output="screen", parameters=[robot_description]),
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen",
             parameters=[robot_description, robot_description_semantic, kinematics,
                         ompl, controllers, joint_limits,
                         {"publish_robot_description_semantic": True}]),
        Node(package="skate_ros2", executable="driver", name="skate_driver",
             output="screen",
             parameters=[{"robot_host": robot_host, "robot_port": 2000}]),
        Node(package="skate_ros2", executable="moveit_bridge",
             name="skate_moveit_bridge", output="screen"),
    ]
    if use_rviz:
        nodes.append(Node(
            package="rviz2", executable="rviz2", output="screen",
            arguments=["-d", os.path.join(pkg, "config", "moveit.rviz")],
            parameters=[robot_description, robot_description_semantic, kinematics]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("model_path",
                              description="path to skate_teleop/skt_v3 (holds skt_v3.urdf)"),
        DeclareLaunchArgument("robot_host", default_value="127.0.0.1",
                              description="127.0.0.1 = sim endpoint, r.local = robot"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
