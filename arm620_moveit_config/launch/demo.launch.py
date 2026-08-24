from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    MoveItConfigsBuilder("arm620", package_name="arm620_moveit_config") \
        .robot_description(file_path="config/arm620.urdf") \
        .robot_description_semantic(file_path="config/arm620.srdf") \
        .robot_description_kinematics(file_path="config/kinematics.yaml") \
        .joint_limits(file_path="config/joint_limits.yaml") \
        .trajectory_execution(file_path="config/moveit_controllers.yaml") \
        .planning_pipelines(default_planning_pipeline="ompl", pipelines=["ompl"]) \
        .to_moveit_configs()

    from pathlib import Path
    from ament_index_python.packages import get_package_share_directory
    config_pkg = Path(get_package_share_directory("arm620_moveit_config"))
    ld = LaunchDescription()
    for name in [
        "static_virtual_joint_tfs.launch.py",
        "rsp.launch.py",
        "move_group.launch.py",
        "moveit_rviz.launch.py",
    ]:
        ld.add_action(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(config_pkg / "launch" / name))
            )
        )
    return ld
