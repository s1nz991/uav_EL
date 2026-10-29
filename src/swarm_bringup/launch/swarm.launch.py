from __future__ import annotations

import json
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _package_root() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("swarm_bringup"))
    except Exception:
        return Path(__file__).resolve().parent.parent


def _launch_setup(context, *args, **kwargs):
    package_root = _package_root()
    scenario_value = LaunchConfiguration("scenario").perform(context).strip()
    scenario_path = Path(scenario_value).expanduser() if scenario_value else package_root / "config" / "scenario.json"
    world_value = LaunchConfiguration("world").perform(context).strip()
    world_path = Path(world_value).expanduser() if world_value else package_root / "worlds" / "swarm.sdf"

    scenario = json.loads(scenario_path.read_text())
    starts = scenario["starts"]
    goals = scenario["goals"]
    if len(starts) != 3 or len(goals) != 3:
        raise ValueError("the launch file requires exactly three starts and three goals")

    px4_dir = Path(
        LaunchConfiguration("px4_dir").perform(context)
        or os.environ.get("PX4_DIR", "/home/siddharthg/PX4-Autopilot")
    ).expanduser()
    px4_build = px4_dir / "build" / "px4_sitl_default"
    px4_binary = px4_build / "bin" / "px4"
    if not px4_binary.is_file():
        raise FileNotFoundError(f"PX4 SITL binary not found: {px4_binary}")

    mode = LaunchConfiguration("mode").perform(context)
    checkpoint = LaunchConfiguration("checkpoint").perform(context).strip()
    if not checkpoint:
        checkpoint = str(package_root / "checkpoints" / "swarm_macro_policy.pt")
    allow_legacy = LaunchConfiguration("allow_legacy_checkpoint").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    enable_shield = LaunchConfiguration("enable_shield").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    auto_arm = LaunchConfiguration("auto_arm").perform(context).lower() in (
        "1",
        "true",
        "yes",
    )
    headless = LaunchConfiguration("headless").perform(context).lower() in ("1", "true", "yes")
    takeoff_altitude = float(LaunchConfiguration("takeoff_altitude").perform(context))
    control_rate = float(LaunchConfiguration("control_rate").perform(context))
    speed = 0.8
    accel = 1.2
    yaw_rate = 0.6

    agent_ids = ["drone_0", "drone_1", "drone_2"]

    actions = [
        SetEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            f"{px4_dir}/Tools/simulation/gz/models:{px4_dir}/Tools/simulation/gz/worlds:{package_root}/worlds",
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SYSTEM_PLUGIN_PATH",
            f"{px4_build}/src/modules/simulation/gz_plugins",
        ),
        SetEnvironmentVariable(
            "GZ_SIM_SERVER_CONFIG_PATH",
            f"{px4_dir}/src/modules/simulation/gz_bridge/server.config",
        ),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        SetEnvironmentVariable("GZ_PARTITION", "swarm"),
        SetEnvironmentVariable("PX4_GZ_WORLD", "swarm"),
        ExecuteProcess(
            cmd=[
                "bash",
                str(package_root / "scripts" / "install_px4_airframe.sh"),
                str(px4_dir),
            ],
            output="log",
            name="install_px4_airframe",
        ),
        ExecuteProcess(
            cmd=["MicroXRCEAgent", "udp4", "-p", "8888"],
            output="log",
            name="micro_xrce_agent",
        ),
        ExecuteProcess(
            cmd=["gz", "sim", "-r", "-s", str(world_path), "--verbose=1"],
            output="screen",
            name="gz_server",
        ),
    ]

    if not headless:
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    ExecuteProcess(
                        cmd=["gz", "sim", "-g", "--verbose=1"],
                        output="log",
                        name="gz_gui",
                    )
                ],
            )
        )

    for index, start in enumerate(starts):
        pose = f"{start[0]},{start[1]},0.0,0,0,0"
        actions.append(
            TimerAction(
                period=6.0 + 3.0 * index,
                actions=[
                    ExecuteProcess(
                        cmd=[str(px4_binary), "-i", str(index)],
                        cwd=str(px4_build),
                        additional_env={
                            "PX4_SYS_AUTOSTART": "4002",
                            "PX4_SIM_MODEL": "gz_x500",
                            "PX4_GZ_STANDALONE": "1",
                            "PX4_GZ_WORLD": "swarm",
                            "PX4_GZ_MODEL_POSE": pose,
                            "PX4_UXRCE_DDS_NS": f"uav_{index}",
                            "GZ_IP": "127.0.0.1",
                            "GZ_PARTITION": "swarm",
                        },
                        output="screen",
                        name=f"px4_{index}",
                    )
                ],
            )
        )

    for index, start in enumerate(starts):
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="px4_adapter",
                        executable="adapter_node",
                        name=f"adapter_{index}",
                        output="screen",
                        parameters=[
                            {
                                "drone_id": agent_ids[index],
                                "px4_namespace": f"/uav_{index}",
                                "system_id": index + 1,
                                "spawn_x": float(start[0]),
                                "spawn_y": float(start[1]),
                                "spawn_z": 0.0,
                                "takeoff_altitude": takeoff_altitude,
                                "takeoff_velocity": 0.8,
                                "auto_arm": auto_arm,
                            }
                        ],
                    )
                ],
            )
        )

    for index, goal in enumerate(goals):
        actions.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package="swarm_control",
                        executable="controller",
                        name=f"control_{index}",
                        output="screen",
                        parameters=[
                            {
                                "drone_id": agent_ids[index],
                                "agent_index": index,
                                "agent_ids": agent_ids,
                                "slot": [float(value) for value in goal],
                                "start": [float(value) for value in starts[index]],
                                "scenario": str(scenario_path),
                                "mode": mode,
                                "checkpoint": checkpoint,
                                "allow_legacy_checkpoint": allow_legacy,
                                "actor_index": index,
                                "control_rate": control_rate,
                                "enable_shield": enable_shield,
                                "altitude": float(goal[2]),
                                "speed": speed,
                                "accel": accel,
                                "yaw_rate": yaw_rate,
                            }
                        ],
                    )
                ],
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "scenario",
                default_value="",
                description="Scenario JSON; empty uses the packaged seed-7 scenario.",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="",
                description="Gazebo world SDF; empty uses the packaged swarm world.",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="learned",
                description="learned, classical, or zero.",
            ),
            DeclareLaunchArgument(
                "checkpoint",
                default_value="",
                description="Actor checkpoint for mode:=learned; empty uses the packaged swarm_macro_policy.pt.",
            ),
            DeclareLaunchArgument(
                "allow_legacy_checkpoint",
                default_value="false",
                description="Allow a state_dict without contract metadata (legacy maddpg_iter00460.pt).",
            ),
            DeclareLaunchArgument(
                "enable_shield",
                default_value="true",
                description="Enable the per-drone velocity CBF filter.",
            ),
            DeclareLaunchArgument(
                "auto_arm",
                default_value="true",
                description="Automatically arm and enter offboard after the setpoint stream is stable.",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Do not start the Gazebo GUI.",
            ),
            DeclareLaunchArgument(
                "px4_dir",
                default_value="",
                description="PX4-Autopilot checkout; empty uses $PX4_DIR or the default path.",
            ),
            DeclareLaunchArgument(
                "takeoff_altitude",
                default_value="2.5",
                description="World ENU takeoff altitude in metres.",
            ),
            DeclareLaunchArgument(
                "control_rate",
                default_value="10.0",
                description="Controller rate in Hz.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
