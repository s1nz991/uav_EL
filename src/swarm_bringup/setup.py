from setuptools import setup

package_name = "swarm_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/swarm.launch.py"]),
        ("share/" + package_name + "/config", ["config/scenario.json"]),
        ("share/" + package_name + "/worlds", ["worlds/swarm.sdf"]),
        (
            "share/" + package_name + "/checkpoints",
            ["checkpoints/swarm_macro_policy.pt", "checkpoints/maddpg_iter00460.pt"],
        ),
        ("share/" + package_name + "/px4", ["px4/4002_gz_x500_swarm"]),
        ("share/" + package_name + "/scripts", ["scripts/install_px4_airframe.sh"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Swarm Integration",
    maintainer_email="swarm@example.invalid",
    description="Bring-up for the three-drone PX4/Gazebo swarm simulation.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "generate_world = swarm_bringup.scenario:main",
        ],
    },
)
