import os
from glob import glob

from setuptools import find_packages, setup

package_name = "px4_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Swarm Integration",
    maintainer_email="swarm@example.invalid",
    description="PX4 offboard adapter and ENU telemetry bridge for the swarm controller.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "adapter_node = px4_adapter.adapter_node:main",
        ],
    },
)
