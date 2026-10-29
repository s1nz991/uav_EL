from setuptools import setup

package_name = "swarm_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Swarm Integration",
    maintainer_email="swarm@example.invalid",
    description="Decentralized swarm controller with docking supervisor and safety filter.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "controller = swarm_control.node:main",
        ],
    },
)
