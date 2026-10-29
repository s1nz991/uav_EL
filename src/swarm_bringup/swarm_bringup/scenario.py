from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from swarm_control.planning import astar


@dataclass
class Scenario:
    seed: int
    starts: list
    goals: list
    obstacles: list

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        return cls(**json.loads(Path(path).read_text()))


def triangle(side: float = 1.5) -> np.ndarray:
    radius = side / math.sqrt(3.0)
    return np.array([[radius, 0.0, 0.0], [-radius / 2.0, side / 2.0, 0.0], [-radius / 2.0, -side / 2.0, 0.0]])


def generate(seed: int = 7, count: int = 7) -> Scenario:
    from swarm_control.observation import Obstacle

    rng = np.random.default_rng(seed)
    for _ in range(200):
        angle = rng.uniform(-math.pi, math.pi)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        center = np.r_[rng.uniform(-0.5, 0.5, 2), 0.0]
        slots = triangle() @ rotation.T
        starts = slots + np.array([-6.0, 0.0, 0.45]) @ rotation.T + center
        goals = slots + np.array([6.0, 0.0, 2.5]) @ rotation.T + center
        obstacles: list[list[float]] = []
        for _ in range(500):
            if len(obstacles) == count:
                break
            xy = rng.uniform(-6.7, 6.7, 2)
            radius = rng.uniform(0.45, 0.85)
            if min(
                np.linalg.norm(starts[:, :2] - xy, axis=1).min(),
                np.linalg.norm(goals[:, :2] - xy, axis=1).min(),
            ) < radius + 1.8:
                continue
            if any(
                np.linalg.norm(xy - np.array(obstacle[:2])) < radius + obstacle[2] + 0.65
                for obstacle in obstacles
            ):
                continue
            obstacles.append([float(xy[0]), float(xy[1]), float(radius), 5.0])
        if len(obstacles) != count:
            continue
        typed = [Obstacle(x, y, r, h) for x, y, r, h in obstacles]
        if all(astar(start, goal, typed) is not None for start, goal in zip(starts, goals)):
            return Scenario(int(seed), starts.tolist(), goals.tolist(), obstacles)
    raise RuntimeError("unable to sample a reachable map")


def _visual(name: str, geometry: str, color: str, pose: str = "0 0 0 0 0 0") -> str:
    return (
        f'<visual name="{name}"><pose>{pose}</pose><geometry>{geometry}</geometry>'
        f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse>"
        f"<specular>{color}</specular></material></visual>"
    )


def _cylinder(radius: float, height: float) -> str:
    return f"<cylinder><radius>{radius}</radius><length>{height}</length></cylinder>"


def _marker(name: str, x: float, y: float, radius: float, color: str) -> str:
    return (
        f'<model name="{name}"><static>true</static><pose>{x} {y} 0.026 0 0 0</pose>'
        f'<link name="marker">{_visual("disk", _cylinder(radius, 0.025), color)}</link></model>'
    )


def write_world(scenario: Scenario, path: str | Path) -> None:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sdf version="1.9"><world name="swarm">',
        '<physics type="ode"><max_step_size>0.004</max_step_size>'
        '<real_time_factor>1.0</real_time_factor>'
        '<real_time_update_rate>250</real_time_update_rate></physics>',
        "<gravity>0 0 -9.8</gravity>",
        "<magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>",
        '<atmosphere type="adiabatic"/>',
        "<spherical_coordinates>"
        "<surface_model>EARTH_WGS84</surface_model>"
        "<world_frame_orientation>ENU</world_frame_orientation>"
        "<latitude_deg>47.397971057728974</latitude_deg>"
        "<longitude_deg>8.546163739800146</longitude_deg>"
        "<elevation>488</elevation>"
        "</spherical_coordinates>",
        '<scene><grid>false</grid><ambient>0.4 0.4 0.4 1</ambient>'
        "<background>0.7 0.7 0.7 1</background><shadows>true</shadows></scene>",
        '<light name="sun" type="directional"><pose>0 0 500 0 -0 0</pose>'
        "<cast_shadows>true</cast_shadows><intensity>1</intensity>"
        "<direction>0.001 0.625 -0.78</direction>"
        "<diffuse>0.904 0.904 0.904 1</diffuse>"
        "<specular>0.271 0.271 0.271 1</specular>"
        "<attenuation><range>2000</range><linear>0</linear><constant>1</constant>"
        "<quadratic>0</quadratic></attenuation></light>",
        '<model name="ground_plane"><static>true</static><link name="link">'
        '<collision name="collision"><geometry><plane><normal>0 0 1</normal>'
        "<size>1 1</size></plane></geometry><surface><friction><ode/></friction>"
        "<bounce/><contact/></surface></collision>"
        + _visual("visual", "<plane><normal>0 0 1</normal><size>500 500</size></plane>", "0.8 0.8 0.8 1")
        + "</link></model>",
    ]

    for k in range(-10, 11):
        for axis in range(2):
            size = "0.015 20 0.002" if axis == 0 else "20 0.015 0.002"
            pose = f"{k if axis == 0 else 0} {k if axis == 1 else 0} .002 0 0 0"
            parts.append(
                f'<model name="grid_{axis}_{k + 10}"><static>true</static><link name="line">'
                + _visual("line", f"<box><size>{size}</size></box>", "0.25 0.29 0.34 1", pose)
                + "</link></model>"
            )

    start_x = float(np.mean([start[0] for start in scenario.starts]))
    start_y = float(np.mean([start[1] for start in scenario.starts]))
    goal_x = float(np.mean([goal[0] for goal in scenario.goals]))
    goal_y = float(np.mean([goal[1] for goal in scenario.goals]))
    parts.append(_marker("START_ZONE", start_x, start_y, 1.6, "0.05 0.35 0.65 1"))
    parts.append(_marker("GOAL_ZONE", goal_x, goal_y, 1.6, "0.05 0.55 0.30 1"))

    for index, (x, y, radius, height) in enumerate(scenario.obstacles):
        geometry = _cylinder(radius, height)
        parts.append(
            f'<model name="obstacle_{index}"><static>true</static>'
            f"<pose>{x} {y} {height / 2} 0 0 0</pose><link name=\"pillar\">"
            f'<collision name="collision"><geometry>{geometry}</geometry></collision>'
            + _visual("pillar", geometry, "0.65 0.25 0.18 1")
            + "</link></model>"
        )

    for index, goal in enumerate(scenario.goals):
        color = ["0.10 0.75 1.0 1", "1.0 0.45 0.15 1", "0.7 0.30 1.0 1"][index]
        parts.append(
            f'<model name="goal_beacon_{index}"><static>true</static>'
            f"<pose>{goal[0]} {goal[1]} {goal[2]} 0 0 0</pose><link name=\"beacon\">"
            + _visual("beacon", "<sphere><radius>.08</radius></sphere>", color)
            + "</link></model>"
        )

    parts.append("</world></sdf>")
    Path(path).write_text("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--obstacles", type=int, default=7)
    parser.add_argument("--out", default="generated")
    args = parser.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    scenario = generate(args.seed, args.obstacles)
    scenario.save(folder / "scenario.json")
    write_world(scenario, folder / "swarm.sdf")
    print(folder.resolve())


if __name__ == "__main__":
    main()
