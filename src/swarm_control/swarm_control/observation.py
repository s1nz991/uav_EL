from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import Contract, DEFAULT_CONTRACT, normalize_observation


@dataclass
class Telemetry:
    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    stamp: float = 0.0
    active: bool = True

    @classmethod
    def zero(cls) -> "Telemetry":
        return cls(np.zeros(3), np.zeros(3), 0.0, 0.0, False)


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    height: float = 4.0


def build_observation(
    agent_index: int,
    telemetry: list[Telemetry],
    goal: np.ndarray,
    obstacles: list[Obstacle],
    contract: Contract = DEFAULT_CONTRACT,
) -> np.ndarray:
    own = telemetry[agent_index]
    peers = [telemetry[i] for i in range(len(telemetry)) if i != agent_index]
    if len(peers) < 2:
        raise ValueError("the v1 contract requires exactly two peers")

    obs = np.zeros(contract.obs_dim, dtype=np.float32)
    obs[0:3] = (np.asarray(goal, dtype=np.float64) - own.position) / contract.goal_scale
    obs[3:6] = own.velocity / contract.vel_scale

    for k, peer in enumerate(peers[:2]):
        base = 6 + 6 * k
        obs[base : base + 3] = (peer.position - own.position) / contract.goal_scale
        obs[base + 3 : base + 6] = (peer.velocity - own.velocity) / contract.peer_vel_scale

    if obstacles:
        best = min(
            obstacles,
            key=lambda o: np.linalg.norm(own.position[:2] - np.array([o.x, o.y])) - o.radius,
        )
        delta = np.array([best.x, best.y]) - own.position[:2]
        distance = float(np.linalg.norm(delta))
        obs[18:20] = delta / contract.obstacle_scale
        obs[20] = (distance - best.radius - contract.radius) / contract.obstacle_scale
    else:
        obs[20] = 1.0

    obs[21] = np.sin(own.yaw)
    obs[22] = np.cos(own.yaw)
    return normalize_observation(obs, contract)
