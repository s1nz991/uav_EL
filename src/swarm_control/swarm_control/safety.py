from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .observation import Obstacle, Telemetry


@dataclass
class ShieldConfig:
    speed: float = 0.8
    accel: float = 1.2
    pair_safe: float = 1.1
    radius: float = 0.45
    obstacle_margin: float = 0.25
    workspace: tuple[tuple[float, float], ...] = ((-9.3, 9.3), (-9.3, 9.3), (0.35, 4.0))
    alpha: float = 1.4
    iterations: int = 60


@dataclass
class ShieldResult:
    velocity: np.ndarray
    feasible: bool
    intervention: float


def _clip_box(x: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)


def safe_velocity(
    position: np.ndarray,
    velocity: np.ndarray,
    desired: np.ndarray,
    peers: list[Telemetry],
    obstacles: list[Obstacle],
    dt: float,
    config: ShieldConfig | None = None,
) -> ShieldResult:
    cfg = config or ShieldConfig()
    position = np.asarray(position, dtype=np.float64).reshape(3)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    target = np.asarray(desired, dtype=np.float64).reshape(3)

    current = np.clip(velocity, -cfg.speed, cfg.speed)
    low = np.maximum(-cfg.speed, current - cfg.accel * dt)
    high = np.minimum(cfg.speed, current + cfg.accel * dt)
    low, high = np.minimum(low, high), np.maximum(low, high)

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for peer in peers:
        delta = position - peer.position
        distance = float(np.linalg.norm(delta))
        normal = delta / max(distance, 1e-9)
        rows.append(normal)
        rhs.append(-cfg.alpha * (distance - cfg.pair_safe) + float(np.dot(normal, peer.velocity)))

    for obstacle in obstacles:
        delta = position[:2] - np.array([obstacle.x, obstacle.y])
        distance = float(np.linalg.norm(delta))
        normal = np.array([delta[0] / max(distance, 1e-9), delta[1] / max(distance, 1e-9), 0.0])
        rows.append(normal)
        rhs.append(
            -cfg.alpha * (distance - obstacle.radius - cfg.radius - cfg.obstacle_margin)
        )

    for axis, (minimum, maximum) in enumerate(cfg.workspace):
        row = np.zeros(3)
        row[axis] = 1.0
        rows.append(row)
        rhs.append(-cfg.alpha * (position[axis] - minimum))
        rows.append(-row)
        rhs.append(-cfg.alpha * (maximum - position[axis]))

    matrix = np.asarray(rows)
    bounds = np.asarray(rhs)

    candidate = _clip_box(target, low, high)
    for _ in range(cfg.iterations):
        worst = 0.0
        for row, bound in zip(matrix, bounds):
            violation = float(bound - np.dot(row, candidate))
            if violation > 1e-9:
                candidate = candidate + (violation / float(np.dot(row, row))) * row
                candidate = _clip_box(candidate, low, high)
                worst = max(worst, violation)
        if worst <= 1e-9:
            break

    residual = float(np.min(matrix @ candidate - bounds))
    if residual < -1e-6:
        braking = np.zeros(3)
        return ShieldResult(braking, False, float(np.linalg.norm(braking - target)))

    intervention = float(np.linalg.norm(candidate - _clip_box(target, low, high)))
    return ShieldResult(candidate, True, intervention)
