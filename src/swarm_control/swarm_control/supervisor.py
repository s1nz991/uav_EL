from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Mode(str, Enum):
    APPROACH = "APPROACH"
    DOCKING = "DOCKING"
    HOLD = "HOLD"
    RECOVERY = "RECOVERY"


@dataclass
class SupervisorConfig:
    brake_accel: float = 0.8
    stop_margin: float = 0.65
    dock_kp: float = 0.9
    dock_kd: float = 0.55
    dock_speed: float = 0.8
    hold_dwell: float = 1.5
    hold_error: float = 0.12
    hold_speed: float = 0.10
    exit_error: float = 0.25
    exit_speed: float = 0.25
    recovery_distance: float = 2.2
    docking_timeout: float = 15.0
    recovery_time: float = 1.0


class Supervisor:
    def __init__(self, config: SupervisorConfig | None = None):
        self.config = config or SupervisorConfig()
        self.mode = Mode.APPROACH
        self.elapsed = 0.0
        self.settled = 0.0
        self.cooldown = 0.0

    def reset(self) -> None:
        self.mode = Mode.APPROACH
        self.elapsed = 0.0
        self.settled = 0.0
        self.cooldown = 0.0

    def step(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
        macro_velocity: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.config
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        macro_velocity = np.asarray(macro_velocity, dtype=np.float64)

        error = goal - position
        distance = float(np.linalg.norm(error))
        closing = max(0.0, float(np.dot(velocity, error / max(distance, 1e-9))))
        stop = closing * closing / (2.0 * cfg.brake_accel) + cfg.stop_margin
        self.cooldown = max(0.0, self.cooldown - dt)

        if self.mode in (Mode.DOCKING, Mode.HOLD):
            self.elapsed += dt
            if distance > cfg.recovery_distance or (
                self.mode == Mode.DOCKING and self.elapsed > cfg.docking_timeout
            ):
                self.mode = Mode.RECOVERY
                self.cooldown = cfg.recovery_time
                self.elapsed = 0.0
                self.settled = 0.0

        if self.mode == Mode.RECOVERY:
            if self.cooldown > 0.0:
                return macro_velocity, stop
            self.mode = Mode.APPROACH

        if self.mode == Mode.APPROACH and distance < stop:
            self.mode = Mode.DOCKING
            self.elapsed = 0.0
            self.settled = 0.0

        if self.mode in (Mode.DOCKING, Mode.HOLD):
            settled = distance < cfg.hold_error and float(np.linalg.norm(velocity)) < cfg.hold_speed
            self.settled = self.settled + dt if settled else 0.0
            if self.settled >= cfg.hold_dwell:
                self.mode = Mode.HOLD
            if self.mode == Mode.HOLD and (
                distance > cfg.exit_error or float(np.linalg.norm(velocity)) > cfg.exit_speed
            ):
                self.mode = Mode.DOCKING
                self.elapsed = 0.0
                self.settled = 0.0
            docking = cfg.dock_kp * error - cfg.dock_kd * velocity
            return np.clip(docking, -cfg.dock_speed, cfg.dock_speed), stop

        return macro_velocity, stop
