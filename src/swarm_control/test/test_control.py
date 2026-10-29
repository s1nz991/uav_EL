import numpy as np

from swarm_control.observation import Obstacle, Telemetry, build_observation
from swarm_control.safety import safe_velocity
from swarm_control.supervisor import Mode, Supervisor


def test_observation_contract_layout():
    telemetry = [
        Telemetry(np.array([0.0, 0.0, 2.0]), np.array([0.8, 0.0, 0.0]), np.pi / 2),
        Telemetry(np.array([1.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.0]), 0.0),
        Telemetry(np.array([0.0, 1.0, 2.0]), np.array([0.0, 0.8, 0.0]), 0.0),
    ]
    obs = build_observation(
        0,
        telemetry,
        np.array([15.0, 0.0, 2.0]),
        [Obstacle(3.0, 0.0, 0.5, 5.0)],
    )
    assert obs.shape == (23,)
    assert obs.dtype == np.float32
    assert np.allclose(obs[0:3], [1.0, 0.0, 0.0])
    assert np.allclose(obs[3:6], [1.0, 0.0, 0.0])
    assert np.allclose(obs[6:9], [1.0 / 15.0, 0.0, 0.0])
    assert np.allclose(obs[9:12], [-0.5, 0.0, 0.0])
    assert np.allclose(obs[12:15], [0.0, 1.0 / 15.0, 0.0])
    assert np.allclose(obs[15:18], [-0.5, 0.5, 0.0])
    assert np.allclose(obs[18:20], [0.2, 0.0])
    assert np.isclose(obs[20], (3.0 - 0.5 - 0.3) / 15.0)
    assert np.isclose(obs[21], 1.0)
    assert np.isclose(obs[22], 0.0)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_supervisor_approach_docking_hold_and_recovery():
    supervisor = Supervisor()
    goal = np.array([5.0, 0.0, 2.5])
    position = np.array([0.0, 0.0, 2.5])
    velocity = np.array([0.8, 0.0, 0.0])
    command, _ = supervisor.step(position, velocity, goal, velocity, 0.1)
    assert supervisor.mode == Mode.APPROACH
    assert np.allclose(command, velocity)

    position = np.array([4.0, 0.0, 2.5])
    command, stop = supervisor.step(position, velocity, goal, velocity, 0.1)
    assert supervisor.mode == Mode.DOCKING
    assert stop > 0.0
    assert command[0] < velocity[0]

    position = np.array([5.0, 0.0, 2.5])
    for _ in range(16):
        supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.HOLD

    position = np.array([5.3, 0.0, 2.5])
    supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.DOCKING

    position = np.array([2.0, 0.0, 2.5])
    supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.RECOVERY
    command, _ = supervisor.step(position, np.zeros(3), goal, np.array([0.8, 0.0, 0.0]), 0.1)
    assert np.allclose(command, [0.8, 0.0, 0.0])
    for _ in range(11):
        supervisor.step(position, np.zeros(3), goal, np.zeros(3), 0.1)
    assert supervisor.mode == Mode.APPROACH


def test_shield_blocks_head_on_pair_closing():
    own = np.array([0.0, 0.0, 2.5])
    peer = Telemetry(np.array([1.1, 0.0, 2.5]), np.zeros(3), 0.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.array([0.8, 0.0, 0.0]),
        [peer],
        [],
        0.1,
    )
    assert result.feasible
    assert result.velocity[0] <= 0.01


def test_shield_requires_separation_when_inside_pair_envelope():
    own = np.array([0.0, 0.0, 2.5])
    peer = Telemetry(np.array([0.5, 0.0, 2.5]), np.zeros(3), 0.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.zeros(3),
        [peer],
        [],
        0.1,
    )
    assert not result.feasible
    assert np.linalg.norm(result.velocity) <= 1e-9


def test_shield_avoids_obstacle_cylinder():
    own = np.array([0.0, 0.0, 2.5])
    obstacle = Obstacle(1.2, 0.0, 0.5, 5.0)
    result = safe_velocity(
        own,
        np.zeros(3),
        np.array([0.8, 0.0, 0.0]),
        [],
        [obstacle],
        0.1,
    )
    assert result.feasible
    assert result.velocity[0] <= 0.01


import unittest


class TestControlContract(unittest.TestCase):
    def test_observation_contract_layout(self):
        test_observation_contract_layout()

    def test_supervisor_approach_docking_hold_and_recovery(self):
        test_supervisor_approach_docking_hold_and_recovery()

    def test_shield_blocks_head_on_pair_closing(self):
        test_shield_blocks_head_on_pair_closing()

    def test_shield_requires_separation_when_inside_pair_envelope(self):
        test_shield_requires_separation_when_inside_pair_envelope()

    def test_shield_avoids_obstacle_cylinder(self):
        test_shield_avoids_obstacle_cylinder()
