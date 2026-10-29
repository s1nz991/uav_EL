from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from swarm_interfaces.msg import AgentTelemetry

from .contract import ActorPolicy, Contract, DEFAULT_CONTRACT, action_to_velocity
from .observation import Obstacle, Telemetry, build_observation
from .planning import ClassicalMacro
from .safety import ShieldConfig, safe_velocity
from .supervisor import Mode, Supervisor, SupervisorConfig


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SwarmControlNode(Node):
    def __init__(self):
        super().__init__("swarm_control")

        self.declare_parameter("drone_id", "drone_0")
        self.declare_parameter("agent_index", 0)
        self.declare_parameter("agent_ids", ["drone_0", "drone_1", "drone_2"])
        self.declare_parameter("slot", [0.0, 0.0, 2.5])
        self.declare_parameter("start", [0.0, 0.0, 0.1])
        self.declare_parameter("scenario", "")
        self.declare_parameter("mode", "learned")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("allow_legacy_checkpoint", False)
        self.declare_parameter("actor_index", 0)
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("enable_shield", True)
        self.declare_parameter("altitude", 2.5)
        self.declare_parameter("speed", 0.8)
        self.declare_parameter("accel", 1.2)
        self.declare_parameter("yaw_rate", 0.6)
        self.declare_parameter("docking_timeout", 15.0)
        self.declare_parameter("hold_dwell", 1.5)
        self.declare_parameter("telemetry_timeout", 0.5)
        self.declare_parameter("stall_recovery", True)
        self.declare_parameter("stall_time", 1.0)
        self.declare_parameter("stall_recover_time", 1.5)
        self.declare_parameter("stall_clearance", 0.6)

        self.drone_id = self.get_parameter("drone_id").value
        self.agent_index = int(self.get_parameter("agent_index").value)
        self.agent_ids = list(self.get_parameter("agent_ids").value)
        self.slot = np.asarray(self.get_parameter("slot").value, dtype=np.float64)
        self.start = np.asarray(self.get_parameter("start").value, dtype=np.float64)
        self.scenario_path = str(self.get_parameter("scenario").value)
        self.mode_name = str(self.get_parameter("mode").value)
        self.checkpoint = str(self.get_parameter("checkpoint").value)
        self.allow_legacy = bool(self.get_parameter("allow_legacy_checkpoint").value)
        self.actor_index = int(self.get_parameter("actor_index").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.enable_shield = bool(self.get_parameter("enable_shield").value)
        self.altitude = float(self.get_parameter("altitude").value)
        self.speed = float(self.get_parameter("speed").value)
        self.accel = float(self.get_parameter("accel").value)
        self.yaw_rate_max = float(self.get_parameter("yaw_rate").value)
        self.telemetry_timeout = float(self.get_parameter("telemetry_timeout").value)
        self.stall_recovery = bool(self.get_parameter("stall_recovery").value)
        self.stall_time = float(self.get_parameter("stall_time").value)
        self.stall_recover_time = float(self.get_parameter("stall_recover_time").value)
        self.stall_clearance = float(self.get_parameter("stall_clearance").value)

        if len(self.agent_ids) != 3:
            raise ValueError("this build requires exactly three agents")
        if not 0 <= self.agent_index < 3:
            raise ValueError("agent_index must be 0, 1 or 2")
        if self.drone_id != self.agent_ids[self.agent_index]:
            raise ValueError("drone_id must equal agent_ids[agent_index]")
        if self.slot.shape != (3,) or self.start.shape != (3,):
            raise ValueError("slot and start must each have three values")

        self.contract = Contract(
            speed=self.speed,
            accel=self.accel,
            yaw_rate=self.yaw_rate_max,
            alt=self.altitude,
        )
        self.obstacles = self._load_obstacles()
        self.policy = None
        self.classical = None
        if self.mode_name == "learned":
            if not self.checkpoint:
                raise RuntimeError(
                    "mode:=learned requires checkpoint:=/path/to/actor.pt. "
                    "The original maddpg_iter00460.pt is not present in this workspace."
                )
            self.policy = ActorPolicy(
                self.checkpoint,
                contract=self.contract,
                actor_index=self.actor_index,
                allow_legacy=self.allow_legacy,
            )
            self.get_logger().info(
                f"loaded actor {self.actor_index} from {self.checkpoint} "
                f"(schema={self.contract.schema})"
            )
        elif self.mode_name == "classical":
            self.classical = ClassicalMacro(
                self.start,
                self.slot,
                self.obstacles,
                speed=self.speed,
                altitude=self.altitude,
            )
        elif self.mode_name != "zero":
            raise ValueError("mode must be learned, classical or zero")

        self.supervisor = Supervisor(
            SupervisorConfig(
                brake_accel=0.8,
                docking_timeout=float(self.get_parameter("docking_timeout").value),
                hold_dwell=float(self.get_parameter("hold_dwell").value),
            )
        )
        self.shield = ShieldConfig(
            speed=self.speed,
            accel=self.accel,
            workspace=(
                (-9.3, 9.3),
                (-9.3, 9.3),
                (0.35, max(4.0, self.altitude + 1.0)),
            ),
        )

        telemetry_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.telemetry: dict[str, Telemetry] = {
            agent_id: Telemetry.zero() for agent_id in self.agent_ids
        }
        self.last_rx: dict[str, float] = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.telemetry_subs = []
        for agent_id in self.agent_ids:
            self.telemetry_subs.append(
                self.create_subscription(
                    AgentTelemetry,
                    f"/{agent_id}/telemetry",
                    lambda msg, aid=agent_id: self._on_telemetry(aid, msg),
                    telemetry_qos,
                )
            )

        self.cmd_pub = self.create_publisher(Twist, f"/{self.drone_id}/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, f"/{self.drone_id}/status", 10)
        self.dt = 1.0 / self.control_rate
        self.last_log = 0.0
        self.stall_timer = 0.0
        self.recover_timer = 0.0
        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f"{self.drone_id} index={self.agent_index} mode={self.mode_name} "
            f"slot={self.slot.tolist()} shield={self.enable_shield}"
        )

    def _load_obstacles(self) -> list[Obstacle]:
        if not self.scenario_path:
            return []
        payload = json.loads(Path(self.scenario_path).read_text())
        return [
            Obstacle(float(x), float(y), float(radius), float(height))
            for x, y, radius, height in payload.get("obstacles", [])
        ]

    def _on_telemetry(self, agent_id: str, msg: AgentTelemetry) -> None:
        position = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float64)
        velocity = np.array([msg.velocity.x, msg.velocity.y, msg.velocity.z], dtype=np.float64)
        stamp = _stamp_seconds(msg.header.stamp)
        self.telemetry[agent_id] = Telemetry(
            position, velocity, float(msg.yaw), stamp, bool(msg.active_status)
        )
        self.last_rx[agent_id] = time.monotonic()

    def _fresh(self) -> bool:
        now = time.monotonic()
        return all(
            self.telemetry[agent_id].active
            and self.last_rx[agent_id] > 0.0
            and now - self.last_rx[agent_id] <= self.telemetry_timeout
            for agent_id in self.agent_ids
        )

    def _publish(self, velocity: np.ndarray, yaw_rate: float) -> None:
        command = Twist()
        command.linear.x = float(velocity[0])
        command.linear.y = float(velocity[1])
        command.linear.z = float(velocity[2])
        command.angular.z = float(yaw_rate)
        self.cmd_pub.publish(command)

    def tick(self) -> None:
        if not self._fresh():
            self._publish(np.zeros(3), 0.0)
            return

        own = self.telemetry[self.drone_id]
        peers = [
            self.telemetry[agent_id]
            for agent_id in self.agent_ids
            if agent_id != self.drone_id
        ]
        observation = build_observation(
            self.agent_index,
            [self.telemetry[agent_id] for agent_id in self.agent_ids],
            self.slot,
            self.obstacles,
            self.contract,
        )

        if self.policy is not None:
            action = self.policy(observation)
            macro_velocity, macro_yaw_rate = action_to_velocity(action, self.contract)
        elif self.classical is not None:
            macro_velocity = self.classical(own.position, own.velocity, peers, self.dt)
            macro_yaw_rate = 0.0
        else:
            macro_velocity = np.zeros(3)
            macro_yaw_rate = 0.0

        desired, stop_distance = self.supervisor.step(
            own.position, own.velocity, self.slot, macro_velocity, self.dt
        )

        stall_active = False
        if (
            self.stall_recovery
            and self.supervisor.mode == Mode.APPROACH
            and self.obstacles
            and own.position[2] > self.altitude - 0.25
        ):
            distance = float(np.linalg.norm(self.slot - own.position))
            speed = float(np.linalg.norm(own.velocity[:2]))
            clearance = float(observation[20]) * self.contract.obstacle_scale
            if distance > 1.0 and speed < 0.05 and clearance < self.stall_clearance:
                self.stall_timer += self.dt
            else:
                self.stall_timer = max(0.0, self.stall_timer - self.dt)
            if self.stall_timer >= self.stall_time:
                self.recover_timer = self.stall_recover_time
                self.stall_timer = 0.0
            if self.recover_timer > 0.0:
                nearest = min(
                    self.obstacles,
                    key=lambda o: float(
                        np.linalg.norm(own.position[:2] - np.array([o.x, o.y]))
                    )
                    - o.radius,
                )
                delta = own.position[:2] - np.array([nearest.x, nearest.y])
                length = float(np.linalg.norm(delta))
                if length < 1e-9:
                    delta = np.array([1.0, 0.0])
                    length = 1.0
                desired = np.array(
                    [
                        delta[0] / length * self.speed,
                        delta[1] / length * self.speed,
                        float(
                            np.clip(
                                1.1 * (self.altitude - own.position[2]),
                                -self.speed,
                                self.speed,
                            )
                        ),
                    ]
                )
                self.recover_timer = max(0.0, self.recover_timer - self.dt)
                stall_active = True

        yaw_rate = (
            0.0
            if stall_active or self.supervisor.mode in (Mode.DOCKING, Mode.HOLD)
            else float(np.clip(macro_yaw_rate, -self.yaw_rate_max, self.yaw_rate_max))
        )

        if self.enable_shield:
            result = safe_velocity(
                own.position,
                own.velocity,
                desired,
                peers,
                self.obstacles,
                self.dt,
                self.shield,
            )
            velocity, feasible, intervention = result.velocity, result.feasible, result.intervention
        else:
            low = np.maximum(-self.speed, own.velocity - self.accel * self.dt)
            high = np.minimum(self.speed, own.velocity + self.accel * self.dt)
            low, high = np.minimum(low, high), np.maximum(low, high)
            velocity = np.minimum(np.maximum(desired, low), high)
            feasible, intervention = True, 0.0

        self._publish(velocity, yaw_rate)

        now = time.monotonic()
        if now - self.last_log >= 1.0:
            self.last_log = now
            distance = float(np.linalg.norm(self.slot - own.position))
            status = {
                "drone_id": self.drone_id,
                "mode": self.supervisor.mode.value,
                "position": own.position.round(3).tolist(),
                "velocity": own.velocity.round(3).tolist(),
                "goal_distance": round(distance, 3),
                "d_stop": round(stop_distance, 3),
                "shield_feasible": bool(feasible),
                "shield_intervention": round(float(intervention), 4),
                "stall_recovery": bool(stall_active),
            }
            message = String()
            message.data = json.dumps(status)
            self.status_pub.publish(message)
            self.get_logger().info(message.data)


def main(args=None):
    rclpy.init(args=args)
    node = SwarmControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node._publish(np.zeros(3), 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
