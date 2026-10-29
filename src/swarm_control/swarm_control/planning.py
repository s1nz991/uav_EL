from __future__ import annotations

import heapq
import math

import numpy as np

from .observation import Obstacle, Telemetry


def segment_distance(a, b, c) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    ab = b - a
    t = np.clip(np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - c))


def clear_segment(a, b, obstacles: list[Obstacle], margin: float) -> bool:
    return all(
        segment_distance(np.asarray(a)[:2], np.asarray(b)[:2], np.array([o.x, o.y]))
        > o.radius + margin
        for o in obstacles
    )


def astar(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[Obstacle],
    resolution: float = 0.4,
    margin: float = 1.3,
) -> np.ndarray | None:
    lo, hi = -9.2, 9.2
    size = int(round((hi - lo) / resolution)) + 1

    def cell(point) -> tuple[int, int]:
        return tuple(
            np.clip(np.rint((np.asarray(point)[:2] - lo) / resolution), 0, size - 1).astype(int)
        )

    def point(cell_xy: tuple[int, int]) -> np.ndarray:
        return np.array(cell_xy, dtype=np.float64) * resolution + lo

    def free(cell_xy: tuple[int, int]) -> bool:
        return all(0 <= k < size for k in cell_xy) and clear_segment(
            point(cell_xy), point(cell_xy), obstacles, margin
        )

    source, target = cell(start), cell(goal)
    if not free(source) or not clear_segment(start, point(source), obstacles, margin):
        candidates = [
            (source[0] + dx, source[1] + dy)
            for dx in range(-4, 5)
            for dy in range(-4, 5)
        ]
        candidates.sort(key=lambda c: np.linalg.norm(point(c) - np.asarray(start)[:2]))
        source = next(
            (c for c in candidates if free(c) and clear_segment(start, point(c), obstacles, margin)),
            None,
        )
        if source is None:
            return None
    if not free(target) or not clear_segment(point(target), goal, obstacles, margin):
        return None

    queue: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
    costs = {source: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == target:
            cells = [current]
            while current != source:
                current = parent[current]
                cells.append(current)
            route = [np.asarray(start)[:2]] + [point(c) for c in cells[::-1]] + [
                np.asarray(goal)[:2]
            ]
            simplified = [route[0]]
            i = 0
            while i < len(route) - 1:
                j = len(route) - 1
                while j > i + 1 and not clear_segment(route[i], route[j], obstacles, margin):
                    j -= 1
                simplified.append(route[j])
                i = j
            return np.asarray(simplified)

        for dx, dy in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            neighbor = (current[0] + dx, current[1] + dy)
            if not free(neighbor) or not clear_segment(
                point(current), point(neighbor), obstacles, margin
            ):
                continue
            cost = costs[current] + math.hypot(dx, dy)
            if cost < costs.get(neighbor, float("inf")):
                costs[neighbor] = cost
                parent[neighbor] = current
                heapq.heappush(
                    queue, (cost + math.dist(neighbor, target), neighbor)
                )
    return None


class ClassicalMacro:
    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[Obstacle],
        speed: float = 0.8,
        altitude: float = 2.5,
    ):
        self.start = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.obstacles = list(obstacles)
        self.speed = float(speed)
        self.altitude = float(altitude)
        self.path = astar(self.start, self.goal, self.obstacles)
        if self.path is None:
            raise ValueError("classical macro could not find a route to the assigned slot")
        self.index = 1
        self.stalled = 0.0

    def __call__(self, position, velocity, peers: list[Telemetry], dt: float) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        stalled = (
            float(np.linalg.norm(velocity[:2])) < 0.08
            and float(np.linalg.norm(self.goal[:2] - position[:2])) > 1.0
            and position[2] > self.altitude - 0.25
        )
        self.stalled = self.stalled + dt if stalled else 0.0
        if self.stalled > 2.0:
            dynamic = self.obstacles + [
                Obstacle(peer.position[0], peer.position[1], 0.45, 5.0) for peer in peers
            ]
            replanned = astar(position, self.goal, dynamic, margin=1.25)
            if replanned is not None:
                self.path = replanned
                self.index = 1
            self.stalled = 0.0

        while self.index < len(self.path) - 1:
            waypoint = self.path[self.index]
            previous = self.path[self.index - 1]
            delta = waypoint - previous
            distance = float(np.linalg.norm(position[:2] - waypoint))
            passed = float(np.dot(position[:2] - previous, delta)) >= 0.95 * float(
                np.dot(delta, delta)
            )
            if distance < 0.5 or passed:
                self.index += 1
            else:
                break

        target = np.r_[self.path[self.index], self.altitude]
        error = target - position
        remaining = float(np.linalg.norm(error[:2]))
        speed_scale = min(1.0, max(0.25, remaining / 1.5))
        command = np.clip(1.1 * speed_scale * error, -self.speed, self.speed)
        if position[2] < self.altitude - 0.25:
            command[:2] = 0.0
        return command
