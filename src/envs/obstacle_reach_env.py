from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@dataclass
class ObstacleStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class ObstacleReachEnv:
    """2D reaching with one circular obstacle.

    Observation: [x, y, gx, gy, ox, oy, r]
    Action: [dx, dy]
    """

    def __init__(
        self,
        workspace_low: float = -1.0,
        workspace_high: float = 1.0,
        max_action: float = 0.08,
        goal_threshold: float = 0.06,
        max_steps: int = 100,
        obstacle_center: tuple[float, float] = (0.0, 0.0),
        obstacle_radius: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        if workspace_low >= workspace_high:
            raise ValueError("workspace_low must be smaller than workspace_high.")
        if max_action <= 0:
            raise ValueError("max_action must be positive.")
        if goal_threshold <= 0:
            raise ValueError("goal_threshold must be positive.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if obstacle_radius <= 0:
            raise ValueError("obstacle_radius must be positive.")

        self.workspace_low = float(workspace_low)
        self.workspace_high = float(workspace_high)
        self.max_action = float(max_action)
        self.goal_threshold = float(goal_threshold)
        self.max_steps = int(max_steps)
        self.obstacle_center = np.asarray(obstacle_center, dtype=np.float32).reshape(2)
        self.obstacle_radius = float(obstacle_radius)
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.trajectory: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def reset(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        obstacle_center: Optional[np.ndarray] = None,
        obstacle_radius: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if obstacle_center is not None:
            self.obstacle_center = np.asarray(obstacle_center, dtype=np.float32).reshape(2)
        if obstacle_radius is not None:
            if obstacle_radius <= 0:
                raise ValueError("obstacle_radius must be positive.")
            self.obstacle_radius = float(obstacle_radius)

        self.state = self._sample_start() if start is None else self._validate_point(start, "start")
        self.goal = self._sample_goal() if goal is None else self._validate_point(goal, "goal")
        if self.is_inside_obstacle(self.state):
            raise ValueError("start is inside the obstacle.")
        if self.is_inside_obstacle(self.goal):
            raise ValueError("goal is inside the obstacle.")

        self.steps = 0
        self.trajectory = [self.state.copy()]
        self.actions = []
        return self.observation(), self.goal.copy()

    def observation(self) -> np.ndarray:
        return np.asarray(
            [
                self.state[0],
                self.state[1],
                self.goal[0],
                self.goal[1],
                self.obstacle_center[0],
                self.obstacle_center[1],
                self.obstacle_radius,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> ObstacleStepResult:
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -self.max_action, self.max_action)
        previous = self.state.copy()
        proposed = np.clip(previous + action, self.workspace_low, self.workspace_high).astype(np.float32)

        self.state = proposed
        self.steps += 1
        self.trajectory.append(self.state.copy())
        self.actions.append(action.copy())

        distance = float(np.linalg.norm(self.goal - self.state))
        collision = self.segment_collides(previous, proposed)
        success = distance < self.goal_threshold and not collision
        timeout = self.steps >= self.max_steps
        done = success or collision or timeout
        reward = -distance - (1.0 if collision else 0.0)
        info = {
            "distance": distance,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "steps": self.steps,
        }
        return ObstacleStepResult(self.observation(), reward, done, info)

    def is_inside_obstacle(self, point: np.ndarray, margin: float = 0.0) -> bool:
        point = np.asarray(point, dtype=np.float32).reshape(2)
        return float(np.linalg.norm(point - self.obstacle_center)) <= self.obstacle_radius + margin

    def segment_collides(self, start: np.ndarray, end: np.ndarray, margin: float = 0.0) -> bool:
        start = np.asarray(start, dtype=np.float32).reshape(2)
        end = np.asarray(end, dtype=np.float32).reshape(2)
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-12:
            return self.is_inside_obstacle(end, margin=margin)
        t = float(np.dot(self.obstacle_center - start, segment) / denom)
        t = float(np.clip(t, 0.0, 1.0))
        closest = start + t * segment
        return self.is_inside_obstacle(closest, margin=margin)

    def rollout_open_loop(self, start: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, bool]:
        points = [np.asarray(start, dtype=np.float32).reshape(2)]
        collision = False
        current = points[0].copy()
        for action in np.asarray(actions, dtype=np.float32):
            action = np.clip(action.reshape(2), -self.max_action, self.max_action)
            next_point = np.clip(current + action, self.workspace_low, self.workspace_high).astype(np.float32)
            if self.segment_collides(current, next_point):
                collision = True
            points.append(next_point.copy())
            current = next_point
        return np.asarray(points, dtype=np.float32), collision

    def expert_trajectory(
        self,
        route: str,
        horizon: int,
        noise_std: float = 0.006,
        waypoint_y: float = 0.55,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if route not in {"upper", "lower"}:
            raise ValueError("route must be 'upper' or 'lower'.")
        waypoint = np.asarray([0.0, waypoint_y if route == "upper" else -waypoint_y], dtype=np.float32)
        target_index = 0
        targets = [waypoint, self.goal.copy()]
        states = [self.state.copy()]
        observations = [self.observation()]
        actions = []

        for _ in range(self.max_steps):
            target = targets[target_index]
            if np.linalg.norm(target - self.state) < self.goal_threshold and target_index == 0:
                target_index = 1
                target = targets[target_index]

            action = self._move_toward(target)
            if noise_std > 0:
                action = action + self.rng.normal(0.0, noise_std, size=2).astype(np.float32)
            action = np.clip(action, -self.max_action, self.max_action).astype(np.float32)
            result = self.step(action)
            actions.append(action.copy())
            states.append(self.state.copy())
            observations.append(result.observation.copy())
            if result.done:
                break

        action_array = np.asarray(actions, dtype=np.float32)
        state_array = np.asarray(states, dtype=np.float32)
        obs_array = np.asarray(observations, dtype=np.float32)
        if len(action_array) < horizon:
            pad = np.zeros((horizon - len(action_array), 2), dtype=np.float32)
            action_array = np.concatenate([action_array, pad], axis=0)
        return state_array, obs_array, action_array

    def render_trajectory(
        self,
        trajectory: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        save_path: Optional[str | Path] = None,
        ax: Optional["plt.Axes"] = None,
        title: str = "Obstacle Reaching Trajectory",
        color: str = "#1f77b4",
        label: str = "trajectory",
    ) -> "plt.Axes":
        import matplotlib.pyplot as plt

        points = np.asarray(trajectory if trajectory is not None else self.trajectory, dtype=np.float32)
        target = np.asarray(goal if goal is not None else self.goal, dtype=np.float32).reshape(2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("trajectory must have shape (T, 2).")

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(5, 5))
        assert ax is not None

        obstacle = plt.Circle(
            self.obstacle_center,
            self.obstacle_radius,
            color="#444444",
            alpha=0.22,
            ec="#111111",
            linewidth=1.5,
        )
        ax.add_patch(obstacle)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=label)
        ax.scatter(points[0, 0], points[0, 1], color=color, s=45, alpha=0.85)
        ax.scatter(target[0], target[1], marker="*", color="#d62728", s=120, zorder=3, label="goal")
        ax.set_xlim(self.workspace_low - 0.05, self.workspace_high + 0.05)
        ax.set_ylim(self.workspace_low - 0.05, self.workspace_high + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.tight_layout()
            ax.figure.savefig(save_path, dpi=160)
        if created_fig:
            plt.close(ax.figure)
        return ax

    def _move_toward(self, target: np.ndarray) -> np.ndarray:
        direction = np.asarray(target, dtype=np.float32).reshape(2) - self.state
        distance = float(np.linalg.norm(direction))
        if distance < 1e-8:
            return np.zeros(2, dtype=np.float32)
        return (direction / distance * min(self.max_action, distance)).astype(np.float32)

    def _sample_start(self) -> np.ndarray:
        return np.asarray(
            [
                self.rng.uniform(-0.9, -0.7),
                self.rng.uniform(-0.3, 0.3),
            ],
            dtype=np.float32,
        )

    def _sample_goal(self) -> np.ndarray:
        return np.asarray(
            [
                self.rng.uniform(0.7, 0.9),
                self.rng.uniform(-0.3, 0.3),
            ],
            dtype=np.float32,
        )

    def _validate_point(self, value: np.ndarray, name: str) -> np.ndarray:
        point = np.asarray(value, dtype=np.float32).reshape(2)
        if np.any(point < self.workspace_low) or np.any(point > self.workspace_high):
            raise ValueError(f"{name} must be inside [{self.workspace_low}, {self.workspace_high}].")
        return point.astype(np.float32)
