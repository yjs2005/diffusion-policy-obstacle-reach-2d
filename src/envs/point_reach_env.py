from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict


class PointReachEnv:
    """A minimal 2D point reaching environment.

    State: current point position (x, y)
    Goal: target point position (gx, gy)
    Action: displacement command (dx, dy), clipped to max_action
    """

    def __init__(
        self,
        workspace_low: float = -1.0,
        workspace_high: float = 1.0,
        max_action: float = 0.08,
        goal_threshold: float = 0.05,
        max_steps: int = 80,
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

        self.workspace_low = float(workspace_low)
        self.workspace_high = float(workspace_high)
        self.max_action = float(max_action)
        self.goal_threshold = float(goal_threshold)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.trajectory: list[np.ndarray] = []

    def reset(
        self,
        start: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset the environment and return observation and goal."""

        self.state = self._sample_or_validate_point(start, "start")
        self.goal = self._sample_or_validate_point(goal, "goal")

        # Avoid trivially solved episodes when sampling both points.
        retries = 0
        while np.linalg.norm(self.goal - self.state) < self.goal_threshold * 2 and retries < 100:
            self.goal = self._sample_or_validate_point(None, "goal")
            retries += 1

        self.steps = 0
        self.trajectory = [self.state.copy()]
        return self.state.copy(), self.goal.copy()

    def step(self, action: np.ndarray) -> StepResult:
        """Apply one action and return the transition result."""

        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -self.max_action, self.max_action)

        self.state = self.state + action
        self.state = np.clip(self.state, self.workspace_low, self.workspace_high).astype(np.float32)
        self.steps += 1
        self.trajectory.append(self.state.copy())

        distance = float(np.linalg.norm(self.goal - self.state))
        success = distance < self.goal_threshold
        timeout = self.steps >= self.max_steps
        done = success or timeout
        reward = -distance
        info = {"distance": distance, "success": success, "timeout": timeout, "steps": self.steps}
        return StepResult(self.state.copy(), reward, done, info)

    def render_trajectory(
        self,
        trajectory: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        save_path: Optional[str | Path] = None,
        ax: Optional[plt.Axes] = None,
        title: str = "Point Reach Trajectory",
    ) -> plt.Axes:
        """Render a trajectory to an existing axes or save it as an image."""

        import matplotlib.pyplot as plt

        points = np.asarray(trajectory if trajectory is not None else self.trajectory, dtype=np.float32)
        target = np.asarray(goal if goal is not None else self.goal, dtype=np.float32).reshape(2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("trajectory must have shape (T, 2).")

        created_fig = ax is None
        if created_fig:
            _, ax = plt.subplots(figsize=(5, 5))

        assert ax is not None
        ax.plot(points[:, 0], points[:, 1], "-o", markersize=2.5, linewidth=1.5, label="trajectory")
        ax.scatter(points[0, 0], points[0, 1], c="#2ca02c", s=55, label="start", zorder=3)
        ax.scatter(target[0], target[1], c="#d62728", marker="*", s=120, label="goal", zorder=3)
        ax.set_xlim(self.workspace_low - 0.05, self.workspace_high + 0.05)
        ax.set_ylim(self.workspace_low - 0.05, self.workspace_high + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            ax.figure.tight_layout()
            ax.figure.savefig(save_path, dpi=160)
        if created_fig:
            plt.close(ax.figure)
        return ax

    def expert_action(self, noise_std: float = 0.01) -> np.ndarray:
        """Move toward the current goal with optional Gaussian exploration noise."""

        direction = self.goal - self.state
        distance = float(np.linalg.norm(direction))
        if distance < 1e-8:
            action = np.zeros(2, dtype=np.float32)
        else:
            action = direction / distance * min(self.max_action, distance)
        if noise_std > 0:
            action = action + self.rng.normal(0.0, noise_std, size=2).astype(np.float32)
        return np.clip(action, -self.max_action, self.max_action).astype(np.float32)

    def _sample_or_validate_point(self, value: Optional[np.ndarray], name: str) -> np.ndarray:
        if value is None:
            return self.rng.uniform(self.workspace_low, self.workspace_high, size=2).astype(np.float32)

        point = np.asarray(value, dtype=np.float32).reshape(2)
        if np.any(point < self.workspace_low) or np.any(point > self.workspace_high):
            raise ValueError(f"{name} must be inside [{self.workspace_low}, {self.workspace_high}].")
        return point.astype(np.float32)
