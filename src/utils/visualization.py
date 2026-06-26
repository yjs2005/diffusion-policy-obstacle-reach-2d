from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(
    history: list[dict],
    save_path: str | Path,
    title: str = "Diffusion Policy Training Loss",
    ylabel: str = "Noise prediction MSE",
) -> None:
    """Save a train/validation loss curve."""

    if not history:
        raise ValueError("history is empty.")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item.get("val_loss") for item in history]

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_loss, label="train loss", linewidth=2)
    if any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label="val loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_eval_trajectories(
    trajectories: Iterable[np.ndarray],
    goals: Iterable[np.ndarray],
    successes: Iterable[bool],
    save_path: str | Path,
    workspace_low: float = -1.0,
    workspace_high: float = 1.0,
    max_trajectories: int = 16,
    title: str = "Closed-loop Diffusion Policy Rollouts",
) -> None:
    """Save a multi-episode rollout plot."""

    trajectories = list(trajectories)[:max_trajectories]
    goals = list(goals)[:max_trajectories]
    successes = list(successes)[:max_trajectories]
    if not trajectories:
        raise ValueError("No trajectories to plot.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    for traj, goal, success in zip(trajectories, goals, successes):
        traj = np.asarray(traj, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        color = "#2ca02c" if success else "#d62728"
        ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.75, linewidth=1.4)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, s=18, alpha=0.7)
        ax.scatter(goal[0], goal[1], marker="*", color=color, s=55, alpha=0.85)

    ax.set_xlim(workspace_low - 0.05, workspace_high + 0.05)
    ax.set_ylim(workspace_low - 0.05, workspace_high + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def save_rollout_gif(
    trajectory: np.ndarray,
    goal: np.ndarray,
    save_path: str | Path,
    workspace_low: float = -1.0,
    workspace_high: float = 1.0,
    fps: int = 8,
) -> None:
    """Render a single rollout as an animated GIF."""

    imageio = None
    pil_image = None
    try:
        import imageio.v2 as imageio
    except ImportError:
        try:
            from PIL import Image as pil_image
        except ImportError as exc:
            raise ImportError(
                "Saving GIF requires imageio or Pillow. Install dependencies from requirements.txt "
                "or run eval.py without --save_gif."
            ) from exc

    trajectory = np.asarray(trajectory, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape (T, 2).")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for idx in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots(figsize=(5, 5))
        prefix = trajectory[:idx]
        ax.plot(prefix[:, 0], prefix[:, 1], color="#1f77b4", linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color="#2ca02c", s=55, label="start")
        ax.scatter(prefix[-1, 0], prefix[-1, 1], color="#1f77b4", s=45, label="robot")
        ax.scatter(goal[0], goal[1], marker="*", color="#d62728", s=120, label="goal")
        ax.set_xlim(workspace_low - 0.05, workspace_high + 0.05)
        ax.set_ylim(workspace_low - 0.05, workspace_high + 0.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"Rollout step {idx - 1}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    if imageio is not None:
        imageio.mimsave(save_path, frames, fps=fps)
    else:
        pil_frames = [pil_image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            save_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )
