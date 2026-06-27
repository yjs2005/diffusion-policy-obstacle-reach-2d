from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_obstacle import (
    DDPMScheduler,
    load_bc_policy,
    load_dp_policy,
    rollout_policy,
    sample_dp_chunks,
)
from src.envs.obstacle_reach_env import ObstacleReachEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot obstacle multi-path experiment results.")
    parser.add_argument("--dp_checkpoint", type=str, default="outputs/checkpoints/obstacle_dp/best.pt")
    parser.add_argument("--bc_checkpoint", type=str, default="outputs/checkpoints/obstacle_bc_best.pt")
    parser.add_argument("--metrics_path", type=str, default="outputs/logs/obstacle_eval_metrics.json")
    parser.add_argument("--fig_dir", type=str, default="outputs/figures")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--multi_samples", type=int, default=16)
    parser.add_argument("--obstacle_radius", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=321)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def fixed_case() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([-0.82, 0.0], dtype=np.float32), np.asarray([0.82, 0.0], dtype=np.float32)


def setup_axes(ax: plt.Axes, env: ObstacleReachEnv, title: str) -> None:
    obstacle = plt.Circle(
        env.obstacle_center,
        env.obstacle_radius,
        color="#444444",
        alpha=0.22,
        ec="#111111",
        linewidth=1.5,
    )
    ax.add_patch(obstacle)
    ax.set_xlim(env.workspace_low - 0.05, env.workspace_high + 0.05)
    ax.set_ylim(env.workspace_low - 0.05, env.workspace_high + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_expert_modes(fig_dir: Path, obstacle_radius: float) -> None:
    start, goal = fixed_case()
    env = ObstacleReachEnv(obstacle_radius=obstacle_radius, seed=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    setup_axes(ax, env, "Expert Demonstrations: Upper vs Lower Route")

    for route, color in [("upper", "#1f77b4"), ("lower", "#ff7f0e")]:
        env.reset(start=start, goal=goal)
        states, _, _ = env.expert_trajectory(route=route, horizon=16, noise_std=0.0)
        ax.plot(states[:, 0], states[:, 1], color=color, linewidth=2.5, label=f"{route} expert")

    ax.scatter(start[0], start[1], color="#2ca02c", s=70, label="start", zorder=3)
    ax.scatter(goal[0], goal[1], color="#d62728", marker="*", s=140, label="goal", zorder=3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "obstacle_expert_modes.png", dpi=170)
    plt.close(fig)


def load_models(dp_checkpoint: str, bc_checkpoint: str, device: torch.device):
    dp_model, dp_ckpt = load_dp_policy(dp_checkpoint, device)
    bc_model, bc_ckpt = load_bc_policy(bc_checkpoint, device)
    action_scale = float(dp_ckpt.get("action_scale", bc_ckpt.get("action_scale", 0.08)))
    diffusion_steps = int(dp_ckpt.get("diffusion_steps", 50))
    return dp_model, dp_ckpt, bc_model, bc_ckpt, action_scale, diffusion_steps


def plot_bc_vs_dp(
    fig_dir: Path,
    dp_model,
    bc_model,
    scheduler: DDPMScheduler,
    action_scale: float,
    device: torch.device,
    obstacle_radius: float,
    multi_samples: int,
    seed: int,
) -> dict[str, dict]:
    start, goal = fixed_case()
    methods = [
        ("bc", "BC", "#d62728"),
        ("dp_single", "DP single", "#1f77b4"),
        ("dp_multi", "DP multi-sample", "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    env_for_axes = ObstacleReachEnv(obstacle_radius=obstacle_radius, seed=seed)
    setup_axes(ax, env_for_axes, "BC vs Diffusion Policy on Obstacle Reaching")
    rollouts = {}

    for method, label, color in methods:
        env = ObstacleReachEnv(obstacle_radius=obstacle_radius, seed=seed)
        rollout = rollout_policy(
            method=method,
            start=start,
            goal=goal,
            env=env,
            device=device,
            bc_model=bc_model,
            dp_model=dp_model,
            scheduler=scheduler,
            action_scale=action_scale,
            execute_steps=4,
            multi_samples=multi_samples,
        )
        rollouts[method] = rollout
        traj = rollout["trajectory"]
        suffix = "success" if rollout["success"] else ("collision" if rollout["collision"] else "timeout")
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.2, label=f"{label} ({suffix})")

    ax.scatter(start[0], start[1], color="#2ca02c", s=65, zorder=3)
    ax.scatter(goal[0], goal[1], color="#d62728", marker="*", s=140, zorder=3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "bc_vs_dp_obstacle_rollouts.png", dpi=170)
    plt.close(fig)
    return rollouts


def plot_dp_multisample(
    fig_dir: Path,
    dp_model,
    scheduler: DDPMScheduler,
    action_scale: float,
    device: torch.device,
    obstacle_radius: float,
    multi_samples: int,
    seed: int,
) -> None:
    start, goal = fixed_case()
    env = ObstacleReachEnv(obstacle_radius=obstacle_radius, seed=seed)
    observation, _ = env.reset(start=start, goal=goal)
    chunks = sample_dp_chunks(dp_model, scheduler, observation, action_scale, device, num_samples=multi_samples)

    fig, ax = plt.subplots(figsize=(6, 6))
    setup_axes(ax, env, "Diffusion Policy Multi-sample Diversity")
    for idx, chunk in enumerate(chunks):
        points, collision = env.rollout_open_loop(start, chunk)
        color = "#d62728" if collision else "#1f77b4"
        ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.45, linewidth=1.5)
        if idx == 0:
            ax.plot([], [], color="#1f77b4", alpha=0.7, label="candidate chunk")
            ax.plot([], [], color="#d62728", alpha=0.7, label="colliding candidate")

    ax.scatter(start[0], start[1], color="#2ca02c", s=65, zorder=3, label="start")
    ax.scatter(goal[0], goal[1], color="#d62728", marker="*", s=140, zorder=3, label="goal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "dp_multisample_diversity.png", dpi=170)
    plt.close(fig)


def plot_metrics_bar(fig_dir: Path, metrics_path: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    methods = [
        ("bc", "BC"),
        ("dp_single", "DP single"),
        ("dp_multi", "DP multi"),
    ]
    metric_names = ["success_rate", "collision_rate", "avg_final_distance"]
    labels = ["Success", "Collision", "Final distance"]
    values = np.asarray(
        [[metrics["methods"][method][name] for name in metric_names] for method, _ in methods],
        dtype=np.float32,
    )

    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for idx, (_, label) in enumerate(methods):
        ax.bar(x + (idx - 1) * width, values[idx], width=width, label=label, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(1.05, float(values.max()) * 1.15))
    ax.set_title("Obstacle Multi-path Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "obstacle_metrics_bar.png", dpi=170)
    plt.close(fig)


def save_rollout_gif(fig_dir: Path, trajectory: np.ndarray, obstacle_radius: float, fps: int = 8) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
        from PIL import Image

    start = trajectory[0]
    goal = fixed_case()[1]
    env = ObstacleReachEnv(obstacle_radius=obstacle_radius)
    frames = []
    for idx in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots(figsize=(5, 5))
        setup_axes(ax, env, f"Obstacle rollout step {idx - 1}")
        prefix = trajectory[:idx]
        ax.plot(prefix[:, 0], prefix[:, 1], color="#2ca02c", linewidth=2.2)
        ax.scatter(start[0], start[1], color="#2ca02c", s=65, label="start")
        ax.scatter(prefix[-1, 0], prefix[-1, 1], color="#1f77b4", s=55, label="robot")
        ax.scatter(goal[0], goal[1], color="#d62728", marker="*", s=135, label="goal")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    gif_path = fig_dir / "obstacle_rollout.gif"
    if imageio is not None:
        imageio.mimsave(gif_path, frames, fps=fps)
    else:
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / max(fps, 1)),
            loop=0,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    device = resolve_device(args.device)
    fig_dir = PROJECT_ROOT / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    dp_model, dp_ckpt, bc_model, _, action_scale, ckpt_steps = load_models(
        args.dp_checkpoint,
        args.bc_checkpoint,
        device,
    )
    diffusion_steps = int(args.diffusion_steps or ckpt_steps)
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps, device=device)

    plot_expert_modes(fig_dir, obstacle_radius=args.obstacle_radius)
    rollouts = plot_bc_vs_dp(
        fig_dir,
        dp_model=dp_model,
        bc_model=bc_model,
        scheduler=scheduler,
        action_scale=action_scale,
        device=device,
        obstacle_radius=args.obstacle_radius,
        multi_samples=max(args.multi_samples, 2),
        seed=args.seed,
    )
    plot_dp_multisample(
        fig_dir,
        dp_model=dp_model,
        scheduler=scheduler,
        action_scale=action_scale,
        device=device,
        obstacle_radius=args.obstacle_radius,
        multi_samples=args.multi_samples,
        seed=args.seed,
    )
    plot_metrics_bar(fig_dir, PROJECT_ROOT / args.metrics_path)
    save_rollout_gif(fig_dir, rollouts["dp_multi"]["trajectory"], obstacle_radius=args.obstacle_radius)
    print(f"Saved obstacle figures to {fig_dir}")


if __name__ == "__main__":
    main()
