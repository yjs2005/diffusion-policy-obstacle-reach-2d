from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.scheduler import DDPMScheduler
from src.envs.point_reach_env import PointReachEnv
from src.models.diffusion_policy import DiffusionPolicyMLP
from src.utils.visualization import plot_eval_trajectories, save_rollout_gif


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D Diffusion Policy.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt", help="Checkpoint path.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of closed-loop episodes.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Evaluation device.")
    parser.add_argument("--execute_steps", type=int, default=4, help="Actions to execute from each sampled chunk.")
    parser.add_argument("--max_episode_steps", type=int, default=80, help="Max environment steps per episode.")
    parser.add_argument("--diffusion_steps", type=int, default=None, help="Override diffusion steps for sampling.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default="outputs/figures/eval_trajectories.png",
        help="Rollout figure path.",
    )
    parser.add_argument("--save_gif", action="store_true", help="Save one rollout GIF.")
    parser.add_argument("--gif_path", type=str, default="outputs/figures/rollout.gif", help="GIF save path.")
    parser.add_argument("--metrics_path", type=str, default="outputs/logs/eval_metrics.json", help="Metrics path.")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[DiffusionPolicyMLP, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing model_state_dict or model_config.")

    model = DiffusionPolicyMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def sample_action_chunk(
    model: DiffusionPolicyMLP,
    scheduler: DDPMScheduler,
    observation: np.ndarray,
    goal: np.ndarray,
    action_scale: float,
    device: torch.device,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(device)
    goal_tensor = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
    normalized_actions = scheduler.sample(
        model=model,
        observation=obs_tensor,
        goal=goal_tensor,
        action_shape=(model.horizon, model.action_dim),
        clamp=1.0,
    )
    actions = normalized_actions.squeeze(0).cpu().numpy() * action_scale
    return np.clip(actions, -action_scale, action_scale).astype(np.float32)


def rollout_episode(
    model: DiffusionPolicyMLP,
    scheduler: DDPMScheduler,
    env: PointReachEnv,
    action_scale: float,
    execute_steps: int,
    device: torch.device,
) -> dict:
    observation, goal = env.reset()
    done = False
    info = {"success": False, "distance": float(np.linalg.norm(goal - observation)), "steps": 0}

    while not done:
        action_chunk = sample_action_chunk(model, scheduler, observation, goal, action_scale, device)
        for action in action_chunk[:execute_steps]:
            result = env.step(action)
            observation = result.observation
            done = result.done
            info = result.info
            if done:
                break

    return {
        "trajectory": np.asarray(env.trajectory, dtype=np.float32),
        "goal": goal.copy(),
        "success": bool(info["success"]),
        "final_distance": float(info["distance"]),
        "steps": int(info["steps"]),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_episodes <= 0:
        raise SystemExit("--num_episodes must be positive.")
    if args.execute_steps <= 0:
        raise SystemExit("--execute_steps must be positive.")

    set_seed(args.seed)
    device = resolve_device(args.device)

    try:
        model, checkpoint = load_policy(args.checkpoint, device)
    except Exception as exc:
        raise SystemExit(f"Failed to load checkpoint: {exc}") from exc

    diffusion_steps = int(args.diffusion_steps or checkpoint.get("diffusion_steps", 50))
    action_scale = float(checkpoint.get("action_scale", 0.08))
    execute_steps = min(args.execute_steps, model.horizon)
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps, device=device)
    env = PointReachEnv(max_action=action_scale, max_steps=args.max_episode_steps, seed=args.seed)

    rollouts = []
    for _ in range(args.num_episodes):
        rollouts.append(rollout_episode(model, scheduler, env, action_scale, execute_steps, device))

    successes = [item["success"] for item in rollouts]
    final_distances = [item["final_distance"] for item in rollouts]
    steps = [item["steps"] for item in rollouts]
    metrics = {
        "num_episodes": args.num_episodes,
        "success_rate": float(np.mean(successes)),
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_steps": float(np.mean(steps)),
        "diffusion_steps": diffusion_steps,
        "execute_steps": execute_steps,
        "checkpoint": str(args.checkpoint),
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_eval_trajectories(
        trajectories=[item["trajectory"] for item in rollouts],
        goals=[item["goal"] for item in rollouts],
        successes=successes,
        save_path=args.trajectory_path,
        workspace_low=env.workspace_low,
        workspace_high=env.workspace_high,
    )

    if args.save_gif:
        first = rollouts[0]
        save_rollout_gif(
            trajectory=first["trajectory"],
            goal=first["goal"],
            save_path=args.gif_path,
            workspace_low=env.workspace_low,
            workspace_high=env.workspace_high,
        )

    print(json.dumps(metrics, indent=2))
    print(f"Saved trajectory plot to {args.trajectory_path}")
    if args.save_gif:
        print(f"Saved rollout GIF to {args.gif_path}")


if __name__ == "__main__":
    main()
