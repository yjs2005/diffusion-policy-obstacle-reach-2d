from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval import resolve_device, set_seed
from src.envs.point_reach_env import PointReachEnv
from src.models.bc_policy import BCPolicyMLP
from src.utils.visualization import plot_eval_trajectories, save_rollout_gif


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an MLP Behavior Cloning baseline.")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/bc_best.pt", help="Checkpoint path.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of closed-loop episodes.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Evaluation device.")
    parser.add_argument("--execute_steps", type=int, default=4, help="Actions to execute from each predicted chunk.")
    parser.add_argument("--max_episode_steps", type=int, default=80, help="Max environment steps per episode.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--trajectory_path",
        type=str,
        default="outputs/figures/bc_eval_trajectories.png",
        help="Rollout figure path.",
    )
    parser.add_argument("--save_gif", action="store_true", help="Save one rollout GIF.")
    parser.add_argument("--gif_path", type=str, default="outputs/figures/bc_rollout.gif", help="GIF save path.")
    parser.add_argument("--metrics_path", type=str, default="outputs/logs/bc_eval_metrics.json", help="Metrics path.")
    return parser


def load_policy(checkpoint_path: str | Path, device: torch.device) -> tuple[BCPolicyMLP, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError("Invalid checkpoint: missing model_state_dict or model_config.")

    model = BCPolicyMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_action_chunk(
    model: BCPolicyMLP,
    observation: np.ndarray,
    goal: np.ndarray,
    action_scale: float,
    device: torch.device,
) -> np.ndarray:
    obs_tensor = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(device)
    goal_tensor = torch.from_numpy(goal.astype(np.float32)).unsqueeze(0).to(device)
    normalized_actions = model(obs_tensor, goal_tensor)
    actions = normalized_actions.squeeze(0).cpu().numpy() * action_scale
    return np.clip(actions, -action_scale, action_scale).astype(np.float32)


def rollout_episode(
    model: BCPolicyMLP,
    env: PointReachEnv,
    action_scale: float,
    execute_steps: int,
    device: torch.device,
) -> dict:
    observation, goal = env.reset()
    done = False
    info = {"success": False, "distance": float(np.linalg.norm(goal - observation)), "steps": 0}

    while not done:
        action_chunk = predict_action_chunk(model, observation, goal, action_scale, device)
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

    action_scale = float(checkpoint.get("action_scale", 0.08))
    execute_steps = min(args.execute_steps, model.horizon)
    env = PointReachEnv(max_action=action_scale, max_steps=args.max_episode_steps, seed=args.seed)

    rollouts = []
    for _ in range(args.num_episodes):
        rollouts.append(rollout_episode(model, env, action_scale, execute_steps, device))

    successes = [item["success"] for item in rollouts]
    final_distances = [item["final_distance"] for item in rollouts]
    steps = [item["steps"] for item in rollouts]
    metrics = {
        "num_episodes": args.num_episodes,
        "success_rate": float(np.mean(successes)),
        "avg_final_distance": float(np.mean(final_distances)),
        "avg_steps": float(np.mean(steps)),
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
        title="Closed-loop Behavior Cloning Rollouts",
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
    print(f"Saved BC trajectory plot to {args.trajectory_path}")
    if args.save_gif:
        print(f"Saved BC rollout GIF to {args.gif_path}")


if __name__ == "__main__":
    main()
