from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.scheduler import DDPMScheduler
from src.envs.obstacle_reach_env import ObstacleReachEnv
from src.models.bc_policy import BCPolicyMLP
from src.models.diffusion_policy import DiffusionPolicyMLP


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate obstacle multi-path reaching policies.")
    parser.add_argument("--dp_checkpoint", type=str, default="outputs/checkpoints/obstacle_dp/best.pt")
    parser.add_argument("--bc_checkpoint", type=str, default="outputs/checkpoints/obstacle_bc_best.pt")
    parser.add_argument("--metrics_path", type=str, default="outputs/logs/obstacle_eval_metrics.json")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--multi_samples", type=int, default=8)
    parser.add_argument("--execute_steps", type=int, default=4)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--obstacle_radius", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=321)
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


def load_torch_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_dp_policy(path: str | Path, device: torch.device) -> tuple[DiffusionPolicyMLP, dict]:
    checkpoint = load_torch_checkpoint(path, device)
    model = DiffusionPolicyMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_bc_policy(path: str | Path, device: torch.device) -> tuple[BCPolicyMLP, dict]:
    checkpoint = load_torch_checkpoint(path, device)
    model = BCPolicyMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def make_eval_cases(num_episodes: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(num_episodes):
        start = np.asarray([rng.uniform(-0.9, -0.7), rng.uniform(-0.25, 0.25)], dtype=np.float32)
        goal = np.asarray([rng.uniform(0.7, 0.9), rng.uniform(-0.25, 0.25)], dtype=np.float32)
        cases.append((start, goal))
    return cases


def empty_goal(batch_size: int, goal_dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch_size, goal_dim), dtype=torch.float32, device=device)


@torch.no_grad()
def predict_bc_chunk(
    model: BCPolicyMLP,
    observation: np.ndarray,
    action_scale: float,
    device: torch.device,
) -> np.ndarray:
    obs = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(device)
    goal = empty_goal(1, model.goal_dim, device)
    normalized = model(obs, goal)
    return np.clip(normalized.squeeze(0).cpu().numpy() * action_scale, -action_scale, action_scale)


@torch.no_grad()
def sample_dp_chunks(
    model: DiffusionPolicyMLP,
    scheduler: DDPMScheduler,
    observation: np.ndarray,
    action_scale: float,
    device: torch.device,
    num_samples: int = 1,
) -> np.ndarray:
    obs = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).repeat(num_samples, 1).to(device)
    goal = empty_goal(num_samples, model.goal_dim, device)
    normalized = scheduler.sample(
        model=model,
        observation=obs,
        goal=goal,
        action_shape=(model.horizon, model.action_dim),
        clamp=1.0,
    )
    actions = normalized.cpu().numpy() * action_scale
    return np.clip(actions, -action_scale, action_scale).astype(np.float32)


def action_diversity(chunks: np.ndarray) -> float:
    if len(chunks) <= 1:
        return 0.0
    flat = chunks.reshape(len(chunks), -1)
    distances = [float(np.linalg.norm(flat[i] - flat[j]) / np.sqrt(flat.shape[1])) for i, j in combinations(range(len(flat)), 2)]
    return float(np.mean(distances)) if distances else 0.0


def candidate_score(env: ObstacleReachEnv, start: np.ndarray, goal: np.ndarray, actions: np.ndarray) -> float:
    points, collision = env.rollout_open_loop(start, actions)
    final_distance = float(np.linalg.norm(points[-1] - goal))
    path_length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    collision_penalty = 10.0 if collision else 0.0
    return collision_penalty + final_distance + 0.02 * path_length


def select_safe_chunk(env: ObstacleReachEnv, observation: np.ndarray, chunks: np.ndarray) -> np.ndarray:
    start = observation[:2]
    goal = observation[2:4]
    scores = [candidate_score(env, start, goal, chunk) for chunk in chunks]
    return chunks[int(np.argmin(scores))]


def path_length(points: np.ndarray) -> float:
    if len(points) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def smoothness(actions: np.ndarray) -> float:
    if len(actions) <= 1:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))


def route_mode(points: np.ndarray, obstacle_center: np.ndarray) -> str:
    points = np.asarray(points, dtype=np.float32)
    near = points[np.abs(points[:, 0] - obstacle_center[0]) < 0.35]
    y_value = float(np.mean(near[:, 1])) if len(near) else float(np.mean(points[:, 1]))
    return "upper" if y_value >= obstacle_center[1] else "lower"


def rollout_policy(
    method: str,
    start: np.ndarray,
    goal: np.ndarray,
    env: ObstacleReachEnv,
    device: torch.device,
    bc_model: BCPolicyMLP | None = None,
    dp_model: DiffusionPolicyMLP | None = None,
    scheduler: DDPMScheduler | None = None,
    action_scale: float = 0.08,
    execute_steps: int = 4,
    multi_samples: int = 8,
) -> dict:
    observation, _ = env.reset(start=start, goal=goal)
    done = False
    info = {"success": False, "collision": False, "distance": float(np.linalg.norm(goal - start)), "steps": 0}
    diversities = []

    while not done:
        if method == "bc":
            assert bc_model is not None
            action_chunk = predict_bc_chunk(bc_model, observation, action_scale, device)
        else:
            assert dp_model is not None and scheduler is not None
            sample_count = multi_samples if method == "dp_multi" else 1
            chunks = sample_dp_chunks(dp_model, scheduler, observation, action_scale, device, num_samples=sample_count)
            if sample_count > 1:
                diversities.append(action_diversity(chunks))
            action_chunk = select_safe_chunk(env, observation, chunks) if method == "dp_multi" else chunks[0]

        for action in action_chunk[:execute_steps]:
            result = env.step(action)
            observation = result.observation
            done = result.done
            info = result.info
            if done:
                break

    trajectory = np.asarray(env.trajectory, dtype=np.float32)
    executed_actions = np.asarray(env.actions, dtype=np.float32)
    mode = route_mode(trajectory, env.obstacle_center)
    return {
        "trajectory": trajectory,
        "actions": executed_actions,
        "success": bool(info["success"]),
        "collision": bool(info["collision"]),
        "final_distance": float(info["distance"]),
        "steps": int(info["steps"]),
        "path_length": path_length(trajectory),
        "smoothness": smoothness(executed_actions),
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "route_mode": mode,
    }


def summarize_rollouts(rollouts: list[dict]) -> dict:
    if not rollouts:
        raise ValueError("No rollouts to summarize.")
    upper_count = sum(item["route_mode"] == "upper" for item in rollouts)
    lower_count = sum(item["route_mode"] == "lower" for item in rollouts)
    total = len(rollouts)
    return {
        "success_rate": float(np.mean([item["success"] for item in rollouts])),
        "collision_rate": float(np.mean([item["collision"] for item in rollouts])),
        "avg_final_distance": float(np.mean([item["final_distance"] for item in rollouts])),
        "avg_path_length": float(np.mean([item["path_length"] for item in rollouts])),
        "avg_smoothness": float(np.mean([item["smoothness"] for item in rollouts])),
        "diversity": float(np.mean([item["diversity"] for item in rollouts])),
        "upper_route_ratio": float(upper_count / total),
        "lower_route_ratio": float(lower_count / total),
    }


def evaluate_obstacle(
    dp_checkpoint: str | Path,
    bc_checkpoint: str | Path,
    num_episodes: int = 50,
    diffusion_steps: int | None = None,
    multi_samples: int = 8,
    execute_steps: int = 4,
    max_episode_steps: int = 100,
    obstacle_radius: float = 0.25,
    device: torch.device | str = "cpu",
    seed: int = 321,
) -> tuple[dict, dict[str, list[dict]]]:
    device = torch.device(device)
    bc_model, bc_ckpt = load_bc_policy(bc_checkpoint, device)
    dp_model, dp_ckpt = load_dp_policy(dp_checkpoint, device)
    action_scale = float(dp_ckpt.get("action_scale", bc_ckpt.get("action_scale", 0.08)))
    diffusion_steps = int(diffusion_steps or dp_ckpt.get("diffusion_steps", 50))
    scheduler = DDPMScheduler(num_train_timesteps=diffusion_steps, device=device)
    cases = make_eval_cases(num_episodes, seed)

    all_rollouts: dict[str, list[dict]] = {"bc": [], "dp_single": [], "dp_multi": []}
    for method in all_rollouts:
        for start, goal in cases:
            env = ObstacleReachEnv(
                max_action=action_scale,
                max_steps=max_episode_steps,
                obstacle_radius=obstacle_radius,
                seed=seed,
            )
            all_rollouts[method].append(
                rollout_policy(
                    method=method,
                    start=start,
                    goal=goal,
                    env=env,
                    device=device,
                    bc_model=bc_model,
                    dp_model=dp_model,
                    scheduler=scheduler,
                    action_scale=action_scale,
                    execute_steps=execute_steps,
                    multi_samples=multi_samples,
                )
            )

    metrics = {
        "num_episodes": num_episodes,
        "diffusion_steps": diffusion_steps,
        "multi_samples": multi_samples,
        "execute_steps": execute_steps,
        "obstacle_radius": obstacle_radius,
        "dp_checkpoint": str(dp_checkpoint),
        "bc_checkpoint": str(bc_checkpoint),
        "methods": {method: summarize_rollouts(rollouts) for method, rollouts in all_rollouts.items()},
    }
    return metrics, all_rollouts


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_episodes <= 0:
        raise SystemExit("--num_episodes must be positive.")
    if args.multi_samples <= 0:
        raise SystemExit("--multi_samples must be positive.")
    if args.execute_steps <= 0:
        raise SystemExit("--execute_steps must be positive.")

    set_seed(args.seed)
    device = resolve_device(args.device)
    try:
        metrics, _ = evaluate_obstacle(
            dp_checkpoint=args.dp_checkpoint,
            bc_checkpoint=args.bc_checkpoint,
            num_episodes=args.num_episodes,
            diffusion_steps=args.diffusion_steps,
            multi_samples=args.multi_samples,
            execute_steps=args.execute_steps,
            max_episode_steps=args.max_episode_steps,
            obstacle_radius=args.obstacle_radius,
            device=device,
            seed=args.seed,
        )
    except Exception as exc:
        raise SystemExit(f"Obstacle evaluation failed: {exc}") from exc

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved obstacle metrics to {metrics_path}")


if __name__ == "__main__":
    main()
