from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.obstacle_reach_env import ObstacleReachEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate obstacle multi-path reaching demonstrations.")
    parser.add_argument("--num_pairs", type=int, default=1000, help="Number of start-goal pairs.")
    parser.add_argument("--horizon", type=int, default=16, help="Future action sequence length.")
    parser.add_argument("--save_path", type=str, default="data/obstacle_demos.npz", help="Output .npz path.")
    parser.add_argument("--obstacle_radius", type=float, default=0.25, help="Circular obstacle radius.")
    parser.add_argument("--noise_std", type=float, default=0.006, help="Expert action noise standard deviation.")
    parser.add_argument("--samples_per_route", type=int, default=8, help="Chunks sampled from each route demo.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--max_action", type=float, default=0.08, help="Maximum action magnitude per axis.")
    return parser


def sample_chunk_starts(actual_len: int, samples_per_route: int) -> np.ndarray:
    if actual_len <= 0:
        return np.zeros(1, dtype=np.int64)
    count = min(samples_per_route, actual_len)
    return np.unique(np.linspace(0, max(actual_len - 1, 0), count, dtype=np.int64))


def make_action_chunk(actions: np.ndarray, start_idx: int, horizon: int) -> np.ndarray:
    chunk = actions[start_idx : start_idx + horizon]
    if len(chunk) < horizon:
        pad = np.zeros((horizon - len(chunk), 2), dtype=np.float32)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def generate_obstacle_dataset(
    num_pairs: int,
    horizon: int,
    save_path: str | Path,
    obstacle_radius: float = 0.25,
    noise_std: float = 0.006,
    samples_per_route: int = 8,
    seed: int = 7,
    max_action: float = 0.08,
) -> None:
    if num_pairs <= 0:
        raise ValueError("--num_pairs must be positive.")
    if horizon <= 0:
        raise ValueError("--horizon must be positive.")
    if obstacle_radius <= 0:
        raise ValueError("--obstacle_radius must be positive.")
    if samples_per_route <= 0:
        raise ValueError("--samples_per_route must be positive.")

    rng = np.random.default_rng(seed)
    env = ObstacleReachEnv(
        max_action=max_action,
        obstacle_radius=obstacle_radius,
        max_steps=100,
        seed=seed,
    )

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    route_labels: list[int] = []
    pair_ids: list[int] = []

    for pair_id in range(num_pairs):
        start = np.asarray([rng.uniform(-0.9, -0.7), rng.uniform(-0.25, 0.25)], dtype=np.float32)
        goal = np.asarray([rng.uniform(0.7, 0.9), rng.uniform(-0.25, 0.25)], dtype=np.float32)
        obstacle_center = np.asarray([0.0, 0.0], dtype=np.float32)

        for route_label, route in enumerate(("upper", "lower")):
            env.reset(start=start, goal=goal, obstacle_center=obstacle_center, obstacle_radius=obstacle_radius)
            states, route_observations, route_actions = env.expert_trajectory(
                route=route,
                horizon=horizon,
                noise_std=noise_std,
            )
            actual_len = max(len(states) - 1, 1)
            for chunk_start in sample_chunk_starts(actual_len, samples_per_route):
                observations.append(route_observations[min(chunk_start, len(route_observations) - 1)])
                actions.append(make_action_chunk(route_actions, int(chunk_start), horizon))
                starts.append(start.copy())
                goals.append(goal.copy())
                route_labels.append(route_label)
                pair_ids.append(pair_id)

    metadata = {
        "task": "obstacle_multi_path_reaching",
        "observation": "[x, y, gx, gy, ox, oy, r]",
        "route_labels": {"upper": 0, "lower": 1},
        "upper_waypoint": [0.0, 0.55],
        "lower_waypoint": [0.0, -0.55],
        "samples_per_route": samples_per_route,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        starts=np.asarray(starts, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        obstacle_center=np.asarray([0.0, 0.0], dtype=np.float32),
        obstacle_radius=np.asarray(obstacle_radius, dtype=np.float32),
        route_labels=np.asarray(route_labels, dtype=np.int64),
        pair_ids=np.asarray(pair_ids, dtype=np.int64),
        horizon=np.asarray(horizon, dtype=np.int64),
        max_action=np.asarray(max_action, dtype=np.float32),
        seed=np.asarray(seed, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"Saved {len(observations)} obstacle action chunks to {save_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        generate_obstacle_dataset(
            num_pairs=args.num_pairs,
            horizon=args.horizon,
            save_path=args.save_path,
            obstacle_radius=args.obstacle_radius,
            noise_std=args.noise_std,
            samples_per_route=args.samples_per_route,
            seed=args.seed,
            max_action=args.max_action,
        )
    except Exception as exc:
        raise SystemExit(f"Obstacle dataset generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
