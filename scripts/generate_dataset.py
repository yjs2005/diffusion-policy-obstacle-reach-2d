from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.point_reach_env import PointReachEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 2D reaching expert demonstrations.")
    parser.add_argument("--num_demos", type=int, default=1000, help="Number of action chunks to save.")
    parser.add_argument("--horizon", type=int, default=16, help="Future action sequence length.")
    parser.add_argument("--save_path", type=str, default="data/demos.npz", help="Output .npz path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--noise_std", type=float, default=0.008, help="Expert action noise standard deviation.")
    parser.add_argument("--max_action", type=float, default=0.08, help="Maximum action magnitude per axis.")
    return parser


def generate_dataset(
    num_demos: int,
    horizon: int,
    save_path: str | Path,
    seed: int = 42,
    noise_std: float = 0.008,
    max_action: float = 0.08,
) -> None:
    if num_demos <= 0:
        raise ValueError("--num_demos must be positive.")
    if horizon <= 0:
        raise ValueError("--horizon must be positive.")

    rng = np.random.default_rng(seed)
    env = PointReachEnv(max_action=max_action, max_steps=max(horizon + 10, 80), seed=seed)

    observations = np.zeros((num_demos, 2), dtype=np.float32)
    goals = np.zeros((num_demos, 2), dtype=np.float32)
    action_sequences = np.zeros((num_demos, horizon, 2), dtype=np.float32)

    for demo_idx in range(num_demos):
        start = rng.uniform(env.workspace_low, env.workspace_high, size=2).astype(np.float32)
        goal = rng.uniform(env.workspace_low, env.workspace_high, size=2).astype(np.float32)
        observation, goal = env.reset(start=start, goal=goal)

        observations[demo_idx] = observation
        goals[demo_idx] = goal

        for t in range(horizon):
            action = env.expert_action(noise_std=noise_std)
            result = env.step(action)
            action_sequences[demo_idx, t] = action

            if result.done:
                # Pad the remaining chunk with zeros once the target is reached.
                break

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        observation=observations,
        goal=goals,
        action_sequence=action_sequences,
        horizon=np.array(horizon, dtype=np.int64),
        max_action=np.array(max_action, dtype=np.float32),
        seed=np.array(seed, dtype=np.int64),
    )
    print(f"Saved {num_demos} demos to {save_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        generate_dataset(
            num_demos=args.num_demos,
            horizon=args.horizon,
            save_path=args.save_path,
            seed=args.seed,
            noise_std=args.noise_std,
            max_action=args.max_action,
        )
    except Exception as exc:
        raise SystemExit(f"Dataset generation failed: {exc}") from exc


if __name__ == "__main__":
    main()
