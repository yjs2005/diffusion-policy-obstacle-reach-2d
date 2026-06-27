from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the obstacle multi-path reaching experiment.")
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--multi_samples", type=int, default=8)
    parser.add_argument("--obstacle_radius", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    return parser


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def run_command(command: list[str]) -> None:
    print("\n[run_obstacle_experiment] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validate_args(args: argparse.Namespace) -> None:
    for name in ["num_pairs", "horizon", "diffusion_steps", "epochs", "batch_size", "eval_episodes", "multi_samples"]:
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name} must be positive.")
    if args.obstacle_radius <= 0:
        raise SystemExit("--obstacle_radius must be positive.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    if args.quick:
        args.num_pairs = 20
        args.epochs = 1
        args.eval_episodes = 3
        args.multi_samples = 2
        args.batch_size = min(args.batch_size, 32)
        print("[run_obstacle_experiment] Quick mode: 20 pairs, 1 epoch, 3 eval episodes.", flush=True)

    data_path = PROJECT_ROOT / "data" / "obstacle_demos.npz"
    dp_checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_dp"
    bc_best_path = PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_bc_best.pt"
    bc_last_path = PROJECT_ROOT / "outputs" / "checkpoints" / "obstacle_bc_last.pt"
    metrics_path = PROJECT_ROOT / "outputs" / "logs" / "obstacle_eval_metrics.json"

    run_command(
        [
            sys.executable,
            "scripts/generate_obstacle_dataset.py",
            "--num_pairs",
            str(args.num_pairs),
            "--horizon",
            str(args.horizon),
            "--save_path",
            rel(data_path),
            "--obstacle_radius",
            str(args.obstacle_radius),
        ]
    )

    run_command(
        [
            sys.executable,
            "train.py",
            "--data_path",
            rel(data_path),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--diffusion_steps",
            str(args.diffusion_steps),
            "--device",
            args.device,
            "--checkpoint_dir",
            rel(dp_checkpoint_dir),
            "--log_path",
            "outputs/logs/obstacle_train_log.json",
            "--loss_figure_path",
            "outputs/figures/obstacle_training_loss.png",
        ]
    )

    run_command(
        [
            sys.executable,
            "train_bc.py",
            "--data_path",
            rel(data_path),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--checkpoint_path",
            rel(bc_best_path),
            "--last_checkpoint_path",
            rel(bc_last_path),
            "--log_path",
            "outputs/logs/obstacle_bc_train_log.json",
            "--loss_figure_path",
            "outputs/figures/obstacle_bc_training_loss.png",
        ]
    )

    run_command(
        [
            sys.executable,
            "eval_obstacle.py",
            "--dp_checkpoint",
            rel(dp_checkpoint_dir / "best.pt"),
            "--bc_checkpoint",
            rel(bc_best_path),
            "--metrics_path",
            rel(metrics_path),
            "--num_episodes",
            str(args.eval_episodes),
            "--diffusion_steps",
            str(args.diffusion_steps),
            "--multi_samples",
            str(args.multi_samples),
            "--obstacle_radius",
            str(args.obstacle_radius),
            "--device",
            args.device,
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/plot_obstacle_results.py",
            "--dp_checkpoint",
            rel(dp_checkpoint_dir / "best.pt"),
            "--bc_checkpoint",
            rel(bc_best_path),
            "--metrics_path",
            rel(metrics_path),
            "--diffusion_steps",
            str(args.diffusion_steps),
            "--multi_samples",
            str(max(args.multi_samples, 8)),
            "--obstacle_radius",
            str(args.obstacle_radius),
            "--device",
            args.device,
        ]
    )

    print("\n[run_obstacle_experiment] Done.")
    print(f"[run_obstacle_experiment] Metrics: {rel(metrics_path)}")


if __name__ == "__main__":
    main()
