from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight Diffusion Policy ablations.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32], help="Action horizons to test.")
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="DDPM denoising steps to test.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per configuration.")
    parser.add_argument("--num_demos", type=int, default=1000, help="Generated demos per configuration.")
    parser.add_argument("--num_episodes", type=int, default=30, help="Eval episodes per configuration.")
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training/eval device.")
    parser.add_argument("--execute_steps", type=int, default=4, help="Actions executed per sampled chunk.")
    parser.add_argument("--max_episode_steps", type=int, default=80, help="Maximum env steps per eval episode.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset and training seed.")
    parser.add_argument("--eval_seed", type=int, default=123, help="Evaluation seed.")
    parser.add_argument("--quick", action="store_true", help="Run only the smallest configuration as a smoke test.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="outputs/logs/ablation_results.csv",
        help="CSV summary output path.",
    )
    parser.add_argument(
        "--md_path",
        type=str,
        default="outputs/logs/ablation_results.md",
        help="Markdown summary output path.",
    )
    return parser


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def validate_args(args: argparse.Namespace) -> None:
    if any(h <= 0 for h in args.horizons):
        raise SystemExit("--horizons must contain only positive integers.")
    if any(s <= 1 for s in args.diffusion_steps):
        raise SystemExit("--diffusion_steps must contain integers greater than 1.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.num_demos <= 0:
        raise SystemExit("--num_demos must be positive.")
    if args.num_episodes <= 0:
        raise SystemExit("--num_episodes must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    if args.execute_steps <= 0:
        raise SystemExit("--execute_steps must be positive.")
    if args.max_episode_steps <= 0:
        raise SystemExit("--max_episode_steps must be positive.")


def run_command(command: list[str]) -> None:
    printable = " ".join(command)
    print(f"\n[run_ablation] {printable}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing eval metrics: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict, *names: str) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Missing metric. Expected one of: {', '.join(names)}")


def write_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "horizon",
        "diffusion_steps",
        "num_demos",
        "epochs",
        "num_episodes",
        "success_rate",
        "avg_final_distance",
        "avg_steps",
        "checkpoint",
        "metrics_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_markdown(rows: list[dict], md_path: Path, quick: bool) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "quick smoke test" if quick else "full grid"
    lines = [
        "# Ablation Results",
        "",
        f"Mode: `{mode}`",
        "",
        "| Horizon | Diffusion Steps | Success Rate | Avg Final Distance | Avg Steps | Episodes | Checkpoint |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {horizon} | {diffusion_steps} | {success_rate:.4f} | {avg_final_distance:.4f} | "
            "{avg_steps:.2f} | {num_episodes} | `{checkpoint}` |".format(**row)
        )
    lines.extend(
        [
            "",
            "This ablation compares how action horizon and diffusion denoising steps affect closed-loop reaching success rate and final error.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_config(
    horizon: int,
    diffusion_steps: int,
    args: argparse.Namespace,
) -> dict:
    tag = f"h{horizon}_d{diffusion_steps}"
    data_path = PROJECT_ROOT / "data" / "ablations" / tag / "demos.npz"
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints" / "ablations" / tag
    train_log_path = PROJECT_ROOT / "outputs" / "logs" / "ablations" / f"{tag}_train_log.json"
    loss_figure_path = PROJECT_ROOT / "outputs" / "figures" / "ablations" / f"{tag}_training_loss.png"
    metrics_path = PROJECT_ROOT / "outputs" / "logs" / "ablations" / f"{tag}_eval_metrics.json"
    trajectory_path = PROJECT_ROOT / "outputs" / "figures" / "ablations" / f"{tag}_eval_trajectories.png"

    print(f"\n[run_ablation] Starting config: horizon={horizon}, diffusion_steps={diffusion_steps}", flush=True)

    run_command(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--num_demos",
            str(args.num_demos),
            "--horizon",
            str(horizon),
            "--save_path",
            rel(data_path),
            "--seed",
            str(args.seed + horizon),
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
            str(diffusion_steps),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--checkpoint_dir",
            rel(checkpoint_dir),
            "--log_path",
            rel(train_log_path),
            "--loss_figure_path",
            rel(loss_figure_path),
        ]
    )

    run_command(
        [
            sys.executable,
            "eval.py",
            "--checkpoint",
            rel(checkpoint_dir / "best.pt"),
            "--num_episodes",
            str(args.num_episodes),
            "--device",
            args.device,
            "--execute_steps",
            str(min(args.execute_steps, horizon)),
            "--max_episode_steps",
            str(args.max_episode_steps),
            "--diffusion_steps",
            str(diffusion_steps),
            "--seed",
            str(args.eval_seed),
            "--trajectory_path",
            rel(trajectory_path),
            "--metrics_path",
            rel(metrics_path),
        ]
    )

    metrics = load_metrics(metrics_path)
    return {
        "horizon": horizon,
        "diffusion_steps": diffusion_steps,
        "num_demos": args.num_demos,
        "epochs": args.epochs,
        "num_episodes": int(metrics.get("num_episodes", args.num_episodes)),
        "success_rate": metric_value(metrics, "success_rate"),
        "avg_final_distance": metric_value(metrics, "avg_final_distance", "average_final_distance"),
        "avg_steps": metric_value(metrics, "avg_steps", "average_steps"),
        "checkpoint": rel(checkpoint_dir / "best.pt"),
        "metrics_path": rel(metrics_path),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    horizons = sorted(set(args.horizons))
    diffusion_steps_values = sorted(set(args.diffusion_steps))

    if args.quick:
        horizons = [min(horizons)]
        diffusion_steps_values = [min(diffusion_steps_values)]
        args.epochs = 1
        args.num_demos = 64
        args.num_episodes = 3
        args.batch_size = min(args.batch_size, 32)
        print(
            "[run_ablation] Quick mode: "
            f"horizon={horizons[0]}, diffusion_steps={diffusion_steps_values[0]}, "
            "1 epoch, 64 demos, 3 episodes.",
            flush=True,
        )

    rows: list[dict] = []
    total = len(horizons) * len(diffusion_steps_values)
    done = 0
    for horizon in horizons:
        for diffusion_steps in diffusion_steps_values:
            done += 1
            print(f"\n[run_ablation] Progress {done}/{total}", flush=True)
            rows.append(run_config(horizon, diffusion_steps, args))
            write_csv(rows, PROJECT_ROOT / args.csv_path)
            write_markdown(rows, PROJECT_ROOT / args.md_path, quick=args.quick)

    write_csv(rows, PROJECT_ROOT / args.csv_path)
    write_markdown(rows, PROJECT_ROOT / args.md_path, quick=args.quick)
    print(f"\n[run_ablation] Saved CSV summary to {args.csv_path}", flush=True)
    print(f"[run_ablation] Saved Markdown summary to {args.md_path}", flush=True)


if __name__ == "__main__":
    main()
