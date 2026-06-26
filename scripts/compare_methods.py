from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Diffusion Policy and BC evaluation metrics.")
    parser.add_argument(
        "--diffusion_metrics",
        type=str,
        default="outputs/logs/eval_metrics.json",
        help="Diffusion Policy eval metrics JSON.",
    )
    parser.add_argument(
        "--bc_metrics",
        type=str,
        default="outputs/logs/bc_eval_metrics.json",
        help="Behavior Cloning eval metrics JSON.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/logs/compare_results.md",
        help="Markdown comparison table path.",
    )
    return parser


def load_metrics(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: dict, *names: str) -> float:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    raise KeyError(f"Missing metric. Expected one of: {', '.join(names)}")


def build_table(rows: list[dict]) -> str:
    lines = [
        "| Method | Success Rate | Avg Final Distance | Avg Steps | Episodes | Checkpoint |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {success_rate:.4f} | {avg_final_distance:.4f} | "
            "{avg_steps:.2f} | {num_episodes} | `{checkpoint}` |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        diffusion = load_metrics(args.diffusion_metrics)
        bc = load_metrics(args.bc_metrics)
    except Exception as exc:
        raise SystemExit(f"Failed to load metrics: {exc}") from exc

    try:
        rows = [
            {
                "method": "Diffusion Policy",
                "success_rate": metric_value(diffusion, "success_rate"),
                "avg_final_distance": metric_value(diffusion, "avg_final_distance", "average_final_distance"),
                "avg_steps": metric_value(diffusion, "avg_steps", "average_steps"),
                "num_episodes": int(diffusion.get("num_episodes", 0)),
                "checkpoint": diffusion.get("checkpoint", ""),
            },
            {
                "method": "MLP BC",
                "success_rate": metric_value(bc, "success_rate"),
                "avg_final_distance": metric_value(bc, "avg_final_distance", "average_final_distance"),
                "avg_steps": metric_value(bc, "avg_steps", "average_steps"),
                "num_episodes": int(bc.get("num_episodes", 0)),
                "checkpoint": bc.get("checkpoint", ""),
            },
        ]
    except Exception as exc:
        raise SystemExit(f"Failed to build comparison table: {exc}") from exc

    table = build_table(rows)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(table, encoding="utf-8")

    print(table)
    print(f"Saved comparison table to {save_path}")


if __name__ == "__main__":
    main()
