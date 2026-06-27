from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bc_policy import BCPolicyMLP
from src.utils.visualization import plot_training_loss
from train import DemoDataset, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MLP Behavior Cloning baseline.")
    parser.add_argument("--data_path", type=str, default="data/demos.npz", help="Path to generated .npz data.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="MLP hidden dimension.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/checkpoints/bc_best.pt",
        help="Best checkpoint path.",
    )
    parser.add_argument(
        "--last_checkpoint_path",
        type=str,
        default="outputs/checkpoints/bc_last.pt",
        help="Last checkpoint path.",
    )
    parser.add_argument("--log_path", type=str, default="outputs/logs/bc_train_log.json", help="JSON log path.")
    parser.add_argument(
        "--loss_figure_path",
        type=str,
        default="outputs/figures/bc_training_loss.png",
        help="BC training loss figure path.",
    )
    return parser


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    iterator = tqdm(dataloader, leave=False, desc="train" if training else "val")
    for batch in iterator:
        observation = batch["observation"].to(device)
        goal = batch["goal"].to(device)
        target_actions = batch["action_sequence"].to(device)
        pred_actions = model(observation, goal)
        loss = torch.nn.functional.mse_loss(pred_actions, target_actions)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        batch_size = target_actions.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        iterator.set_postfix(loss=float(loss.detach().cpu()))

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive.")
    if not (0.0 <= args.val_split < 1.0):
        raise SystemExit("--val_split must be in [0, 1).")

    set_seed(args.seed)
    device = resolve_device(args.device)

    try:
        dataset = DemoDataset(args.data_path)
    except Exception as exc:
        raise SystemExit(f"Failed to load dataset: {exc}") from exc

    horizon = int(dataset.action_sequence.shape[1])
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise SystemExit("Dataset is too small for the requested validation split.")

    split_generator = torch.Generator().manual_seed(args.seed)
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset is not None else None
    )

    model = BCPolicyMLP(
        horizon=horizon,
        action_dim=dataset.action_dim,
        obs_dim=dataset.obs_dim,
        goal_dim=dataset.goal_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_path = Path(args.checkpoint_path)
    last_path = Path(args.last_checkpoint_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device)
        val_loss = run_epoch(model, val_loader, None, device) if val_loader is not None else train_loss

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "action_scale": dataset.action_scale,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "history": history,
        }
        torch.save(checkpoint, last_path)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, best_path)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": str(args.data_path),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": str(device),
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "obs_dim": dataset.obs_dim,
                "goal_dim": dataset.goal_dim,
                "action_dim": dataset.action_dim,
                "action_scale": dataset.action_scale,
                "best_val_loss": best_val,
                "history": history,
            },
            f,
            indent=2,
        )

    plot_training_loss(
        history,
        args.loss_figure_path,
        title="Behavior Cloning Training Loss",
        ylabel="Action sequence MSE",
    )
    print(f"Saved best BC checkpoint to {best_path}")
    print(f"Saved BC training log to {log_path}")
    print(f"Saved BC loss curve to {args.loss_figure_path}")


if __name__ == "__main__":
    main()
