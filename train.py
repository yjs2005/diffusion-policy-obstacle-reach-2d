from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.scheduler import DDPMScheduler
from src.models.diffusion_policy import DiffusionPolicyMLP
from src.utils.visualization import plot_training_loss


class DemoDataset(Dataset):
    def __init__(self, data_path: str | Path, action_scale: float | None = None) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        data = np.load(data_path)
        required = {"observation", "goal", "action_sequence"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Dataset is missing keys: {sorted(missing)}")

        self.observation = data["observation"].astype(np.float32)
        self.goal = data["goal"].astype(np.float32)
        self.action_sequence = data["action_sequence"].astype(np.float32)
        if self.observation.ndim != 2 or self.observation.shape[1] != 2:
            raise ValueError("observation must have shape (N, 2).")
        if self.goal.shape != self.observation.shape:
            raise ValueError("goal must have shape (N, 2).")
        if self.action_sequence.ndim != 3 or self.action_sequence.shape[-1] != 2:
            raise ValueError("action_sequence must have shape (N, horizon, 2).")
        if len(self.observation) != len(self.action_sequence):
            raise ValueError("observation and action_sequence must have the same length.")

        stored_scale = float(data["max_action"]) if "max_action" in data.files else 0.08
        self.action_scale = float(action_scale) if action_scale is not None else stored_scale
        if self.action_scale <= 0:
            raise ValueError("action_scale must be positive.")

        self.action_sequence = np.clip(self.action_sequence / self.action_scale, -1.0, 1.0)

    def __len__(self) -> int:
        return len(self.observation)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": torch.from_numpy(self.observation[index]),
            "goal": torch.from_numpy(self.goal[index]),
            "action_sequence": torch.from_numpy(self.action_sequence[index]),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight 2D Diffusion Policy.")
    parser.add_argument("--data_path", type=str, default="data/demos.npz", help="Path to generated .npz data.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--diffusion_steps", type=int, default=50, help="Number of DDPM timesteps.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training device.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="MLP hidden dimension.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints", help="Checkpoint directory.")
    parser.add_argument("--log_path", type=str, default="outputs/logs/train_log.json", help="JSON log path.")
    parser.add_argument(
        "--loss_figure_path",
        type=str,
        default="outputs/figures/training_loss.png",
        help="Training loss figure path.",
    )
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


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    scheduler: DDPMScheduler,
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
        clean_actions = batch["action_sequence"].to(device)
        batch_size = clean_actions.shape[0]

        timesteps = torch.randint(
            low=0,
            high=scheduler.num_train_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_actions)
        noisy_actions = scheduler.add_noise(clean_actions, noise, timesteps)
        noise_pred = model(noisy_actions, observation, goal, timesteps)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

    model = DiffusionPolicyMLP(horizon=horizon, hidden_dim=args.hidden_dim).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    best_val = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, scheduler, optimizer, device)
        val_loss = run_epoch(model, val_loader, scheduler, None, device) if val_loader is not None else train_loss

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(f"Epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model.config(),
            "diffusion_steps": args.diffusion_steps,
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
                "diffusion_steps": args.diffusion_steps,
                "device": str(device),
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "action_scale": dataset.action_scale,
                "best_val_loss": best_val,
                "history": history,
            },
            f,
            indent=2,
        )

    plot_training_loss(history, args.loss_figure_path)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved training log to {log_path}")
    print(f"Saved loss curve to {args.loss_figure_path}")


if __name__ == "__main__":
    main()
