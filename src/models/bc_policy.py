from __future__ import annotations

import torch
from torch import nn


class BCPolicyMLP(nn.Module):
    """Behavior Cloning baseline that predicts a full action chunk."""

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 2,
        obs_dim: int = 2,
        goal_dim: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if horizon <= 0 or action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive.")

        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)

        output_dim = self.horizon * self.action_dim
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim + self.goal_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or goal.ndim != 2:
            raise ValueError("observation and goal must have shape (B, dim).")
        batch_size = observation.shape[0]
        x = torch.cat([observation, goal], dim=-1)
        actions = self.net(x)
        return actions.reshape(batch_size, self.horizon, self.action_dim)

    def config(self) -> dict:
        return {
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "hidden_dim": self.hidden_dim,
        }
