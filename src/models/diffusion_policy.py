from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding used by the MLP denoiser."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("embedding dimension must be positive.")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps[:, None]

        exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        ) / max(half_dim - 1, 1)
        emb = timesteps[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class DiffusionPolicyMLP(nn.Module):
    """CPU-friendly conditional denoising network for 2D action chunks."""

    def __init__(
        self,
        horizon: int = 16,
        action_dim: int = 2,
        obs_dim: int = 2,
        goal_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        if horizon <= 0 or action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive.")

        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_embed_dim = int(time_embed_dim)

        flat_action_dim = self.horizon * self.action_dim
        input_dim = flat_action_dim + self.obs_dim + self.goal_dim + self.time_embed_dim
        output_dim = flat_action_dim

        self.time_embedding = SinusoidalTimestepEmbedding(self.time_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        observation: torch.Tensor,
        goal: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape (B, horizon, action_dim).")
        batch_size = noisy_actions.shape[0]
        flat_actions = noisy_actions.reshape(batch_size, self.horizon * self.action_dim)
        time_emb = self.time_embedding(timesteps)
        x = torch.cat([flat_actions, observation, goal, time_emb], dim=-1)
        pred = self.net(x)
        return pred.reshape(batch_size, self.horizon, self.action_dim)

    def config(self) -> dict:
        return {
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "goal_dim": self.goal_dim,
            "hidden_dim": self.hidden_dim,
            "time_embed_dim": self.time_embed_dim,
        }
