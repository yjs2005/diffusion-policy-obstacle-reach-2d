from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DDPMScheduler:
    """Small DDPM scheduler for action-sequence diffusion."""

    num_train_timesteps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.num_train_timesteps <= 1:
            raise ValueError("num_train_timesteps must be greater than 1.")
        if not (0 < self.beta_start < self.beta_end < 1):
            raise ValueError("Require 0 < beta_start < beta_end < 1.")

        self.device = torch.device(self.device)
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_train_timesteps,
            dtype=torch.float32,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: str | torch.device) -> "DDPMScheduler":
        return DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device,
        )

    def add_noise(
        self,
        clean_actions: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """q(x_t | x_0): add noise to clean action sequences."""

        timesteps = timesteps.long().to(clean_actions.device)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, clean_actions.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean_actions.shape
        )
        return sqrt_alpha * clean_actions + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        clip_sample: bool = True,
    ) -> torch.Tensor:
        """p(x_{t-1} | x_t): one reverse denoising step."""

        if isinstance(timestep, int):
            timestep = torch.full((sample.shape[0],), timestep, device=sample.device, dtype=torch.long)
        else:
            timestep = timestep.long().to(sample.device)

        beta_t = self._extract(self.betas, timestep, sample.shape)
        alpha_t = self._extract(self.alphas, timestep, sample.shape)
        alpha_prod_t = self._extract(self.alphas_cumprod, timestep, sample.shape)
        alpha_prod_t_prev = self._extract(self.alphas_cumprod_prev, timestep, sample.shape)
        sqrt_one_minus_alpha_prod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timestep, sample.shape
        )

        pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * model_output) / torch.sqrt(alpha_prod_t)
        if clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        pred_original_coeff = beta_t * torch.sqrt(alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        current_sample_coeff = torch.sqrt(alpha_t) * (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t)
        mean = pred_original_coeff * pred_original_sample + current_sample_coeff * sample

        nonzero_mask = (timestep != 0).float().view(sample.shape[0], *([1] * (sample.ndim - 1)))
        variance = self._extract(self.posterior_variance, timestep, sample.shape)
        noise = torch.randn(sample.shape, device=sample.device, generator=generator)
        return mean + nonzero_mask * torch.sqrt(torch.clamp(variance, min=1e-20)) * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        observation: torch.Tensor,
        goal: torch.Tensor,
        action_shape: tuple[int, int],
        generator: torch.Generator | None = None,
        clamp: float | None = None,
    ) -> torch.Tensor:
        """Generate action sequences by iterative reverse diffusion."""

        model.eval()
        batch_size = observation.shape[0]
        sample = torch.randn(
            (batch_size, action_shape[0], action_shape[1]),
            device=observation.device,
            generator=generator,
        )

        for t in reversed(range(self.num_train_timesteps)):
            timesteps = torch.full((batch_size,), t, device=observation.device, dtype=torch.long)
            noise_pred = model(sample, observation, goal, timesteps)
            sample = self.step(noise_pred, t, sample, generator=generator)
            if clamp is not None:
                sample = torch.clamp(sample, -clamp, clamp)
        return sample

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        gathered = values.to(timesteps.device).gather(0, timesteps)
        return gathered.view(timesteps.shape[0], *([1] * (len(target_shape) - 1)))
