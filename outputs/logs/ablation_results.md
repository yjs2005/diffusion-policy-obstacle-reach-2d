# Ablation Results

Mode: `full grid`

| Horizon | Diffusion Steps | Success Rate | Avg Final Distance | Avg Steps | Episodes | Checkpoint |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 20 | 0.1333 | 0.8848 | 75.53 | 30 | `outputs/checkpoints/ablations/h8_d20/best.pt` |
| 8 | 50 | 0.1000 | 0.5001 | 78.70 | 30 | `outputs/checkpoints/ablations/h8_d50/best.pt` |
| 8 | 100 | 0.2000 | 0.2675 | 73.57 | 30 | `outputs/checkpoints/ablations/h8_d100/best.pt` |
| 16 | 20 | 0.0333 | 0.9760 | 79.03 | 30 | `outputs/checkpoints/ablations/h16_d20/best.pt` |
| 16 | 50 | 0.0667 | 0.7977 | 77.27 | 30 | `outputs/checkpoints/ablations/h16_d50/best.pt` |
| 16 | 100 | 0.0667 | 0.7592 | 78.77 | 30 | `outputs/checkpoints/ablations/h16_d100/best.pt` |
| 32 | 20 | 0.0333 | 1.0211 | 79.23 | 30 | `outputs/checkpoints/ablations/h32_d20/best.pt` |
| 32 | 50 | 0.1000 | 1.0091 | 74.40 | 30 | `outputs/checkpoints/ablations/h32_d50/best.pt` |
| 32 | 100 | 0.0667 | 0.9966 | 79.13 | 30 | `outputs/checkpoints/ablations/h32_d100/best.pt` |

This ablation compares how action horizon and diffusion denoising steps affect closed-loop reaching success rate and final error.
