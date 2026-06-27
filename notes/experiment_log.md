# Experiment Log

## 目标

在一个轻量 2D 点机器人项目中复现并比较：

1. Diffusion Policy 的条件去噪动作序列生成。
2. MLP Behavior Cloning baseline。
3. Receding horizon control 闭环执行。
4. Horizon / diffusion steps ablation。
5. 带圆形障碍物的多路径 reaching，用于观察多模态动作分布。

## 推荐命令

Simple reaching：

```powershell
python scripts/generate_dataset.py --num_demos 1000 --horizon 16 --save_path data/demos.npz
python train.py --data_path data/demos.npz --epochs 50 --batch_size 128 --diffusion_steps 50 --device cpu
python train_bc.py --data_path data/demos.npz --epochs 50 --batch_size 128 --device cpu
python eval.py --checkpoint outputs/checkpoints/best.pt --num_episodes 50 --save_gif
python eval_bc.py --checkpoint outputs/checkpoints/bc_best.pt --num_episodes 50 --save_gif
```

Obstacle multi-path reaching：

```powershell
python scripts/run_obstacle_experiment.py
```

Ablation：

```powershell
python scripts/run_ablation.py
```

## 实验记录

| Date | Task | Data | Epochs | Diffusion Steps | Horizon | Episodes | Main Result | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2026-06-26 | Simple Reaching / DP | 1000 demos | 50 | 50 | 16 | 50 | success 0.04, final distance 0.8731 | 低维单模态任务中，小型 MLP DDPM 表现弱于 BC |
| 2026-06-26 | Simple Reaching / BC | 1000 demos | 50 | - | 16 | 50 | success 1.00, final distance 0.0409 | BC 对简单 reaching 很强 |
| 2026-06-27 | Ablation | 1000 demos per config | 30 | 20/50/100 | 8/16/32 | 30 | best: h=8, steps=100, success 0.20 | 用于比较 horizon 和 denoising steps 的影响 |
| 2026-06-27 | Obstacle Multi-path / BC | 1000 pairs | 50 | - | 16 | 50 | success 0.86, collision 0.14 | BC 在多路径任务中仍能完成较多样本，但有碰撞 |
| 2026-06-27 | Obstacle Multi-path / DP single | 1000 pairs | 50 | 50 | 16 | 50 | success 0.18, collision 0.82 | 单次采样不稳定 |
| 2026-06-27 | Obstacle Multi-path / DP multi | 1000 pairs | 50 | 50 | 16 | 50 | success 0.94, collision 0.06 | 多次采样 + 碰撞筛选效果最好 |

## 观察

- Simple reaching 是低维、单模态回归问题，BC baseline 更直接、更稳定。
- Obstacle multi-path reaching 在同一 start-goal-obstacle 输入下有上绕和下绕两种专家路径，更适合展示 Diffusion Policy 的采样式动作生成。
- DP single-sample 在当前小模型下不稳定，但 DP multi-sample selection 能从多个候选 action chunk 中选出更安全的轨迹。
- 这个项目目前更适合作为“最小可复现 + 对比实验 + 可视化展示”，不是完整视觉机器人系统。
