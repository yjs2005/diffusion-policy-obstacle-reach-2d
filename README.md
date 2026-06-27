# diffusion-policy-obstacle-reach-2d

一个面向学习项目整理的轻量级 Diffusion Policy 复现。项目用 PyTorch 和一个手写 2D 点机器人环境，在普通笔记本 CPU 上跑通：

```text
专家数据生成 -> Diffusion Policy 训练 -> 闭环评估 -> BC baseline 对比 -> Ablation -> 障碍物多路径实验
```

## 1. 项目背景：为什么 Diffusion Policy 能用于机器人动作生成

机器人控制可以看成一个条件动作生成问题：给定当前观测、目标和任务上下文，策略需要生成接下来可执行的动作。传统 Behavior Cloning 常用 MSE 直接回归动作，但当同一个状态下存在多种合理动作时，例如绕障碍物的上路径和下路径，单一均值回归容易学到“平均动作”，这个动作可能正好不可执行。

Diffusion Policy 的核心思想是把机器人动作序列看成连续数据分布来建模。训练时给专家 action chunk 加噪声，让网络学习在观测条件下预测噪声；推理时从高斯噪声开始，多步去噪生成未来动作序列。扩散模型最初常见于图像生成，但它本质上学习的是连续数据分布，因此也可以用于生成连续机器人动作。

本项目把真实机器人系统压缩到两个 CPU 可跑的小任务：

- Simple Reaching：点机器人从随机起点移动到随机目标。
- Obstacle Multi-path Reaching：点机器人需要绕过圆形障碍物，同一个输入下存在上绕和下绕两种专家路径。

## 2. 方法简介：conditional denoising + action chunk + receding horizon control

### Conditional Denoising

Diffusion Policy 的 denoising network 输入：

```text
noisy_action_sequence + observation + goal/task_condition + diffusion_timestep
```

训练目标是预测加到专家动作序列中的噪声。推理时，模型从随机噪声开始逐步去噪，生成符合当前观测和目标的动作序列。

### Action Chunk

策略不是只输出一步动作，而是输出未来一段动作：

```text
[a_t, a_{t+1}, ..., a_{t+horizon-1}]
```

本项目默认 `horizon=16`。Action chunk 让策略有短期规划能力，也让动作更连续。

### Receding Horizon Control

评估时不会一次性执行完整 action chunk，而是默认每次执行前 `execute_steps=4` 步，然后重新观测当前状态并重新生成动作序列。这种闭环执行方式可以不断修正偏差。

## 3. 环境安装

建议使用 Python 3.10 或更新版本。

```powershell
cd diffusion-policy-obstacle-reach-2d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

已有 Python 环境时也可以直接安装依赖：

```powershell
pip install -r requirements.txt
```

依赖保持轻量：`numpy`、`torch`、`matplotlib`、`tqdm`、`imageio`。

## 4. 数据生成

Simple Reaching 数据：

```powershell
python scripts/generate_dataset.py --num_demos 1000 --horizon 16 --save_path data/demos.npz
```

Obstacle Multi-path Reaching 数据：

```powershell
python scripts/generate_obstacle_dataset.py --num_pairs 1000 --horizon 16 --save_path data/obstacle_demos.npz
```

障碍物任务中，观察量为：

```text
[x, y, gx, gy, ox, oy, r]
```

其中 `(x, y)` 是当前位置，`(gx, gy)` 是目标，`(ox, oy, r)` 是圆形障碍物中心和半径。注意：上绕/下绕 mode 不作为输入，这样同一个状态下会保留多模态动作分布。

## 5. 训练命令

训练 Diffusion Policy：

```powershell
python train.py --data_path data/demos.npz --epochs 50 --batch_size 128 --diffusion_steps 50 --device cpu
```

训练 MLP Behavior Cloning baseline：

```powershell
python train_bc.py --data_path data/demos.npz --epochs 50 --batch_size 128 --device cpu
```

训练障碍物多路径任务可以直接使用一键脚本：

```powershell
python scripts/run_obstacle_experiment.py
```

快速 smoke test：

```powershell
python scripts/run_obstacle_experiment.py --quick
```

运行 ablation study：

```powershell
python scripts/run_ablation.py
```

快速检查 ablation 流程：

```powershell
python scripts/run_ablation.py --quick
```

## 6. 评估命令

评估 Diffusion Policy：

```powershell
python eval.py --checkpoint outputs/checkpoints/best.pt --num_episodes 50 --save_gif
```

评估 BC baseline：

```powershell
python eval_bc.py --checkpoint outputs/checkpoints/bc_best.pt --num_episodes 50 --save_gif
```

生成普通 reaching 的方法对比表：

```powershell
python scripts/compare_methods.py --diffusion_metrics outputs/logs/eval_metrics.json --bc_metrics outputs/logs/bc_eval_metrics.json
```

评估障碍物多路径任务：

```powershell
python eval_obstacle.py --dp_checkpoint outputs/checkpoints/obstacle_dp/best.pt --bc_checkpoint outputs/checkpoints/obstacle_bc_best.pt
```

## 7. 实验结果 / Results

当前结果已经在本地 CPU 上实际跑出。

### Simple Reaching：Diffusion Policy vs BC

配置：`num_demos=1000`，`horizon=16`，`diffusion_steps=50`，`epochs=50`，`num_episodes=50`。

| Method               | Success Rate | Avg Final Distance | Avg Steps | Episodes |
| -------------------- | -----------: | -----------------: | --------: | -------: |
| Diffusion Policy     |       0.0400 |             0.8731 |     77.64 |       50 |
| MLP Behavior Cloning |       1.0000 |             0.0409 |     16.04 |       50 |

这个任务是低维、近似单模态的 reaching，因此直接回归专家动作的 BC baseline 表现更强。这个结果也说明：Diffusion Policy 不应该被硬套到所有任务上，它更适合需要建模多模态动作分布的场景。

### Obstacle Multi-path Reaching：多模态动作生成

配置：`num_pairs=1000`，`horizon=16`，`diffusion_steps=50`，`epochs=50`，`num_episodes=50`，`multi_samples=8`。

| Method                    | Success Rate | Collision Rate | Avg Final Distance | Diversity | Upper Route | Lower Route |
| ------------------------- | -----------: | -------------: | -----------------: | --------: | ----------: | ----------: |
| BC                        |       0.8600 |         0.1400 |             0.1791 |    0.0000 |        0.52 |        0.48 |
| DP single-sample          |       0.1800 |         0.8200 |             0.8005 |    0.0000 |        0.58 |        0.42 |
| DP multi-sample selection |       0.9400 |         0.0600 |             0.0992 |    0.0168 |        0.68 |        0.32 |

观察：在 simple reaching 上 BC 更强；加入障碍物和上下两种专家路径后，Diffusion Policy 的多次采样加碰撞筛选版本在成功率、碰撞率和最终误差上优于 BC。这个实验更能体现扩散策略“生成多个候选 action chunk，再根据约束选择可执行轨迹”的价值。

### Ablation Study

项目加入了两个轻量 ablation：

- `horizon = 8, 16, 32`
- `diffusion_steps = 20, 50, 100`

每组默认使用 `num_demos=1000`、`epochs=30`、`num_episodes=30`。

| Horizon | Diffusion Steps | Success Rate | Avg Final Distance | Avg Steps |
| ------: | --------------: | -----------: | -----------------: | --------: |
|       8 |              20 |       0.1333 |             0.8848 |     75.53 |
|       8 |              50 |       0.1000 |             0.5001 |     78.70 |
|       8 |             100 |       0.2000 |             0.2675 |     73.57 |
|      16 |              20 |       0.0333 |             0.9760 |     79.03 |
|      16 |              50 |       0.0667 |             0.7977 |     77.27 |
|      16 |             100 |       0.0667 |             0.7592 |     78.77 |
|      32 |              20 |       0.0333 |             1.0211 |     79.23 |
|      32 |              50 |       0.1000 |             1.0091 |     74.40 |
|      32 |             100 |       0.0667 |             0.9966 |     79.13 |

观察：在当前小型 MLP DDPM 设置下，较短 action horizon 搭配更多 denoising steps 的表现更好；`horizon=8, diffusion_steps=100` 在 ablation 中取得最高成功率和最低最终误差。

## 8. 可视化图

### Training Loss

![training_loss](outputs/figures/training_loss.png)

### Evaluation Trajectories

![eval_trajectories](outputs/figures/eval_trajectories.png)

### Rollout Demo

![rollout](outputs/figures/rollout.gif)

### BC Training Loss

![bc_training_loss](outputs/figures/bc_training_loss.png)

### BC Evaluation Trajectories

![bc_eval_trajectories](outputs/figures/bc_eval_trajectories.png)

### BC Rollout Demo

![bc_rollout](outputs/figures/bc_rollout.gif)

### Obstacle DP Training Loss

![obstacle_training_loss](outputs/figures/obstacle_training_loss.png)

### Obstacle BC Training Loss

![obstacle_bc_training_loss](outputs/figures/obstacle_bc_training_loss.png)

### Obstacle Expert Modes

![obstacle_expert_modes](outputs/figures/obstacle_expert_modes.png)

### BC vs Diffusion Policy on Obstacle Reaching

![bc_vs_dp_obstacle_rollouts](outputs/figures/bc_vs_dp_obstacle_rollouts.png)

### Diffusion Policy Multi-sample Diversity

![dp_multisample_diversity](outputs/figures/dp_multisample_diversity.png)

### Obstacle Metrics Bar

![obstacle_metrics_bar](outputs/figures/obstacle_metrics_bar.png)

### Obstacle Rollout Demo

![obstacle_rollout](outputs/figures/obstacle_rollout.gif)

<details>
<summary>Ablation 全部结果图</summary>


| Config          | Training Loss                                                | Evaluation Trajectories                                      |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| h=8, steps=20   | ![h8_d20_training_loss](outputs/figures/ablations/h8_d20_training_loss.png) | ![h8_d20_eval_trajectories](outputs/figures/ablations/h8_d20_eval_trajectories.png) |
| h=8, steps=50   | ![h8_d50_training_loss](outputs/figures/ablations/h8_d50_training_loss.png) | ![h8_d50_eval_trajectories](outputs/figures/ablations/h8_d50_eval_trajectories.png) |
| h=8, steps=100  | ![h8_d100_training_loss](outputs/figures/ablations/h8_d100_training_loss.png) | ![h8_d100_eval_trajectories](outputs/figures/ablations/h8_d100_eval_trajectories.png) |
| h=16, steps=20  | ![h16_d20_training_loss](outputs/figures/ablations/h16_d20_training_loss.png) | ![h16_d20_eval_trajectories](outputs/figures/ablations/h16_d20_eval_trajectories.png) |
| h=16, steps=50  | ![h16_d50_training_loss](outputs/figures/ablations/h16_d50_training_loss.png) | ![h16_d50_eval_trajectories](outputs/figures/ablations/h16_d50_eval_trajectories.png) |
| h=16, steps=100 | ![h16_d100_training_loss](outputs/figures/ablations/h16_d100_training_loss.png) | ![h16_d100_eval_trajectories](outputs/figures/ablations/h16_d100_eval_trajectories.png) |
| h=32, steps=20  | ![h32_d20_training_loss](outputs/figures/ablations/h32_d20_training_loss.png) | ![h32_d20_eval_trajectories](outputs/figures/ablations/h32_d20_eval_trajectories.png) |
| h=32, steps=50  | ![h32_d50_training_loss](outputs/figures/ablations/h32_d50_training_loss.png) | ![h32_d50_eval_trajectories](outputs/figures/ablations/h32_d50_eval_trajectories.png) |
| h=32, steps=100 | ![h32_d100_training_loss](outputs/figures/ablations/h32_d100_training_loss.png) | ![h32_d100_eval_trajectories](outputs/figures/ablations/h32_d100_eval_trajectories.png) |

</details>

完整日志和结果表保存在：

```text
outputs/logs/train_log.json
outputs/logs/bc_train_log.json
outputs/logs/eval_metrics.json
outputs/logs/bc_eval_metrics.json
outputs/logs/compare_results.md
outputs/logs/ablation_results.csv
outputs/logs/ablation_results.md
outputs/logs/obstacle_eval_metrics.json
```

## 9. 本地Diffusion Policy 的策略

| 项目     | 本项目                    |
| -------- | ------------------------- |
| 输入     | 2D 坐标、目标和障碍物参数 |
| 动作     | 2D 位移 `(dx, dy)`        |
| 数据     | 手写专家自动生成          |
| 环境     | 手写 2D reaching          |
| 模型     | 小型 MLP denoiser         |
| 训练成本 | CPU 可运行                |
| 目标     | 理解核心算法闭环          |

因此，本项目是把 Diffusion Policy 的动作生成思想压缩到可本地复现、可展示、可扩展的最小版本。

## 项目结构

```text
diffusion-policy-obstacle-reach-2d/
├── README.md
├── requirements.txt
├── train.py
├── train_bc.py
├── eval.py
├── eval_bc.py
├── eval_obstacle.py
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_obstacle_dataset.py
│   ├── compare_methods.py
│   ├── run_ablation.py
│   ├── run_obstacle_experiment.py
│   └── plot_obstacle_results.py
├── src/
│   ├── envs/
│   │   ├── point_reach_env.py
│   │   └── obstacle_reach_env.py
│   ├── models/
│   │   ├── diffusion_policy.py
│   │   └── bc_policy.py
│   ├── diffusion/
│   │   └── scheduler.py
│   └── utils/
│       └── visualization.py
├── data/
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
└── notes/
    ├── experiment_log.md
    └── diffusion_policy_reading_notes.md
```
