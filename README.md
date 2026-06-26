# mini-diffusion-policy-2d

一个面向学习和展示的轻量级 Diffusion Policy 复现项目。项目不下载大模型，不下载外部机器人数据集，不依赖 MuJoCo、Isaac Sim 或真实机械臂，只在一个 2D 点机器人 reaching 环境中跑通：

```text
专家数据生成 -> Diffusion Policy 训练 -> 闭环评估 -> BC baseline 对比 -> Ablation study
```

## 1. 项目背景：为什么 Diffusion Policy 能用于机器人动作生成

机器人控制可以看成条件动作生成问题：给定当前观测、目标和任务上下文，策略需要生成下一段可执行动作。传统行为克隆通常直接回归单步动作，但真实机器人任务常存在多种可行轨迹，例如从左侧或右侧接近目标、绕开障碍物、选择不同抓取姿态等。单步 MSE 回归容易学到多个专家动作的平均值，导致动作不稳定。

Diffusion Policy 的核心思想是：把机器人动作序列当作连续数据分布来建模。训练时给专家动作序列加入高斯噪声，让神经网络学习在观测条件下预测噪声；推理时从随机噪声开始逐步去噪，生成一段未来动作序列。这个机制最早常用于图像生成，但同样适用于连续机器人动作。

本项目把真实机器人系统简化为 2D reaching：

- observation：当前位置 `(x, y)`
- goal：目标位置 `(gx, gy)`
- action：位置增量 `(dx, dy)`
- expert：每步朝目标方向移动，并加入少量噪声
- policy：输入 `observation + goal`，输出未来 `horizon` 步动作序列

## 2. 方法简介：conditional denoising + action chunk + receding horizon control

### Conditional Denoising

模型输入包括：

```text
noisy_action_sequence + observation + goal + diffusion_timestep
```

训练目标是预测加入动作序列中的噪声。推理时，模型从高斯噪声开始，经过多步反向去噪，生成符合当前状态和目标的动作序列。

### Action Chunk

策略不是只输出一步动作，而是输出一段未来动作：

```text
[a_t, a_{t+1}, ..., a_{t+horizon-1}]
```

本项目默认 `horizon=16`。Action chunk 让策略具有短期规划能力，动作也更容易保持连续。

### Receding Horizon Control

评估时不会一次性执行完整 action chunk。默认每次只执行前 `execute_steps=4` 步，然后重新观测当前位置、重新生成动作序列。这种闭环执行方式可以不断修正偏差。

## 3. 环境安装

建议使用 Python 3.10 或更新版本。

```powershell
cd mini-diffusion-policy-2d
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

```powershell
python scripts/generate_dataset.py --num_demos 1000 --horizon 16 --save_path data/demos.npz
```

生成的 `.npz` 文件包含：

- `observation`：`(N, 2)`
- `goal`：`(N, 2)`
- `action_sequence`：`(N, horizon, 2)`

## 5. 训练命令

训练 Diffusion Policy：

```powershell
python train.py --data_path data/demos.npz --epochs 50 --batch_size 128 --diffusion_steps 50 --device cpu
```

训练 MLP Behavior Cloning baseline：

```powershell
python train_bc.py --data_path data/demos.npz --epochs 50 --batch_size 128 --device cpu
```

运行 ablation study：

```powershell
python scripts/run_ablation.py
```

快速检查 ablation 脚本链路：

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

生成方法对比表：

```powershell
python scripts/compare_methods.py --diffusion_metrics outputs/logs/eval_metrics.json --bc_metrics outputs/logs/bc_eval_metrics.json
```

## Results

当前结果已在 CPU 上实际跑出。主线设置为 `num_demos=1000`、`horizon=16`、`diffusion_steps=50`、`epochs=50`、`num_episodes=50`。

| Method | Success Rate | Avg Final Distance | Avg Steps | Episodes |
| --- | ---: | ---: | ---: | ---: |
| Diffusion Policy | 0.0400 | 0.8731 | 77.64 | 50 |
| MLP Behavior Cloning | 1.0000 | 0.0409 | 16.04 | 50 |

这个 toy 任务是低维、近似单模态的 reaching，因此直接回归专家动作的 BC baseline 很强。当前小型 MLP DDPM 版本完整复现了 Diffusion Policy 的训练和采样流程，但在该低维任务上没有追平 BC；这也说明该项目不仅是 demo，还能用于观察方法适用场景和失败模式。

### Training Loss

![training_loss](outputs/figures/training_loss.png)

### Evaluation Trajectories

![eval_trajectories](outputs/figures/eval_trajectories.png)

### Rollout Demo

![rollout](outputs/figures/rollout.gif)

### BC Baseline Visuals

BC baseline 的训练曲线和闭环轨迹：

![bc_training_loss](outputs/figures/bc_training_loss.png)

![bc_eval_trajectories](outputs/figures/bc_eval_trajectories.png)

完整对比表保存于：

```text
outputs/logs/compare_results.md
```

### Ablation Study

为了更接近论文实验，本项目加入两个轻量 ablation：

- `horizon = 8, 16, 32`
- `diffusion_steps = 20, 50, 100`

每组配置默认使用 `num_demos=1000`、`epochs=30`、`num_episodes=30`。

| Horizon | Diffusion Steps | Success Rate | Avg Final Distance | Avg Steps |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 20 | 0.1333 | 0.8848 | 75.53 |
| 8 | 50 | 0.1000 | 0.5001 | 78.70 |
| 8 | 100 | 0.2000 | 0.2675 | 73.57 |
| 16 | 20 | 0.0333 | 0.9760 | 79.03 |
| 16 | 50 | 0.0667 | 0.7977 | 77.27 |
| 16 | 100 | 0.0667 | 0.7592 | 78.77 |
| 32 | 20 | 0.0333 | 1.0211 | 79.23 |
| 32 | 50 | 0.1000 | 1.0091 | 74.40 |
| 32 | 100 | 0.0667 | 0.9966 | 79.13 |

观察：在当前轻量 MLP DDPM 设置下，较短 action horizon 配合更多 denoising steps 表现更好；`horizon=8, diffusion_steps=100` 在 ablation 中取得最高成功率和最低最终误差。

完整结果保存于：

```text
outputs/logs/ablation_results.csv
outputs/logs/ablation_results.md
```

## 8. 可视化图

主要可视化产物：

- Diffusion loss：`outputs/figures/training_loss.png`
- Diffusion rollout trajectories：`outputs/figures/eval_trajectories.png`
- Diffusion rollout GIF：`outputs/figures/rollout.gif`
- BC loss：`outputs/figures/bc_training_loss.png`
- BC rollout trajectories：`outputs/figures/bc_eval_trajectories.png`
- BC rollout GIF：`outputs/figures/bc_rollout.gif`
- Ablation 每组轨迹图：`outputs/figures/ablations/`

## 9. 与正式视觉版 Diffusion Policy 的区别

本项目是教学和简历展示用的轻量复现，保留核心算法结构，但刻意去掉了高成本部分。

| 项目 | 本项目 | 正式视觉版 Diffusion Policy |
| --- | --- | --- |
| 输入 | 2D 坐标 `(x, y, gx, gy)` | RGB/RGB-D 图像、机器人 proprioception、历史观测 |
| 动作 | 2D 位移 `(dx, dy)` | 末端位姿、关节速度、夹爪动作等 |
| 数据 | 手写专家自动生成 | 遥操作、真实机器人、仿真数据 |
| 环境 | 手写 2D reaching | 真实机械臂或高保真仿真 |
| 模型 | 小型 MLP denoiser | 视觉编码器 + 1D U-Net / Transformer 等 |
| 训练成本 | CPU 可运行 | 通常需要 GPU 和大规模数据 |
| 目标 | 理解算法闭环 | 完成真实机器人操作任务 |

因此，本项目不是完整机器人系统，而是把 Diffusion Policy 的动作生成思想压缩到可本地复现的最小版本。

## 10. 后续计划

- 接入视觉输入：把 2D 坐标观测替换为简单图像，例如渲染点机器人和目标点，再加入 CNN encoder。
- 引入障碍物：让 reaching 从单模态路径变成多模态路径，更适合展示 diffusion policy 的优势。
- 替换 MLP denoiser：实现 1D U-Net action denoiser，贴近正式 Diffusion Policy。
- 加入历史观测：使用过去几帧状态作为条件，模拟真实机器人中的时序信息。
- 迁移到真实 XY 平台：把 `(dx, dy)` 输出映射到二维滑台或桌面 XY gantry 的控制接口。
- 增加 real-time controller：加入速度限制、动作平滑、安全边界和异常停止逻辑。

## 项目结构

```text
mini-diffusion-policy-2d/
├── README.md
├── requirements.txt
├── train.py
├── train_bc.py
├── eval.py
├── eval_bc.py
├── scripts/
│   ├── generate_dataset.py
│   ├── compare_methods.py
│   └── run_ablation.py
├── src/
│   ├── envs/
│   │   └── point_reach_env.py
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

## 简历表述建议

> 基于 PyTorch 复现轻量级 Diffusion Policy 模仿学习框架，构建 2D reaching 环境并自动生成专家 demonstrations；实现条件 DDPM 动作序列生成、MLP Behavior Cloning baseline、receding horizon 闭环控制与 horizon / diffusion steps ablation study，在 CPU 环境下完成训练、评估和可视化分析。
