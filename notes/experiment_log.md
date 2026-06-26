# Experiment Log

## 目标

在 2D 点机器人 reaching 任务上复现一个轻量级 Diffusion Policy 流程：

1. 自动生成专家 demonstrations
2. 训练条件扩散策略模型
3. 使用 receding horizon control 做闭环评估
4. 保存 loss 曲线、轨迹图和评估指标

## 推荐命令

```powershell
python scripts/generate_dataset.py --num_demos 1000 --horizon 16 --save_path data/demos.npz
python train.py --data_path data/demos.npz --epochs 50 --batch_size 128 --diffusion_steps 50 --device cpu
python eval.py --checkpoint outputs/checkpoints/best.pt --num_episodes 50 --save_gif
```

## 记录模板

| Date | Data | Epochs | Diffusion Steps | Horizon | Execute Steps | Success Rate | Avg Final Distance | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-06-26 | 1000 demos | 50 | 50 | 16 | 4 | TBD | TBD | Initial run |

## 观察项

- 训练 loss 是否平稳下降
- 评估轨迹是否朝目标移动
- `execute_steps` 太大时是否更容易偏离
- `diffusion_steps` 减少后速度是否提升、成功率是否下降
- 数据量减少时是否出现明显过拟合
