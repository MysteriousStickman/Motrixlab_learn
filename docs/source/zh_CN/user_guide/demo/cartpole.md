# 倒立摆训练示例

倒立摆（CartPole）是强化学习中的经典控制任务，目标是通过控制小车左右移动来保持杆子平衡。
![cartpole](/_static/images/poster/cartpole.jpg)

## 任务描述

-   **状态空间**：小车位置、小车速度、杆子角度、杆子角速度
-   **动作空间**：向左或向右施加力
-   **奖励函数**：每一步保持杆子不倒下获得+1 奖励
-   **终止条件**：杆子角度超过 ±15 度或 episode 长度超过 10 秒

## 快速开始

### 1. 环境预览

```bash
uv run scripts/view.py --env cartpole
```

### 2. 开始训练

```bash
# 使用默认参数训练
uv run scripts/train.py --env cartpole

# 自定义环境数量
uv run scripts/train.py --env cartpole --num-envs 1024

# 启用渲染（训练时可视化）
uv run scripts/train.py --env cartpole --render
```

### 3. 查看训练进度

```bash
uv run tensorboard --logdir runs/cartpole
```

### 4. 测试训练结果

```bash
# 自动寻找最佳策略测试（推荐）
uv run scripts/play.py --env cartpole

# 手动指定策略文件测试
uv run scripts/play.py --env cartpole --policy runs/cartpole/nn/best_agent.pickle

```

> **提示**：系统会自动在 `runs/cartpole/` 目录下寻找最新、最佳的策略文件进行测试。您也可以通过 `--policy` 参数手动指定特定的策略文件。

## 配置参数

倒立摆环境的主要配置参数：

```python
@dataclass
class CartPoleEnvCfg(EnvCfg):
    model_file: str = "path/to/inverted_pendulum.xml"  # MJCF模型文件
    reset_noise_scale: float = 0.01                    # 重置噪声
    max_episode_seconds: float = 10.0                 # 最大episode长度
```

训练配置参数：

```python
@dataclass
class CartPoleRLCfg(BaseRLCfg):
    num_envs: int = 2048                    # 并行环境数量
    learning_rate: float = 3e-4             # 学习率
    batch_size: int = 2048                  # 批大小
    max_epochs: int = 500                   # 最大训练轮数
```

## 自定义训练

您可以通过命令行参数覆盖默认配置：

```bash
uv run scripts/train.py --env cartpole \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np
```

## 预期结果

-   杆子角度大部分时间保持在 ±5 度以内
-   小车位移范围适中

## 故障排除

如果训练效果不佳，可以尝试：

1. 调整学习率（尝试 1e-4 到 1e-3）
2. 增加环境数量（更多并行训练）
3. 调整奖励函数权重
4. 检查物理参数设置是否合理
