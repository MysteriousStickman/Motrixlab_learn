# 二维步行机器人训练示例

二维步行机器人（Walker2D）是基于 DeepMind Control Suite 的经典机器人控制任务，目标是通过控制机器人关节来实现站立、行走和奔跑。

```{video} /_static/videos/dm_walker.mp4
:poster: _static/images/poster/dm_walker.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Walker2D 是一个二维平面的双足机器人，具有多个关节和执行器：

-   **状态空间**：包括机器人各部位的旋转角度、角速度、躯干高度和速度等
-   **动作空间**：控制各个关节的力矩
-   **奖励函数**：主要由保持站立、前进速度等组成
-   **终止条件**：机器人摔倒或关节达到极限位置

### 三种任务模式

1. **dm-stander**: 静止站立任务 (move_speed = 0.0)

```bash
uv run scripts/train.py --env dm-stander
```

2. **dm-walker**: 行走任务 (move_speed = 1.0)

```bash
uv run scripts/train.py --env dm-walker
```

3. **dm-runner**: 奔跑任务 (move_speed = 5.0)

```bash
uv run scripts/train.py --env dm-runner
```

## 配置参数

### 环境配置

```python
@dataclass
class WalkerEnvCfg(EnvCfg):
    model_file: str = "walker.xml"           # MJCF模型文件
    max_episode_seconds: float = 25.0        # 最大episode长度
    sim_dt: float = 0.0125                   # 仿真时间步
    ctrl_dt: float = 0.025                   # 控制时间步
    move_speed: float = 1.0                  # 目标移动速度
    stand_height: float = 1.2                # 目标站立高度
```

### 训练配置

```python
@dataclass
class WalkerRLCfg(BaseRLCfg):
    num_envs: int = 512                      # 并行环境数量
    learning_rate: float = 3e-4              # 学习率
    batch_size: int = 512                    # 批大小
    max_epochs: int = 1000                   # 最大训练轮数
```

## 奖励函数设计

Walker2D 的奖励函数由以下几个部分组成：

### 基础站立奖励

```python
# 高度奖励：保持躯干在目标高度
# 直立奖励：保持躯干直立
```

### 移动奖励（行走和奔跑任务）

```python
# 速度奖励：追踪目标速度
# 总奖励 = 站立奖励 * 移动权重
```

## 预期结果

1. **dm-stander**：

    - 躯干高度保持在 1.0-1.4m 范围
    - 躯干直立角度偏差小于 15 度

2. **dm-walker**：

    - 实际行走速度接近 1.0 m/s
    - 步态协调，无明显摔倒

3. **dm-runner**：
    - 奔跑速度达到 4.0-5.0 m/s
    - 出现飞行相（双脚同时离地）
