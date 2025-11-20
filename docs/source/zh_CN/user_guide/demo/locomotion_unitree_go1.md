# Unitree GO1 机器人行走训练示例

Unitree GO1 是一个四足机器人平台，本示例展示了如何训练 GO1 在平坦地形上实现稳定的步态行走。

```{video} /_static/videos/go1_walk.mp4
:poster: _static/images/poster/go1_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

GO1 四足机器人具有 12 个自由度（每条腿 3 个关节），需要通过深度强化学习学习协调的步态控制：

-   **状态空间**：48 维，包含机器人线速度、角速度、姿态、关节角度、关节速度、动作和命令等
-   **动作空间**：12 维，控制各个关节的目标位置（通过 PD 控制器转换为力矩）
-   **奖励函数**：复合奖励，包含速度跟踪、姿态稳定、能量效率等多个组件
-   **终止条件**：机器人躯干接触地面或其他不稳定状态

### 训练任务

```bash
uv run scripts/train.py --env go1-flat-terrain-walk
```

## 配置参数

### 环境配置

```python
@dataclass
class Go1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0      # 最大episode长度
    model_file: str = "scene_motor_actuator.xml"
    sim_dt: float = 0.01                   # 仿真时间步
    ctrl_dt: float = 0.01                  # 控制时间步
```

### 控制配置

```python
@dataclass
class ControlConfig:
    stiffness = 80                         # PD 控制器刚度 [N*m/rad]
    damping = 1                            # PD 控制器阻尼 [N*m*s/rad]
    action_scale = 0.05                    # 动作缩放因子
```

### 初始关节角度

```python
default_joint_angles = {
    "FL_hip": 0.0,      # 前左髋关节
    "RL_hip": 0.0,      # 后左髋关节
    "FR_hip": -0.0,     # 前右髋关节
    "RR_hip": -0.0,     # 后右髋关节
    "FL_thigh": 0.9,    # 前左大腿
    "RL_thigh": 0.9,    # 后左大腿
    "FR_thigh": 0.9,    # 前右大腿
    "RR_thigh": 0.9,    # 后右大腿
    "FL_calf": -1.8,    # 前左小腿
    "RL_calf": -1.8,    # 后左小腿
    "FR_calf": -1.8,    # 前右小腿
    "RR_calf": -1.8,    # 后右小腿
}
```

## 奖励函数设计

GO1 的奖励函数是一个复杂的复合函数，包含多个组件：

### 主要奖励组件

```python
reward_config.scales = {
    "tracking_lin_vel": 1.0,      # 线速度跟踪奖励
    "tracking_ang_vel": 0.5,      # 角速度跟踪奖励
    "feet_air_time": 1.0,         # 足部空中时间奖励
    "lin_vel_z": -2.0,            # Z轴线速度惩罚
    "ang_vel_xy": -0.05,          # XY轴角速度惩罚
    "orientation": -0.0,          # 姿态偏离惩罚
    "torques": -0.00001,          # 力矩消耗惩罚
    "dof_acc": -2.5e-7,           # 关节加速度惩罚
    "action_rate": -0.001,        # 动作变化率惩罚
    "hip_pos": -1,                # 髋关节位置惩罚
    "calf_pos": -0.3,             # 腿关节位置惩罚
}
```

### 关键奖励函数

#### 速度跟踪奖励

```python
# 跟踪线速度命令（xy平面）
def _reward_tracking_lin_vel(self, data, commands):

# 跟踪角速度命令（偏航）
def _reward_tracking_ang_vel(self, data, commands):
```

#### 足部空中时间奖励

```python
def _reward_feet_air_time(self, commands, info):
```

## 观察空间构成

GO1 的观察空间为 48 维，包含以下信息：

```python
obs = np.hstack([
    noisy_linvel,        # 3维：局部坐标系线速度
    noisy_gyro,          # 3维：陀螺仪数据
    local_gravity,       # 3维：局部重力方向
    noisy_joint_angle,   # 12维：关节角度（相对于默认值）
    noisy_joint_vel,     # 12维：关节速度
    last_actions,        # 12维：上一帧动作
    command,             # 3维：速度命令 [vx, vy, vyaw]
])
```

## 运动速度命令生成

训练过程中随机生成速度命令，确保智能体能够跟踪不同的移动速度：

```python
def resample_commands(self, num_envs: int):
```

## 预期训练结果

1. 稳定的四足步态
2. 良好的速度跟踪
