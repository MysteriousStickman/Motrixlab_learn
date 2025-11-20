# Unitree GO1 Robot Walking Training Example

Unitree GO1 is a quadruped robot platform. This example demonstrates how to train GO1 to achieve stable gait walking on flat terrain.

```{video} /_static/videos/go1_walk.mp4
:poster: _static/images/poster/go1_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

The GO1 quadruped robot has 12 degrees of freedom (3 joints per leg) and needs to learn coordinated gait control through deep reinforcement learning:

-   **State Space**: 48-dimensional, including robot linear velocity, angular velocity, posture, joint angles, joint velocities, actions, and commands
-   **Action Space**: 12-dimensional, controlling target positions of each joint (converted to torques through PD controller)
-   **Reward Function**: Composite reward including speed tracking, posture stability, energy efficiency, and other components
-   **Termination Conditions**: Robot trunk contacts ground or other unstable states

### Training Task

```bash
uv run scripts/train.py --env go1-flat-terrain-walk
```

## Configuration Parameters

### Environment Configuration

```python
@dataclass
class Go1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0      # Maximum episode length
    model_file: str = "scene_motor_actuator.xml"
    sim_dt: float = 0.01                   # Simulation time step
    ctrl_dt: float = 0.01                  # Control time step
```

### Training Configuration

```python
from dataclasses import dataclass
from motrix_rl.skrl.cfg import PPOCfg
from motrix_rl import registry

@registry.rlcfg("go1-flat-terrain-walk")
@dataclass
class Go1WalkPPO(PPOCfg):
    """
    GO1 quadruped robot walking training configuration
    """

    seed = 42
    max_env_steps: int = 40960000          # Maximum training steps
    num_envs: int = 2048                   # Number of parallel environments

    # Large network structure (suitable for complex robot control tasks)
    policy_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)
    value_hidden_layer_sizes: tuple[int, ...] = (512, 256, 128)

    # PPO parameters (optimized for robot tasks)
    learning_epochs: int = 2               # Training rounds
    mini_batches: int = 32                 # Number of mini-batches
    learning_rate: float = 1e-3            # Learning rate
```

**Note**: GO1 is a complex task that uses large network structures. If you need to create specialized configurations for different training backends (JAX/Torch), refer to the environment configuration documentation examples.

### Control Configuration

```python
@dataclass
class ControlConfig:
    stiffness = 80                         # PD controller stiffness [N*m/rad]
    damping = 1                            # PD controller damping [N*m*s/rad]
    action_scale = 0.1                     # Action scaling factor
```

### Initial Joint Angles

```python
default_joint_angles = {
    "FL_hip": 0.0,      # Front left hip joint
    "RL_hip": 0.0,      # Rear left hip joint
    "FR_hip": -0.0,     # Front right hip joint
    "RR_hip": -0.0,     # Rear right hip joint
    "FL_thigh": 0.9,    # Front left thigh
    "RL_thigh": 0.9,    # Rear left thigh
    "FR_thigh": 0.9,    # Front right thigh
    "RR_thigh": 0.9,    # Rear right thigh
    "FL_calf": -1.8,    # Front left calf
    "RL_calf": -1.8,    # Rear left calf
    "FR_calf": -1.8,    # Front right calf
    "RR_calf": -1.8,    # Rear right calf
}
```

## Reward Function Design

GO1's reward function is a complex composite function containing multiple components:

### Main Reward Components

```python
reward_config.scales = {
    "tracking_lin_vel": 1.0,      # Linear velocity tracking reward
    "tracking_ang_vel": 0.5,      # Angular velocity tracking reward
    "feet_air_time": 1.0,         # Foot air time reward
    "lin_vel_z": -2.0,            # Z-axis linear velocity penalty
    "ang_vel_xy": -0.05,          # XY-axis angular velocity penalty
    "orientation": -0.0,          # Posture deviation penalty
    "torques": -0.00001,          # Torque consumption penalty
    "dof_acc": -2.5e-7,           # Joint acceleration penalty
    "action_rate": -0.001,        # Action change rate penalty
    "hip_pos": -1,                # Hip joint position penalty
    "calf_pos": -0.3,             # Calf joint position penalty
}
```

### Key Reward Functions

#### Velocity Tracking Reward

```python
# Track linear velocity commands (xy plane)
def _reward_tracking_lin_vel(self, data, commands):

# Track angular velocity commands (yaw)
def _reward_tracking_ang_vel(self, data, commands):
```

#### Foot Air Time Reward

```python
def _reward_feet_air_time(self, commands, info):
```

## Observation Space Composition

GO1's observation space is 48-dimensional, containing the following information:

```python
obs = np.hstack([
    noisy_linvel,        # 3D: Local coordinate system linear velocity
    noisy_gyro,          # 3D: Gyroscope data
    local_gravity,       # 3D: Local gravity direction
    noisy_joint_angle,   # 12D: Joint angles (relative to default values)
    noisy_joint_vel,     # 12D: Joint velocities
    last_actions,        # 12D: Previous frame actions
    command,             # 3D: Velocity commands [vx, vy, vyaw]
])
```

## Motion Velocity Command Generation

Random velocity commands are generated during training to ensure the agent can track different movement speeds:

```python
def resample_commands(self, num_envs: int):
```

## Expected Training Results

1. Stable quadruped gait
2. Good speed tracking
