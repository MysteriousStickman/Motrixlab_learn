# 2D Walker Robot Training Example

The 2D Walker Robot (Walker2D) is a classic robot control task from DeepMind Control Suite. The goal is to achieve standing, walking, and running by controlling the robot's joints.

```{video} /_static/videos/dm_walker.mp4
:poster: _static/images/poster/dm_walker.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## Task Description

Walker2D is a 2D planar bipedal robot with multiple joints and actuators:

-   **State Space**: Includes rotation angles and angular velocities of various robot parts, torso height and velocity, etc.
-   **Action Space**: Control torques for each joint
-   **Reward Function**: Mainly composed of maintaining standing balance and forward speed
-   **Termination Conditions**: Robot falls or joints reach limit positions

### Three Task Modes

1. **dm-stander**: Static standing task (move_speed = 0.0)

```bash
uv run scripts/train.py --env dm-stander
```

2. **dm-walker**: Walking task (move_speed = 1.0)

```bash
uv run scripts/train.py --env dm-walker
```

3. **dm-runner**: Running task (move_speed = 5.0)

```bash
uv run scripts/train.py --env dm-runner
```

## Quick Start

### 1. Environment Preview

```bash
# View standing task
uv run scripts/view.py --env dm-stander

# View walking task
uv run scripts/view.py --env dm-walker

# View running task
uv run scripts/view.py --env dm-runner
```

### 2. Start Training

```bash
# Train standing task
uv run scripts/train.py --env dm-stander

# Train walking task (default)
uv run scripts/train.py --env dm-walker

# Train running task
uv run scripts/train.py --env dm-runner

# Customize number of environments
uv run scripts/train.py --env dm-walker --num-envs 512

# Enable rendering (visualize during training)
uv run scripts/train.py --env dm-walker --render
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/dm-walker
```

### 4. Test Training Results

```bash
# Automatically find best policy for testing (recommended)
uv run scripts/play.py --env dm-walker

# Manually specify policy file for testing
uv run scripts/play.py --env dm-walker --policy runs/dm-walker/nn/best_policy.pickle
```

> **Tip**: The system will automatically find the latest and best policy files in the `runs/dm-walker/` directory for testing. Supports dm-stander, dm-walker, dm-runner three task modes.

## Configuration Parameters

### Environment Configuration

```python
@dataclass
class WalkerEnvCfg(EnvCfg):
    model_file: str = "walker.xml"           # MJCF model file
    max_episode_seconds: float = 25.0        # Maximum episode length
    sim_dt: float = 0.0125                   # Simulation time step
    ctrl_dt: float = 0.025                   # Control time step
    move_speed: float = 1.0                  # Target movement speed
    stand_height: float = 1.2                # Target standing height
```

### Training Configuration

```python
@dataclass
class WalkerRLCfg(BaseRLCfg):
    num_envs: int = 512                      # Number of parallel environments
    learning_rate: float = 3e-4              # Learning rate
    batch_size: int = 512                    # Batch size
    max_epochs: int = 1000                   # Maximum training epochs
```

## Reward Function Design

Walker2D's reward function consists of the following components:

### Basic Standing Reward

```python
# Height reward: keep torso at target height

# Upright reward: keep torso upright
```

### Movement Reward (walking and running tasks)

```python
# Speed reward: track target speed

# Total reward = standing reward * movement weight
```

## Expected Results

1. **dm-stander**:

    - Torso height maintained in 1.0-1.4m range

2. **dm-walker**:

    - Actual walking speed close to 1.0 m/s

3. **dm-runner**:
    - Running speed reaches 4.0-5.0 m/s
