# CartPole Training Example

CartPole is a classic control task in reinforcement learning. The goal is to keep the pole balanced by controlling the cart's left-right movement.
![cartpole](/_static/images/poster/cartpole.jpg)

## Task Description

-   **State Space**: Cart position, cart velocity, pole angle, pole angular velocity
-   **Action Space**: Apply force left or right
-   **Reward Function**: +1 reward for each step the pole stays upright
-   **Termination Conditions**: Pole angle exceeds ±15 degrees or episode length exceeds 10 seconds

## Quick Start

### 1. Environment Preview

```bash
uv run scripts/view.py --env cartpole
```

### 2. Start Training

```bash
# Train with default parameters
uv run scripts/train.py --env cartpole

# Customize number of environments
uv run scripts/train.py --env cartpole --num-envs 1024

# Enable rendering (visualize during training)
uv run scripts/train.py --env cartpole --render
```

### 3. View Training Progress

```bash
uv run tensorboard --logdir runs/cartpole
```

### 4. Test Training Results

```bash
# Automatically find best policy for testing (recommended)
uv run scripts/play.py --env cartpole

# Manually specify policy file for testing
uv run scripts/play.py --env cartpole --policy runs/cartpole/nn/best_policy.pickle
```

> **Tip**: The system will automatically find the latest and best policy files in the `runs/cartpole/` directory for testing. You can also manually specify specific policy files using the `--policy` parameter.

## Configuration Parameters

Main configuration parameters for the CartPole environment:

```python
@dataclass
class CartPoleEnvCfg(EnvCfg):
    model_file: str = "path/to/inverted_pendulum.xml"  # MJCF model file
    reset_noise_scale: float = 0.01                    # Reset noise
    max_episode_seconds: float = 10.0                 # Maximum episode length
```

Training configuration parameters:

```python
from dataclasses import dataclass
from motrix_rl.skrl.cfg import PPOCfg
from motrix_rl import registry

@registry.rlcfg("cartpole")
@dataclass
class CartPolePPO(PPOCfg):
    max_env_steps: int = 10_000_000          # Maximum environment steps
    check_point_interval: int = 500          # Checkpoint interval

    # Network structure (small network suitable for simple tasks)
    policy_hidden_layer_sizes: tuple[int, ...] = (32, 32)
    value_hidden_layer_sizes: tuple[int, ...] = (32, 32)

    # PPO parameters
    rollouts: int = 32                       # Experience replay rounds
    learning_epochs: int = 5                 # Training rounds
    mini_batches: int = 4                    # Number of mini-batches
```

**Note**: CartPole is a simple task and currently uses universal configuration. If you need to create specialized configurations for different training backends (JAX/Torch), refer to the environment configuration documentation examples.

## Custom Training

You can override default configurations through command line arguments:

```bash
uv run scripts/train.py --env cartpole \
  --num-envs 1024 \
  --train-backend jax \
  --sim-backend np
```

## Expected Results

-   Pole angle stays within ±5 degrees most of the time
-   Cart displacement range is reasonable

## Troubleshooting

If training performance is poor, you can try:

1. Adjust learning rate (try 1e-4 to 1e-3)
2. Increase number of environments (more parallel training)
3. Adjust reward function weights
4. Check if physical parameters are reasonable
