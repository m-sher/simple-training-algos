# GRPO for Atari Breakout

Group Relative Policy Optimization (GRPO) implementation for Atari Breakout using TensorFlow.

### Disclaimer

Originally intended to add more algorithms to this repo, but no longer have plans to do so.

## Project Structure

```
grpo_atari/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── environment.py       # Environment wrappers with seeding support
├── model.py             # Policy network architecture
├── trajectory.py        # Trajectory collection and data structures
├── grpo_loss.py         # GRPO loss and gradient computation
├── trainer.py           # Training loop orchestration
└── utils.py             # Utility functions

tests/
├── __init__.py          # Test package
└── test_grpo.py         # Comprehensive test suite

main.py                  # Main entry point
```

### Module Descriptions

#### `config.py`
Defines `GRPOConfig` dataclass with all hyperparameters:
- Environment settings (env name, frame skip, frame stack)
- GRPO parameters (group size, number of groups)
- Optimization settings (learning rate, epsilon clip, entropy coefficient)
- Training loop settings (iterations, epochs, minibatch size)

#### `environment.py`
- `create_atari_env()`: Creates preprocessed Atari environment
- `SeededEnvGroup`: Group of environments sharing the same seed
- `ParallelEnvGroups`: Manages multiple groups for batch collection

#### `model.py`
- `NatureCNN`: Standard DQN-style convolutional feature extractor
- `PolicyNetwork`: Policy network with action logit output
- No value head needed for pure GRPO (uses group-relative advantages)

#### `trajectory.py`
- `Trajectory`: Single trajectory data container
- `GroupedTrajectories`: Trajectories from same initial state
- `TrajectoryBatch`: Complete batch for updates
- `TrajectoryCollector`: Collects trajectories from parallel envs

#### `grpo_loss.py`
- `compute_group_advantages()`: Core GRPO advantage computation
- `GRPOLoss`: Computes clipped surrogate loss with entropy bonus
- `GRPOOptimizer`: Handles complete update process

#### `trainer.py`
- `GRPOTrainer`: Main training orchestration class
- `Logger`: Metrics logging (TensorBoard + CSV)
- `Checkpointer`: Model checkpointing
- `Evaluator`: Policy evaluation

#### `utils.py`
- Learning rate schedules (linear, cosine, exponential)
- Data utilities (advantage normalization, discounted returns)
- Experiment management (config saving, seeding)

## Usage

### Command-Line Interface

The package provides three main entry points that can be run with `uv run`:

#### Training (`grpo-train`)

```bash
# Train with default settings
uv run grpo-train

# Custom group size (more trajectories per initial state)
uv run grpo-train --group-size 16

# Faster iteration (fewer steps per trajectory)
uv run grpo-train --steps-per-trajectory 64 --minibatch-size 128

# Resume training from checkpoint
uv run grpo-train --resume

# See all options
uv run grpo-train --help
```

#### Evaluation (`grpo-eval`)

```bash
# Evaluate a trained policy
uv run grpo-eval --weights checkpoints/policy

# Evaluate with more episodes
uv run grpo-eval --weights checkpoints/policy --episodes 100 -v

# Evaluate a random policy (baseline)
uv run grpo-eval --random --episodes 10

# Use deterministic (greedy) actions
uv run grpo-eval --weights checkpoints/policy --deterministic

# See all options
uv run grpo-eval --help
```

#### Testing (`grpo-test`)

```bash
# Run all tests
uv run grpo-test

# Run with verbose output
uv run grpo-test -v

# Run specific module
uv run grpo-test --module model

# List available test modules
uv run grpo-test --list

# Run only integration tests
uv run grpo-test --markers integration

# See all options
uv run grpo-test --help
```

### Training Options

```
--env-name            Atari environment name (default: ALE/Breakout-v5)
--group-size          Trajectories per group (default: 8)
--num-groups          Number of environment groups (default: 4)
--num-iterations      Training iterations (default: 10000)
--steps-per-trajectory Steps per trajectory (default: 128)
--learning-rate       Learning rate (default: 2.5e-4)
--gamma               Discount factor (default: 0.99)
--epsilon-clip        Policy ratio clip (default: 0.1)
--entropy-coef        Entropy bonus (default: 0.01)
--seed                Random seed (default: 42)
--resume              Resume from checkpoint
```

### Programmatic Usage

```python
from grpo_atari import GRPOConfig, GRPOTrainer

# Create configuration
config = GRPOConfig(
    env_name="ALE/Breakout-v5",
    group_size=8,
    num_groups=4,
    num_iterations=5000,
)

# Create and run trainer
trainer = GRPOTrainer(config)
history = trainer.train()

# Get trained policy
policy = trainer.get_policy()

# Save/load weights
trainer.save_policy("policy_weights")
trainer.load_policy("policy_weights")
```

### Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with detailed logging (shows comparisons and values)
pytest tests/ -v --log-cli-level=INFO

# Run with DEBUG level for maximum verbosity
pytest tests/ -v --log-cli-level=DEBUG

# Run specific test module
pytest tests/test_model/ -v

# Run specific test class
pytest tests/test_model/test_model.py::TestPolicyNetworkCreation -v

# Run only integration tests
pytest tests/ -v -m integration

# Run tests excluding slow tests
pytest tests/ -v -m "not slow"

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=grpo_atari
```

## Dependencies

- TensorFlow 2.15+
- TF-Agents 0.19+
- Gymnasium 1.2+
- ALE-Py 0.11+
- NumPy 1.26+
