"""
GRPO (Group Relative Policy Optimization) for Atari Breakout.

This package implements GRPO adapted for reinforcement learning in Atari games.
GRPO is typically used for LLM training, where multiple responses are sampled
from the same prompt. For RL, we adapt this by running parallel environments
with the same seed to sample multiple trajectories from the same initial state.

Key components:
- environment: Environment wrapper with seeding support
- model: Policy network definition
- trajectory: Trajectory collection from grouped environments
- grpo_loss: GRPO loss computation and gradient updates
- trainer: Training loop orchestration
- config: Configuration settings
- demo: Demo recording utilities
"""

from grpo_atari.config import GRPOConfig
from grpo_atari.environment import create_atari_env, SeededEnvGroup
from grpo_atari.model import PolicyNetwork
from grpo_atari.trajectory import TrajectoryCollector, Trajectory
from grpo_atari.grpo_loss import GRPOLoss, compute_group_advantages
from grpo_atari.trainer import GRPOTrainer
from grpo_atari.demo import DemoRecorder, record_demo, save_gif

__version__ = "0.1.0"
__all__ = [
    "GRPOConfig",
    "create_atari_env",
    "SeededEnvGroup", 
    "PolicyNetwork",
    "TrajectoryCollector",
    "Trajectory",
    "GRPOLoss",
    "compute_group_advantages",
    "GRPOTrainer",
    "DemoRecorder",
    "record_demo",
    "save_gif",
]
