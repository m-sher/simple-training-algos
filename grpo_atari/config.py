"""
Configuration settings for GRPO training on Atari Breakout.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GRPOConfig:
    """
    Configuration for GRPO (Group Relative Policy Optimization) training.
    
    GRPO samples multiple trajectories from the same initial state and uses
    group-relative advantages instead of a learned value baseline.
    
    Attributes:
        env_name: Name of the Atari environment.
        group_size: Number of parallel trajectories per initial state (G in GRPO).
                   Each group shares the same initial random seed.
        num_groups: Number of different initial states to sample per update.
        max_episode_steps: Maximum steps per episode.
        frame_skip: Number of frames to skip (action repeat).
        frame_stack: Number of frames to stack for observation.
        
        learning_rate: Learning rate for policy optimization.
        gamma: Discount factor for returns.
        epsilon_clip: Clipping parameter for policy ratio (similar to PPO).
        entropy_coef: Coefficient for entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
        
        num_iterations: Total number of training iterations.
        steps_per_trajectory: Number of steps to collect per trajectory.
        update_epochs: Number of epochs to update on collected data.
        minibatch_size: Size of minibatches for updates.
        
        log_interval: How often to log training metrics.
        save_interval: How often to save model checkpoints.
        eval_interval: How often to run evaluation.
        eval_episodes: Number of episodes for evaluation.
        
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save model checkpoints.
    """
    
    # Environment settings
    env_name: str = "ALE/Breakout-v5"
    group_size: int = 8  # Number of trajectories per group (same initial state)
    num_groups: int = 4  # Number of different initial states per update
    max_episode_steps: int = 27000  # ~30 minutes at 60fps with frame_skip=4
    frame_skip: int = 4
    frame_stack: int = 4
    
    # Observation preprocessing
    obs_height: int = 84
    obs_width: int = 84
    grayscale: bool = True
    
    # Optimizer settings
    learning_rate: float = 2.5e-4
    adam_epsilon: float = 1e-5
    
    # GRPO hyperparameters
    gamma: float = 0.99  # Discount factor for returns
    epsilon_clip: float = 0.1  # Policy ratio clipping
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5
    
    # Advantage normalization (GRPO-specific)
    advantage_eps: float = 1e-8  # Small constant for numerical stability
    normalize_advantages: bool = True  # Whether to normalize group advantages
    
    # Training loop settings
    num_iterations: int = 10000
    steps_per_trajectory: int = 128  # Steps collected per trajectory
    update_epochs: int = 4  # Epochs per update
    minibatch_size: int = 256
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 10
    
    # Demo recording during training
    demo_interval: int = 100  # Record demo every N iterations (0 to disable)
    demo_steps: int = 100  # Number of steps per demo
    demo_fps: int = 30  # FPS for demo GIFs
    demo_dir: str = "demos"  # Directory for training demos
    
    # Reproducibility
    seed: Optional[int] = 42
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    @property
    def total_envs(self) -> int:
        """Total number of parallel environments."""
        return self.group_size * self.num_groups
    
    @property
    def batch_size(self) -> int:
        """Total batch size per update (all trajectories combined)."""
        return self.total_envs * self.steps_per_trajectory
    
    @property
    def obs_shape(self) -> Tuple[int, int, int]:
        """Observation shape after preprocessing."""
        channels = self.frame_stack if self.grayscale else self.frame_stack * 3
        return (self.obs_height, self.obs_width, channels)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.group_size >= 2, "group_size must be at least 2 for GRPO"
        assert self.num_groups >= 1, "num_groups must be at least 1"
        assert self.steps_per_trajectory > 0, "steps_per_trajectory must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert self.epsilon_clip > 0, "epsilon_clip must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.minibatch_size <= self.batch_size, \
            f"minibatch_size ({self.minibatch_size}) must be <= batch_size ({self.batch_size})"


def get_default_config() -> GRPOConfig:
    """Get the default GRPO configuration."""
    config = GRPOConfig()
    config.validate()
    return config
