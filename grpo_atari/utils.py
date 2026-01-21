"""
Utility functions for GRPO training.

This module contains helper functions for:
- Learning rate scheduling
- Data preprocessing
- Visualization
- Miscellaneous utilities
"""

import numpy as np
import tensorflow as tf
from typing import Callable, Optional, List, Dict, Any
import json
from pathlib import Path


# ==============================================================================
# Learning Rate Scheduling
# ==============================================================================

def linear_schedule(
    initial_lr: float,
    final_lr: float = 0.0,
    total_steps: int = 1000000,
) -> Callable[[int], float]:
    """
    Create a linear learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate.
        final_lr: Final learning rate.
        total_steps: Total number of steps.
        
    Returns:
        Function that takes step and returns learning rate.
    """
    def schedule(step: int) -> float:
        progress = min(1.0, step / total_steps)
        return initial_lr + progress * (final_lr - initial_lr)
    return schedule


def cosine_schedule(
    initial_lr: float,
    final_lr: float = 0.0,
    total_steps: int = 1000000,
    warmup_steps: int = 0,
) -> Callable[[int], float]:
    """
    Create a cosine annealing learning rate schedule with optional warmup.
    
    Args:
        initial_lr: Initial learning rate (after warmup).
        final_lr: Final learning rate.
        total_steps: Total number of steps.
        warmup_steps: Number of warmup steps.
        
    Returns:
        Function that takes step and returns learning rate.
    """
    def schedule(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return initial_lr * step / warmup_steps
        
        # Cosine decay
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return final_lr + (initial_lr - final_lr) * cosine_decay
    
    return schedule


def exponential_schedule(
    initial_lr: float,
    decay_rate: float = 0.99,
    decay_steps: int = 1000,
    min_lr: float = 1e-6,
) -> Callable[[int], float]:
    """
    Create an exponential decay learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate.
        decay_rate: Decay rate per decay_steps.
        decay_steps: Steps between decay applications.
        min_lr: Minimum learning rate.
        
    Returns:
        Function that takes step and returns learning rate.
    """
    def schedule(step: int) -> float:
        lr = initial_lr * (decay_rate ** (step / decay_steps))
        return max(lr, min_lr)
    return schedule


class LearningRateScheduler:
    """Wrapper for learning rate scheduling with optimizer integration."""
    
    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        schedule_fn: Callable[[int], float],
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: TensorFlow optimizer.
            schedule_fn: Function mapping step to learning rate.
        """
        self.optimizer = optimizer
        self.schedule_fn = schedule_fn
        self.current_step = 0
    
    def step(self) -> float:
        """Advance one step and update learning rate."""
        self.current_step += 1
        lr = self.schedule_fn(self.current_step)
        self.optimizer.learning_rate.assign(lr)
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.learning_rate)


# ==============================================================================
# Entropy Scheduling
# ==============================================================================

def entropy_schedule(
    initial_coef: float = 0.01,
    final_coef: float = 0.001,
    total_steps: int = 1000000,
    schedule_type: str = 'linear',
) -> Callable[[int], float]:
    """
    Create an entropy coefficient schedule.
    
    It's often helpful to decrease the entropy bonus over training
    to encourage more deterministic policies as training progresses.
    
    Args:
        initial_coef: Initial entropy coefficient.
        final_coef: Final entropy coefficient.
        total_steps: Total steps for the schedule.
        schedule_type: 'linear' or 'exponential'.
        
    Returns:
        Function mapping step to entropy coefficient.
    """
    if schedule_type == 'linear':
        return linear_schedule(initial_coef, final_coef, total_steps)
    elif schedule_type == 'exponential':
        decay_rate = (final_coef / initial_coef) ** (1.0 / total_steps)
        return lambda step: max(initial_coef * (decay_rate ** step), final_coef)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


# ==============================================================================
# Data Utilities
# ==============================================================================

def normalize_advantages(
    advantages: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize advantages to zero mean and unit variance.
    
    Args:
        advantages: Advantage values.
        eps: Small constant for numerical stability.
        
    Returns:
        Normalized advantages.
    """
    mean = np.mean(advantages)
    std = np.std(advantages)
    return (advantages - mean) / (std + eps)


def explained_variance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute explained variance.
    
    ev = 1 - Var(y_true - y_pred) / Var(y_true)
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        
    Returns:
        Explained variance in [-inf, 1]. 1 is perfect prediction.
    """
    var_true = np.var(y_true)
    if var_true == 0:
        return 0.0
    return 1.0 - np.var(y_true - y_pred) / var_true


def compute_discounted_cumsum(
    x: np.ndarray,
    discount: float,
) -> np.ndarray:
    """
    Compute discounted cumulative sum.
    
    For array [x0, x1, x2, ...] with discount gamma, computes:
    [x0 + gamma*x1 + gamma^2*x2 + ..., x1 + gamma*x2 + ..., ...]
    
    Args:
        x: Input array.
        discount: Discount factor.
        
    Returns:
        Discounted cumulative sums.
    """
    result = np.zeros_like(x)
    running_sum = 0.0
    for i in reversed(range(len(x))):
        running_sum = x[i] + discount * running_sum
        result[i] = running_sum
    return result


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def plot_training_curves(
    history: List[Dict[str, float]],
    metrics: List[str],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training curves from history.
    
    Args:
        history: List of metric dictionaries.
        metrics: List of metric names to plot.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return
    
    steps = [h.get('step', i) for i, h in enumerate(history)]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        values = [h.get(metric, np.nan) for h in history]
        ax.plot(steps, values)
        ax.set_xlabel('Step')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()


# ==============================================================================
# Experiment Management
# ==============================================================================

def save_config(config: Any, path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration object (must have __dict__ or be a dict).
        path: Path to save to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(config, '__dict__'):
        config_dict = {k: v for k, v in config.__dict__.items() 
                      if not k.startswith('_')}
    else:
        config_dict = dict(config)
    
    # Convert non-serializable types
    for key, value in config_dict.items():
        if isinstance(value, np.ndarray):
            config_dict[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            config_dict[key] = value.item()
    
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to load from.
        
    Returns:
        Configuration dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Try to set Python's random seed too
    import random
    random.seed(seed)


# ==============================================================================
# Performance Monitoring
# ==============================================================================

class Timer:
    """Simple timer for profiling."""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        """Start timing a named operation."""
        import time
        self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record duration."""
        import time
        duration = time.perf_counter() - self._start_times[name]
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(duration)
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named operation."""
        if name not in self.times:
            return {}
        times = self.times[name]
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'count': len(times),
        }
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all operations."""
        return {name: self.get_stats(name) for name in self.times}


class MovingAverage:
    """Exponential moving average for metrics."""
    
    def __init__(self, decay: float = 0.99):
        """
        Initialize moving average.
        
        Args:
            decay: Decay factor (higher = slower update).
        """
        self.decay = decay
        self.value: Optional[float] = None
        self.count = 0
    
    def update(self, value: float) -> float:
        """
        Update with new value and return current average.
        
        Args:
            value: New value to incorporate.
            
        Returns:
            Current moving average.
        """
        if self.value is None:
            self.value = value
        else:
            self.value = self.decay * self.value + (1 - self.decay) * value
        self.count += 1
        return self.value
    
    def get(self) -> float:
        """Get current moving average."""
        return self.value if self.value is not None else 0.0


# ==============================================================================
# Model Utilities
# ==============================================================================

def count_parameters(model: tf.keras.Model) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: Keras model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(np.prod(v.shape) for v in model.trainable_variables)


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get a string summary of the model.
    
    Args:
        model: Keras model.
        
    Returns:
        Summary string.
    """
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return '\n'.join(lines)
