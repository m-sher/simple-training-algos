"""
Pytest configuration and shared fixtures for GRPO Atari tests.

This module provides:
- Common fixtures for environments, policies, and configurations
- Logging utilities for verbose test output
- Cleanup utilities for temporary resources
"""

import os
import sys
import logging
import tempfile
import shutil
from typing import Callable, Generator, Any

import pytest
import numpy as np

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grpo_atari.config import GRPOConfig
from grpo_atari.environment import create_atari_env, ParallelEnvGroups
from grpo_atari.model import PolicyNetwork, create_policy_network


# Configure logging for tests
logger = logging.getLogger(__name__)


class TestLogger:
    """
    Utility class for verbose test logging.
    
    Provides methods to log test values, comparisons, and results
    in a consistent format that's visible with pytest -v flags.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def log_value(self, description: str, value: Any, level: int = logging.INFO) -> None:
        """Log a single value with description."""
        if isinstance(value, np.ndarray):
            value_str = f"shape={value.shape}, dtype={value.dtype}, min={value.min():.4f}, max={value.max():.4f}"
        elif isinstance(value, tf.Tensor):
            value_str = f"shape={value.shape}, dtype={value.dtype}"
        elif isinstance(value, float):
            value_str = f"{value:.6f}"
        else:
            value_str = str(value)
        self.logger.log(level, f"  {description}: {value_str}")
    
    def log_comparison(
        self, 
        description: str, 
        expected: Any, 
        actual: Any, 
        passed: bool,
        level: int = logging.INFO
    ) -> None:
        """Log a comparison between expected and actual values."""
        status = "✓ PASS" if passed else "✗ FAIL"
        self.logger.log(level, f"  {status} | {description}")
        self.logger.log(level, f"       Expected: {expected}")
        self.logger.log(level, f"       Actual:   {actual}")
    
    def log_assert_equal(self, description: str, expected: Any, actual: Any) -> None:
        """Log and assert equality."""
        passed = expected == actual
        self.log_comparison(description, expected, actual, passed)
        assert passed, f"{description}: expected {expected}, got {actual}"
    
    def log_assert_shape(self, description: str, expected_shape: tuple, tensor: Any) -> None:
        """Log and assert tensor shape."""
        if isinstance(tensor, tf.Tensor):
            actual_shape = tuple(tensor.shape.as_list())
        else:
            actual_shape = tensor.shape
        passed = expected_shape == actual_shape
        self.log_comparison(f"{description} shape", expected_shape, actual_shape, passed)
        assert passed, f"{description}: expected shape {expected_shape}, got {actual_shape}"
    
    def log_assert_close(
        self, 
        description: str, 
        expected: float, 
        actual: float, 
        tolerance: float = 1e-5
    ) -> None:
        """Log and assert approximate equality."""
        diff = abs(expected - actual)
        passed = diff <= tolerance
        self.log_comparison(
            f"{description} (tol={tolerance})", 
            f"{expected:.6f}", 
            f"{actual:.6f} (diff={diff:.2e})",
            passed
        )
        assert passed, f"{description}: expected {expected}, got {actual}, diff {diff} > tol {tolerance}"
    
    def log_assert_range(
        self, 
        description: str, 
        value: float, 
        min_val: float, 
        max_val: float
    ) -> None:
        """Log and assert value is within range."""
        passed = min_val <= value <= max_val
        self.log_comparison(
            f"{description} in [{min_val}, {max_val}]",
            f"[{min_val}, {max_val}]",
            f"{value:.6f}",
            passed
        )
        assert passed, f"{description}: {value} not in [{min_val}, {max_val}]"
    
    def log_section(self, title: str) -> None:
        """Log a section header."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"  {title}")
        self.logger.info(f"{'='*60}")
    
    def log_subsection(self, title: str) -> None:
        """Log a subsection header."""
        self.logger.info(f"\n  --- {title} ---")


@pytest.fixture
def test_logger(request) -> TestLogger:
    """Provide a test logger for verbose output."""
    return TestLogger(request.node.name)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory that's cleaned up after the test."""
    temp_path = tempfile.mkdtemp(prefix="grpo_test_")
    logger.debug(f"Created temp directory: {temp_path}")
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)
    logger.debug(f"Cleaned up temp directory: {temp_path}")


@pytest.fixture
def default_config() -> GRPOConfig:
    """Provide a default GRPO configuration."""
    config = GRPOConfig()
    logger.debug(f"Created default config: group_size={config.group_size}, num_groups={config.num_groups}")
    return config


@pytest.fixture
def small_config(temp_dir: str) -> GRPOConfig:
    """Provide a small configuration for fast testing."""
    config = GRPOConfig(
        group_size=2,
        num_groups=2,
        steps_per_trajectory=16,
        num_iterations=10,
        update_epochs=1,
        minibatch_size=32,
        log_interval=5,
        save_interval=5,
        eval_interval=100,  # Skip eval during tests
        checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
        log_dir=os.path.join(temp_dir, 'logs'),
        seed=42,
    )
    logger.debug(f"Created small config: batch_size={config.batch_size}")
    return config


@pytest.fixture
def obs_shape() -> tuple:
    """Standard observation shape for Atari."""
    return (84, 84, 4)


@pytest.fixture
def num_actions() -> int:
    """Number of actions for Breakout."""
    return 4


@pytest.fixture
def policy_network(obs_shape: tuple, num_actions: int) -> PolicyNetwork:
    """Create a policy network for testing."""
    policy = create_policy_network(obs_shape=obs_shape, num_actions=num_actions)
    logger.debug(f"Created policy network: {sum(v.numpy().size for v in policy.trainable_variables):,} parameters")
    return policy


@pytest.fixture
def env_factory() -> Callable:
    """Factory function for creating Atari environments."""
    def factory():
        return create_atari_env('ALE/Breakout-v5')
    return factory


@pytest.fixture
def parallel_env_groups(env_factory: Callable) -> Generator[ParallelEnvGroups, None, None]:
    """Create parallel environment groups for testing."""
    env_groups = ParallelEnvGroups(
        env_fn=env_factory,
        num_groups=2,
        group_size=2,
        base_seed=42,
    )
    logger.debug(f"Created parallel env groups: {env_groups.total_envs} total envs")
    yield env_groups
    env_groups.close()
    logger.debug("Closed parallel env groups")


@pytest.fixture
def random_observations(obs_shape: tuple) -> np.ndarray:
    """Generate random observations for testing."""
    obs = np.random.rand(4, *obs_shape).astype(np.float32)
    return obs


@pytest.fixture
def random_actions(num_actions: int) -> np.ndarray:
    """Generate random actions for testing."""
    return np.random.randint(0, num_actions, size=4)


# Hooks for better test output
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
