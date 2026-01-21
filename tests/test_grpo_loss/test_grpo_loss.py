"""
Tests for GRPO loss computation.

Tests cover:
- Group-relative advantage computation
- GRPO loss computation
- Gradient computation
- GRPOOptimizer updates
"""

import pytest
import logging
import numpy as np
import tensorflow as tf

from grpo_atari.grpo_loss import (
    compute_trajectory_returns,
    compute_group_advantages,
    GRPOLoss,
    GRPOOptimizer,
)
from grpo_atari.trajectory import TrajectoryBatch, GroupedTrajectories, Trajectory
from grpo_atari.model import create_policy_network

logger = logging.getLogger(__name__)


class TestComputeTrajectoryReturns:
    """Tests for trajectory return computation."""
    
    def test_trajectory_returns_shape(self, test_logger):
        """Test compute_trajectory_returns output shape."""
        test_logger.log_section("Trajectory Returns Shape")
        
        num_groups = 2
        group_size = 4
        T = 10
        
        rewards = np.random.randn(num_groups, group_size, T).astype(np.float32) * 0.1
        dones = np.zeros((num_groups, group_size, T), dtype=bool)
        gamma = 0.99
        
        returns = compute_trajectory_returns(rewards, dones, gamma)
        
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("returns shape", returns.shape)
        
        test_logger.log_assert_shape("returns", rewards.shape, returns)
    
    def test_trajectory_returns_values(self, test_logger):
        """Test trajectory return values are correct."""
        test_logger.log_section("Trajectory Returns Values")
        
        rewards = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        dones = np.zeros((1, 1, 3), dtype=bool)
        gamma = 0.5
        
        returns = compute_trajectory_returns(rewards, dones, gamma)
        
        # R[0] = 1 + 0.5 + 0.25 = 1.75
        # R[1] = 1 + 0.5 = 1.5
        # R[2] = 1
        expected = np.array([[[1.75, 1.5, 1.0]]])
        
        test_logger.log_value("rewards", rewards.tolist())
        test_logger.log_value("gamma", gamma)
        test_logger.log_value("expected returns", expected.tolist())
        test_logger.log_value("actual returns", returns.tolist())
        
        np.testing.assert_array_almost_equal(returns, expected, decimal=5)
        logger.info("  ✓ Trajectory returns computed correctly")


class TestComputeGroupAdvantages:
    """Tests for group-relative advantage computation."""
    
    def test_group_advantages_shape(self, test_logger):
        """Test compute_group_advantages output shape."""
        test_logger.log_section("Group Advantages Shape")
        
        num_groups = 2
        group_size = 4
        T = 10
        
        rewards = np.random.randn(num_groups, group_size, T).astype(np.float32) * 0.1
        dones = np.zeros((num_groups, group_size, T), dtype=bool)
        gamma = 0.99
        
        advantages, stats = compute_group_advantages(rewards, dones, gamma)
        
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("advantages shape", advantages.shape)
        
        test_logger.log_assert_shape("advantages", rewards.shape, advantages)
    
    def test_group_advantages_normalized(self, test_logger):
        """Test that advantages are normalized (mean≈0, std≈1)."""
        test_logger.log_section("Group Advantages Normalized")
        
        np.random.seed(42)
        num_groups = 3
        group_size = 8
        T = 20
        
        rewards = np.random.randn(num_groups, group_size, T).astype(np.float32) * 0.5
        dones = np.zeros((num_groups, group_size, T), dtype=bool)
        
        advantages, stats = compute_group_advantages(rewards, dones, gamma=0.99)
        
        test_logger.log_value("advantage_mean", stats['advantage_mean'])
        test_logger.log_value("advantage_std", stats['advantage_std'])
        test_logger.log_value("advantage_min", stats['advantage_min'])
        test_logger.log_value("advantage_max", stats['advantage_max'])
        
        test_logger.log_assert_close("advantage_mean", 0.0, stats['advantage_mean'], tolerance=1e-5)
        test_logger.log_assert_close("advantage_std", 1.0, stats['advantage_std'], tolerance=0.1)
    
    def test_group_advantages_group_relative(self, test_logger):
        """Test that advantages are relative to group mean (key GRPO property)."""
        test_logger.log_section("Group-Relative Property")
        
        np.random.seed(42)
        num_groups = 3
        group_size = 6
        T = 15
        
        rewards = np.random.randn(num_groups, group_size, T).astype(np.float32)
        dones = np.zeros((num_groups, group_size, T), dtype=bool)
        
        advantages, _ = compute_group_advantages(rewards, dones, gamma=0.99, normalize=False)
        
        # Within each group, mean advantage should be ~0
        for g in range(num_groups):
            group_mean = np.mean(advantages[g])
            test_logger.log_value(f"group {g} advantage mean", group_mean)
            test_logger.log_assert_close(f"group {g} mean", 0.0, group_mean, tolerance=1e-5)
        
        logger.info("  ✓ Advantages are group-relative (mean=0 within each group)")
    
    def test_group_advantages_statistics(self, test_logger):
        """Test that statistics dictionary contains expected keys."""
        test_logger.log_section("Group Advantages Statistics")
        
        rewards = np.random.randn(2, 4, 10).astype(np.float32)
        dones = np.zeros((2, 4, 10), dtype=bool)
        
        _, stats = compute_group_advantages(rewards, dones, gamma=0.99)
        
        expected_keys = [
            'advantage_mean', 'advantage_std', 'advantage_min', 'advantage_max',
            'return_mean', 'return_std', 'group_return_spread'
        ]
        
        for key in expected_keys:
            test_logger.log_value(key, stats.get(key, 'MISSING'))
            assert key in stats, f"Missing key: {key}"
        
        logger.info(f"  ✓ All {len(expected_keys)} expected statistics present")


class TestGRPOLoss:
    """Tests for GRPO loss computation."""
    
    @pytest.fixture
    def grpo_loss(self, policy_network):
        """Create a GRPOLoss instance."""
        return GRPOLoss(
            policy=policy_network,
            epsilon_clip=0.1,
            entropy_coef=0.01,
            max_grad_norm=0.5,
        )
    
    def test_grpo_loss_computation(self, test_logger, grpo_loss, obs_shape):
        """Test GRPO loss computation."""
        test_logger.log_section("GRPO Loss Computation")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        actions = tf.random.uniform((batch_size,), 0, 4, dtype=tf.int32)
        old_log_probs = tf.random.uniform((batch_size,), -2, 0)
        advantages = tf.random.normal((batch_size,))
        
        loss, metrics = grpo_loss.compute_loss(obs, actions, old_log_probs, advantages)
        
        test_logger.log_value("loss shape", loss.shape)
        test_logger.log_value("loss value", float(loss))
        test_logger.log_value("loss dtype", loss.dtype)
        
        # Check loss is scalar
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        
        # Check metrics
        test_logger.log_subsection("Metrics")
        for key, value in metrics.items():
            test_logger.log_value(key, float(value))
        
        expected_metrics = ['policy_loss', 'entropy_loss', 'entropy', 'total_loss', 'approx_kl', 'clip_fraction']
        for key in expected_metrics:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_grpo_loss_entropy_contribution(self, test_logger, policy_network, obs_shape):
        """Test that entropy coefficient affects loss."""
        test_logger.log_section("Entropy Coefficient Effect")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        actions = tf.random.uniform((batch_size,), 0, 4, dtype=tf.int32)
        old_log_probs = tf.random.uniform((batch_size,), -2, 0)
        advantages = tf.random.normal((batch_size,))
        
        # Low entropy coefficient
        loss_fn_low = GRPOLoss(policy=policy_network, entropy_coef=0.001)
        loss_low, metrics_low = loss_fn_low.compute_loss(obs, actions, old_log_probs, advantages)
        
        # High entropy coefficient
        loss_fn_high = GRPOLoss(policy=policy_network, entropy_coef=0.1)
        loss_high, metrics_high = loss_fn_high.compute_loss(obs, actions, old_log_probs, advantages)
        
        test_logger.log_value("entropy_coef=0.001 loss", float(loss_low))
        test_logger.log_value("entropy_coef=0.1 loss", float(loss_high))
        test_logger.log_value("entropy_loss (low)", float(metrics_low['entropy_loss']))
        test_logger.log_value("entropy_loss (high)", float(metrics_high['entropy_loss']))
        
        # Higher entropy coef should have more negative entropy_loss (entropy bonus)
        assert float(metrics_high['entropy_loss']) < float(metrics_low['entropy_loss'])
        logger.info("  ✓ Higher entropy coefficient produces larger entropy bonus")
    
    def test_grpo_gradient_computation(self, test_logger, grpo_loss, policy_network, obs_shape):
        """Test gradient computation."""
        test_logger.log_section("Gradient Computation")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        actions = tf.random.uniform((batch_size,), 0, 4, dtype=tf.int32)
        old_log_probs = tf.random.uniform((batch_size,), -2, 0)
        advantages = tf.random.normal((batch_size,))
        
        gradients, loss, metrics = grpo_loss.compute_gradients(
            obs, actions, old_log_probs, advantages
        )
        
        num_vars = len(policy_network.trainable_variables)
        num_grads = len(gradients)
        
        test_logger.log_value("num trainable variables", num_vars)
        test_logger.log_value("num gradients", num_grads)
        
        test_logger.log_assert_equal("gradient count", num_vars, num_grads)
        
        # Check no None gradients
        none_count = sum(1 for g in gradients if g is None)
        test_logger.log_value("None gradients", none_count)
        assert none_count == 0, f"Found {none_count} None gradients"
        
        # Check grad_norm in metrics
        assert 'grad_norm' in metrics
        test_logger.log_value("grad_norm", float(metrics['grad_norm']))


class TestGRPOOptimizer:
    """Tests for GRPOOptimizer."""
    
    def _make_trajectory_batch(self, num_groups=2, group_size=3, T=16):
        """Helper to create a trajectory batch."""
        grouped_list = []
        for g in range(num_groups):
            trajectories = [
                Trajectory(
                    observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
                    actions=np.random.randint(0, 4, size=T),
                    rewards=np.random.rand(T).astype(np.float32) * 0.1,
                    dones=np.zeros(T, dtype=bool),
                    log_probs=np.random.rand(T).astype(np.float32) * -1,
                )
                for _ in range(group_size)
            ]
            grouped_list.append(GroupedTrajectories(
                trajectories=trajectories,
                group_seed=np.random.randint(0, 2**31),
                group_id=g,
            ))
        return TrajectoryBatch(grouped_trajectories=grouped_list)
    
    def test_optimizer_initialization(self, test_logger, policy_network):
        """Test GRPOOptimizer initialization."""
        test_logger.log_section("GRPOOptimizer Initialization")
        
        optimizer = GRPOOptimizer(
            policy=policy_network,
            learning_rate=2.5e-4,
            epsilon_clip=0.1,
            entropy_coef=0.01,
        )
        
        test_logger.log_value("learning_rate", optimizer.learning_rate)
        
        test_logger.log_assert_equal("learning_rate", 2.5e-4, optimizer.learning_rate)
    
    def test_optimizer_update(self, test_logger, policy_network):
        """Test full optimizer update."""
        test_logger.log_section("GRPOOptimizer Update")
        
        optimizer = GRPOOptimizer(
            policy=policy_network,
            learning_rate=2.5e-4,
            epsilon_clip=0.1,
            entropy_coef=0.01,
        )
        
        # Create trajectory batch
        batch = self._make_trajectory_batch(num_groups=2, group_size=4, T=16)
        
        test_logger.log_value("batch num_groups", batch.num_groups)
        test_logger.log_value("batch group_size", batch.group_size)
        
        # Get initial weights
        initial_weights = [v.numpy().copy() for v in policy_network.trainable_variables]
        
        # Perform update
        metrics = optimizer.update(
            trajectory_batch=batch,
            gamma=0.99,
            update_epochs=2,
            minibatch_size=32,
        )
        
        # Check metrics
        test_logger.log_subsection("Update Metrics")
        for key in ['policy_loss', 'entropy', 'advantage_mean']:
            if key in metrics:
                test_logger.log_value(key, metrics[key])
        
        # Check weights changed
        weights_changed = False
        for init_w, final_w in zip(initial_weights, policy_network.trainable_variables):
            if not np.allclose(init_w, final_w.numpy()):
                weights_changed = True
                break
        
        test_logger.log_value("weights changed", weights_changed)
        assert weights_changed, "Optimizer should update weights"
        logger.info("  ✓ Optimizer successfully updated policy weights")
    
    def test_optimizer_learning_rate_setter(self, test_logger, policy_network):
        """Test learning rate can be updated."""
        test_logger.log_section("Learning Rate Update")
        
        optimizer = GRPOOptimizer(
            policy=policy_network,
            learning_rate=1e-4,
        )
        
        initial_lr = optimizer.learning_rate
        test_logger.log_value("initial learning_rate", initial_lr)
        
        new_lr = 5e-5
        optimizer.set_learning_rate(new_lr)
        
        test_logger.log_value("new learning_rate", new_lr)
        test_logger.log_value("actual learning_rate", float(optimizer.optimizer.learning_rate))
        
        test_logger.log_assert_close(
            "learning_rate after update",
            new_lr,
            float(optimizer.optimizer.learning_rate),
            tolerance=1e-10
        )
