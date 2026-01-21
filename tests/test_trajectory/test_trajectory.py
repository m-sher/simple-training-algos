"""
Tests for trajectory collection and data structures.

Tests cover:
- Trajectory dataclass
- GroupedTrajectories
- TrajectoryBatch
- Return computation
- TrajectoryCollector
"""

import pytest
import logging
import numpy as np
import tensorflow as tf

from grpo_atari.trajectory import (
    Trajectory,
    GroupedTrajectories,
    TrajectoryBatch,
    TrajectoryCollector,
    compute_returns,
)
from grpo_atari.environment import ParallelEnvGroups
from grpo_atari.model import create_policy_network

logger = logging.getLogger(__name__)


class TestTrajectoryDataclass:
    """Tests for Trajectory dataclass."""
    
    def test_trajectory_creation(self, test_logger):
        """Test Trajectory creation."""
        test_logger.log_section("Trajectory Creation")
        
        T = 10
        traj = Trajectory(
            observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
            actions=np.random.randint(0, 4, size=T),
            rewards=np.random.rand(T).astype(np.float32),
            dones=np.zeros(T, dtype=bool),
            log_probs=np.random.rand(T).astype(np.float32) * -1,
        )
        
        test_logger.log_value("observations shape", traj.observations.shape)
        test_logger.log_value("actions shape", traj.actions.shape)
        test_logger.log_value("rewards shape", traj.rewards.shape)
        test_logger.log_value("dones shape", traj.dones.shape)
        test_logger.log_value("log_probs shape", traj.log_probs.shape)
        
        test_logger.log_assert_equal("length", T, traj.length)
    
    def test_trajectory_length_property(self, test_logger):
        """Test Trajectory length property."""
        test_logger.log_section("Trajectory Length")
        
        for T in [5, 10, 20, 100]:
            traj = Trajectory(
                observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
                actions=np.random.randint(0, 4, size=T),
                rewards=np.random.rand(T).astype(np.float32),
                dones=np.zeros(T, dtype=bool),
                log_probs=np.random.rand(T).astype(np.float32) * -1,
            )
            test_logger.log_value(f"T={T} length", traj.length)
            test_logger.log_assert_equal(f"T={T}", T, traj.length)
    
    def test_trajectory_total_reward(self, test_logger):
        """Test Trajectory total_reward property."""
        test_logger.log_section("Trajectory Total Reward")
        
        T = 10
        rewards = np.array([1.0, 0.5, 0.0, -0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5], dtype=np.float32)
        
        traj = Trajectory(
            observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
            actions=np.random.randint(0, 4, size=T),
            rewards=rewards,
            dones=np.zeros(T, dtype=bool),
            log_probs=np.random.rand(T).astype(np.float32) * -1,
        )
        
        expected_total = float(np.sum(rewards))
        actual_total = traj.total_reward
        
        test_logger.log_value("rewards", rewards.tolist())
        test_logger.log_value("expected sum", expected_total)
        test_logger.log_value("actual total_reward", actual_total)
        
        test_logger.log_assert_close("total_reward", expected_total, actual_total)


class TestComputeReturns:
    """Tests for return computation."""
    
    def test_compute_returns_shape(self, test_logger):
        """Test compute_returns output shape."""
        test_logger.log_section("Compute Returns Shape")
        
        T = 10
        rewards = np.random.rand(T).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        gamma = 0.99
        
        returns = compute_returns(rewards, dones, gamma)
        
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("returns shape", returns.shape)
        test_logger.log_value("returns dtype", returns.dtype)
        
        test_logger.log_assert_equal("returns shape", (T,), returns.shape)
        test_logger.log_assert_equal("returns dtype", np.float32, returns.dtype)
    
    def test_compute_returns_no_discount(self, test_logger):
        """Test returns with gamma=1 (no discounting)."""
        test_logger.log_section("Returns No Discount (gamma=1)")
        
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        dones = np.zeros(5, dtype=bool)
        gamma = 1.0
        
        returns = compute_returns(rewards, dones, gamma)
        
        # With gamma=1, return at t=0 should be sum of all rewards
        expected_returns = [15.0, 14.0, 12.0, 9.0, 5.0]
        
        test_logger.log_value("rewards", rewards.tolist())
        test_logger.log_value("gamma", gamma)
        test_logger.log_value("expected returns", expected_returns)
        test_logger.log_value("actual returns", returns.tolist())
        
        for t, (exp, act) in enumerate(zip(expected_returns, returns)):
            test_logger.log_assert_close(f"return[{t}]", exp, act)
    
    def test_compute_returns_with_discount(self, test_logger):
        """Test returns with discounting."""
        test_logger.log_section("Returns With Discount")
        
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        dones = np.zeros(3, dtype=bool)
        gamma = 0.5
        
        returns = compute_returns(rewards, dones, gamma)
        
        # R[0] = 1 + 0.5*1 + 0.25*1 = 1.75
        # R[1] = 1 + 0.5*1 = 1.5
        # R[2] = 1
        expected_returns = [1.75, 1.5, 1.0]
        
        test_logger.log_value("rewards", rewards.tolist())
        test_logger.log_value("gamma", gamma)
        test_logger.log_value("expected returns", expected_returns)
        test_logger.log_value("actual returns", returns.tolist())
        
        for t, (exp, act) in enumerate(zip(expected_returns, returns)):
            test_logger.log_assert_close(f"return[{t}]", exp, act)
    
    def test_compute_returns_with_done(self, test_logger):
        """Test returns reset at episode boundaries."""
        test_logger.log_section("Returns With Episode Termination")
        
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        dones = np.array([False, False, True, False, False], dtype=bool)
        gamma = 1.0
        
        returns = compute_returns(rewards, dones, gamma)
        
        # Episode ends at t=2, so returns don't cross boundary
        # R[0] = 1 + 1 + 1 = 3 (stops at done)
        # R[1] = 1 + 1 = 2
        # R[2] = 1 (done, return = immediate reward)
        # R[3] = 1 + 1 = 2
        # R[4] = 1
        expected_returns = [3.0, 2.0, 1.0, 2.0, 1.0]
        
        test_logger.log_value("rewards", rewards.tolist())
        test_logger.log_value("dones", dones.tolist())
        test_logger.log_value("expected returns", expected_returns)
        test_logger.log_value("actual returns", returns.tolist())
        
        for t, (exp, act) in enumerate(zip(expected_returns, returns)):
            test_logger.log_assert_close(f"return[{t}]", exp, act)
    
    def test_compute_returns_decreasing(self, test_logger):
        """Test that returns are generally decreasing within episodes."""
        test_logger.log_section("Returns Decreasing")
        
        T = 20
        rewards = np.ones(T, dtype=np.float32)
        dones = np.zeros(T, dtype=bool)
        gamma = 0.99
        
        returns = compute_returns(rewards, dones, gamma)
        
        test_logger.log_value("first return", float(returns[0]))
        test_logger.log_value("last return", float(returns[-1]))
        
        # Each return should be >= the next one (for positive rewards)
        for t in range(T - 1):
            assert returns[t] >= returns[t + 1], f"return[{t}] < return[{t+1}]"
        
        logger.info("  ✓ Returns are monotonically decreasing")


class TestGroupedTrajectories:
    """Tests for GroupedTrajectories."""
    
    def _make_trajectory(self, T=10, total_reward=1.0):
        """Helper to create a trajectory with specified total reward."""
        return Trajectory(
            observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
            actions=np.random.randint(0, 4, size=T),
            rewards=np.full(T, total_reward / T, dtype=np.float32),
            dones=np.zeros(T, dtype=bool),
            log_probs=np.random.rand(T).astype(np.float32) * -1,
        )
    
    def test_grouped_trajectories_creation(self, test_logger):
        """Test GroupedTrajectories creation."""
        test_logger.log_section("GroupedTrajectories Creation")
        
        trajectories = [self._make_trajectory() for _ in range(4)]
        
        grouped = GroupedTrajectories(
            trajectories=trajectories,
            group_seed=42,
            group_id=0,
        )
        
        test_logger.log_value("group_size", grouped.group_size)
        test_logger.log_value("group_seed", grouped.group_seed)
        test_logger.log_value("group_id", grouped.group_id)
        
        test_logger.log_assert_equal("group_size", 4, grouped.group_size)
    
    def test_grouped_trajectories_returns(self, test_logger):
        """Test returns property of GroupedTrajectories."""
        test_logger.log_section("GroupedTrajectories Returns")
        
        total_rewards = [1.0, 2.0, 3.0, 4.0]
        trajectories = [self._make_trajectory(total_reward=r) for r in total_rewards]
        
        grouped = GroupedTrajectories(
            trajectories=trajectories,
            group_seed=42,
            group_id=0,
        )
        
        returns = grouped.returns
        
        test_logger.log_value("expected returns", total_rewards)
        test_logger.log_value("actual returns", returns.tolist())
        
        for i, (exp, act) in enumerate(zip(total_rewards, returns)):
            test_logger.log_assert_close(f"return[{i}]", exp, act, tolerance=1e-4)
    
    def test_grouped_trajectories_statistics(self, test_logger):
        """Test get_group_statistics method."""
        test_logger.log_section("GroupedTrajectories Statistics")
        
        total_rewards = [1.0, 2.0, 3.0, 4.0]
        trajectories = [self._make_trajectory(total_reward=r) for r in total_rewards]
        
        grouped = GroupedTrajectories(
            trajectories=trajectories,
            group_seed=42,
            group_id=0,
        )
        
        stats = grouped.get_group_statistics()
        
        expected_mean = np.mean(total_rewards)
        expected_std = np.std(total_rewards)
        
        test_logger.log_value("mean_return", stats['mean_return'])
        test_logger.log_value("std_return", stats['std_return'])
        test_logger.log_value("min_return", stats['min_return'])
        test_logger.log_value("max_return", stats['max_return'])
        
        test_logger.log_assert_close("mean_return", expected_mean, stats['mean_return'], tolerance=1e-4)
        test_logger.log_assert_close("std_return", expected_std, stats['std_return'], tolerance=1e-4)


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch."""
    
    def _make_grouped_trajectories(self, group_size=3, T=10):
        """Helper to create grouped trajectories."""
        trajectories = [
            Trajectory(
                observations=np.random.rand(T, 84, 84, 4).astype(np.float32),
                actions=np.random.randint(0, 4, size=T),
                rewards=np.random.rand(T).astype(np.float32),
                dones=np.zeros(T, dtype=bool),
                log_probs=np.random.rand(T).astype(np.float32) * -1,
            )
            for _ in range(group_size)
        ]
        return GroupedTrajectories(
            trajectories=trajectories,
            group_seed=np.random.randint(0, 2**31),
            group_id=0,
        )
    
    def test_trajectory_batch_properties(self, test_logger):
        """Test TrajectoryBatch properties."""
        test_logger.log_section("TrajectoryBatch Properties")
        
        num_groups = 3
        group_size = 4
        
        grouped_list = [self._make_grouped_trajectories(group_size) for _ in range(num_groups)]
        batch = TrajectoryBatch(grouped_trajectories=grouped_list)
        
        test_logger.log_value("num_groups", batch.num_groups)
        test_logger.log_value("group_size", batch.group_size)
        test_logger.log_value("total_trajectories", batch.total_trajectories)
        
        test_logger.log_assert_equal("num_groups", num_groups, batch.num_groups)
        test_logger.log_assert_equal("group_size", group_size, batch.group_size)
        test_logger.log_assert_equal("total_trajectories", num_groups * group_size, batch.total_trajectories)
    
    def test_trajectory_batch_get_structured_data(self, test_logger):
        """Test get_structured_data method."""
        test_logger.log_section("TrajectoryBatch Structured Data")
        
        num_groups = 2
        group_size = 3
        T = 10
        
        grouped_list = [self._make_grouped_trajectories(group_size, T) for _ in range(num_groups)]
        batch = TrajectoryBatch(grouped_trajectories=grouped_list)
        
        obs, actions, rewards, dones, log_probs = batch.get_structured_data()
        
        test_logger.log_value("observations shape", obs.shape)
        test_logger.log_value("actions shape", actions.shape)
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("dones shape", dones.shape)
        test_logger.log_value("log_probs shape", log_probs.shape)
        
        expected_shape = (num_groups, group_size, T)
        test_logger.log_assert_shape("observations", expected_shape + (84, 84, 4), obs)
        test_logger.log_assert_shape("actions", expected_shape, actions)
        test_logger.log_assert_shape("rewards", expected_shape, rewards)


class TestTrajectoryCollector:
    """Tests for TrajectoryCollector."""
    
    def test_collector_initialization(self, test_logger, env_factory, policy_network):
        """Test TrajectoryCollector initialization."""
        test_logger.log_section("TrajectoryCollector Initialization")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=2,
            base_seed=42,
        )
        
        collector = TrajectoryCollector(
            env_groups=env_groups,
            policy=policy_network,
            steps_per_trajectory=16,
            gamma=0.99,
        )
        
        test_logger.log_value("steps_per_trajectory", collector.steps_per_trajectory)
        test_logger.log_value("gamma", collector.gamma)
        
        test_logger.log_assert_equal("steps_per_trajectory", 16, collector.steps_per_trajectory)
        test_logger.log_assert_equal("gamma", 0.99, collector.gamma)
        
        env_groups.close()
    
    def test_collector_collect(self, test_logger, env_factory, policy_network):
        """Test trajectory collection."""
        test_logger.log_section("TrajectoryCollector Collect")
        
        num_groups = 2
        group_size = 3
        steps_per_trajectory = 16
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=num_groups,
            group_size=group_size,
            base_seed=42,
        )
        
        collector = TrajectoryCollector(
            env_groups=env_groups,
            policy=policy_network,
            steps_per_trajectory=steps_per_trajectory,
            gamma=0.99,
        )
        
        batch = collector.collect()
        
        test_logger.log_value("batch.num_groups", batch.num_groups)
        test_logger.log_value("batch.group_size", batch.group_size)
        test_logger.log_value("batch.total_trajectories", batch.total_trajectories)
        
        test_logger.log_assert_equal("num_groups", num_groups, batch.num_groups)
        test_logger.log_assert_equal("group_size", group_size, batch.group_size)
        
        # Check data shapes
        obs, actions, rewards, dones, log_probs = batch.get_structured_data()
        expected_shape = (num_groups, group_size, steps_per_trajectory)
        
        test_logger.log_value("observations shape", obs.shape)
        test_logger.log_assert_shape("observations", expected_shape + (84, 84, 4), obs)
        test_logger.log_assert_shape("actions", expected_shape, actions)
        
        env_groups.close()
    
    def test_collector_episode_statistics(self, test_logger, env_factory, policy_network):
        """Test episode statistics tracking."""
        test_logger.log_section("TrajectoryCollector Episode Statistics")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=2,
            base_seed=42,
        )
        
        collector = TrajectoryCollector(
            env_groups=env_groups,
            policy=policy_network,
            steps_per_trajectory=32,
        )
        
        # Collect some trajectories
        collector.collect()
        
        stats = collector.get_episode_statistics()
        
        test_logger.log_value("mean_episode_return", stats.get('mean_episode_return', 0))
        test_logger.log_value("mean_episode_length", stats.get('mean_episode_length', 0))
        test_logger.log_value("num_episodes", stats.get('num_episodes', 0))
        
        assert 'mean_episode_return' in stats
        assert 'num_episodes' in stats
        
        env_groups.close()
