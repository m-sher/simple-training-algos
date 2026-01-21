"""
Tests for environment wrappers and seeding.

Tests cover:
- Single environment creation and preprocessing
- Seeded environment groups (same initial state)
- Parallel environment groups management
- Environment stepping and reset behavior
"""

import pytest
import logging
import numpy as np

from grpo_atari.environment import (
    create_atari_env,
    SeededEnvGroup,
    ParallelEnvGroups,
)

logger = logging.getLogger(__name__)


class TestSingleEnvironment:
    """Tests for single Atari environment creation."""
    
    def test_create_atari_env(self, test_logger):
        """Test single environment creation with correct preprocessing."""
        test_logger.log_section("Create Atari Environment")
        
        env = create_atari_env('ALE/Breakout-v5', seed=42)
        obs, info = env.reset(seed=42)
        
        test_logger.log_value("observation shape", obs.shape)
        test_logger.log_value("observation dtype", obs.dtype)
        test_logger.log_value("observation min", float(obs.min()))
        test_logger.log_value("observation max", float(obs.max()))
        test_logger.log_value("action space", env.action_space)
        
        test_logger.log_assert_equal("obs shape", (84, 84, 4), obs.shape)
        test_logger.log_assert_equal("obs dtype", np.float32, obs.dtype)
        
        # Check normalization to [0, 1]
        test_logger.log_assert_range("obs min", float(obs.min()), 0.0, 1.0)
        test_logger.log_assert_range("obs max", float(obs.max()), 0.0, 1.0)
        
        # Check action space
        test_logger.log_assert_equal("num actions", 4, env.action_space.n)
        
        env.close()
        logger.info("  ✓ Environment created and closed successfully")
    
    def test_environment_step(self, test_logger):
        """Test environment stepping returns correct types."""
        test_logger.log_section("Environment Step")
        
        env = create_atari_env('ALE/Breakout-v5', seed=42)
        obs, _ = env.reset(seed=42)
        
        action = 0  # NOOP
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        test_logger.log_value("action taken", action)
        test_logger.log_value("next_obs shape", next_obs.shape)
        test_logger.log_value("reward", reward)
        test_logger.log_value("reward type", type(reward).__name__)
        test_logger.log_value("terminated", terminated)
        test_logger.log_value("truncated", truncated)
        
        test_logger.log_assert_equal("next_obs shape", (84, 84, 4), next_obs.shape)
        assert isinstance(reward, (int, float)), f"reward should be numeric, got {type(reward)}"
        assert isinstance(terminated, bool), f"terminated should be bool, got {type(terminated)}"
        assert isinstance(truncated, bool), f"truncated should be bool, got {type(truncated)}"
        
        logger.info("  ✓ Step returns correct types")
        env.close()
    
    def test_environment_determinism(self, test_logger):
        """Test that same seed produces same initial state."""
        test_logger.log_section("Environment Determinism")
        
        env1 = create_atari_env('ALE/Breakout-v5')
        env2 = create_atari_env('ALE/Breakout-v5')
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        are_equal = np.array_equal(obs1, obs2)
        max_diff = np.max(np.abs(obs1 - obs2))
        
        test_logger.log_value("observations equal", are_equal)
        test_logger.log_value("max difference", max_diff)
        
        assert are_equal, f"Same seed should produce same obs, max diff: {max_diff}"
        logger.info("  ✓ Same seed produces identical observations")
        
        env1.close()
        env2.close()


class TestSeededEnvGroup:
    """Tests for SeededEnvGroup (same initial state)."""
    
    def test_seeded_group_creation(self, test_logger, env_factory):
        """Test SeededEnvGroup creation."""
        test_logger.log_section("SeededEnvGroup Creation")
        
        group = SeededEnvGroup(env_factory, group_size=3, seed=42)
        
        test_logger.log_value("group_size", group.group_size)
        test_logger.log_value("seed", group.seed)
        test_logger.log_value("num_actions", group.num_actions)
        
        test_logger.log_assert_equal("group_size", 3, group.group_size)
        test_logger.log_assert_equal("seed", 42, group.seed)
        test_logger.log_assert_equal("num_actions", 4, group.num_actions)
        
        group.close()
    
    def test_seeded_group_same_initial_state(self, test_logger, env_factory):
        """Test that all envs in group have identical initial state."""
        test_logger.log_section("Seeded Group Same Initial State")
        
        group = SeededEnvGroup(env_factory, group_size=4, seed=42)
        obs = group.reset()
        
        test_logger.log_value("observations shape", obs.shape)
        test_logger.log_assert_shape("observations", (4, 84, 84, 4), obs)
        
        # Check all observations are identical
        test_logger.log_subsection("Pairwise Comparison")
        for i in range(1, 4):
            are_equal = np.array_equal(obs[0], obs[i])
            max_diff = np.max(np.abs(obs[0].astype(float) - obs[i].astype(float)))
            test_logger.log_value(f"obs[0] == obs[{i}]", are_equal)
            test_logger.log_value(f"max diff [0] vs [{i}]", max_diff)
            assert are_equal, f"obs[0] != obs[{i}], max diff: {max_diff}"
        
        logger.info("  ✓ All observations in group are identical")
        group.close()
    
    def test_seeded_group_step(self, test_logger, env_factory):
        """Test stepping all environments in a group."""
        test_logger.log_section("Seeded Group Step")
        
        group = SeededEnvGroup(env_factory, group_size=3, seed=42)
        obs = group.reset()
        
        actions = np.array([0, 1, 2])
        next_obs, rewards, terminateds, truncateds, infos = group.step(actions)
        
        test_logger.log_value("actions", actions)
        test_logger.log_value("next_obs shape", next_obs.shape)
        test_logger.log_value("rewards", rewards)
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("terminateds", terminateds)
        
        test_logger.log_assert_shape("next_obs", (3, 84, 84, 4), next_obs)
        test_logger.log_assert_shape("rewards", (3,), rewards)
        test_logger.log_assert_shape("terminateds", (3,), terminateds)
        
        group.close()
    
    def test_seeded_group_different_seeds_differ(self, test_logger, env_factory):
        """Test that different seeds produce different initial states."""
        test_logger.log_section("Different Seeds Differ")
        
        group1 = SeededEnvGroup(env_factory, group_size=2, seed=42)
        group2 = SeededEnvGroup(env_factory, group_size=2, seed=123)
        
        obs1 = group1.reset()
        obs2 = group2.reset()
        
        are_equal = np.array_equal(obs1[0], obs2[0])
        max_diff = np.max(np.abs(obs1[0].astype(float) - obs2[0].astype(float)))
        
        test_logger.log_value("group1 seed", 42)
        test_logger.log_value("group2 seed", 123)
        test_logger.log_value("observations equal", are_equal)
        test_logger.log_value("max difference", max_diff)
        
        assert not are_equal, "Different seeds should produce different observations"
        logger.info("  ✓ Different seeds produce different observations")
        
        group1.close()
        group2.close()


class TestParallelEnvGroups:
    """Tests for ParallelEnvGroups management."""
    
    def test_parallel_groups_creation(self, test_logger, env_factory):
        """Test ParallelEnvGroups creation."""
        test_logger.log_section("ParallelEnvGroups Creation")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=3,
            group_size=4,
            base_seed=42,
        )
        
        test_logger.log_value("num_groups", env_groups.num_groups)
        test_logger.log_value("group_size", env_groups.group_size)
        test_logger.log_value("total_envs", env_groups.total_envs)
        test_logger.log_value("num_actions", env_groups.num_actions)
        
        test_logger.log_assert_equal("num_groups", 3, env_groups.num_groups)
        test_logger.log_assert_equal("group_size", 4, env_groups.group_size)
        test_logger.log_assert_equal("total_envs", 12, env_groups.total_envs)
        
        env_groups.close()
    
    def test_parallel_groups_reset_shape(self, test_logger, env_factory):
        """Test that reset returns correct observation shape."""
        test_logger.log_section("ParallelEnvGroups Reset Shape")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=3,
            base_seed=42,
        )
        
        obs = env_groups.reset()
        
        expected_shape = (2, 3, 84, 84, 4)
        test_logger.log_value("observations shape", obs.shape)
        test_logger.log_assert_shape("observations", expected_shape, obs)
        
        env_groups.close()
    
    def test_parallel_groups_within_group_same(self, test_logger, env_factory):
        """Test that within each group, initial states are identical."""
        test_logger.log_section("Within-Group Same State")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=3,
            base_seed=42,
        )
        
        obs = env_groups.reset()
        
        for g in range(2):
            test_logger.log_subsection(f"Group {g}")
            for e in range(1, 3):
                are_equal = np.array_equal(obs[g, 0], obs[g, e])
                test_logger.log_value(f"obs[{g},0] == obs[{g},{e}]", are_equal)
                assert are_equal, f"Within group {g}: obs[0] != obs[{e}]"
            logger.info(f"  ✓ Group {g}: all envs have identical initial state")
        
        env_groups.close()
    
    def test_parallel_groups_between_groups_differ(self, test_logger, env_factory):
        """Test that between groups, initial states differ."""
        test_logger.log_section("Between-Group Different States")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=2,
            base_seed=42,
        )
        
        obs = env_groups.reset()
        
        are_equal = np.array_equal(obs[0, 0], obs[1, 0])
        max_diff = np.max(np.abs(obs[0, 0].astype(float) - obs[1, 0].astype(float)))
        
        test_logger.log_value("group 0 seed", int(env_groups.group_seeds[0]))
        test_logger.log_value("group 1 seed", int(env_groups.group_seeds[1]))
        test_logger.log_value("obs[0,0] == obs[1,0]", are_equal)
        test_logger.log_value("max difference", max_diff)
        
        assert not are_equal, "Groups should have different initial states"
        logger.info("  ✓ Different groups have different initial states")
        
        env_groups.close()
    
    def test_parallel_groups_step(self, test_logger, env_factory):
        """Test stepping all environment groups."""
        test_logger.log_section("ParallelEnvGroups Step")
        
        env_groups = ParallelEnvGroups(
            env_fn=env_factory,
            num_groups=2,
            group_size=2,
            base_seed=42,
        )
        
        obs = env_groups.reset()
        actions = np.array([[0, 1], [2, 0]])
        
        next_obs, rewards, terminateds, truncateds, infos = env_groups.step(actions)
        
        test_logger.log_value("actions shape", actions.shape)
        test_logger.log_value("next_obs shape", next_obs.shape)
        test_logger.log_value("rewards shape", rewards.shape)
        test_logger.log_value("rewards", rewards)
        
        test_logger.log_assert_shape("next_obs", (2, 2, 84, 84, 4), next_obs)
        test_logger.log_assert_shape("rewards", (2, 2), rewards)
        test_logger.log_assert_shape("terminateds", (2, 2), terminateds)
        
        env_groups.close()
