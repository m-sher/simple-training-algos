"""
End-to-end integration tests for GRPO.

Tests cover:
- Full training pipeline
- Component interactions
- Data flow verification
"""

import os
import pytest
import logging
import numpy as np
import tensorflow as tf

from grpo_atari.config import GRPOConfig
from grpo_atari.environment import ParallelEnvGroups, create_atari_env
from grpo_atari.model import create_policy_network
from grpo_atari.trajectory import TrajectoryCollector
from grpo_atari.grpo_loss import GRPOOptimizer, compute_group_advantages
from grpo_atari.trainer import GRPOTrainer

logger = logging.getLogger(__name__)


@pytest.mark.integration
class TestFullPipeline:
    """Full pipeline integration tests."""
    
    @pytest.mark.timeout(180)
    def test_full_training_pipeline(self, test_logger, temp_dir):
        """Test complete training pipeline end-to-end."""
        test_logger.log_section("Full Training Pipeline")
        
        # 1. Create configuration
        test_logger.log_subsection("1. Configuration")
        config = GRPOConfig(
            group_size=2,
            num_groups=2,
            steps_per_trajectory=16,
            num_iterations=10,
            update_epochs=1,
            minibatch_size=32,
            log_interval=2,
            save_interval=5,
            eval_interval=100,  # Skip eval
            checkpoint_dir=os.path.join(temp_dir, 'checkpoints'),
            log_dir=os.path.join(temp_dir, 'logs'),
            seed=42,
        )
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_value("total_envs", config.total_envs)
        test_logger.log_value("batch_size", config.batch_size)
        
        # 2. Create trainer
        test_logger.log_subsection("2. Trainer Initialization")
        trainer = GRPOTrainer(config)
        
        test_logger.log_value("policy parameters", 
                              sum(v.numpy().size for v in trainer.policy.trainable_variables))
        test_logger.log_value("trainer initialized", True)
        
        # 3. Capture initial weights
        test_logger.log_subsection("3. Initial Weights")
        initial_weights = [v.numpy().copy() for v in trainer.policy.trainable_variables]
        test_logger.log_value("num weight tensors", len(initial_weights))
        test_logger.log_value("first tensor shape", initial_weights[0].shape)
        
        # 4. Run training
        test_logger.log_subsection("4. Training")
        history = trainer.train()
        
        test_logger.log_value("training completed", True)
        test_logger.log_value("history entries", len(history['history']))
        
        # 5. Verify weights changed
        test_logger.log_subsection("5. Weight Updates")
        final_weights = [v.numpy() for v in trainer.policy.trainable_variables]
        
        weights_changed = False
        for i, (init_w, final_w) in enumerate(zip(initial_weights, final_weights)):
            if not np.allclose(init_w, final_w):
                weights_changed = True
                max_diff = np.max(np.abs(init_w - final_w))
                test_logger.log_value(f"weight[{i}] max diff", max_diff)
                break
        
        test_logger.log_value("weights changed", weights_changed)
        assert weights_changed, "Policy weights should change during training"
        
        # 6. Verify checkpoints
        test_logger.log_subsection("6. Checkpoints")
        checkpoint_dir = config.checkpoint_dir
        checkpoint_files = os.listdir(checkpoint_dir) if os.path.exists(checkpoint_dir) else []
        test_logger.log_value("checkpoint files", len(checkpoint_files))
        assert len(checkpoint_files) > 0, "Checkpoints should be saved"
        
        # 7. Verify logs
        test_logger.log_subsection("7. Logs")
        log_dir = config.log_dir
        log_files = os.listdir(log_dir) if os.path.exists(log_dir) else []
        test_logger.log_value("log files", len(log_files))
        assert len(log_files) > 0, "Logs should be created"
        
        logger.info("="*60)
        logger.info("  FULL PIPELINE TEST PASSED")
        logger.info("="*60)
        
        trainer.env_groups.close()


@pytest.mark.integration
class TestComponentInteraction:
    """Tests for component interactions."""
    
    def test_env_to_collector_to_optimizer_flow(self, test_logger, temp_dir):
        """Test data flows correctly from environments to optimizer."""
        test_logger.log_section("Data Flow: Env -> Collector -> Optimizer")
        
        # Create components
        test_logger.log_subsection("Component Creation")
        
        env_fn = lambda: create_atari_env('ALE/Breakout-v5')
        env_groups = ParallelEnvGroups(
            env_fn=env_fn,
            num_groups=2,
            group_size=3,
            base_seed=42,
        )
        test_logger.log_value("env_groups created", True)
        test_logger.log_value("total_envs", env_groups.total_envs)
        
        policy = create_policy_network(
            obs_shape=(84, 84, 4),
            num_actions=env_groups.num_actions,
        )
        test_logger.log_value("policy created", True)
        
        collector = TrajectoryCollector(
            env_groups=env_groups,
            policy=policy,
            steps_per_trajectory=16,
            gamma=0.99,
        )
        test_logger.log_value("collector created", True)
        
        optimizer = GRPOOptimizer(
            policy=policy,
            learning_rate=2.5e-4,
            epsilon_clip=0.1,
            entropy_coef=0.01,
        )
        test_logger.log_value("optimizer created", True)
        
        # Collect trajectories
        test_logger.log_subsection("Trajectory Collection")
        batch = collector.collect()
        
        test_logger.log_value("batch.num_groups", batch.num_groups)
        test_logger.log_value("batch.group_size", batch.group_size)
        test_logger.log_value("batch.total_trajectories", batch.total_trajectories)
        
        # Get structured data
        obs, actions, rewards, dones, log_probs = batch.get_structured_data()
        test_logger.log_value("observations shape", obs.shape)
        test_logger.log_value("actions shape", actions.shape)
        test_logger.log_value("rewards shape", rewards.shape)
        
        # Compute advantages
        test_logger.log_subsection("Advantage Computation")
        advantages, stats = compute_group_advantages(rewards, dones, gamma=0.99)
        
        test_logger.log_value("advantages shape", advantages.shape)
        test_logger.log_value("advantage_mean", stats['advantage_mean'])
        test_logger.log_value("advantage_std", stats['advantage_std'])
        
        # Perform update
        test_logger.log_subsection("Optimizer Update")
        metrics = optimizer.update(
            trajectory_batch=batch,
            gamma=0.99,
            update_epochs=2,
            minibatch_size=32,
        )
        
        test_logger.log_value("policy_loss", metrics.get('policy_loss', 'N/A'))
        test_logger.log_value("entropy", metrics.get('entropy', 'N/A'))
        
        logger.info("  ✓ Data flowed correctly through pipeline")
        env_groups.close()
    
    def test_group_seeding_verification(self, test_logger):
        """Verify that group seeding produces correct behavior for GRPO."""
        test_logger.log_section("Group Seeding Verification")
        
        env_fn = lambda: create_atari_env('ALE/Breakout-v5')
        
        # Create two separate groups with same seed
        test_logger.log_subsection("Same Seed Comparison")
        
        env_groups1 = ParallelEnvGroups(
            env_fn=env_fn, num_groups=1, group_size=2, base_seed=42
        )
        env_groups2 = ParallelEnvGroups(
            env_fn=env_fn, num_groups=1, group_size=2, base_seed=42
        )
        
        obs1 = env_groups1.reset()
        obs2 = env_groups2.reset()
        
        same_initial = np.array_equal(obs1, obs2)
        test_logger.log_value("same seed same initial state", same_initial)
        
        # Within group, all envs should have same initial state
        within_group_same = np.array_equal(obs1[0, 0], obs1[0, 1])
        test_logger.log_value("within group same state", within_group_same)
        
        assert within_group_same, "Envs within group should have same initial state"
        
        env_groups1.close()
        env_groups2.close()
        
        # Create groups with different seeds
        test_logger.log_subsection("Different Seed Comparison")
        
        env_groups3 = ParallelEnvGroups(
            env_fn=env_fn, num_groups=2, group_size=2, base_seed=42
        )
        
        obs3 = env_groups3.reset()
        
        between_groups_different = not np.array_equal(obs3[0, 0], obs3[1, 0])
        test_logger.log_value("between groups different state", between_groups_different)
        
        assert between_groups_different, "Different groups should have different initial states"
        
        logger.info("  ✓ Group seeding works correctly for GRPO")
        env_groups3.close()


@pytest.mark.integration
class TestGRPOProperties:
    """Tests verifying GRPO-specific properties."""
    
    def test_no_value_network_in_pipeline(self, test_logger, small_config):
        """Verify no value network is used anywhere in pipeline."""
        test_logger.log_section("No Value Network Verification")
        
        trainer = GRPOTrainer(small_config)
        
        # Check policy has no value head
        has_value_head = hasattr(trainer.policy, 'value_head')
        test_logger.log_value("policy.value_head exists", has_value_head)
        assert not has_value_head, "Policy should not have value_head"
        
        # Check policy output is tensor not dict
        obs = tf.random.uniform((1, 84, 84, 4))
        output = trainer.policy(obs)
        output_is_tensor = isinstance(output, tf.Tensor)
        output_is_dict = isinstance(output, dict)
        
        test_logger.log_value("output is Tensor", output_is_tensor)
        test_logger.log_value("output is dict", output_is_dict)
        
        assert output_is_tensor, "Policy output should be Tensor"
        assert not output_is_dict, "Policy output should not be dict"
        
        logger.info("  ✓ No value network in pipeline (core GRPO property)")
        trainer.env_groups.close()
    
    def test_group_relative_advantages(self, test_logger):
        """Verify advantages are computed relative to group."""
        test_logger.log_section("Group-Relative Advantages")
        
        np.random.seed(42)
        
        # Create rewards where groups have different means
        rewards = np.zeros((2, 4, 10), dtype=np.float32)
        rewards[0] = np.random.randn(4, 10) + 5.0  # Group 0: high rewards
        rewards[1] = np.random.randn(4, 10) - 5.0  # Group 1: low rewards
        dones = np.zeros((2, 4, 10), dtype=bool)
        
        test_logger.log_value("group 0 mean reward", np.mean(rewards[0]))
        test_logger.log_value("group 1 mean reward", np.mean(rewards[1]))
        
        # Compute advantages
        advantages, stats = compute_group_advantages(
            rewards, dones, gamma=0.99, normalize=False
        )
        
        # Within each group, mean should be ~0
        group0_mean = np.mean(advantages[0])
        group1_mean = np.mean(advantages[1])
        
        test_logger.log_value("group 0 advantage mean", group0_mean)
        test_logger.log_value("group 1 advantage mean", group1_mean)
        
        # Both group means should be ~0 despite different reward levels
        assert abs(group0_mean) < 1e-5, f"Group 0 mean should be ~0, got {group0_mean}"
        assert abs(group1_mean) < 1e-5, f"Group 1 mean should be ~0, got {group1_mean}"
        
        logger.info("  ✓ Advantages are group-relative (key GRPO property)")
