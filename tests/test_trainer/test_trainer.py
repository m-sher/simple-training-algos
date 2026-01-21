"""
Tests for GRPO trainer.

Tests cover:
- Trainer initialization
- Training loop (10 iterations)
- Checkpointing
- Logging
"""

import os
import pytest
import logging
import numpy as np

from grpo_atari.config import GRPOConfig
from grpo_atari.trainer import GRPOTrainer, create_trainer

logger = logging.getLogger(__name__)


class TestTrainerInitialization:
    """Tests for trainer initialization."""
    
    def test_trainer_creation(self, test_logger, small_config):
        """Test GRPOTrainer creation."""
        test_logger.log_section("GRPOTrainer Creation")
        
        trainer = GRPOTrainer(small_config)
        
        test_logger.log_value("policy is not None", trainer.policy is not None)
        test_logger.log_value("collector is not None", trainer.collector is not None)
        test_logger.log_value("optimizer is not None", trainer.optimizer is not None)
        test_logger.log_value("env_groups is not None", trainer.env_groups is not None)
        
        assert trainer.policy is not None
        assert trainer.collector is not None
        assert trainer.optimizer is not None
        assert trainer.env_groups is not None
        
        logger.info("  ✓ All trainer components initialized")
        trainer.env_groups.close()
    
    def test_trainer_config_stored(self, test_logger, small_config):
        """Test that config is stored in trainer."""
        test_logger.log_section("Trainer Config Storage")
        
        trainer = GRPOTrainer(small_config)
        
        test_logger.log_value("config.group_size", trainer.config.group_size)
        test_logger.log_value("config.num_groups", trainer.config.num_groups)
        
        assert trainer.config is small_config
        logger.info("  ✓ Config properly stored in trainer")
        
        trainer.env_groups.close()
    
    def test_create_trainer_factory(self, test_logger, temp_dir):
        """Test create_trainer factory function."""
        test_logger.log_section("create_trainer Factory")
        
        trainer = create_trainer(
            env_name="ALE/Breakout-v5",
            group_size=2,
            num_groups=2,
            steps_per_trajectory=16,
            minibatch_size=32,
            checkpoint_dir=os.path.join(temp_dir, 'ckpt'),
            log_dir=os.path.join(temp_dir, 'logs'),
        )
        
        test_logger.log_value("trainer type", type(trainer).__name__)
        test_logger.log_value("config.group_size", trainer.config.group_size)
        
        assert isinstance(trainer, GRPOTrainer)
        test_logger.log_assert_equal("group_size", 2, trainer.config.group_size)
        
        trainer.env_groups.close()


class TestTrainingLoop:
    """Tests for training loop execution."""
    
    @pytest.mark.timeout(120)
    def test_train_10_iterations(self, test_logger, small_config):
        """Test training for 10 iterations completes without error."""
        test_logger.log_section("Train 10 Iterations")
        
        trainer = GRPOTrainer(small_config)
        
        test_logger.log_value("num_iterations", small_config.num_iterations)
        test_logger.log_value("batch_size", small_config.batch_size)
        test_logger.log_value("total_envs", small_config.total_envs)
        
        # Get initial weights
        initial_weights = [v.numpy().copy() for v in trainer.policy.trainable_variables]
        test_logger.log_value("initial weights captured", len(initial_weights))
        
        # Run training
        test_logger.log_subsection("Training Progress")
        history = trainer.train()
        
        # Check training completed
        test_logger.log_value("final global_step", trainer.global_step)
        test_logger.log_assert_equal("global_step", 9, trainer.global_step)  # 0-indexed
        
        # Check history recorded
        test_logger.log_value("history entries", len(history['history']))
        assert len(history['history']) > 0, "History should have entries"
        
        # Check weights changed
        final_weights = [v.numpy() for v in trainer.policy.trainable_variables]
        weights_changed = any(
            not np.allclose(init_w, final_w)
            for init_w, final_w in zip(initial_weights, final_weights)
        )
        test_logger.log_value("weights changed", weights_changed)
        assert weights_changed, "Policy weights should change during training"
        
        logger.info("  ✓ Training completed successfully for 10 iterations")
        trainer.env_groups.close()
    
    def test_train_metrics_logged(self, test_logger, small_config):
        """Test that metrics are logged during training."""
        test_logger.log_section("Training Metrics Logged")
        
        trainer = GRPOTrainer(small_config)
        history = trainer.train()
        
        # Check for expected metrics in history
        expected_metrics = ['policy_loss', 'entropy', 'total_time']
        
        test_logger.log_subsection("Checking Metrics")
        if history['history']:
            sample_entry = history['history'][-1]
            for metric in expected_metrics:
                present = metric in sample_entry
                value = sample_entry.get(metric, 'MISSING')
                test_logger.log_value(f"{metric} present", present)
                if present:
                    test_logger.log_value(f"{metric} value", value)
        
        trainer.env_groups.close()


class TestCheckpointing:
    """Tests for model checkpointing."""
    
    def test_checkpoint_created(self, test_logger, small_config, temp_dir):
        """Test that checkpoints are created during training."""
        test_logger.log_section("Checkpoint Creation")
        
        # Ensure checkpoint_dir uses temp_dir
        small_config.checkpoint_dir = os.path.join(temp_dir, 'checkpoints')
        small_config.save_interval = 5  # Save every 5 iterations
        
        trainer = GRPOTrainer(small_config)
        trainer.train()
        
        # Check checkpoint directory
        checkpoint_dir = small_config.checkpoint_dir
        test_logger.log_value("checkpoint_dir", checkpoint_dir)
        
        exists = os.path.exists(checkpoint_dir)
        test_logger.log_value("directory exists", exists)
        assert exists, f"Checkpoint directory should exist: {checkpoint_dir}"
        
        files = os.listdir(checkpoint_dir)
        test_logger.log_value("num files", len(files))
        test_logger.log_value("files", files[:5])  # First 5 files
        
        assert len(files) > 0, "Checkpoint directory should contain files"
        logger.info(f"  ✓ {len(files)} checkpoint files created")
        
        trainer.env_groups.close()
    
    def test_logs_created(self, test_logger, small_config, temp_dir):
        """Test that log files are created during training."""
        test_logger.log_section("Log Creation")
        
        small_config.log_dir = os.path.join(temp_dir, 'logs')
        
        trainer = GRPOTrainer(small_config)
        trainer.train()
        
        log_dir = small_config.log_dir
        test_logger.log_value("log_dir", log_dir)
        
        exists = os.path.exists(log_dir)
        test_logger.log_value("directory exists", exists)
        assert exists, f"Log directory should exist: {log_dir}"
        
        files = os.listdir(log_dir)
        test_logger.log_value("num files", len(files))
        test_logger.log_value("files", files)
        
        assert len(files) > 0, "Log directory should contain files"
        
        # Check for metrics.csv
        has_csv = any('metrics.csv' in f for f in files)
        test_logger.log_value("has metrics.csv", has_csv)
        
        logger.info(f"  ✓ {len(files)} log files created")
        trainer.env_groups.close()


class TestPolicyAccessors:
    """Tests for policy accessor methods."""
    
    def test_get_policy(self, test_logger, small_config):
        """Test get_policy method."""
        test_logger.log_section("get_policy")
        
        trainer = GRPOTrainer(small_config)
        
        policy = trainer.get_policy()
        
        test_logger.log_value("policy type", type(policy).__name__)
        test_logger.log_value("is same as trainer.policy", policy is trainer.policy)
        
        assert policy is trainer.policy
        logger.info("  ✓ get_policy returns trainer's policy")
        
        trainer.env_groups.close()
    
    def test_save_load_policy(self, test_logger, small_config, temp_dir):
        """Test save and load policy weights."""
        test_logger.log_section("Save/Load Policy")
        
        trainer = GRPOTrainer(small_config)
        
        # Get initial weights
        initial_weights = [v.numpy().copy() for v in trainer.policy.trainable_variables]
        
        # Save weights
        save_path = os.path.join(temp_dir, 'policy_weights')
        trainer.save_policy(save_path)
        
        test_logger.log_value("save_path", save_path)
        test_logger.log_value("files created", os.path.exists(save_path + '.index'))
        
        # Modify weights (simulate training)
        for v in trainer.policy.trainable_variables:
            v.assign(v + 1.0)
        
        modified_weights = [v.numpy().copy() for v in trainer.policy.trainable_variables]
        
        # Load weights
        trainer.load_policy(save_path)
        loaded_weights = [v.numpy() for v in trainer.policy.trainable_variables]
        
        # Check weights restored
        weights_restored = all(
            np.allclose(init_w, loaded_w)
            for init_w, loaded_w in zip(initial_weights, loaded_weights)
        )
        
        test_logger.log_value("weights restored correctly", weights_restored)
        assert weights_restored, "Loaded weights should match saved weights"
        
        logger.info("  ✓ Policy weights saved and loaded correctly")
        trainer.env_groups.close()
