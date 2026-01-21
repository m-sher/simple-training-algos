"""
Tests for GRPO configuration.

Tests cover:
- Default configuration creation
- Configuration validation
- Computed properties
- Configuration edge cases
"""

import pytest
import logging

from grpo_atari.config import GRPOConfig, get_default_config

logger = logging.getLogger(__name__)


class TestConfigCreation:
    """Tests for configuration creation."""
    
    def test_default_config_creation(self, test_logger):
        """Test that default config can be created with expected values."""
        test_logger.log_section("Default Config Creation")
        
        config = GRPOConfig()
        
        test_logger.log_value("env_name", config.env_name)
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_value("gamma", config.gamma)
        test_logger.log_value("learning_rate", config.learning_rate)
        
        test_logger.log_assert_equal("env_name", "ALE/Breakout-v5", config.env_name)
        test_logger.log_assert_equal("group_size", 8, config.group_size)
        test_logger.log_assert_equal("num_groups", 4, config.num_groups)
        test_logger.log_assert_equal("gamma", 0.99, config.gamma)
    
    def test_custom_config_creation(self, test_logger):
        """Test config creation with custom values."""
        test_logger.log_section("Custom Config Creation")
        
        config = GRPOConfig(
            group_size=16,
            num_groups=8,
            learning_rate=1e-4,
            gamma=0.95,
        )
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_value("learning_rate", config.learning_rate)
        test_logger.log_value("gamma", config.gamma)
        
        test_logger.log_assert_equal("group_size", 16, config.group_size)
        test_logger.log_assert_equal("num_groups", 8, config.num_groups)
        test_logger.log_assert_equal("learning_rate", 1e-4, config.learning_rate)
        test_logger.log_assert_equal("gamma", 0.95, config.gamma)
    
    def test_get_default_config_utility(self, test_logger):
        """Test get_default_config utility function."""
        test_logger.log_section("get_default_config Utility")
        
        config = get_default_config()
        
        test_logger.log_value("type", type(config).__name__)
        test_logger.log_value("is GRPOConfig", isinstance(config, GRPOConfig))
        
        assert isinstance(config, GRPOConfig)
        logger.info("  ✓ get_default_config returns valid GRPOConfig")


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_valid_config_passes_validation(self, test_logger):
        """Test that valid config passes validation."""
        test_logger.log_section("Valid Config Validation")
        
        config = GRPOConfig()
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_value("minibatch_size", config.minibatch_size)
        test_logger.log_value("batch_size", config.batch_size)
        
        # Should not raise
        config.validate()
        logger.info("  ✓ Valid config passed validation")
    
    def test_group_size_too_small_fails(self, test_logger):
        """Test that group_size < 2 fails validation."""
        test_logger.log_section("Group Size Validation")
        
        config = GRPOConfig(group_size=1)
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("expected", "AssertionError")
        
        with pytest.raises(AssertionError) as exc_info:
            config.validate()
        
        logger.info(f"  ✓ Raised AssertionError: {exc_info.value}")
        assert "group_size must be at least 2" in str(exc_info.value)
    
    def test_invalid_gamma_fails(self, test_logger):
        """Test that gamma outside (0, 1] fails validation."""
        test_logger.log_section("Gamma Validation")
        
        test_cases = [
            (0.0, "gamma = 0"),
            (1.5, "gamma > 1"),
            (-0.1, "gamma < 0"),
        ]
        
        for gamma_val, desc in test_cases:
            config = GRPOConfig(gamma=gamma_val)
            test_logger.log_value(desc, gamma_val)
            
            with pytest.raises(AssertionError):
                config.validate()
            logger.info(f"  ✓ gamma={gamma_val} correctly rejected")
    
    def test_minibatch_size_validation(self, test_logger):
        """Test that minibatch_size <= batch_size is enforced."""
        test_logger.log_section("Minibatch Size Validation")
        
        config = GRPOConfig(
            group_size=2,
            num_groups=2,
            steps_per_trajectory=16,
            minibatch_size=256,  # > batch_size of 64
        )
        
        test_logger.log_value("batch_size", config.batch_size)
        test_logger.log_value("minibatch_size", config.minibatch_size)
        test_logger.log_value("minibatch > batch", config.minibatch_size > config.batch_size)
        
        with pytest.raises(AssertionError) as exc_info:
            config.validate()
        
        logger.info(f"  ✓ Raised AssertionError: {exc_info.value}")


class TestConfigProperties:
    """Tests for computed configuration properties."""
    
    def test_total_envs_property(self, test_logger):
        """Test total_envs computed property."""
        test_logger.log_section("total_envs Property")
        
        config = GRPOConfig(group_size=4, num_groups=3)
        
        expected = 4 * 3
        actual = config.total_envs
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_assert_equal("total_envs", expected, actual)
    
    def test_batch_size_property(self, test_logger):
        """Test batch_size computed property."""
        test_logger.log_section("batch_size Property")
        
        config = GRPOConfig(
            group_size=4,
            num_groups=2,
            steps_per_trajectory=64,
        )
        
        expected = 4 * 2 * 64
        actual = config.batch_size
        
        test_logger.log_value("group_size", config.group_size)
        test_logger.log_value("num_groups", config.num_groups)
        test_logger.log_value("steps_per_trajectory", config.steps_per_trajectory)
        test_logger.log_assert_equal("batch_size", expected, actual)
    
    def test_obs_shape_property(self, test_logger):
        """Test obs_shape computed property."""
        test_logger.log_section("obs_shape Property")
        
        config = GRPOConfig(
            obs_height=84,
            obs_width=84,
            frame_stack=4,
            grayscale=True,
        )
        
        expected = (84, 84, 4)
        actual = config.obs_shape
        
        test_logger.log_value("obs_height", config.obs_height)
        test_logger.log_value("obs_width", config.obs_width)
        test_logger.log_value("frame_stack", config.frame_stack)
        test_logger.log_value("grayscale", config.grayscale)
        test_logger.log_assert_equal("obs_shape", expected, actual)
    
    def test_obs_shape_rgb(self, test_logger):
        """Test obs_shape for RGB observations."""
        test_logger.log_section("obs_shape RGB")
        
        config = GRPOConfig(
            obs_height=84,
            obs_width=84,
            frame_stack=4,
            grayscale=False,
        )
        
        expected = (84, 84, 12)  # 4 frames * 3 channels
        actual = config.obs_shape
        
        test_logger.log_value("grayscale", config.grayscale)
        test_logger.log_value("expected channels", 4 * 3)
        test_logger.log_assert_equal("obs_shape", expected, actual)
