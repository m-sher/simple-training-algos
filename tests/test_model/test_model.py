"""
Tests for policy network model.

Tests cover:
- Network creation and architecture
- Forward pass shapes
- Action sampling
- Log probability computation
- Entropy computation
- Verification that no value head exists
"""

import pytest
import logging
import numpy as np
import tensorflow as tf

from grpo_atari.model import PolicyNetwork, NatureCNN, create_policy_network

logger = logging.getLogger(__name__)


class TestNatureCNN:
    """Tests for NatureCNN feature extractor."""
    
    def test_nature_cnn_creation(self, test_logger, obs_shape):
        """Test NatureCNN creation."""
        test_logger.log_section("NatureCNN Creation")
        
        cnn = NatureCNN()
        
        # Build with dummy input
        dummy_input = tf.zeros((1,) + obs_shape, dtype=tf.float32)
        output = cnn(dummy_input)
        
        test_logger.log_value("input shape", dummy_input.shape)
        test_logger.log_value("output shape", output.shape)
        test_logger.log_value("output dtype", output.dtype)
        
        # NatureCNN should output 512-dim features
        test_logger.log_assert_shape("output", (1, 512), output)
    
    def test_nature_cnn_batch_processing(self, test_logger, obs_shape):
        """Test NatureCNN with batch inputs."""
        test_logger.log_section("NatureCNN Batch Processing")
        
        cnn = NatureCNN()
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            inputs = tf.random.uniform((batch_size,) + obs_shape)
            output = cnn(inputs)
            
            expected_shape = (batch_size, 512)
            test_logger.log_value(f"batch_size={batch_size} output shape", output.shape)
            test_logger.log_assert_shape(f"batch_size={batch_size}", expected_shape, output)


class TestPolicyNetworkCreation:
    """Tests for PolicyNetwork creation."""
    
    def test_policy_network_creation(self, test_logger, obs_shape, num_actions):
        """Test PolicyNetwork creation."""
        test_logger.log_section("PolicyNetwork Creation")
        
        policy = create_policy_network(obs_shape=obs_shape, num_actions=num_actions)
        
        test_logger.log_value("num_actions", policy.num_actions)
        test_logger.log_value("model name", policy.name)
        
        num_params = sum(v.numpy().size for v in policy.trainable_variables)
        test_logger.log_value("trainable parameters", f"{num_params:,}")
        
        test_logger.log_assert_equal("num_actions", num_actions, policy.num_actions)
        assert num_params > 0, "Model should have trainable parameters"
        logger.info(f"  ✓ PolicyNetwork created with {num_params:,} parameters")
    
    def test_policy_network_no_value_head(self, test_logger, policy_network):
        """Test that PolicyNetwork has no value head (core GRPO property)."""
        test_logger.log_section("No Value Head Verification")
        
        has_value_head = hasattr(policy_network, 'value_head')
        test_logger.log_value("has value_head attribute", has_value_head)
        
        assert not has_value_head, "GRPO PolicyNetwork should not have value_head"
        logger.info("  ✓ PolicyNetwork correctly has no value head")
    
    def test_policy_network_output_type(self, test_logger, policy_network, random_observations):
        """Test that forward pass returns tensor, not dict."""
        test_logger.log_section("Output Type Verification")
        
        obs_tensor = tf.convert_to_tensor(random_observations, dtype=tf.float32)
        output = policy_network(obs_tensor)
        
        test_logger.log_value("output type", type(output).__name__)
        test_logger.log_value("is Tensor", isinstance(output, tf.Tensor))
        
        assert isinstance(output, tf.Tensor), f"Output should be Tensor, got {type(output)}"
        assert not isinstance(output, dict), "Output should not be dict (no value head)"
        logger.info("  ✓ Output is Tensor (not dict with value)")


class TestPolicyNetworkForward:
    """Tests for PolicyNetwork forward pass."""
    
    def test_forward_pass_shape(self, test_logger, policy_network, obs_shape):
        """Test forward pass returns correct logits shape."""
        test_logger.log_section("Forward Pass Shape")
        
        batch_size = 8
        obs = tf.random.uniform((batch_size,) + obs_shape)
        logits = policy_network(obs)
        
        test_logger.log_value("input shape", obs.shape)
        test_logger.log_value("output shape", logits.shape)
        test_logger.log_value("output dtype", logits.dtype)
        
        expected_shape = (batch_size, policy_network.num_actions)
        test_logger.log_assert_shape("logits", expected_shape, logits)
    
    def test_forward_pass_various_batch_sizes(self, test_logger, policy_network, obs_shape):
        """Test forward pass with various batch sizes."""
        test_logger.log_section("Various Batch Sizes")
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            obs = tf.random.uniform((batch_size,) + obs_shape)
            logits = policy_network(obs)
            
            expected_shape = (batch_size, policy_network.num_actions)
            test_logger.log_value(f"batch_size={batch_size} shape", logits.shape)
            test_logger.log_assert_shape(f"batch_size={batch_size}", expected_shape, logits)


class TestActionSampling:
    """Tests for action sampling."""
    
    def test_sample_actions_shapes(self, test_logger, policy_network, obs_shape):
        """Test that sample_actions returns correct shapes."""
        test_logger.log_section("Sample Actions Shapes")
        
        batch_size = 8
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        actions, log_probs = policy_network.sample_actions(obs)
        
        test_logger.log_value("actions shape", actions.shape)
        test_logger.log_value("actions dtype", actions.dtype)
        test_logger.log_value("log_probs shape", log_probs.shape)
        test_logger.log_value("log_probs dtype", log_probs.dtype)
        
        test_logger.log_assert_shape("actions", (batch_size,), actions)
        test_logger.log_assert_shape("log_probs", (batch_size,), log_probs)
    
    def test_sample_actions_valid_range(self, test_logger, policy_network, obs_shape, num_actions):
        """Test that sampled actions are in valid range."""
        test_logger.log_section("Actions Valid Range")
        
        batch_size = 100
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        actions, _ = policy_network.sample_actions(obs)
        actions_np = actions.numpy()
        
        min_action = int(np.min(actions_np))
        max_action = int(np.max(actions_np))
        
        test_logger.log_value("num_actions", num_actions)
        test_logger.log_value("sampled min action", min_action)
        test_logger.log_value("sampled max action", max_action)
        test_logger.log_value("unique actions", np.unique(actions_np).tolist())
        
        test_logger.log_assert_range("min action", min_action, 0, num_actions - 1)
        test_logger.log_assert_range("max action", max_action, 0, num_actions - 1)
    
    def test_sample_actions_log_probs_negative(self, test_logger, policy_network, obs_shape):
        """Test that log probabilities are non-positive."""
        test_logger.log_section("Log Probs Non-Positive")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        _, log_probs = policy_network.sample_actions(obs)
        log_probs_np = log_probs.numpy()
        
        max_log_prob = float(np.max(log_probs_np))
        min_log_prob = float(np.min(log_probs_np))
        
        test_logger.log_value("max log_prob", max_log_prob)
        test_logger.log_value("min log_prob", min_log_prob)
        
        assert max_log_prob <= 0.0, f"Log probs should be <= 0, got max {max_log_prob}"
        logger.info("  ✓ All log probabilities are non-positive")


class TestLogProbabilities:
    """Tests for log probability computation."""
    
    def test_get_log_probs_shape(self, test_logger, policy_network, random_observations, random_actions):
        """Test get_log_probs returns correct shape."""
        test_logger.log_section("get_log_probs Shape")
        
        obs_tensor = tf.convert_to_tensor(random_observations, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(random_actions, dtype=tf.int32)
        
        log_probs = policy_network.get_log_probs(obs_tensor, actions_tensor)
        
        test_logger.log_value("observations shape", obs_tensor.shape)
        test_logger.log_value("actions shape", actions_tensor.shape)
        test_logger.log_value("log_probs shape", log_probs.shape)
        
        expected_shape = (len(random_actions),)
        test_logger.log_assert_shape("log_probs", expected_shape, log_probs)
    
    def test_get_log_probs_consistency(self, test_logger, policy_network, obs_shape):
        """Test that get_log_probs is consistent with sample_actions."""
        test_logger.log_section("Log Probs Consistency")
        
        batch_size = 16
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        # Sample actions and get log probs in one call
        actions, log_probs_sampled = policy_network.sample_actions(obs)
        
        # Get log probs separately for the same actions
        log_probs_computed = policy_network.get_log_probs(obs, actions)
        
        # They should be identical
        diff = tf.abs(log_probs_sampled - log_probs_computed)
        max_diff = float(tf.reduce_max(diff))
        
        test_logger.log_value("max difference", max_diff)
        test_logger.log_assert_close("max difference", 0.0, max_diff, tolerance=1e-5)


class TestEntropy:
    """Tests for entropy computation."""
    
    def test_entropy_shape(self, test_logger, policy_network, obs_shape):
        """Test entropy returns correct shape."""
        test_logger.log_section("Entropy Shape")
        
        batch_size = 8
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        entropy = policy_network.get_entropy(obs)
        
        test_logger.log_value("entropy shape", entropy.shape)
        test_logger.log_value("entropy dtype", entropy.dtype)
        
        test_logger.log_assert_shape("entropy", (batch_size,), entropy)
    
    def test_entropy_non_negative(self, test_logger, policy_network, obs_shape):
        """Test that entropy is non-negative."""
        test_logger.log_section("Entropy Non-Negative")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        entropy = policy_network.get_entropy(obs)
        entropy_np = entropy.numpy()
        
        min_entropy = float(np.min(entropy_np))
        max_entropy = float(np.max(entropy_np))
        mean_entropy = float(np.mean(entropy_np))
        
        test_logger.log_value("min entropy", min_entropy)
        test_logger.log_value("max entropy", max_entropy)
        test_logger.log_value("mean entropy", mean_entropy)
        
        assert min_entropy >= 0.0, f"Entropy should be >= 0, got min {min_entropy}"
        logger.info("  ✓ All entropy values are non-negative")
    
    def test_entropy_upper_bound(self, test_logger, policy_network, obs_shape, num_actions):
        """Test that entropy is bounded by log(num_actions)."""
        test_logger.log_section("Entropy Upper Bound")
        
        batch_size = 32
        obs = tf.random.uniform((batch_size,) + obs_shape)
        
        entropy = policy_network.get_entropy(obs)
        entropy_np = entropy.numpy()
        
        max_possible_entropy = np.log(num_actions)
        max_observed_entropy = float(np.max(entropy_np))
        
        test_logger.log_value("num_actions", num_actions)
        test_logger.log_value("max possible entropy (ln(n))", max_possible_entropy)
        test_logger.log_value("max observed entropy", max_observed_entropy)
        
        # Allow small numerical tolerance
        assert max_observed_entropy <= max_possible_entropy + 1e-5, \
            f"Entropy {max_observed_entropy} exceeds max {max_possible_entropy}"
        logger.info(f"  ✓ Entropy bounded by ln({num_actions}) = {max_possible_entropy:.4f}")


class TestGreedyAction:
    """Tests for greedy action selection."""
    
    def test_get_greedy_action(self, test_logger, policy_network, obs_shape, num_actions):
        """Test greedy action selection."""
        test_logger.log_section("Greedy Action")
        
        obs = tf.random.uniform(obs_shape)
        
        action = policy_network.get_greedy_action(obs)
        
        test_logger.log_value("action", action)
        test_logger.log_value("action type", type(action).__name__)
        
        assert isinstance(action, int), f"Action should be int, got {type(action)}"
        test_logger.log_assert_range("action", action, 0, num_actions - 1)
    
    def test_greedy_action_deterministic(self, test_logger, policy_network, obs_shape):
        """Test that greedy action is deterministic for same input."""
        test_logger.log_section("Greedy Action Deterministic")
        
        obs = tf.random.uniform(obs_shape)
        
        actions = [policy_network.get_greedy_action(obs) for _ in range(10)]
        
        test_logger.log_value("actions", actions)
        test_logger.log_value("all same", len(set(actions)) == 1)
        
        assert len(set(actions)) == 1, f"Greedy action should be deterministic, got {set(actions)}"
        logger.info("  ✓ Greedy action is deterministic")
