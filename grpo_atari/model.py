"""
Policy network model for GRPO on Atari.

This module defines the neural network architecture for the policy.
GRPO does not require a separate value network since it uses group-relative
advantages instead of a learned baseline.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import numpy as np


class NatureCNN(keras.Model):
    """
    The CNN architecture from the DQN Nature paper.
    
    This is the standard feature extractor for Atari games, consisting of
    three convolutional layers followed by a fully connected layer.
    
    Architecture:
    - Conv2D: 32 filters, 8x8 kernel, stride 4, ReLU
    - Conv2D: 64 filters, 4x4 kernel, stride 2, ReLU
    - Conv2D: 64 filters, 3x3 kernel, stride 1, ReLU
    - Flatten
    - Dense: 512 units, ReLU
    """
    
    def __init__(self, name: str = "nature_cnn"):
        super().__init__(name=name)
        
        # Convolutional layers
        self.conv1 = layers.Conv2D(
            32, kernel_size=8, strides=4, activation='relu',
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
            name='conv1'
        )
        self.conv2 = layers.Conv2D(
            64, kernel_size=4, strides=2, activation='relu',
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
            name='conv2'
        )
        self.conv3 = layers.Conv2D(
            64, kernel_size=3, strides=1, activation='relu',
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
            name='conv3'
        )
        
        # Flatten and fully connected
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(
            512, activation='relu',
            kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
            name='fc'
        )
        
    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the CNN.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels).
               Expected to be normalized to [0, 1].
            training: Whether in training mode (unused, for API compatibility).
            
        Returns:
            Feature tensor of shape (batch, 512).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class PolicyNetwork(keras.Model):
    """
    Policy network for GRPO.
    
    This network outputs action logits given observations. GRPO uses only
    a policy network as advantages are computed from group-relative returns,
    eliminating the need for a learned baseline.
    
    Architecture:
    - NatureCNN feature extractor
    - Linear layer to action logits
    """
    
    def __init__(
        self,
        num_actions: int,
        name: str = "policy_network"
    ):
        """
        Initialize the policy network.
        
        Args:
            num_actions: Number of discrete actions.
            name: Model name.
        """
        super().__init__(name=name)
        
        self.num_actions = int(num_actions)  # Ensure Python int for Keras
        
        # Feature extractor
        self.feature_extractor = NatureCNN()
        
        # Policy head (action logits)
        self.policy_head = layers.Dense(
            self.num_actions,
            kernel_initializer=keras.initializers.Orthogonal(0.01),
            name='policy_logits'
        )
    
    def call(
        self, 
        observations: tf.Tensor, 
        training: bool = False
    ) -> tf.Tensor:
        """
        Forward pass through the policy network.
        
        Args:
            observations: Batch of observations, shape (batch, H, W, C).
            training: Whether in training mode.
            
        Returns:
            Action logits, shape (batch, num_actions).
        """
        features = self.feature_extractor(observations, training=training)
        logits = self.policy_head(features)
        return logits
    
    def get_action_probabilities(
        self, 
        observations: tf.Tensor
    ) -> tf.Tensor:
        """
        Get action probabilities for given observations.
        
        Args:
            observations: Batch of observations.
            
        Returns:
            Action probabilities, shape (batch, num_actions).
        """
        logits = self(observations, training=False)
        return tf.nn.softmax(logits, axis=-1)
    
    @tf.function
    def sample_actions(
        self, 
        observations: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            observations: Batch of observations, shape (batch, H, W, C).
            
        Returns:
            actions: Sampled actions, shape (batch,).
            log_probs: Log probabilities of sampled actions, shape (batch,).
        """
        logits = self(observations, training=False)
        
        # Sample from categorical distribution
        actions = tf.random.categorical(logits, num_samples=1)
        actions = tf.squeeze(actions, axis=-1)
        
        # Compute log probabilities
        log_probs = self._log_prob(logits, actions)
        
        return actions, log_probs
    
    @tf.function
    def get_log_probs(
        self, 
        observations: tf.Tensor, 
        actions: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute log probabilities of given actions.
        
        Args:
            observations: Batch of observations.
            actions: Batch of actions.
            
        Returns:
            Log probabilities, shape (batch,).
        """
        logits = self(observations, training=False)
        return self._log_prob(logits, actions)
    
    def _log_prob(self, logits: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """
        Compute log probability of actions given logits.
        
        Uses log-softmax for numerical stability.
        """
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        action_log_probs = tf.gather(
            log_probs, 
            tf.cast(actions, tf.int32), 
            batch_dims=1
        )
        return action_log_probs
    
    @tf.function
    def get_entropy(self, observations: tf.Tensor) -> tf.Tensor:
        """
        Compute entropy of the action distribution.
        
        Args:
            observations: Batch of observations.
            
        Returns:
            Entropy, shape (batch,).
        """
        logits = self(observations, training=False)
        
        # Compute entropy: -sum(p * log(p))
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.nn.softmax(logits, axis=-1)
        entropy = -tf.reduce_sum(probs * log_probs, axis=-1)
        
        return entropy
    
    def get_greedy_action(self, observation: tf.Tensor) -> int:
        """
        Get the greedy (argmax) action for a single observation.
        
        Args:
            observation: Single observation, shape (H, W, C).
            
        Returns:
            Greedy action.
        """
        # Add batch dimension
        obs_batch = tf.expand_dims(observation, axis=0)
        logits = self(obs_batch, training=False)
        action = tf.argmax(logits, axis=-1)
        return int(action[0])


def create_policy_network(
    obs_shape: Tuple[int, int, int],
    num_actions: int,
) -> PolicyNetwork:
    """
    Factory function to create and build a policy network.
    
    Args:
        obs_shape: Observation shape (H, W, C).
        num_actions: Number of discrete actions.
        
    Returns:
        Built PolicyNetwork instance.
    """
    network = PolicyNetwork(num_actions=num_actions)
    
    # Build the network with a dummy input
    dummy_input = tf.zeros((1,) + obs_shape, dtype=tf.float32)
    _ = network(dummy_input)
    
    return network

