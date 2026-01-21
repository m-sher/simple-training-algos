"""
GRPO (Group Relative Policy Optimization) loss computation.

This module implements the core GRPO algorithm, which uses group-relative
advantages instead of a learned value baseline. The key insight is that
by sampling multiple trajectories from the same initial state, we can
compute advantages relative to the group's mean performance.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Tuple, Optional, List

from grpo_atari.trajectory import TrajectoryBatch, compute_returns
from grpo_atari.model import PolicyNetwork


def compute_trajectory_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute discounted returns for trajectories.
    
    Args:
        rewards: Rewards array, shape (..., T).
        dones: Done flags, shape (..., T).
        gamma: Discount factor.
        
    Returns:
        Returns array, shape (..., T).
    """
    # Handle arbitrary leading dimensions
    original_shape = rewards.shape
    T = original_shape[-1]
    
    # Flatten leading dimensions
    flat_rewards = rewards.reshape(-1, T)
    flat_dones = dones.reshape(-1, T)
    batch_size = flat_rewards.shape[0]
    
    returns = np.zeros_like(flat_rewards)
    
    for b in range(batch_size):
        running_return = 0.0
        for t in reversed(range(T)):
            if flat_dones[b, t]:
                running_return = 0.0
            running_return = flat_rewards[b, t] + gamma * running_return
            returns[b, t] = running_return
    
    return returns.reshape(original_shape)


def compute_group_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    eps: float = 1e-8,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute group-relative advantages for GRPO.
    
    This is the core of GRPO: advantages are computed relative to the
    group's mean return, eliminating the need for a learned value baseline.
    
    For each timestep t, the advantage is computed as:
        A_t = (R_t - mean(R_t across group)) / (std(R_t across group) + eps)
    
    where R_t is the return from timestep t onwards.
    
    Args:
        rewards: Rewards array, shape (num_groups, group_size, T).
        dones: Done flags, shape (num_groups, group_size, T).
        gamma: Discount factor.
        eps: Small constant for numerical stability.
        normalize: Whether to normalize advantages by std.
        
    Returns:
        advantages: Group-relative advantages, shape (num_groups, group_size, T).
        stats: Dictionary of advantage statistics.
    """
    num_groups, group_size, T = rewards.shape
    
    # Compute returns for all trajectories
    # Shape: (num_groups, group_size, T)
    returns = compute_trajectory_returns(rewards, dones, gamma)
    
    # Compute group statistics at each timestep
    # Mean and std across group_size dimension
    group_mean = np.mean(returns, axis=1, keepdims=True)  # (num_groups, 1, T)
    group_std = np.std(returns, axis=1, keepdims=True)    # (num_groups, 1, T)
    
    # Compute group-relative advantages
    advantages = returns - group_mean
    
    if normalize:
        advantages = advantages / (group_std + eps)
    
    # Compute statistics for logging
    stats = {
        'advantage_mean': float(np.mean(advantages)),
        'advantage_std': float(np.std(advantages)),
        'advantage_min': float(np.min(advantages)),
        'advantage_max': float(np.max(advantages)),
        'return_mean': float(np.mean(returns)),
        'return_std': float(np.std(returns)),
        'group_return_spread': float(np.mean(group_std)),
    }
    
    return advantages, stats


def compute_group_advantages_per_trajectory(
    trajectory_batch: TrajectoryBatch,
    gamma: float,
    eps: float = 1e-8,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Compute group-relative advantages using total trajectory returns.
    
    Alternative to per-timestep advantages: uses total trajectory returns
    to compute a single advantage value per trajectory, which is then
    applied to all timesteps in that trajectory.
    
    This is closer to the original GRPO formulation for LLMs where each
    "response" gets a single reward.
    
    Args:
        trajectory_batch: Batch of grouped trajectories.
        gamma: Discount factor.
        eps: Small constant for numerical stability.
        normalize: Whether to normalize advantages.
        
    Returns:
        advantages: Shape (num_groups, group_size, T).
        stats: Dictionary of statistics.
    """
    observations, actions, rewards, dones, log_probs = trajectory_batch.get_structured_data()
    num_groups, group_size, T = rewards.shape
    
    # Compute total discounted return for each trajectory
    trajectory_returns = np.zeros((num_groups, group_size), dtype=np.float32)
    
    for g_idx in range(num_groups):
        for e_idx in range(group_size):
            returns = compute_returns(
                rewards[g_idx, e_idx],
                dones[g_idx, e_idx],
                gamma,
            )
            trajectory_returns[g_idx, e_idx] = returns[0]  # Return from start
    
    # Group-relative normalization
    group_mean = np.mean(trajectory_returns, axis=1, keepdims=True)  # (num_groups, 1)
    group_std = np.std(trajectory_returns, axis=1, keepdims=True)    # (num_groups, 1)
    
    trajectory_advantages = trajectory_returns - group_mean
    if normalize:
        trajectory_advantages = trajectory_advantages / (group_std + eps)
    
    # Broadcast to all timesteps
    advantages = np.broadcast_to(
        trajectory_advantages[:, :, np.newaxis],
        (num_groups, group_size, T)
    ).copy()  # Make it writeable
    
    stats = {
        'advantage_mean': float(np.mean(trajectory_advantages)),
        'advantage_std': float(np.std(trajectory_advantages)),
        'trajectory_return_mean': float(np.mean(trajectory_returns)),
        'trajectory_return_std': float(np.std(trajectory_returns)),
        'group_return_spread': float(np.mean(group_std)),
    }
    
    return advantages, stats


class GRPOLoss:
    """
    Computes GRPO loss and gradients for policy updates.
    
    GRPO combines:
    1. Group-relative advantages (no value function baseline)
    2. Policy ratio clipping (similar to PPO)
    3. Entropy regularization
    
    The loss is:
        L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)] - c_ent * H(pi)
    
    where:
        r = pi(a|s) / pi_old(a|s)  (importance sampling ratio)
        A = group-relative advantage
        H(pi) = entropy of policy
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        epsilon_clip: float = 0.1,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        """
        Initialize GRPO loss calculator.
        
        Args:
            policy: The policy network to optimize.
            epsilon_clip: Clipping parameter for policy ratio.
            entropy_coef: Coefficient for entropy bonus.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        self.policy = policy
        self.epsilon_clip = epsilon_clip
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    @tf.function
    def compute_loss(
        self,
        observations: tf.Tensor,
        actions: tf.Tensor,
        old_log_probs: tf.Tensor,
        advantages: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute the GRPO loss.
        
        Args:
            observations: Batch of observations, shape (batch, H, W, C).
            actions: Batch of actions, shape (batch,).
            old_log_probs: Log probs from old policy, shape (batch,).
            advantages: Group-relative advantages, shape (batch,).
            
        Returns:
            loss: Scalar loss value.
            metrics: Dictionary of loss components for logging.
        """
        # Get current policy log probs and logits
        new_log_probs = self.policy.get_log_probs(observations, actions)
        
        # Compute importance sampling ratio
        log_ratio = new_log_probs - old_log_probs
        ratio = tf.exp(log_ratio)
        
        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = tf.clip_by_value(
            ratio,
            1.0 - self.epsilon_clip,
            1.0 + self.epsilon_clip
        ) * advantages
        
        # Policy loss (negative because we want to maximize)
        policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        
        # Entropy bonus
        entropy = self.policy.get_entropy(observations)
        mean_entropy = tf.reduce_mean(entropy)
        entropy_loss = -self.entropy_coef * mean_entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Compute action probabilities for selected actions
        action_probs = tf.exp(new_log_probs)
        old_action_probs = tf.exp(old_log_probs)
        
        # Compute metrics for logging
        with tf.name_scope('metrics'):
            approx_kl = tf.reduce_mean((ratio - 1) - log_ratio)
            clip_fraction = tf.reduce_mean(
                tf.cast(
                    tf.abs(ratio - 1.0) > self.epsilon_clip,
                    tf.float32
                )
            )
        
        metrics = {
            'policy_loss': policy_loss,
            'entropy_loss': entropy_loss,
            'entropy': mean_entropy,
            'total_loss': total_loss,
            'approx_kl': approx_kl,
            'clip_fraction': clip_fraction,
            'ratio_mean': tf.reduce_mean(ratio),
            'ratio_std': tf.math.reduce_std(ratio),
            'ratio_min': tf.reduce_min(ratio),
            'ratio_max': tf.reduce_max(ratio),
            'action_prob_mean': tf.reduce_mean(action_probs),
            'action_prob_std': tf.math.reduce_std(action_probs),
            'old_action_prob_mean': tf.reduce_mean(old_action_probs),
            'new_log_prob_mean': tf.reduce_mean(new_log_probs),
            'old_log_prob_mean': tf.reduce_mean(old_log_probs),
            'advantage_batch_mean': tf.reduce_mean(advantages),
            'advantage_batch_std': tf.math.reduce_std(advantages),
        }
        
        return total_loss, metrics
    
    @tf.function
    def compute_gradients(
        self,
        observations: tf.Tensor,
        actions: tf.Tensor,
        old_log_probs: tf.Tensor,
        advantages: tf.Tensor,
    ) -> Tuple[List[tf.Tensor], tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute gradients of the GRPO loss.
        
        Args:
            observations: Batch of observations.
            actions: Batch of actions.
            old_log_probs: Log probs from old policy.
            advantages: Group-relative advantages.
            
        Returns:
            gradients: List of gradients for each trainable variable.
            loss: Scalar loss value.
            metrics: Dictionary of metrics.
        """
        with tf.GradientTape() as tape:
            loss, metrics = self.compute_loss(
                observations, actions, old_log_probs, advantages
            )
        
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        
        # Clip gradients
        if self.max_grad_norm is not None:
            gradients, grad_norm = tf.clip_by_global_norm(
                gradients, self.max_grad_norm
            )
            metrics['grad_norm'] = grad_norm
        
        return gradients, loss, metrics


class GRPOOptimizer:
    """
    Optimizer for GRPO training.
    
    Handles the complete update process including:
    1. Computing group-relative advantages
    2. Multiple epochs of minibatch updates
    3. Learning rate scheduling (optional)
    """
    
    def __init__(
        self,
        policy: PolicyNetwork,
        learning_rate: float = 2.5e-4,
        epsilon_clip: float = 0.1,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        adam_epsilon: float = 1e-5,
    ):
        """
        Initialize the GRPO optimizer.
        
        Args:
            policy: Policy network to optimize.
            learning_rate: Learning rate.
            epsilon_clip: Clipping parameter.
            entropy_coef: Entropy coefficient.
            max_grad_norm: Max gradient norm.
            adam_epsilon: Adam optimizer epsilon.
        """
        self.policy = policy
        self.learning_rate = learning_rate
        
        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            epsilon=adam_epsilon,
        )
        
        # Create loss calculator
        self.loss_fn = GRPOLoss(
            policy=policy,
            epsilon_clip=epsilon_clip,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        
        # Store old log probs for importance sampling
        self._old_log_probs = None
    
    def store_old_log_probs(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """
        Store log probabilities from current policy before updates.
        
        This is needed for importance sampling ratio computation.
        
        Args:
            observations: All observations.
            actions: All actions.
        """
        obs_tensor = tf.convert_to_tensor(observations, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        self._old_log_probs = self.policy.get_log_probs(
            obs_tensor, actions_tensor
        ).numpy()
    
    def update(
        self,
        trajectory_batch: TrajectoryBatch,
        gamma: float,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        advantage_mode: str = 'per_timestep',
    ) -> Dict[str, float]:
        """
        Perform a complete GRPO update on the collected trajectories.
        
        Args:
            trajectory_batch: Collected trajectory batch.
            gamma: Discount factor.
            update_epochs: Number of epochs over the data.
            minibatch_size: Size of minibatches.
            advantage_mode: 'per_timestep' or 'per_trajectory'.
            
        Returns:
            Dictionary of aggregated metrics.
        """
        # Get structured data
        observations, actions, rewards, dones, log_probs = \
            trajectory_batch.get_structured_data()
        
        num_groups, group_size, T = rewards.shape
        
        # Compute group-relative advantages
        if advantage_mode == 'per_trajectory':
            advantages, adv_stats = compute_group_advantages_per_trajectory(
                trajectory_batch, gamma
            )
        else:
            advantages, adv_stats = compute_group_advantages(
                rewards, dones, gamma
            )
        
        # Flatten for training
        flat_obs = observations.reshape(-1, *observations.shape[3:])
        flat_actions = actions.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_old_log_probs = log_probs.reshape(-1)
        
        # Store old log probs (these are already from collection time)
        # For proper importance sampling, we use the log probs stored during collection
        
        # Training loop
        batch_size = len(flat_actions)
        num_minibatches = max(1, batch_size // minibatch_size)
        
        all_metrics = []
        
        for epoch in range(update_epochs):
            # Shuffle data
            indices = np.random.permutation(batch_size)
            
            for mb in range(num_minibatches):
                start_idx = mb * minibatch_size
                end_idx = min(start_idx + minibatch_size, batch_size)
                mb_indices = indices[start_idx:end_idx]
                
                mb_obs = tf.convert_to_tensor(flat_obs[mb_indices], dtype=tf.float32)
                mb_actions = tf.convert_to_tensor(flat_actions[mb_indices], dtype=tf.int32)
                mb_advantages = tf.convert_to_tensor(flat_advantages[mb_indices], dtype=tf.float32)
                mb_old_log_probs = tf.convert_to_tensor(flat_old_log_probs[mb_indices], dtype=tf.float32)
                
                # Compute gradients and update
                gradients, loss, metrics = self.loss_fn.compute_gradients(
                    mb_obs, mb_actions, mb_old_log_probs, mb_advantages
                )
                
                self.optimizer.apply_gradients(
                    zip(gradients, self.policy.trainable_variables)
                )
                
                # Store metrics
                metrics_np = {k: float(v.numpy()) for k, v in metrics.items()}
                all_metrics.append(metrics_np)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[key] = float(np.mean(values))
        
        # Add advantage stats
        aggregated.update(adv_stats)
        
        return aggregated
    
    def set_learning_rate(self, lr: float) -> None:
        """Update the learning rate."""
        self.optimizer.learning_rate.assign(lr)
