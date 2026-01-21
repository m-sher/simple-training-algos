"""
Trajectory collection for GRPO training.

This module handles collecting trajectories from grouped environments,
where each group shares the same initial state (seed) to enable
group-relative advantage computation.
"""

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from grpo_atari.environment import ParallelEnvGroups
from grpo_atari.model import PolicyNetwork


@dataclass
class Trajectory:
    """
    Container for trajectory data from a single environment.
    
    Stores the sequence of (observation, action, reward, done, log_prob)
    tuples collected during rollout.
    """
    observations: np.ndarray  # (T, H, W, C)
    actions: np.ndarray       # (T,)
    rewards: np.ndarray       # (T,)
    dones: np.ndarray         # (T,)
    log_probs: np.ndarray     # (T,)
    
    @property
    def length(self) -> int:
        return len(self.actions)
    
    @property
    def total_reward(self) -> float:
        return float(np.sum(self.rewards))


@dataclass
class GroupedTrajectories:
    """
    Container for trajectories from a group of environments sharing the same seed.
    
    This is the fundamental unit for GRPO: multiple trajectories from the same
    initial state, enabling group-relative advantage computation.
    
    Attributes:
        trajectories: List of Trajectory objects, one per environment in the group.
        group_seed: The seed shared by all environments in this group.
        group_id: Identifier for this group.
    """
    trajectories: List[Trajectory]
    group_seed: int
    group_id: int
    
    @property
    def group_size(self) -> int:
        return len(self.trajectories)
    
    @property
    def returns(self) -> np.ndarray:
        """Total returns for each trajectory in the group."""
        return np.array([t.total_reward for t in self.trajectories])
    
    def get_group_statistics(self) -> Dict[str, float]:
        """Compute statistics for the group's returns."""
        returns = self.returns
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'min_return': float(np.min(returns)),
            'max_return': float(np.max(returns)),
        }


@dataclass
class TrajectoryBatch:
    """
    Batch of trajectories from all environment groups.
    
    This is the complete data structure used for a single GRPO update,
    containing trajectories organized by groups.
    
    Attributes:
        grouped_trajectories: List of GroupedTrajectories, one per group.
        num_groups: Number of groups (different initial states).
        group_size: Number of trajectories per group.
    """
    grouped_trajectories: List[GroupedTrajectories]
    
    @property
    def num_groups(self) -> int:
        return len(self.grouped_trajectories)
    
    @property
    def group_size(self) -> int:
        if self.grouped_trajectories:
            return self.grouped_trajectories[0].group_size
        return 0
    
    @property
    def total_trajectories(self) -> int:
        return self.num_groups * self.group_size
    
    def flatten(self) -> Tuple[np.ndarray, ...]:
        """
        Flatten all trajectories into arrays for batch processing.
        
        Returns:
            observations: (total_steps, H, W, C)
            actions: (total_steps,)
            rewards: (total_steps,)
            dones: (total_steps,)
            log_probs: (total_steps,)
            group_ids: (total_steps,) - identifies which group each step belongs to
            env_ids: (total_steps,) - identifies which env within group
        """
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_log_probs = []
        all_group_ids = []
        all_env_ids = []
        
        for group_idx, group in enumerate(self.grouped_trajectories):
            for env_idx, traj in enumerate(group.trajectories):
                all_obs.append(traj.observations)
                all_actions.append(traj.actions)
                all_rewards.append(traj.rewards)
                all_dones.append(traj.dones)
                all_log_probs.append(traj.log_probs)
                all_group_ids.append(np.full(traj.length, group_idx))
                all_env_ids.append(np.full(traj.length, env_idx))
        
        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_actions, axis=0),
            np.concatenate(all_rewards, axis=0),
            np.concatenate(all_dones, axis=0),
            np.concatenate(all_log_probs, axis=0),
            np.concatenate(all_group_ids, axis=0),
            np.concatenate(all_env_ids, axis=0),
        )
    
    def get_structured_data(self) -> Tuple[np.ndarray, ...]:
        """
        Get trajectory data in structured format preserving group/env organization.
        
        Returns:
            observations: (num_groups, group_size, T, H, W, C)
            actions: (num_groups, group_size, T)
            rewards: (num_groups, group_size, T)
            dones: (num_groups, group_size, T)
            log_probs: (num_groups, group_size, T)
        """
        T = self.grouped_trajectories[0].trajectories[0].length
        obs_shape = self.grouped_trajectories[0].trajectories[0].observations.shape[1:]
        
        observations = np.zeros(
            (self.num_groups, self.group_size, T) + obs_shape,
            dtype=np.float32
        )
        actions = np.zeros((self.num_groups, self.group_size, T), dtype=np.int32)
        rewards = np.zeros((self.num_groups, self.group_size, T), dtype=np.float32)
        dones = np.zeros((self.num_groups, self.group_size, T), dtype=bool)
        log_probs = np.zeros((self.num_groups, self.group_size, T), dtype=np.float32)
        
        for g_idx, group in enumerate(self.grouped_trajectories):
            for e_idx, traj in enumerate(group.trajectories):
                observations[g_idx, e_idx] = traj.observations
                actions[g_idx, e_idx] = traj.actions
                rewards[g_idx, e_idx] = traj.rewards
                dones[g_idx, e_idx] = traj.dones
                log_probs[g_idx, e_idx] = traj.log_probs
        
        return observations, actions, rewards, dones, log_probs


class TrajectoryCollector:
    """
    Collects trajectories from parallel environment groups for GRPO training.
    
    This collector manages the interaction between the policy and the
    environment groups, collecting synchronized trajectories where each
    group starts from the same initial state.
    """
    
    def __init__(
        self,
        env_groups: ParallelEnvGroups,
        policy: PolicyNetwork,
        steps_per_trajectory: int,
        gamma: float = 0.99,
    ):
        """
        Initialize the trajectory collector.
        
        Args:
            env_groups: Parallel environment groups manager.
            policy: Policy network for action selection.
            steps_per_trajectory: Number of steps to collect per trajectory.
            gamma: Discount factor for return computation.
        """
        self.env_groups = env_groups
        self.policy = policy
        self.steps_per_trajectory = steps_per_trajectory
        self.gamma = gamma
        
        # Current observations for each environment
        self.current_obs = None
        
        # Statistics tracking
        self.episode_returns = []
        self.episode_lengths = []
        
    def reset(self) -> None:
        """Reset all environments with new seeds."""
        self.current_obs = self.env_groups.reset(new_seeds=True)
        self.episode_returns = []
        self.episode_lengths = []
        
    def collect(self) -> TrajectoryBatch:
        """
        Collect a batch of trajectories from all environment groups.
        
        Each group shares the same initial state (seed), enabling
        group-relative advantage computation in GRPO.
        
        Returns:
            TrajectoryBatch containing all collected trajectories.
        """
        if self.current_obs is None:
            self.reset()
        
        num_groups = self.env_groups.num_groups
        group_size = self.env_groups.group_size
        T = self.steps_per_trajectory
        obs_shape = self.current_obs.shape[2:]  # (H, W, C)
        
        # Pre-allocate arrays for efficiency
        # Shape: (num_groups, group_size, T, ...)
        all_observations = np.zeros(
            (num_groups, group_size, T) + obs_shape,
            dtype=np.float32
        )
        all_actions = np.zeros((num_groups, group_size, T), dtype=np.int32)
        all_rewards = np.zeros((num_groups, group_size, T), dtype=np.float32)
        all_dones = np.zeros((num_groups, group_size, T), dtype=bool)
        all_log_probs = np.zeros((num_groups, group_size, T), dtype=np.float32)
        
        # Track episode statistics
        episode_return_accum = np.zeros((num_groups, group_size), dtype=np.float32)
        episode_length_accum = np.zeros((num_groups, group_size), dtype=np.int32)
        
        # Collect trajectories
        for t in range(T):
            # Store current observations
            all_observations[:, :, t] = self.current_obs
            
            # Flatten observations for policy: (num_groups * group_size, H, W, C)
            flat_obs = self.current_obs.reshape(-1, *obs_shape)
            flat_obs_tensor = tf.convert_to_tensor(flat_obs, dtype=tf.float32)
            
            # Sample actions from policy
            actions, log_probs = self.policy.sample_actions(flat_obs_tensor)
            actions = actions.numpy().reshape(num_groups, group_size)
            log_probs = log_probs.numpy().reshape(num_groups, group_size)
            
            # Store actions and log probs
            all_actions[:, :, t] = actions
            all_log_probs[:, :, t] = log_probs
            
            # Step environments
            next_obs, rewards, terminateds, truncateds, infos = self.env_groups.step(actions)
            
            # Store rewards and dones
            all_rewards[:, :, t] = rewards
            dones = terminateds | truncateds
            all_dones[:, :, t] = dones
            
            # Update episode statistics
            episode_return_accum += rewards
            episode_length_accum += 1
            
            # Record completed episodes
            for g_idx in range(num_groups):
                for e_idx in range(group_size):
                    if dones[g_idx, e_idx]:
                        self.episode_returns.append(episode_return_accum[g_idx, e_idx])
                        self.episode_lengths.append(episode_length_accum[g_idx, e_idx])
                        episode_return_accum[g_idx, e_idx] = 0.0
                        episode_length_accum[g_idx, e_idx] = 0
            
            # Handle environment resets (keep same seed within groups)
            next_obs = self.env_groups.reset_done_envs(next_obs, terminateds, truncateds)
            self.current_obs = next_obs
        
        # Build trajectory batch
        grouped_trajectories = []
        for g_idx in range(num_groups):
            trajectories = []
            for e_idx in range(group_size):
                traj = Trajectory(
                    observations=all_observations[g_idx, e_idx],
                    actions=all_actions[g_idx, e_idx],
                    rewards=all_rewards[g_idx, e_idx],
                    dones=all_dones[g_idx, e_idx],
                    log_probs=all_log_probs[g_idx, e_idx],
                )
                trajectories.append(traj)
            
            grouped = GroupedTrajectories(
                trajectories=trajectories,
                group_seed=int(self.env_groups.group_seeds[g_idx]),
                group_id=g_idx,
            )
            grouped_trajectories.append(grouped)
        
        return TrajectoryBatch(grouped_trajectories=grouped_trajectories)
    
    def get_episode_statistics(self) -> Dict[str, float]:
        """Get statistics from recently completed episodes."""
        if not self.episode_returns:
            return {
                'mean_episode_return': 0.0,
                'std_episode_return': 0.0,
                'mean_episode_length': 0.0,
                'num_episodes': 0,
            }
        
        returns = np.array(self.episode_returns)
        lengths = np.array(self.episode_lengths)
        
        return {
            'mean_episode_return': float(np.mean(returns)),
            'std_episode_return': float(np.std(returns)),
            'mean_episode_length': float(np.mean(lengths)),
            'num_episodes': len(returns),
        }
    
    def clear_episode_statistics(self) -> None:
        """Clear accumulated episode statistics."""
        self.episode_returns = []
        self.episode_lengths = []


def compute_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute discounted returns for a trajectory.
    
    GRPO uses raw returns without bootstrapping since advantages are
    computed relative to the group's mean return.
    
    Args:
        rewards: Array of rewards, shape (T,).
        dones: Array of done flags, shape (T,).
        gamma: Discount factor.
        
    Returns:
        Array of discounted returns, shape (T,).
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    
    running_return = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            running_return = 0.0
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns
