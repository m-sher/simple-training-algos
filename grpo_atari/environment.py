"""
Environment wrappers for Atari Breakout with seeding support for GRPO.

GRPO requires sampling multiple trajectories from the same initial state.
This is achieved by creating groups of environments that share the same
random seed, ensuring they start from identical initial states.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    TransformReward,
)
from typing import List, Tuple, Optional, Dict, Any
import ale_py


# Register ALE environments
gym.register_envs(ale_py)


class FireResetWrapper(gym.Wrapper):
    """
    Wrapper that presses FIRE on reset for environments that require it.
    
    Many Atari games require the FIRE action to start the game after reset.
    This wrapper automatically handles that.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # Press FIRE to start the game
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeWrapper(gym.Wrapper):
    """
    Make end-of-life equal to end-of-episode, but only reset on true game over.
    
    This helps with value estimation by making each life a separate episode
    during training, while still allowing the agent to continue after losing
    a life during actual gameplay.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Lost a life, treat as episode end for training
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} for training stability."""
    
    def reward(self, reward: float) -> float:
        return np.sign(reward)


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """Normalize observations to [0, 1] range."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=obs_space.shape,
            dtype=np.float32
        )
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class FrameStackReshapeWrapper(gym.ObservationWrapper):
    """
    Reshape frame-stacked observations from (N, H, W, C) to (H, W, N*C).
    
    Gymnasium's FrameStackObservation creates observations with shape (N, H, W, C)
    but CNNs typically expect (H, W, channels). This wrapper reshapes to combine
    the frame stack dimension with the channel dimension.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        # Expected input shape: (frame_stack, H, W, C)
        frame_stack, height, width, channels = obs_space.shape
        new_channels = frame_stack * channels
        self.observation_space = spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=(height, width, new_channels),
            dtype=obs_space.dtype
        )
        
    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs shape: (frame_stack, H, W, C)
        # Transpose to (H, W, frame_stack, C) then reshape to (H, W, frame_stack*C)
        obs = np.transpose(obs, (1, 2, 0, 3))  # (H, W, frame_stack, C)
        return obs.reshape(obs.shape[0], obs.shape[1], -1)  # (H, W, frame_stack*C)


def create_atari_env(
    env_name: str = "ALE/Breakout-v5",
    seed: Optional[int] = None,
    frame_skip: int = 4,
    frame_stack: int = 4,
    screen_size: int = 84,
    terminal_on_life_loss: bool = True,
    clip_rewards: bool = True,
    normalize_obs: bool = True,
    fire_reset: bool = True,
) -> gym.Env:
    """
    Create an Atari environment with standard preprocessing.
    
    Args:
        env_name: Name of the Atari environment.
        seed: Random seed for the environment.
        frame_skip: Number of frames to skip (action repeat).
        frame_stack: Number of frames to stack.
        screen_size: Size to resize frames to (square).
        terminal_on_life_loss: Whether to treat life loss as episode end.
        clip_rewards: Whether to clip rewards to {-1, 0, +1}.
        normalize_obs: Whether to normalize observations to [0, 1].
        fire_reset: Whether to press FIRE on reset.
        
    Returns:
        Preprocessed Gymnasium environment.
    """
    # Create base environment with NoFrameSkip (we apply our own)
    env = gym.make(
        env_name,
        frameskip=1,  # We handle frame skip in AtariPreprocessing
        repeat_action_probability=0.0,  # Deterministic for GRPO
        full_action_space=False,
    )
    
    # Apply standard Atari preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=frame_skip,
        screen_size=screen_size,
        terminal_on_life_loss=terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=True,
        scale_obs=False,  # We do our own normalization
    )
    
    # Fire reset wrapper for games that need it
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    
    # Clip rewards for training stability
    if clip_rewards:
        env = ClipRewardWrapper(env)
    
    # Frame stacking
    if frame_stack > 1:
        env = FrameStackObservation(env, frame_stack)
        # Reshape from (N, H, W, C) to (H, W, N*C) for CNN compatibility
        env = FrameStackReshapeWrapper(env)
    
    # Normalize observations
    if normalize_obs:
        env = NormalizeObservationWrapper(env)
    
    return env


class SeededEnvGroup:
    """
    A group of environments that share the same random seed for identical initial states.
    
    This is the core mechanism for GRPO in RL: by seeding multiple environments
    with the same seed, we can sample multiple different trajectories from the
    same initial state, enabling group-relative advantage computation.
    
    Attributes:
        envs: List of environment instances in this group.
        group_size: Number of environments in the group.
        seed: The shared seed for this group.
    """
    
    def __init__(
        self,
        env_fn,
        group_size: int,
        seed: int,
    ):
        """
        Initialize a group of seeded environments.
        
        Args:
            env_fn: Factory function to create environments.
            group_size: Number of environments in the group.
            seed: Shared seed for all environments in the group.
        """
        self.group_size = group_size
        self.seed = seed
        
        # Create environments (they will be seeded on reset)
        self.envs = [env_fn() for _ in range(group_size)]
        
        # Cache action space info
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.num_actions = self.action_space.n
        
    def reset(self, new_seed: Optional[int] = None) -> np.ndarray:
        """
        Reset all environments with the same seed.
        
        Args:
            new_seed: Optional new seed. If None, uses the original seed.
            
        Returns:
            Stacked observations from all environments, shape (group_size, *obs_shape).
        """
        if new_seed is not None:
            self.seed = new_seed
            
        observations = []
        for env in self.envs:
            obs, _ = env.reset(seed=self.seed)
            observations.append(obs)
            
        return np.stack(observations, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with the given actions.
        
        Args:
            actions: Array of actions, shape (group_size,).
            
        Returns:
            observations: Shape (group_size, *obs_shape).
            rewards: Shape (group_size,).
            terminateds: Shape (group_size,).
            truncateds: Shape (group_size,).
            infos: List of info dicts.
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
            
        return (
            np.stack(observations, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=bool),
            np.array(truncateds, dtype=bool),
            infos,
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


class ParallelEnvGroups:
    """
    Manager for multiple groups of seeded environments.
    
    This class manages multiple SeededEnvGroups, each with a different seed,
    to enable batch collection of trajectories for GRPO training.
    
    The structure is:
    - num_groups: Number of different initial states
    - group_size: Number of trajectories per initial state
    - Total environments: num_groups * group_size
    """
    
    def __init__(
        self,
        env_fn,
        num_groups: int,
        group_size: int,
        base_seed: int = 42,
    ):
        """
        Initialize parallel environment groups.
        
        Args:
            env_fn: Factory function to create environments.
            num_groups: Number of groups (different initial states).
            group_size: Number of environments per group (same initial state).
            base_seed: Base seed for generating group seeds.
        """
        self.num_groups = num_groups
        self.group_size = group_size
        self.base_seed = base_seed
        
        # Generate unique seeds for each group
        self.rng = np.random.default_rng(base_seed)
        self.group_seeds = self.rng.integers(0, 2**31, size=num_groups)
        
        # Create environment groups
        self.groups = [
            SeededEnvGroup(env_fn, group_size, int(seed))
            for seed in self.group_seeds
        ]
        
        # Cache space info
        self.action_space = self.groups[0].action_space
        self.observation_space = self.groups[0].observation_space
        self.num_actions = self.groups[0].num_actions
        
    @property
    def total_envs(self) -> int:
        """Total number of environments across all groups."""
        return self.num_groups * self.group_size
    
    def reset(self, new_seeds: bool = True) -> np.ndarray:
        """
        Reset all environment groups.
        
        Args:
            new_seeds: If True, generate new seeds for each group.
            
        Returns:
            Observations, shape (num_groups, group_size, *obs_shape).
        """
        if new_seeds:
            self.group_seeds = self.rng.integers(0, 2**31, size=self.num_groups)
            
        observations = []
        for group, seed in zip(self.groups, self.group_seeds):
            obs = group.reset(new_seed=int(seed))
            observations.append(obs)
            
        return np.stack(observations, axis=0)
    
    def step(
        self, 
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Step all environments.
        
        Args:
            actions: Shape (num_groups, group_size).
            
        Returns:
            observations: Shape (num_groups, group_size, *obs_shape).
            rewards: Shape (num_groups, group_size).
            terminateds: Shape (num_groups, group_size).
            truncateds: Shape (num_groups, group_size).
            infos: Nested list of info dicts.
        """
        all_obs = []
        all_rewards = []
        all_terminateds = []
        all_truncateds = []
        all_infos = []
        
        for group, group_actions in zip(self.groups, actions):
            obs, rewards, terminateds, truncateds, infos = group.step(group_actions)
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminateds.append(terminateds)
            all_truncateds.append(truncateds)
            all_infos.append(infos)
            
        return (
            np.stack(all_obs, axis=0),
            np.stack(all_rewards, axis=0),
            np.stack(all_terminateds, axis=0),
            np.stack(all_truncateds, axis=0),
            all_infos,
        )
    
    def reset_done_envs(
        self, 
        observations: np.ndarray,
        terminateds: np.ndarray,
        truncateds: np.ndarray,
    ) -> np.ndarray:
        """
        Reset environments that are done, but keep the same seed within groups.
        
        For GRPO, when an environment in a group terminates, we reset it with
        the SAME seed as other environments in its group, maintaining the
        group's shared initial state property for the next trajectory segment.
        
        Args:
            observations: Current observations.
            terminateds: Termination flags.
            truncateds: Truncation flags.
            
        Returns:
            Updated observations with reset environments.
        """
        dones = terminateds | truncateds
        new_observations = observations.copy()
        
        for g_idx, group in enumerate(self.groups):
            for e_idx, env in enumerate(group.envs):
                if dones[g_idx, e_idx]:
                    # Reset with the group's seed
                    obs, _ = env.reset(seed=int(self.group_seeds[g_idx]))
                    new_observations[g_idx, e_idx] = obs
                    
        return new_observations
    
    def close(self) -> None:
        """Close all environments."""
        for group in self.groups:
            group.close()
