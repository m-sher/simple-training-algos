"""
GRPO Trainer for Atari Breakout.

This module orchestrates the complete GRPO training loop, including:
- Environment management
- Trajectory collection
- Policy updates
- Logging and checkpointing
- Evaluation
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Optional, Callable, List
from pathlib import Path

from grpo_atari.config import GRPOConfig
from grpo_atari.environment import ParallelEnvGroups, create_atari_env
from grpo_atari.model import PolicyNetwork, create_policy_network
from grpo_atari.trajectory import TrajectoryCollector
from grpo_atari.grpo_loss import GRPOOptimizer
from grpo_atari.demo import DemoRecorder


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for logs.
            use_tensorboard: Whether to use TensorBoard.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.summary_writer = tf.summary.create_file_writer(str(self.log_dir))
        
        # CSV logging
        self.csv_path = self.log_dir / "metrics.csv"
        self.csv_header_written = False
        
        # In-memory storage
        self.history: List[Dict] = []
    
    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a given step."""
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        
        # TensorBoard
        if self.use_tensorboard:
            with self.summary_writer.as_default():
                for key, value in metrics.items():
                    if key not in ['step', 'timestamp']:
                        tf.summary.scalar(key, value, step=step)
        
        # CSV
        if not self.csv_header_written:
            with open(self.csv_path, 'w') as f:
                f.write(','.join(metrics.keys()) + '\n')
            self.csv_header_written = True
        
        with open(self.csv_path, 'a') as f:
            f.write(','.join(str(v) for v in metrics.values()) + '\n')
        
        # Memory
        self.history.append(metrics)
    
    def print_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Print metrics to console."""
        msg = f"Step {step:6d} | "
        msg += " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items() 
                         if k not in ['step', 'timestamp'])
        print(msg)


def format_training_stats(
    iteration: int,
    total_iterations: int,
    metrics: Dict[str, float],
    elapsed_time: float,
    learning_rate: float,
) -> str:
    """
    Format training statistics for verbose console output.
    
    Args:
        iteration: Current iteration number.
        total_iterations: Total number of iterations.
        metrics: Dictionary of training metrics.
        elapsed_time: Time elapsed since training start.
        learning_rate: Current learning rate.
        
    Returns:
        Formatted string for printing.
    """
    # Calculate progress
    progress = (iteration + 1) / total_iterations * 100
    
    # Format elapsed time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Build output string
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  Iteration {iteration + 1}/{total_iterations} ({progress:.1f}%) | Elapsed: {time_str}")
    lines.append("=" * 80)
    
    # Episode Statistics
    lines.append("")
    lines.append("  📊 Episode Statistics:")
    lines.append(f"      Mean Episode Return:    {metrics.get('mean_episode_return', 0):10.2f}")
    lines.append(f"      Std Episode Return:     {metrics.get('std_episode_return', 0):10.2f}")
    lines.append(f"      Mean Episode Length:    {metrics.get('mean_episode_length', 0):10.1f}")
    lines.append(f"      Episodes Completed:     {metrics.get('num_episodes', 0):10.0f}")
    
    # Returns and Advantages (GRPO-specific)
    lines.append("")
    lines.append("  🎯 GRPO Advantages:")
    lines.append(f"      Return Mean:            {metrics.get('return_mean', 0):10.4f}")
    lines.append(f"      Return Std:             {metrics.get('return_std', 0):10.4f}")
    lines.append(f"      Group Return Spread:    {metrics.get('group_return_spread', 0):10.4f}")
    lines.append(f"      Advantage Mean:         {metrics.get('advantage_mean', 0):10.4f}")
    lines.append(f"      Advantage Std:          {metrics.get('advantage_std', 0):10.4f}")
    
    # Policy Statistics
    lines.append("")
    lines.append("  🧠 Policy Statistics:")
    lines.append(f"      Policy Loss:            {metrics.get('policy_loss', 0):10.6f}")
    lines.append(f"      Entropy:                {metrics.get('entropy', 0):10.4f}")
    lines.append(f"      Entropy Loss:           {metrics.get('entropy_loss', 0):10.6f}")
    lines.append(f"      Total Loss:             {metrics.get('total_loss', 0):10.6f}")
    
    # Action Probabilities
    lines.append("")
    lines.append("  🎲 Action Probabilities:")
    lines.append(f"      Mean Action Prob:       {metrics.get('action_prob_mean', 0):10.4f}")
    lines.append(f"      Std Action Prob:        {metrics.get('action_prob_std', 0):10.4f}")
    lines.append(f"      Mean Log Prob (new):    {metrics.get('new_log_prob_mean', 0):10.4f}")
    lines.append(f"      Mean Log Prob (old):    {metrics.get('old_log_prob_mean', 0):10.4f}")
    
    # Importance Sampling
    lines.append("")
    lines.append("  ⚖️  Importance Sampling:")
    lines.append(f"      Ratio Mean:             {metrics.get('ratio_mean', 1):10.4f}")
    lines.append(f"      Ratio Std:              {metrics.get('ratio_std', 0):10.4f}")
    lines.append(f"      Ratio Min:              {metrics.get('ratio_min', 1):10.4f}")
    lines.append(f"      Ratio Max:              {metrics.get('ratio_max', 1):10.4f}")
    lines.append(f"      Clip Fraction:          {metrics.get('clip_fraction', 0):10.4f}")
    lines.append(f"      Approx KL:              {metrics.get('approx_kl', 0):10.6f}")
    
    # Optimization
    lines.append("")
    lines.append("  ⚡ Optimization:")
    lines.append(f"      Learning Rate:          {learning_rate:10.2e}")
    lines.append(f"      Gradient Norm:          {metrics.get('grad_norm', 0):10.4f}")
    
    # Performance
    lines.append("")
    lines.append("  ⏱️  Performance:")
    lines.append(f"      Collect Time:           {metrics.get('collect_time', 0):10.2f}s")
    lines.append(f"      Update Time:            {metrics.get('update_time', 0):10.2f}s")
    lines.append(f"      Total Time:             {metrics.get('total_time', 0):10.2f}s")
    lines.append(f"      FPS:                    {metrics.get('fps', 0):10.1f}")
    
    lines.append("")
    lines.append("-" * 80)
    
    return "\n".join(lines)


def format_training_stats_compact(
    iteration: int,
    metrics: Dict[str, float],
    learning_rate: float,
) -> str:
    """
    Format training statistics in a compact single-line format.
    
    Args:
        iteration: Current iteration number.
        metrics: Dictionary of training metrics.
        learning_rate: Current learning rate.
        
    Returns:
        Formatted string for printing.
    """
    return (
        f"[{iteration:6d}] "
        f"ret={metrics.get('mean_episode_return', 0):6.1f} | "
        f"loss={metrics.get('policy_loss', 0):8.5f} | "
        f"ent={metrics.get('entropy', 0):5.3f} | "
        f"clip={metrics.get('clip_fraction', 0):5.3f} | "
        f"kl={metrics.get('approx_kl', 0):7.5f} | "
        f"prob={metrics.get('action_prob_mean', 0):5.3f} | "
        f"lr={learning_rate:.2e} | "
        f"fps={metrics.get('fps', 0):5.0f}"
    )


class Checkpointer:
    """Handles model checkpointing."""
    
    def __init__(self, checkpoint_dir: str, policy: PolicyNetwork):
        """
        Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory for checkpoints.
            policy: Policy network to checkpoint.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.policy = policy
        self.checkpoint = tf.train.Checkpoint(policy=policy)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            str(self.checkpoint_dir),
            max_to_keep=5
        )
    
    def save(self, step: int) -> str:
        """Save a checkpoint."""
        path = self.manager.save(checkpoint_number=step)
        return path
    
    def restore_latest(self) -> Optional[int]:
        """Restore the latest checkpoint. Returns step number or None."""
        latest = self.manager.latest_checkpoint
        if latest:
            self.checkpoint.restore(latest)
            # Extract step from checkpoint name
            step = int(latest.split('-')[-1])
            return step
        return None
    
    def save_weights(self, path: str) -> None:
        """Save policy weights only."""
        # Keras 3 requires .weights.h5 extension
        if not path.endswith('.weights.h5'):
            path = path + '.weights.h5'
        self.policy.save_weights(path)
    
    def load_weights(self, path: str) -> None:
        """Load policy weights only."""
        # Keras 3 requires .weights.h5 extension
        if not path.endswith('.weights.h5'):
            path = path + '.weights.h5'
        self.policy.load_weights(path)


class Evaluator:
    """Evaluates policy performance."""
    
    def __init__(
        self,
        env_fn: Callable,
        policy: PolicyNetwork,
        num_episodes: int = 10,
    ):
        """
        Initialize evaluator.
        
        Args:
            env_fn: Factory function for evaluation environment.
            policy: Policy to evaluate.
            num_episodes: Number of episodes to run.
        """
        self.env_fn = env_fn
        self.policy = policy
        self.num_episodes = num_episodes
    
    def evaluate(self, render: bool = False) -> Dict[str, float]:
        """
        Run evaluation episodes.
        
        Args:
            render: Whether to render (not implemented for headless).
            
        Returns:
            Dictionary of evaluation metrics.
        """
        env = self.env_fn()
        
        episode_returns = []
        episode_lengths = []
        
        for ep in range(self.num_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                # Use greedy action for evaluation
                obs_tensor = tf.expand_dims(
                    tf.convert_to_tensor(obs, dtype=tf.float32), axis=0
                )
                logits = self.policy(obs_tensor, training=False)
                action = int(tf.argmax(logits, axis=-1)[0])
                
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += reward
                episode_length += 1
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        env.close()
        
        returns = np.array(episode_returns)
        lengths = np.array(episode_lengths)
        
        return {
            'eval_mean_return': float(np.mean(returns)),
            'eval_std_return': float(np.std(returns)),
            'eval_min_return': float(np.min(returns)),
            'eval_max_return': float(np.max(returns)),
            'eval_mean_length': float(np.mean(lengths)),
        }


class GRPOTrainer:
    """
    Main trainer class for GRPO on Atari.
    
    Orchestrates the complete training process including:
    - Environment setup with seeded groups
    - Policy initialization
    - Trajectory collection
    - GRPO updates
    - Logging, evaluation, and checkpointing
    """
    
    def __init__(self, config: GRPOConfig, verbose: bool = True):
        """
        Initialize the trainer.
        
        Args:
            config: GRPO configuration.
            verbose: Whether to print detailed training statistics.
        """
        self.config = config
        self.verbose = verbose
        config.validate()
        
        # Set random seeds
        if config.seed is not None:
            np.random.seed(config.seed)
            tf.random.set_seed(config.seed)
        
        # Create environment factory
        self.env_fn = lambda: create_atari_env(
            env_name=config.env_name,
            frame_skip=config.frame_skip,
            frame_stack=config.frame_stack,
            screen_size=config.obs_height,
        )
        
        # Create parallel environment groups
        self.env_groups = ParallelEnvGroups(
            env_fn=self.env_fn,
            num_groups=config.num_groups,
            group_size=config.group_size,
            base_seed=config.seed or 42,
        )
        
        # Create policy network
        self.policy = create_policy_network(
            obs_shape=config.obs_shape,
            num_actions=self.env_groups.num_actions,
        )
        
        # Create trajectory collector
        self.collector = TrajectoryCollector(
            env_groups=self.env_groups,
            policy=self.policy,
            steps_per_trajectory=config.steps_per_trajectory,
            gamma=config.gamma,
        )
        
        # Create optimizer
        self.optimizer = GRPOOptimizer(
            policy=self.policy,
            learning_rate=config.learning_rate,
            epsilon_clip=config.epsilon_clip,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            adam_epsilon=config.adam_epsilon,
        )
        
        # Create logger and checkpointer
        self.logger = Logger(config.log_dir)
        self.checkpointer = Checkpointer(config.checkpoint_dir, self.policy)
        
        # Create evaluator
        self.evaluator = Evaluator(
            env_fn=self.env_fn,
            policy=self.policy,
            num_episodes=config.eval_episodes,
        )
        
        # Create demo recorder for mid-training demos
        self.demo_recorder = None
        if config.demo_interval > 0:
            self.demo_recorder = DemoRecorder(
                env_name=config.env_name,
                demo_dir=config.demo_dir,
                seed=config.seed or 42,
                num_steps=config.demo_steps,
                fps=config.demo_fps,
            )
        
        # Training state
        self.global_step = 0
        self.start_time = None
    
    def train(self, resume: bool = False) -> Dict[str, List]:
        """
        Run the complete training loop.
        
        Args:
            resume: Whether to resume from latest checkpoint.
            
        Returns:
            Dictionary containing training history.
        """
        config = self.config
        
        # Resume from checkpoint if requested
        if resume:
            restored_step = self.checkpointer.restore_latest()
            if restored_step is not None:
                self.global_step = restored_step
                print(f"Resumed from step {self.global_step}")
        
        self.start_time = time.time()
        
        # Initialize environments
        self.collector.reset()
        
        # Print training header
        print()
        print("=" * 80)
        print("  GRPO Training")
        print("=" * 80)
        print(f"  Environment:         {config.env_name}")
        print(f"  Groups:              {config.num_groups}")
        print(f"  Group Size:          {config.group_size}")
        print(f"  Total Environments:  {config.total_envs}")
        print(f"  Steps/Trajectory:    {config.steps_per_trajectory}")
        print(f"  Batch Size:          {config.batch_size}")
        print(f"  Learning Rate:       {config.learning_rate:.2e}")
        print(f"  Epsilon Clip:        {config.epsilon_clip}")
        print(f"  Entropy Coef:        {config.entropy_coef}")
        print(f"  Gamma:               {config.gamma}")
        print(f"  Update Epochs:       {config.update_epochs}")
        print(f"  Minibatch Size:      {config.minibatch_size}")
        print(f"  Total Iterations:    {config.num_iterations}")
        print("=" * 80)
        print()
        
        for iteration in range(self.global_step, config.num_iterations):
            self.global_step = iteration
            iter_start_time = time.time()
            
            # Collect trajectories
            trajectory_batch = self.collector.collect()
            collect_time = time.time() - iter_start_time
            
            # Perform GRPO update
            update_start_time = time.time()
            update_metrics = self.optimizer.update(
                trajectory_batch=trajectory_batch,
                gamma=config.gamma,
                update_epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
                advantage_mode='per_timestep',  # Use per-timestep advantages
            )
            update_time = time.time() - update_start_time
            
            # Get episode statistics
            episode_stats = self.collector.get_episode_statistics()
            self.collector.clear_episode_statistics()
            
            # Get current learning rate
            current_lr = float(self.optimizer.optimizer.learning_rate)
            
            # Combine all metrics
            metrics = {
                **update_metrics,
                **episode_stats,
                'learning_rate': current_lr,
                'collect_time': collect_time,
                'update_time': update_time,
                'total_time': time.time() - iter_start_time,
                'fps': config.batch_size / (time.time() - iter_start_time),
            }
            
            # Always log to file/tensorboard
            self.logger.log(iteration, metrics)
            
            # Console output
            if iteration % config.log_interval == 0:
                elapsed_time = time.time() - self.start_time
                
                if self.verbose:
                    # Detailed multi-line output
                    print(format_training_stats(
                        iteration=iteration,
                        total_iterations=config.num_iterations,
                        metrics=metrics,
                        elapsed_time=elapsed_time,
                        learning_rate=current_lr,
                    ))
                else:
                    # Compact single-line output
                    print(format_training_stats_compact(
                        iteration=iteration,
                        metrics=metrics,
                        learning_rate=current_lr,
                    ))
            
            # Evaluation
            if iteration % config.eval_interval == 0 and iteration > 0:
                eval_metrics = self.evaluator.evaluate()
                self.logger.log(iteration, eval_metrics)
                print()
                print(f"  🎮 Evaluation ({config.eval_episodes} episodes):")
                print(f"      Mean Return:  {eval_metrics['eval_mean_return']:.2f} ± {eval_metrics['eval_std_return']:.2f}")
                print(f"      Min/Max:      {eval_metrics['eval_min_return']:.2f} / {eval_metrics['eval_max_return']:.2f}")
                print(f"      Mean Length:  {eval_metrics['eval_mean_length']:.1f}")
                print()
            
            # Checkpointing
            if iteration % config.save_interval == 0 and iteration > 0:
                path = self.checkpointer.save(iteration)
                print(f"  💾 Saved checkpoint: {path}")
            
            # Demo recording
            if (self.demo_recorder is not None and 
                config.demo_interval > 0 and 
                iteration % config.demo_interval == 0 and 
                iteration > 0):
                demo_path = self.demo_recorder.record_and_save(
                    policy=self.policy,
                    iteration=iteration,
                    deterministic=False,
                )
                print(f"  🎬 Saved demo: {demo_path}")
        
        # Final evaluation
        print()
        print("=" * 80)
        print("  Final Evaluation")
        print("=" * 80)
        final_metrics = self.evaluator.evaluate()
        self.logger.log(self.global_step, final_metrics)
        print(f"  Mean Return:  {final_metrics['eval_mean_return']:.2f} ± {final_metrics['eval_std_return']:.2f}")
        print(f"  Min/Max:      {final_metrics['eval_min_return']:.2f} / {final_metrics['eval_max_return']:.2f}")
        print(f"  Mean Length:  {final_metrics['eval_mean_length']:.1f}")
        
        # Save final model
        self.checkpointer.save(self.global_step)
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print()
        print("=" * 80)
        print(f"  Training Complete!")
        print(f"  Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"  Final Step: {self.global_step}")
        print("=" * 80)
        
        # Cleanup
        self.env_groups.close()
        if self.demo_recorder is not None:
            self.demo_recorder.close()
        
        return {'history': self.logger.history}
    
    def get_policy(self) -> PolicyNetwork:
        """Get the trained policy network."""
        return self.policy
    
    def save_policy(self, path: str) -> None:
        """Save policy weights to a file."""
        # Keras 3 requires .weights.h5 extension
        if not path.endswith('.weights.h5'):
            path = path + '.weights.h5'
        self.policy.save_weights(path)
    
    def load_policy(self, path: str) -> None:
        """Load policy weights from a file."""
        # Keras 3 requires .weights.h5 extension
        if not path.endswith('.weights.h5'):
            path = path + '.weights.h5'
        self.policy.load_weights(path)


def create_trainer(
    env_name: str = "ALE/Breakout-v5",
    group_size: int = 8,
    num_groups: int = 4,
    **kwargs
) -> GRPOTrainer:
    """
    Factory function to create a GRPO trainer with custom settings.
    
    Args:
        env_name: Atari environment name.
        group_size: Number of trajectories per group.
        num_groups: Number of environment groups.
        **kwargs: Additional config overrides.
        
    Returns:
        Configured GRPOTrainer instance.
    """
    config = GRPOConfig(
        env_name=env_name,
        group_size=group_size,
        num_groups=num_groups,
        **kwargs
    )
    return GRPOTrainer(config)
