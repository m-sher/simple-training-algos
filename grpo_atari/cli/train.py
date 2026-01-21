#!/usr/bin/env python3
"""
GRPO Training CLI.

Train a GRPO agent on Atari Breakout.

Usage:
    uv run grpo-train                      # Train with defaults
    uv run grpo-train --group-size 16      # Custom group size
    uv run grpo-train --help               # Show all options
"""

import argparse
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GRPO on Atari Breakout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment settings
    parser.add_argument(
        '--env-name', type=str, default='ALE/Breakout-v5',
        help='Name of the Atari environment'
    )
    
    # GRPO-specific settings
    parser.add_argument(
        '--group-size', type=int, default=8,
        help='Number of trajectories per group (same initial state)'
    )
    parser.add_argument(
        '--num-groups', type=int, default=4,
        help='Number of environment groups (different initial states)'
    )
    
    # Training settings
    parser.add_argument(
        '--num-iterations', type=int, default=10000,
        help='Number of training iterations'
    )
    parser.add_argument(
        '--steps-per-trajectory', type=int, default=128,
        help='Steps to collect per trajectory'
    )
    parser.add_argument(
        '--update-epochs', type=int, default=4,
        help='Number of epochs per update'
    )
    parser.add_argument(
        '--minibatch-size', type=int, default=256,
        help='Minibatch size for updates'
    )
    
    # Optimizer settings
    parser.add_argument(
        '--learning-rate', type=float, default=2.5e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--epsilon-clip', type=float, default=0.1,
        help='Policy ratio clipping parameter'
    )
    parser.add_argument(
        '--entropy-coef', type=float, default=0.01,
        help='Entropy bonus coefficient'
    )
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5,
        help='Maximum gradient norm for clipping'
    )
    
    # Logging and checkpointing
    parser.add_argument(
        '--log-interval', type=int, default=10,
        help='Logging interval (iterations)'
    )
    parser.add_argument(
        '--save-interval', type=int, default=100,
        help='Checkpoint save interval (iterations)'
    )
    parser.add_argument(
        '--eval-interval', type=int, default=50,
        help='Evaluation interval (iterations)'
    )
    parser.add_argument(
        '--eval-episodes', type=int, default=10,
        help='Number of evaluation episodes'
    )
    
    # Paths
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory for saving checkpoints'
    )
    parser.add_argument(
        '--log-dir', type=str, default='logs',
        help='Directory for logs'
    )
    
    # Demo recording during training
    parser.add_argument(
        '--demo-interval', type=int, default=100,
        help='Record demo every N iterations (0 to disable)'
    )
    parser.add_argument(
        '--demo-steps', type=int, default=100,
        help='Number of steps per demo GIF'
    )
    parser.add_argument(
        '--demo-fps', type=int, default=30,
        help='FPS for demo GIFs'
    )
    parser.add_argument(
        '--demo-dir', type=str, default='demos',
        help='Directory for training demo GIFs'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from latest checkpoint'
    )
    parser.add_argument(
        '--load-weights', type=str, default=None,
        help='Path to load policy weights from'
    )
    
    return parser.parse_args(args)


def main(args=None):
    """Main entry point for training."""
    args = parse_args(args)
    
    # Import here to avoid slow imports when just showing help
    import tensorflow as tf
    import numpy as np
    
    from grpo_atari.config import GRPOConfig
    from grpo_atari.trainer import GRPOTrainer
    from grpo_atari.utils import save_config, set_global_seeds
    
    # Set random seeds
    set_global_seeds(args.seed)
    
    # Create configuration
    config = GRPOConfig(
        env_name=args.env_name,
        group_size=args.group_size,
        num_groups=args.num_groups,
        num_iterations=args.num_iterations,
        steps_per_trajectory=args.steps_per_trajectory,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_clip=args.epsilon_clip,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        demo_interval=args.demo_interval,
        demo_steps=args.demo_steps,
        demo_fps=args.demo_fps,
        demo_dir=args.demo_dir,
        seed=args.seed,
    )
    
    # Validate configuration
    try:
        config.validate()
    except AssertionError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("GRPO Training Configuration")
    print("=" * 60)
    print(f"Environment: {config.env_name}")
    print(f"Group size: {config.group_size}")
    print(f"Number of groups: {config.num_groups}")
    print(f"Total environments: {config.total_envs}")
    print(f"Steps per trajectory: {config.steps_per_trajectory}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Gamma: {config.gamma}")
    print(f"Epsilon clip: {config.epsilon_clip}")
    print(f"Entropy coefficient: {config.entropy_coef}")
    print(f"Seed: {config.seed}")
    print("=" * 60)
    print()
    
    # Save configuration
    os.makedirs(config.log_dir, exist_ok=True)
    save_config(config, os.path.join(config.log_dir, 'config.json'))
    
    # Create trainer
    trainer = GRPOTrainer(config)
    
    # Load weights if specified
    if args.load_weights:
        print(f"Loading weights from {args.load_weights}")
        trainer.load_policy(args.load_weights)
    
    # Run training
    try:
        history = trainer.train(resume=args.resume)
        print("\nTraining completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.checkpointer.save(trainer.global_step)
        print("Checkpoint saved")
        return 1
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise


if __name__ == '__main__':
    sys.exit(main())
