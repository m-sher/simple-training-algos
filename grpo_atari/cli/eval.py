#!/usr/bin/env python3
"""
GRPO Evaluation CLI.

Evaluate a trained GRPO policy on Atari Breakout.

Usage:
    uv run grpo-eval --weights checkpoints/policy  # Evaluate saved policy
    uv run grpo-eval --episodes 100                # Run 100 episodes
    uv run grpo-eval --help                        # Show all options
"""

import argparse
import sys
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained GRPO policy on Atari",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment settings
    parser.add_argument(
        '--env-name', type=str, default='ALE/Breakout-v5',
        help='Name of the Atari environment'
    )
    
    # Model settings
    parser.add_argument(
        '--weights', type=str, default=None,
        help='Path to policy weights (required unless --random)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory to look for checkpoints (used if --weights not specified)'
    )
    parser.add_argument(
        '--random', action='store_true',
        help='Evaluate a random policy (for baseline comparison)'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--episodes', '-n', type=int, default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Use deterministic (greedy) actions instead of sampling'
    )
    parser.add_argument(
        '--max-steps', type=int, default=27000,
        help='Maximum steps per episode'
    )
    
    # Output settings
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print per-episode results'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Save results to JSON file'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for environment'
    )
    
    return parser.parse_args(args)


def evaluate_policy(policy, env, num_episodes, deterministic=False, 
                    max_steps=27000, verbose=False):
    """
    Evaluate a policy for N episodes.
    
    Args:
        policy: PolicyNetwork to evaluate
        env: Gymnasium environment
        num_episodes: Number of episodes to run
        deterministic: Whether to use greedy actions
        max_steps: Maximum steps per episode
        verbose: Whether to print per-episode results
        
    Returns:
        Dictionary with evaluation statistics
    """
    import numpy as np
    import tensorflow as tf
    
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        
        while not done and episode_length < max_steps:
            obs_tensor = tf.expand_dims(
                tf.convert_to_tensor(obs, dtype=tf.float32), axis=0
            )
            
            if deterministic:
                # Greedy action
                logits = policy(obs_tensor, training=False)
                action = int(tf.argmax(logits, axis=-1)[0])
            else:
                # Sample from policy
                actions, _ = policy.sample_actions(obs_tensor)
                action = int(actions[0])
            
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
            episode_length += 1
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        if verbose:
            print(f"  Episode {ep + 1:3d}: return={episode_return:6.1f}, length={episode_length:5d}")
    
    returns = np.array(episode_returns)
    lengths = np.array(episode_lengths)
    
    return {
        'num_episodes': num_episodes,
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'median_return': float(np.median(returns)),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'episode_returns': returns.tolist(),
        'episode_lengths': lengths.tolist(),
    }


def main(args=None):
    """Main entry point for evaluation."""
    args = parse_args(args)
    
    # Import here to avoid slow imports when just showing help
    import json
    import numpy as np
    import tensorflow as tf
    
    from grpo_atari.config import GRPOConfig
    from grpo_atari.environment import create_atari_env
    from grpo_atari.model import create_policy_network
    
    # Create environment
    print(f"Creating environment: {args.env_name}")
    env = create_atari_env(args.env_name, seed=args.seed)
    
    # Get observation shape and action count
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create policy
    policy = create_policy_network(obs_shape=obs_shape, num_actions=num_actions)
    
    # Load weights
    if args.random:
        print("\nEvaluating RANDOM policy (baseline)")
    elif args.weights:
        weights_path = args.weights
        # Handle Keras 3 extension
        if not weights_path.endswith('.weights.h5'):
            weights_path = weights_path + '.weights.h5'
        print(f"\nLoading weights from: {weights_path}")
        policy.load_weights(weights_path)
    else:
        # Try to find latest checkpoint
        checkpoint_dir = args.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            import glob
            # Look for Keras 3 format first, then TF checkpoint format
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.weights.h5"))
            if not checkpoints:
                checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.index"))
            if checkpoints:
                latest = max(checkpoints, key=os.path.getctime)
                weights_path = latest.replace('.index', '')
                print(f"\nLoading latest checkpoint: {weights_path}")
                policy.load_weights(weights_path)
            else:
                print(f"\nNo checkpoints found in {checkpoint_dir}")
                print("Using random policy (untrained)")
        else:
            print(f"\nCheckpoint directory not found: {checkpoint_dir}")
            print("Using random policy (untrained)")
    
    # Run evaluation
    print(f"\nRunning {args.episodes} evaluation episodes...")
    print(f"Mode: {'deterministic (greedy)' if args.deterministic else 'stochastic (sampling)'}")
    print("-" * 40)
    
    results = evaluate_policy(
        policy=policy,
        env=env,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    
    env.close()
    
    # Print summary
    print("-" * 40)
    print("\nEvaluation Results:")
    print(f"  Episodes:      {results['num_episodes']}")
    print(f"  Mean Return:   {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Min Return:    {results['min_return']:.2f}")
    print(f"  Max Return:    {results['max_return']:.2f}")
    print(f"  Median Return: {results['median_return']:.2f}")
    print(f"  Mean Length:   {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
