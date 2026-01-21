#!/usr/bin/env python3
"""
GRPO Demo CLI.

Run a visual demo of a trained GRPO policy and save as GIF.

Usage:
    uv run grpo-demo                              # Use latest checkpoint
    uv run grpo-demo --checkpoint path/to/ckpt    # Use specific checkpoint
    uv run grpo-demo --steps 500 --output demo.gif
    uv run grpo-demo --help                       # Show all options
"""

import argparse
import sys
import os
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a visual demo of a trained GRPO policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Environment settings
    parser.add_argument(
        '--env-name', type=str, default='ALE/Breakout-v5',
        help='Name of the Atari environment'
    )
    
    # Model settings
    parser.add_argument(
        '--checkpoint', '-c', type=str, default=None,
        help='Path to checkpoint file (weights or TF checkpoint)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory to search for latest checkpoint'
    )
    
    # Demo settings
    parser.add_argument(
        '--steps', '-n', type=int, default=1000,
        help='Number of steps to run'
    )
    parser.add_argument(
        '--episodes', '-e', type=int, default=None,
        help='Number of episodes to run (overrides --steps)'
    )
    parser.add_argument(
        '--deterministic', '-d', action='store_true',
        help='Use deterministic (greedy) actions instead of sampling'
    )
    parser.add_argument(
        '--fps', type=int, default=30,
        help='Frames per second in output GIF'
    )
    
    # Output settings
    parser.add_argument(
        '--output', '-o', type=str, default='demo.gif',
        help='Output GIF file path'
    )
    parser.add_argument(
        '--resize', type=int, default=None,
        help='Resize frames to this width (maintains aspect ratio)'
    )
    parser.add_argument(
        '--no-overlay', action='store_true',
        help='Disable text overlay on frames'
    )
    
    # Misc
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for environment'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Print detailed progress'
    )
    
    return parser.parse_args(args)


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to the latest checkpoint, or None if not found
    """
    import glob
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for Keras 3 format (.weights.h5)
    h5_checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.weights.h5"))
    if h5_checkpoints:
        return max(h5_checkpoints, key=os.path.getctime)
    
    # Look for TF checkpoint format (.index files)
    index_files = glob.glob(os.path.join(checkpoint_dir, "*.index"))
    if index_files:
        latest = max(index_files, key=os.path.getctime)
        return latest.replace('.index', '')
    
    # Look for TF checkpoint with 'checkpoint' file
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            first_line = f.readline()
            # Parse: model_checkpoint_path: "ckpt-10"
            if 'model_checkpoint_path' in first_line:
                ckpt_name = first_line.split('"')[1]
                return os.path.join(checkpoint_dir, ckpt_name)
    
    return None


def run_demo_with_episodes(policy, obs_env, render_env, num_steps=None, 
                           num_episodes=None, deterministic=False, 
                           show_overlay=True, verbose=False):
    """
    Run demo with support for episode-based stopping.
    
    This extends the basic record_demo to support stopping after N episodes.
    """
    import numpy as np
    import tensorflow as tf
    from grpo_atari.demo import compose_demo_frame
    
    frames = []
    episode_returns = []
    episode_lengths = []
    
    current_return = 0.0
    current_length = 0
    total_steps = 0
    episode_num = 0
    
    # Reset both environments
    obs, _ = obs_env.reset()
    render_env.reset()
    
    # Determine stopping condition
    if num_episodes is not None:
        stop_condition = lambda: episode_num >= num_episodes
        progress_msg = f"Recording {num_episodes} episodes"
    else:
        num_steps = num_steps or 1000
        stop_condition = lambda: total_steps >= num_steps
        progress_msg = f"Recording {num_steps} steps"
    
    if verbose:
        print(progress_msg)
    
    while not stop_condition():
        # Render frame from render environment
        game_frame = render_env.render()
        
        if game_frame is None:
            continue
        
        # Get action and probabilities from policy
        obs_tensor = tf.expand_dims(
            tf.convert_to_tensor(obs, dtype=tf.float32), axis=0
        )
        
        # Get logits and compute probabilities
        logits = policy(obs_tensor, training=False)
        action_probs = tf.nn.softmax(logits, axis=-1)[0].numpy().tolist()
        
        if deterministic:
            action = int(tf.argmax(logits, axis=-1)[0])
        else:
            actions, _ = policy.sample_actions(obs_tensor)
            action = int(actions[0])
        
        # Compose frame with stats header and action probability bar
        if show_overlay:
            composed_frame = compose_demo_frame(
                game_frame=game_frame,
                episode=episode_num + 1,
                step=current_length,
                total_return=current_return,
                action_probs=action_probs,
                selected_action=action,
            )
            frames.append(composed_frame)
        else:
            frames.append(game_frame)
        
        # Step both environments with the same action
        obs, reward, terminated, truncated, _ = obs_env.step(action)
        render_env.step(action)
        
        done = terminated or truncated
        
        current_return += reward
        current_length += 1
        total_steps += 1
        
        # Print progress
        if verbose and total_steps % 100 == 0:
            print(f"  Step {total_steps}, Episode {episode_num + 1}, Return so far: {current_return:.0f}")
        
        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            episode_num += 1
            
            if verbose:
                print(f"  Episode {episode_num} finished: return={current_return:.0f}, length={current_length}")
            
            current_return = 0.0
            current_length = 0
            obs, _ = obs_env.reset()
            render_env.reset()
    
    # Add final partial episode if any
    if current_length > 0:
        episode_returns.append(current_return)
        episode_lengths.append(current_length)
    
    stats = {
        'total_steps': total_steps,
        'num_episodes': len(episode_returns),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'mean_return': float(np.mean(episode_returns)) if episode_returns else 0.0,
        'total_return': float(sum(episode_returns)),
    }
    
    return frames, stats


def main(args=None):
    """Main entry point for demo."""
    args = parse_args(args)
    
    # Import here to avoid slow imports when just showing help
    import numpy as np
    import tensorflow as tf
    
    from grpo_atari.model import create_policy_network
    from grpo_atari.environment import create_atari_env
    from grpo_atari.demo import save_gif
    
    print("=" * 60)
    print("  GRPO Demo")
    print("=" * 60)
    
    # Create environment with rgb_array render mode
    print(f"\nCreating environment: {args.env_name}")
    
    import gymnasium as gym
    from gymnasium.wrappers import AtariPreprocessing
    
    # Create the observation environment using our standard wrapper
    obs_env = create_atari_env(args.env_name, seed=args.seed)
    
    # Create a separate render environment for RGB frames
    render_env = gym.make(
        args.env_name,
        render_mode='rgb_array',
        frameskip=1,
    )
    render_env = AtariPreprocessing(
        render_env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=False,  # Keep RGB for rendering
        scale_obs=False,
    )
    
    # Get observation shape and action count
    obs_shape = obs_env.observation_space.shape
    num_actions = obs_env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create policy
    policy = create_policy_network(obs_shape=obs_shape, num_actions=num_actions)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
    
    if checkpoint_path is not None:
        print(f"\nLoading checkpoint: {checkpoint_path}")
        try:
            # Handle both formats
            if checkpoint_path.endswith('.weights.h5'):
                policy.load_weights(checkpoint_path)
            elif os.path.exists(checkpoint_path + '.weights.h5'):
                policy.load_weights(checkpoint_path + '.weights.h5')
            elif os.path.exists(checkpoint_path + '.index'):
                checkpoint = tf.train.Checkpoint(policy=policy)
                checkpoint.restore(checkpoint_path).expect_partial()
            else:
                checkpoint = tf.train.Checkpoint(policy=policy)
                checkpoint.restore(checkpoint_path).expect_partial()
            print("  ✓ Checkpoint loaded successfully")
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            print("  Using untrained (random) policy")
    else:
        print(f"\n⚠ No checkpoint found in {args.checkpoint_dir}")
        print("  Using untrained (random) policy")
    
    # Run demo
    print(f"\nRunning demo...")
    print(f"  Mode: {'deterministic (greedy)' if args.deterministic else 'stochastic (sampling)'}")
    if args.episodes:
        print(f"  Episodes: {args.episodes}")
    else:
        print(f"  Steps: {args.steps}")
    print(f"  Output: {args.output}")
    print(f"  FPS: {args.fps}")
    print()
    
    frames, stats = run_demo_with_episodes(
        policy=policy,
        obs_env=obs_env,
        render_env=render_env,
        num_steps=args.steps if not args.episodes else None,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        show_overlay=not args.no_overlay,
        verbose=args.verbose,
    )
    
    obs_env.close()
    render_env.close()
    
    # Save GIF
    print(f"\nSaving GIF with {len(frames)} frames...")
    save_gif(
        frames=frames,
        output_path=args.output,
        fps=args.fps,
        resize_width=args.resize,
    )
    
    # Print summary
    print()
    print("=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print(f"  Total Steps:    {stats['total_steps']}")
    print(f"  Episodes:       {stats['num_episodes']}")
    print(f"  Total Return:   {stats['total_return']:.0f}")
    print(f"  Mean Return:    {stats['mean_return']:.2f}")
    print(f"  Output File:    {args.output}")
    print(f"  File Size:      {os.path.getsize(args.output) / 1024:.1f} KB")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
