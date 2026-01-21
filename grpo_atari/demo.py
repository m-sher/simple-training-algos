"""
Demo recording utilities for GRPO.

This module provides reusable functions for recording gameplay demos
as GIF animations. Used by both the CLI demo command and the trainer
for mid-training demo generation.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Import TensorFlow lazily to avoid slow imports
_tf = None
def _get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


def create_stats_header(width: int, episode: int, step: int, total_return: float,
                        height: int = 50, bg_color: Tuple[int, int, int] = (40, 40, 40)):
    """
    Create a stats header bar to display above the game frame.
    
    Args:
        width: Width of the header (should match game frame width)
        episode: Current episode number
        step: Current step in episode
        total_return: Cumulative return in current episode
        height: Height of the header bar
        bg_color: Background color (RGB tuple)
        
    Returns:
        RGB numpy array of the header
    """
    # Create header background
    header = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    try:
        import cv2
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        line_height = 14
        
        # Define stats lines
        stats_lines = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Return: {total_return:.0f}",
        ]
        
        # Calculate starting y position to center the text block vertically
        total_text_height = len(stats_lines) * line_height
        y_start = (height - total_text_height) // 2 + line_height - 2
        
        # Draw each line
        for i, line in enumerate(stats_lines):
            y = y_start + i * line_height
            x = 8  # Left padding
            
            # Draw text with slight shadow for readability
            cv2.putText(header, line, (x + 1, y + 1), font, font_scale, (0, 0, 0), thickness)
            cv2.putText(header, line, (x, y), font, font_scale, (255, 255, 255), thickness)
        
    except ImportError:
        pass  # Return plain header if cv2 not available
    
    return header


def create_action_prob_bar(width: int, action_probs: List[float], 
                           action_names: Optional[List[str]] = None,
                           height: int = 55, 
                           bg_color: Tuple[int, int, int] = (30, 30, 30),
                           bar_color: Tuple[int, int, int] = (66, 133, 244),
                           selected_color: Tuple[int, int, int] = (52, 168, 83),
                           selected_action: Optional[int] = None):
    """
    Create a horizontal bar chart showing action probabilities.
    
    Args:
        width: Width of the bar chart (should match game frame width)
        action_probs: List of action probabilities
        action_names: Optional list of action names
        height: Height of the bar chart section
        bg_color: Background color (RGB tuple)
        bar_color: Color for probability bars (RGB tuple)
        selected_color: Color for the selected action bar (RGB tuple)
        selected_action: Index of the selected action (to highlight)
        
    Returns:
        RGB numpy array of the bar chart
    """
    num_actions = len(action_probs)
    
    # Default action names for Atari Breakout
    if action_names is None:
        if num_actions == 4:
            action_names = ["NOOP", "FIRE", "RIGHT", "LEFT"]
        else:
            action_names = [f"A{i}" for i in range(num_actions)]
    
    # Create bar chart background
    bar_chart = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Calculate bar dimensions
    padding = 10
    bar_width = (width - 2 * padding - (num_actions - 1) * 5) // num_actions
    max_bar_height = height - 30  # Leave space for labels
    
    try:
        import cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        for i, (prob, name) in enumerate(zip(action_probs, action_names)):
            # Calculate bar position
            x_start = padding + i * (bar_width + 5)
            x_end = x_start + bar_width
            
            # Calculate bar height based on probability
            bar_height = int(prob * max_bar_height)
            y_bottom = height - 18  # Leave space for label
            y_top = y_bottom - bar_height
            
            # Choose color (highlight selected action)
            if selected_action is not None and i == selected_action:
                color = selected_color
            else:
                color = bar_color
            
            # Draw bar
            if bar_height > 0:
                cv2.rectangle(bar_chart, (x_start, y_top), (x_end, y_bottom), color, -1)
            
            # Draw bar outline
            cv2.rectangle(bar_chart, (x_start, y_bottom - max_bar_height), 
                         (x_end, y_bottom), (80, 80, 80), 1)
            
            # Draw probability value on top of bar
            prob_text = f"{prob:.2f}"
            (tw, th), _ = cv2.getTextSize(prob_text, font, font_scale, thickness)
            text_x = x_start + (bar_width - tw) // 2
            text_y = y_top - 3 if bar_height > 0 else y_bottom - max_bar_height - 3
            cv2.putText(bar_chart, prob_text, (text_x, text_y), font, font_scale, 
                       (200, 200, 200), thickness)
            
            # Draw action name below bar
            (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
            text_x = x_start + (bar_width - tw) // 2
            cv2.putText(bar_chart, name, (text_x, height - 5), font, font_scale,
                       (180, 180, 180), thickness)
        
    except ImportError:
        pass  # Return plain background if cv2 not available
    
    return bar_chart


def compose_demo_frame(game_frame: np.ndarray, episode: int, step: int, 
                       total_return: float, action_probs: List[float],
                       selected_action: Optional[int] = None,
                       action_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Compose a complete demo frame with stats header, game view, and action prob bar.
    
    Args:
        game_frame: RGB numpy array of the game
        episode: Current episode number
        step: Current step in episode
        total_return: Cumulative return in current episode
        action_probs: List of action probabilities
        selected_action: Index of the selected action
        action_names: Optional list of action names
        
    Returns:
        Composed RGB numpy array
    """
    height, width = game_frame.shape[:2]
    
    # Create header (stats section) - 50px for 3 lines of text
    header = create_stats_header(
        width=width,
        episode=episode,
        step=step,
        total_return=total_return,
        height=50,
    )
    
    # Create action probability bar chart
    action_bar = create_action_prob_bar(
        width=width,
        action_probs=action_probs,
        action_names=action_names,
        height=55,
        selected_action=selected_action,
    )
    
    # Stack vertically: header, game, action bar
    composed = np.vstack([header, game_frame, action_bar])
    
    return composed


def save_gif(frames: List[np.ndarray], output_path: str, fps: int = 30,
             resize_width: Optional[int] = None) -> None:
    """
    Save frames as an animated GIF.
    
    Args:
        frames: List of RGB numpy arrays
        output_path: Path to save GIF
        fps: Frames per second
        resize_width: Optional width to resize to
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL/Pillow is required for GIF creation.")
        print("Install with: pip install Pillow")
        return
    
    if not frames:
        print("Warning: No frames to save")
        return
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        
        if resize_width is not None:
            # Calculate new height maintaining aspect ratio
            aspect = img.height / img.width
            new_height = int(resize_width * aspect)
            img = img.resize((resize_width, new_height), Image.Resampling.LANCZOS)
        
        pil_frames.append(img)
    
    # Calculate duration in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,  # Loop forever
        optimize=True,
    )


def record_demo(policy, obs_env, render_env, num_steps: int = 100,
                deterministic: bool = False, show_overlay: bool = True) -> Tuple[List[np.ndarray], Dict]:
    """
    Record a demo using the given policy and environments.
    
    This function runs the policy in the environment and collects RGB frames
    that can be saved as a GIF.
    
    Args:
        policy: PolicyNetwork to use for action selection
        obs_env: Environment providing preprocessed observations for the policy
        render_env: Environment providing RGB frames for rendering
        num_steps: Number of steps to record
        deterministic: Whether to use greedy action selection
        show_overlay: Whether to compose frames with stats and action probs
        
    Returns:
        Tuple of (frames, stats) where frames is a list of RGB arrays and
        stats is a dictionary of episode statistics
    """
    tf = _get_tf()
    
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
    
    while total_steps < num_steps:
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
        
        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            episode_num += 1
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


class DemoRecorder:
    """
    Helper class for recording demos during training.
    
    Creates and manages environments for demo recording, providing a simple
    interface for the trainer to generate demo GIFs.
    """
    
    def __init__(self, env_name: str, demo_dir: str, seed: int = 42,
                 num_steps: int = 100, fps: int = 30):
        """
        Initialize the demo recorder.
        
        Args:
            env_name: Name of the Atari environment
            demo_dir: Directory to save demo GIFs
            seed: Random seed for environment
            num_steps: Number of steps per demo
            fps: Frames per second for GIF
        """
        self.env_name = env_name
        self.demo_dir = Path(demo_dir)
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.num_steps = num_steps
        self.fps = fps
        
        # Environments will be created lazily
        self._obs_env = None
        self._render_env = None
    
    def _create_environments(self):
        """Create the observation and render environments."""
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
        from grpo_atari.environment import create_atari_env
        
        # Create observation environment (preprocessed for policy)
        self._obs_env = create_atari_env(self.env_name, seed=self.seed)
        
        # Create render environment (RGB frames)
        render_base = gym.make(
            self.env_name,
            render_mode='rgb_array',
            frameskip=1,
        )
        self._render_env = AtariPreprocessing(
            render_base,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=False,  # Keep RGB for rendering
            scale_obs=False,
        )
    
    def record_and_save(self, policy, iteration: int, 
                        deterministic: bool = False) -> str:
        """
        Record a demo and save it as a GIF.
        
        Args:
            policy: PolicyNetwork to use
            iteration: Current training iteration (used in filename)
            deterministic: Whether to use greedy actions
            
        Returns:
            Path to the saved GIF file
        """
        # Create environments if needed
        if self._obs_env is None:
            self._create_environments()
        
        # Record the demo
        frames, stats = record_demo(
            policy=policy,
            obs_env=self._obs_env,
            render_env=self._render_env,
            num_steps=self.num_steps,
            deterministic=deterministic,
            show_overlay=True,
        )
        
        # Save the GIF
        output_path = self.demo_dir / f"demo_iter_{iteration:06d}.gif"
        save_gif(frames, str(output_path), fps=self.fps)
        
        return str(output_path)
    
    def close(self):
        """Close the environments."""
        if self._obs_env is not None:
            self._obs_env.close()
            self._obs_env = None
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
