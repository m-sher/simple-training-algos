"""
Command-line interface modules for GRPO.

Entry points:
- grpo-train: Train a GRPO agent on Atari
- grpo-eval: Evaluate a trained policy
- grpo-test: Run the test suite
- grpo-demo: Run a visual demo and save as GIF
"""

from grpo_atari.cli import train, eval, test, demo

__all__ = ["train", "eval", "test", "demo"]
