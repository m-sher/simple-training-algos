"""
GRPO Atari Test Suite.

This test suite is organized into subdirectories by component:
- test_config/: Configuration tests
- test_environment/: Environment wrapper tests
- test_model/: Policy network tests
- test_trajectory/: Trajectory collection tests
- test_grpo_loss/: GRPO loss computation tests
- test_trainer/: Trainer tests
- test_integration/: End-to-end integration tests

Run with: pytest tests/ -v
Run with extra verbosity: pytest tests/ -v --log-cli-level=DEBUG
"""
