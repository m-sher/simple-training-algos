#!/usr/bin/env python3
"""
GRPO Test Suite CLI.

Run the GRPO test suite.

Usage:
    uv run grpo-test                    # Run all tests
    uv run grpo-test -v                 # Verbose output
    uv run grpo-test --module model     # Test specific module
    uv run grpo-test --help             # Show all options
"""

import argparse
import sys
import os


def parse_args(args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the GRPO test suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Test selection
    parser.add_argument(
        '--module', '-m', type=str, default=None,
        choices=['config', 'environment', 'model', 'trajectory', 
                 'grpo_loss', 'trainer', 'integration', 'all'],
        help='Specific module to test (default: all)'
    )
    parser.add_argument(
        '--pattern', '-k', type=str, default=None,
        help='Only run tests matching this pattern (pytest -k)'
    )
    parser.add_argument(
        '--markers', type=str, default=None,
        help='Only run tests with these markers (pytest -m)'
    )
    
    # Output verbosity
    parser.add_argument(
        '--verbose', '-v', action='count', default=0,
        help='Increase verbosity (-v, -vv, -vvv)'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Minimal output'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level for test output'
    )
    
    # Test execution
    parser.add_argument(
        '--failfast', '-x', action='store_true',
        help='Stop on first failure'
    )
    parser.add_argument(
        '--timeout', type=int, default=120,
        help='Timeout per test in seconds'
    )
    parser.add_argument(
        '--no-capture', '-s', action='store_true',
        help='Disable output capture (show print statements)'
    )
    
    # Coverage
    parser.add_argument(
        '--coverage', '--cov', action='store_true',
        help='Run with coverage reporting (requires pytest-cov)'
    )
    
    # Other options
    parser.add_argument(
        '--collect-only', action='store_true',
        help='Only collect tests, do not run them'
    )
    parser.add_argument(
        '--list', '-l', action='store_true',
        help='List available test modules'
    )
    
    return parser.parse_args(args)


def list_test_modules():
    """List available test modules."""
    print("\nAvailable test modules:")
    print("-" * 40)
    
    modules = [
        ("config", "Configuration validation and properties"),
        ("environment", "Environment wrappers and seeding"),
        ("model", "Policy network architecture"),
        ("trajectory", "Trajectory collection and data"),
        ("grpo_loss", "GRPO advantage and loss computation"),
        ("trainer", "Training loop and checkpointing"),
        ("integration", "End-to-end pipeline tests"),
    ]
    
    for name, description in modules:
        print(f"  {name:15s} - {description}")
    
    print("-" * 40)
    print("\nUse --module <name> to run a specific module")
    print("Use --module all to run all tests (default)")


def main(args=None):
    """Main entry point for test runner."""
    args = parse_args(args)
    
    # Handle --list
    if args.list:
        list_test_modules()
        return 0
    
    # Find the tests directory
    # First try relative to this file, then try current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    tests_dir = os.path.join(project_root, 'tests')
    
    if not os.path.exists(tests_dir):
        # Try current directory
        tests_dir = 'tests'
        if not os.path.exists(tests_dir):
            print("Error: Could not find tests directory")
            print(f"Looked in: {os.path.join(project_root, 'tests')}")
            print(f"And in: {os.path.abspath('tests')}")
            return 1
    
    # Build pytest arguments
    pytest_args = [tests_dir]
    
    # Module selection
    if args.module and args.module != 'all':
        module_path = os.path.join(tests_dir, f'test_{args.module}')
        if os.path.exists(module_path):
            pytest_args = [module_path]
        else:
            print(f"Error: Test module not found: {module_path}")
            return 1
    
    # Verbosity
    if args.quiet:
        pytest_args.append('-q')
    elif args.verbose:
        pytest_args.append('-' + 'v' * min(args.verbose, 3))
    else:
        pytest_args.append('-v')
    
    # Log level
    pytest_args.extend(['--log-cli-level', args.log_level])
    
    # Pattern matching
    if args.pattern:
        pytest_args.extend(['-k', args.pattern])
    
    # Markers
    if args.markers:
        pytest_args.extend(['-m', args.markers])
    
    # Fail fast
    if args.failfast:
        pytest_args.append('-x')
    
    # Timeout
    pytest_args.extend(['--timeout', str(args.timeout)])
    
    # No capture
    if args.no_capture:
        pytest_args.append('-s')
    
    # Coverage
    if args.coverage:
        pytest_args.extend(['--cov=grpo_atari', '--cov-report=term-missing'])
    
    # Collect only
    if args.collect_only:
        pytest_args.append('--collect-only')
    
    # Short traceback
    pytest_args.extend(['--tb', 'short'])
    
    # Print what we're running
    print("=" * 60)
    print("GRPO Test Suite")
    print("=" * 60)
    print(f"Tests directory: {os.path.abspath(tests_dir)}")
    print(f"Pytest args: {' '.join(pytest_args)}")
    print("=" * 60)
    print()
    
    # Import and run pytest
    try:
        import pytest
    except ImportError:
        print("Error: pytest not installed")
        print("Install with: uv pip install pytest pytest-timeout")
        return 1
    
    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    # Print summary
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ Tests failed (exit code: {exit_code})")
    print("=" * 60)
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
