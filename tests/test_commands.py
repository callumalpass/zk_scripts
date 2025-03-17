#!/usr/bin/env python3
"""
Tests for command execution module.
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

from zk_core.commands import CommandExecutor, run_command


def test_command_executor_basic():
    """Test basic command execution."""
    # Test echo command
    rc, stdout, stderr = CommandExecutor.run(["echo", "test"])
    assert rc == 0
    assert stdout.strip() == "test"
    assert stderr.strip() == ""


def test_command_executor_with_error():
    """Test command that produces an error."""
    # Test command that doesn't exist
    rc, stdout, stderr = CommandExecutor.run(["command_that_does_not_exist"])
    assert rc != 0
    assert "not found" in stderr or "No such file" in stderr


def test_command_executor_with_input():
    """Test command with input data."""
    # Test command with input
    rc, stdout, stderr = CommandExecutor.run(["cat"], input_data="test input")
    assert rc == 0
    assert stdout.strip() == "test input"
    assert stderr.strip() == ""


def test_command_executor_with_env():
    """Test command with custom environment variables."""
    # Test with custom environment
    env = os.environ.copy()
    env["TEST_VAR"] = "test_value"
    
    rc, stdout, stderr = CommandExecutor.run(["env"], env=env)
    assert rc == 0
    assert "TEST_VAR=test_value" in stdout


def test_command_executor_with_cwd():
    """Test command with custom working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Run pwd in the temp directory
        rc, stdout, stderr = CommandExecutor.run(["pwd"], cwd=tmpdir)
        assert rc == 0
        assert os.path.samefile(stdout.strip(), tmpdir)


def test_compatibility_with_run_command():
    """Test backward compatibility with run_command function."""
    # The run_command function should provide the same interface
    rc1, stdout1, stderr1 = CommandExecutor.run(["echo", "test"])
    rc2, stdout2, stderr2 = run_command(["echo", "test"])
    
    assert rc1 == rc2
    assert stdout1 == stdout2
    assert stderr1 == stderr2


@patch("subprocess.run")
def test_command_executor_subprocess_error(mock_run):
    """Test handling of subprocess errors."""
    # Set up the mock to raise an exception
    mock_run.side_effect = Exception("Simulated error")
    
    # Run should handle the exception
    rc, stdout, stderr = CommandExecutor.run(["command"])
    
    assert rc != 0
    assert stdout == ""
    assert "Error executing command" in stderr
    assert "Simulated error" in stderr
