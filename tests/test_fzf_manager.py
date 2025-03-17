#!/usr/bin/env python3
"""
Tests for FZF manager module.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

from zk_core.fzf_manager import FzfManager, FzfBinding


def test_fzf_binding_creation():
    """Test creating FzfBinding objects."""
    # Test basic binding
    binding = FzfBinding("ctrl-t", "Test action", "echo 'test'")
    assert binding.key == "ctrl-t"
    assert binding.description == "Test action"
    assert binding.action == "echo 'test'"
    
    # Test string representation
    assert str(binding) == "ctrl-t: Test action"
    
    # Test with execute flag
    binding = FzfBinding("ctrl-e", "Execute", "command", execute=True)
    assert binding.key == "ctrl-e"
    assert binding.execute is True


@patch("subprocess.Popen")
def test_fzf_manager_initialization(mock_popen):
    """Test FzfManager initialization."""
    # Set up the mock
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.communicate.return_value = (b"selected item\n", b"")
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    
    # Create a manager
    manager = FzfManager(
        prompt="Test>",
        multi=True,
        preview_command="echo 'preview'",
        header="Test Header"
    )
    
    # Check properties were set
    assert manager.prompt == "Test>"
    assert manager.multi is True
    assert manager.preview_command == "echo 'preview'"
    assert manager.header == "Test Header"
    assert manager.bindings == []


def test_add_binding():
    """Test adding bindings to the manager."""
    manager = FzfManager(prompt="Test>")
    
    # Add a binding
    manager.add_binding("ctrl-x", "Exit", "exit 0")
    assert len(manager.bindings) == 1
    assert manager.bindings[0].key == "ctrl-x"
    assert manager.bindings[0].description == "Exit"
    
    # Add another binding
    manager.add_binding("ctrl-o", "Open", "open {}", execute=True)
    assert len(manager.bindings) == 2
    assert manager.bindings[1].execute is True


@patch("subprocess.Popen")
def test_run_with_items(mock_popen):
    """Test running FZF with a list of items."""
    # Set up the mock
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stdout.read.return_value = b"selected item\n"
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    
    # Create a manager and run it
    manager = FzfManager(prompt="Test>")
    result = manager.run(["item1", "item2", "item3"])
    
    # Check the result
    assert result == ["selected item"]
    
    # Check FZF was called with expected arguments
    args = mock_popen.call_args[0][0]
    assert "fzf" in args
    assert "--prompt=Test>" in args


@patch("subprocess.Popen")
def test_run_with_bindings(mock_popen):
    """Test running FZF with custom bindings."""
    # Set up the mock
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stdout.read.return_value = b"selected item\n"
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    
    # Create a manager with bindings
    manager = FzfManager(prompt="Test>")
    manager.add_binding("ctrl-t", "Test", "echo test")
    manager.add_binding("ctrl-o", "Open", "open {}", execute=True)
    
    # Run the manager
    result = manager.run(["item1", "item2"])
    
    # Check the arguments passed to FZF
    args = mock_popen.call_args[0][0]
    assert "--bind=ctrl-t:execute(echo test)" in args
    assert "--bind=ctrl-o:execute-silent(open {})" in args


@patch("subprocess.Popen")
def test_run_with_no_selection(mock_popen):
    """Test handling when user makes no selection."""
    # Set up the mock for no selection
    mock_process = MagicMock()
    mock_process.stdin = MagicMock()
    mock_process.stdout = MagicMock()
    mock_process.stdout.read.return_value = b""
    mock_process.returncode = 1  # Non-zero return code
    mock_popen.return_value = mock_process
    
    # Create a manager and run it
    manager = FzfManager(prompt="Test>")
    result = manager.run(["item1", "item2"])
    
    # Check the result
    assert result == []
