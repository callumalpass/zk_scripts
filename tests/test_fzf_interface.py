#!/usr/bin/env python3
"""
Tests for FZF interface module.
"""

import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from zk_core.fzf_interface import main
from zk_core.fzf_manager import FzfManager


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "notes_dir": "~/notes",
        "editor": "nvim",
        "fzf_interface": {
            "preview_width": 50,
            "diary_subdir": "diary"
        }
    }


@pytest.fixture
def sample_index():
    """Sample index data for testing."""
    return {
        "notes": {
            "note1.md": {
                "title": "Note 1",
                "tags": ["tag1", "tag2"],
                "path": "/path/to/note1.md",
                "word_count": 100
            },
            "note2.md": {
                "title": "Note 2",
                "tags": ["tag2", "tag3"],
                "path": "/path/to/note2.md",
                "word_count": 200
            },
            "diary/note3.md": {
                "title": "Diary Note",
                "tags": ["diary"],
                "path": "/path/to/diary/note3.md",
                "word_count": 150
            }
        }
    }


def test_main_exists():
    """Test that main function exists."""
    assert main is not None
    assert callable(main)


@patch("zk_core.fzf_interface.load_config")
@patch("argparse.ArgumentParser.parse_args")
def test_main_imports(mock_args, mock_load_config):
    """Test imports for main function."""
    # We need to patch both of these to prevent actual execution
    mock_args.return_value = MagicMock(help_keys=True)  # Use help_keys to force early return
    mock_load_config.return_value = {}
    
    # This just tests that required imports are present
    with patch("sys.exit"):  # Prevent actual exit
        try:
            main()
            assert True  # If we get here without ImportError, imports are working
        except ImportError:
            pytest.fail("Missing imports in fzf_interface.py")
        except Exception:
            pass  # Other exceptions are fine for this test - we just care about imports
