#!/usr/bin/env python3
"""
Tests for indexing module.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest
from typer.testing import CliRunner

from zk_core.index import app


@pytest.fixture
def sample_note_content():
    """Sample markdown note content for testing."""
    return """---
title: Test Note
tags: [test, sample]
date: 2023-01-01
---

# Test Note

This is a test note with a [[wikilink]] and a citation @test2023.

It also has multiple paragraphs and a [[second|link with alias]].
"""


@pytest.fixture
def temp_note_file(sample_note_content):
    """Create a temporary note file."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp.write(sample_note_content.encode('utf-8'))
        path = tmp.name
    
    yield Path(path)
    
    # Clean up
    if os.path.exists(path):
        os.unlink(path)


def test_app_exists():
    """Test that the app exists."""
    assert app is not None
    assert hasattr(app, "callback")
    assert callable(app)


@patch("zk_core.index.ProcessPoolExecutor")
@patch("zk_core.index.scandir_recursive")
def test_cli_run_command(mock_scandir, mock_executor, temp_note_file):
    """Test the CLI run command."""
    # Set up mocks
    mock_scandir.return_value = [temp_note_file]
    
    # Set up process pool mock
    mock_context = MagicMock()
    mock_executor.return_value = mock_context
    mock_context.__enter__.return_value.map.return_value = [{"filename": "test.md"}]
    
    # Create temporary index file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_index = tmp.name
    
    try:
        # Run the CLI command
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open(tmp_index, 'w') as f:
                json.dump({"notes": {}}, f)
            
            # Mock config loading
            with patch("zk_core.index.load_config", return_value={
                "notes_dir": os.path.dirname(temp_note_file),
                "index_filename": tmp_index
            }):
                result = runner.invoke(app, ["run"])
        
        # Check the command ran successfully
        assert result.exit_code == 0
        assert "Index updated" in result.stdout
    
    finally:
        # Clean up
        if os.path.exists(tmp_index):
            os.unlink(tmp_index)
