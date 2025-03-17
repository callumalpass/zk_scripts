#!/usr/bin/env python3
"""
Tests for config module.
"""

import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_CONFIG_PATH


def test_load_config_with_existing_file():
    """Test loading config from an existing file."""
    config_data = """
    notes_dir: ~/notes
    index_filename: index.json
    log_level: INFO
    """
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
        tmp.write(config_data)
        tmp_path = tmp.name
    
    try:
        # Load the config from the temp file
        config = load_config(tmp_path)
        
        # Check values were loaded correctly
        assert config["notes_dir"] == "~/notes"
        assert config["index_filename"] == "index.json"
        assert config["log_level"] == "INFO"
    finally:
        # Clean up
        os.unlink(tmp_path)


def test_load_config_create_default():
    """Test creating a default config file if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a path to a non-existent config file
        tmp_config = os.path.join(tmpdir, "config.yaml")
        
        # Load config, which should create the file
        config = load_config(tmp_config)
        
        # Check file was created
        assert os.path.exists(tmp_config)
        
        # Check it has some default values
        with open(tmp_config, "r") as f:
            file_content = f.read()
        assert "notes_dir:" in file_content
        assert "# Default configuration" in file_content


def test_get_config_value():
    """Test retrieval of config values with defaults."""
    config = {
        "section": {
            "key": "value"
        },
        "top_level": "top value"
    }
    
    # Test getting existing nested value
    assert get_config_value(config, "section.key") == "value"
    
    # Test getting existing top-level value
    assert get_config_value(config, "top_level") == "top value"
    
    # Test getting non-existent value with default
    assert get_config_value(config, "missing", "default") == "default"
    
    # Test getting non-existent nested value with default
    assert get_config_value(config, "section.missing", "default") == "default"
    
    # Test getting from non-existent section with default
    assert get_config_value(config, "missing.key", "default") == "default"


def test_resolve_path():
    """Test path resolution functionality."""
    # Test with tilde expansion
    home = os.path.expanduser("~")
    assert resolve_path("~/test") == os.path.join(home, "test")
    
    # Test with absolute path
    assert resolve_path("/absolute/path") == "/absolute/path"
    
    # Test with relative path (should remain relative)
    assert resolve_path("relative/path") == "relative/path"
    
    # Test with empty path
    assert resolve_path("") == ""
    
    # Test with None
    assert resolve_path(None) is None
