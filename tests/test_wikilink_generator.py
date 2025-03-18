"""
Tests for the wikilink generator module.
"""

import pytest
from zk_core.wikilink_generator import WikilinkConfig, create_wikilink_from_selection


def test_wikilink_config_init():
    """Test that WikilinkConfig initializes with correct defaults."""
    config = WikilinkConfig(name="test")
    
    assert config.name == "test"
    assert config.filter_tags == []
    assert config.search_fields == ["filename"]
    assert config.display_fields == ["filename"]
    assert config.alias_fields == ["aliases", "title"]
    assert "command" in config.preview_config
    assert "delimiter" in config.fzf_config


def test_wikilink_config_custom_values():
    """Test that WikilinkConfig accepts custom values."""
    config = WikilinkConfig(
        name="custom",
        filter_tags=["book"],
        search_fields=["title", "author"],
        display_fields=["author", "title"],
        alias_fields=["title"],
        preview_config={"command": "cat"},
        fzf_config={"delimiter": ","}
    )
    
    assert config.name == "custom"
    assert config.filter_tags == ["book"]
    assert config.search_fields == ["title", "author"]
    assert config.display_fields == ["author", "title"]
    assert config.alias_fields == ["title"]
    assert config.preview_config["command"] == "cat"
    assert config.fzf_config["delimiter"] == ","


def test_create_wikilink_from_selection_fallback():
    """Test that create_wikilink_from_selection returns a basic wikilink when fields are missing."""
    config = WikilinkConfig(name="test")
    wikilink = create_wikilink_from_selection("myfile", config, "/notes", debug=False)
    
    # Since we're mocking and not actually running the query command,
    # it should fall back to a basic wikilink
    assert wikilink == "[[myfile]]"