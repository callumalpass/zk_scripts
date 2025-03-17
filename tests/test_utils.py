#!/usr/bin/env python3
"""
Tests for utilities module.
"""

import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from zk_core.utils import (
    json_ready, scandir_recursive, extract_frontmatter_and_body,
    extract_wikilinks_filtered, calculate_word_count, extract_citations,
    load_json_file, save_json_file
)


def test_json_ready():
    """Test the json_ready function for serialization."""
    # Test with Path object
    path = Path("/tmp/test.txt")
    assert json_ready(path) == "/tmp/test.txt"
    
    # Test with set
    test_set = {1, 2, 3}
    assert set(json_ready(test_set)) == {1, 2, 3}
    
    # Test with normal values
    assert json_ready(42) == 42
    assert json_ready("string") == "string"
    assert json_ready([1, 2, 3]) == [1, 2, 3]
    assert json_ready({"a": 1}) == {"a": 1}


def test_scandir_recursive():
    """Test recursive directory scanning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test directory structure
        os.makedirs(os.path.join(tmpdir, "subdir1"))
        os.makedirs(os.path.join(tmpdir, "subdir2"))
        
        # Create some test files
        open(os.path.join(tmpdir, "file1.md"), "w").close()
        open(os.path.join(tmpdir, "file2.txt"), "w").close()
        open(os.path.join(tmpdir, "subdir1", "file3.md"), "w").close()
        
        # Test scanning for markdown files
        md_files = list(scandir_recursive(tmpdir, ["*.md"]))
        assert len(md_files) == 2
        assert any(f.name == "file1.md" for f in md_files)
        assert any(f.name == "file3.md" for f in md_files)
        
        # Test scanning for all files
        all_files = list(scandir_recursive(tmpdir, ["*"]))
        assert len(all_files) == 3


def test_extract_frontmatter_and_body():
    """Test frontmatter and body extraction from markdown."""
    # Test with frontmatter
    markdown = """---
title: Test
tags: [one, two]
---

# Content
This is the body."""
    
    meta, body = extract_frontmatter_and_body(markdown)
    assert meta["title"] == "Test"
    assert meta["tags"] == ["one", "two"]
    assert "# Content" in body
    assert "This is the body" in body
    
    # Test without frontmatter
    markdown = "# No Frontmatter\nJust content."
    meta, body = extract_frontmatter_and_body(markdown)
    assert meta == {}
    assert body == markdown


def test_extract_wikilinks_filtered():
    """Test extracting filtered wikilinks from text."""
    text = """This has [[regular link]] and [[filtered|link with alias]].
    Also [[this one|should be filtered]] but not [[keep this]]."""
    
    # Test without ignore pattern
    links = extract_wikilinks_filtered(text)
    assert "regular link" in links
    assert "filtered" in links
    assert "this one" in links
    assert "keep this" in links
    assert len(links) == 4
    
    # Test with ignore pattern
    links = extract_wikilinks_filtered(text, ignore_pattern=r"filter")
    assert "regular link" in links
    assert "filtered" not in links
    assert "this one" not in links
    assert "keep this" in links
    assert len(links) == 2


def test_calculate_word_count():
    """Test word count calculation."""
    # Simple case
    assert calculate_word_count("This is four words.") == 4
    
    # With markdown formatting
    text = """# Heading
    
    This is **bold** and *italic* formatting.
    
    - List item 1
    - List item 2
    
    [[Wikilink]] and normal words."""
    assert calculate_word_count(text) == 15


def test_extract_citations():
    """Test citation extraction from text."""
    text = """This text cites @smith2020 and also @jones2019 p. 42.
    It also has a [[link|[@reference2021]]] citation.
    And another @citation2022."""
    
    citations = extract_citations(text)
    assert "smith2020" in citations
    assert "jones2019" in citations
    assert "reference2021" in citations
    assert "citation2022" in citations
    assert len(citations) == 4


def test_load_save_json_file():
    """Test loading and saving JSON files."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Test saving
        test_data = {"key": "value", "list": [1, 2, 3], "nested": {"a": 1}}
        save_json_file(tmp_path, test_data)
        
        # Verify file exists and contains correct data
        assert os.path.exists(tmp_path)
        with open(tmp_path, "r") as f:
            file_content = json.load(f)
        assert file_content == test_data
        
        # Test loading
        loaded_data = load_json_file(tmp_path)
        assert loaded_data == test_data
        
        # Test loading non-existent file
        non_existent = "/tmp/non_existent_file_12345.json"
        assert load_json_file(non_existent) == {}
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
