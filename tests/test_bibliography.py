#!/usr/bin/env python3
"""
Tests for bibliography modules.
"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

import pytest

from zk_core.bibliography.builder import generate_citation_keys, generate_bibliography
from zk_core.bibliography.viewer import format_bibliography_data


@pytest.fixture
def sample_index():
    """Sample index data with citations."""
    return {
        "notes": {
            "note1.md": {
                "title": "Note 1",
                "path": "/path/to/note1.md",
                "citations": ["smith2020", "jones2019"]
            },
            "note2.md": {
                "title": "Note 2",
                "path": "/path/to/note2.md",
                "citations": ["smith2020", "brown2021"]
            }
        }
    }


@pytest.fixture
def sample_bibkeys():
    """Sample bibliography keys data."""
    return {
        "smith2020": {
            "title": "Smith's Research Paper",
            "author": "John Smith",
            "year": "2020",
            "journal": "Journal of Testing"
        },
        "jones2019": {
            "title": "Jones' Study",
            "author": "Alice Jones",
            "year": "2019",
            "journal": "Research Monthly"
        },
        "brown2021": {
            "title": "Brown's Analysis",
            "author": "Robert Brown",
            "year": "2021",
            "journal": "Science Today"
        }
    }


@patch("subprocess.run")
def test_generate_citation_keys(mock_run, sample_bibkeys):
    """Test generating citation keys from bibliography."""
    # Mock subprocess result
    mock_result = MagicMock()
    mock_result.stdout = json.dumps(sample_bibkeys)
    mock_run.return_value = mock_result
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_output:
        output_path = tmp_output.name
    
    try:
        # Call the function
        result = generate_citation_keys(
            getbibkeys_script="script.py",
            biblib_dir="/path/to/biblib",
            notes_dir="/path/to/notes",
            output_file=output_path
        )
        
        # Check that the script was called
        assert mock_run.called
        
        # Check that output was written
        assert os.path.exists(output_path)
        
        # Check content of output file
        with open(output_path, "r") as f:
            data = json.load(f)
        
        # Verify keys match
        assert set(data.keys()) == {"smith2020", "jones2019", "brown2021"}
        
    finally:
        # Clean up
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_generate_bibliography(sample_index, sample_bibkeys):
    """Test generating bibliography from index and bibkeys."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_index:
        json.dump(sample_index, tmp_index)
        index_path = tmp_index.name
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_bibkeys:
        json.dump(sample_bibkeys, tmp_bibkeys)
        bibkeys_path = tmp_bibkeys.name
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_output:
        output_path = tmp_output.name
    
    try:
        # Call the function
        generate_bibliography(
            index_file=index_path,
            bibkeys_file=bibkeys_path,
            output_paths=[output_path]
        )
        
        # Check that output was written
        assert os.path.exists(output_path)
        
        # Check content of output file
        with open(output_path, "r") as f:
            data = json.load(f)
        
        # Verify all citations from the index are included
        assert set(data.keys()) == {"smith2020", "jones2019", "brown2021"}
        
        # Verify each entry has the right data
        assert data["smith2020"]["title"] == "Smith's Research Paper"
        assert data["smith2020"]["usage_count"] == 2  # Used in 2 notes
        assert data["jones2019"]["usage_count"] == 1  # Used in 1 note
        
    finally:
        # Clean up
        for path in [index_path, bibkeys_path, output_path]:
            if os.path.exists(path):
                os.unlink(path)


def test_format_bibliography_data(sample_bibkeys):
    """Test formatting bibliography data for display."""
    # Add usage counts to the bibkeys
    for key in sample_bibkeys:
        sample_bibkeys[key]["usage_count"] = 1
    
    # Call the formatting function
    formatted = format_bibliography_data(sample_bibkeys, sort_by="author")
    
    # Check that all entries are included
    assert len(formatted) == 3
    
    # Verify sorting by author
    assert formatted[0].startswith("Alice Jones")  # A comes first
    assert formatted[1].startswith("John Smith")   # J comes second
    assert formatted[2].startswith("Robert Brown") # R comes third
    
    # Check that titles are included
    assert "Jones' Study" in formatted[0]
    assert "Smith's Research Paper" in formatted[1]
    assert "Brown's Analysis" in formatted[2]
