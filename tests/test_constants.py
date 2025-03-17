#!/usr/bin/env python3
"""
Tests for constants module.
"""

import re
from zk_core.constants import (
    WIKILINK_RE, INLINE_CITATION_RE, WIKILINKED_CITATION_RE, WIKILINK_ALL_RE,
    CITATION_ALIAS_RE, MARKDOWN_EXTENSIONS, DEFAULT_CONFIG_PATH,
    DEFAULT_NOTES_DIR, DEFAULT_INDEX_FILENAME, DEFAULT_LOG_LEVEL,
    DEFAULT_NVIM_SOCKET, DEFAULT_FILENAME_FORMAT,
    DEFAULT_FILENAME_EXTENSION, DEFAULT_NUM_WORKERS, MAX_CHUNK_SIZE
)


def test_regex_patterns():
    """Test regular expression patterns."""
    # Test WIKILINK_RE
    assert WIKILINK_RE.search("This is a [[link]]").group(1) == "link"
    assert WIKILINK_RE.search("This is a [[link|with alias]]").group(1) == "link"
    assert WIKILINK_RE.search("No link here") is None

    # Test INLINE_CITATION_RE
    assert INLINE_CITATION_RE.search("See @smith2020").group(1) == "smith2020"
    assert INLINE_CITATION_RE.search("See @smith2020 p. 42").group(1) == "smith2020"
    assert INLINE_CITATION_RE.search("No citation") is None

    # Test WIKILINKED_CITATION_RE
    assert WIKILINKED_CITATION_RE.search("[[Title|[@smith2020]]").group(1) == "smith2020"
    assert WIKILINKED_CITATION_RE.search("Normal text") is None

    # Test WIKILINK_ALL_RE
    match = WIKILINK_ALL_RE.search("[[link|alias]]")
    assert match.group(1) == "link"
    assert match.group(2) == "alias"
    
    match = WIKILINK_ALL_RE.search("[[link]]")
    assert match.group(1) == "link"
    assert match.group(2) is None

    # Test CITATION_ALIAS_RE
    assert CITATION_ALIAS_RE.search("[@smith2020]").group(1) == "smith2020"
    assert CITATION_ALIAS_RE.search("Not a citation") is None


def test_file_extensions():
    """Test file extension constants."""
    assert ".md" in MARKDOWN_EXTENSIONS
    assert ".markdown" in MARKDOWN_EXTENSIONS
    assert len(MARKDOWN_EXTENSIONS) >= 2


def test_default_values():
    """Test default configuration values."""
    assert "~/.config" in DEFAULT_CONFIG_PATH
    assert "~/" in DEFAULT_NOTES_DIR
    assert ".json" in DEFAULT_INDEX_FILENAME
    assert DEFAULT_LOG_LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert "/" in DEFAULT_NVIM_SOCKET
    assert "%" in DEFAULT_FILENAME_FORMAT
    assert "." in DEFAULT_FILENAME_EXTENSION
    assert isinstance(DEFAULT_NUM_WORKERS, int)
    assert DEFAULT_NUM_WORKERS > 0
    assert isinstance(MAX_CHUNK_SIZE, int)
    assert MAX_CHUNK_SIZE > 0
