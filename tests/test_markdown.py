#!/usr/bin/env python3
"""
Tests for markdown processing module.
"""

import re
from zk_core.markdown import (
    extract_frontmatter_and_body,
    extract_wikilinks_filtered,
    calculate_word_count,
    extract_citations
)


def test_extract_frontmatter_and_body():
    """Test extraction of frontmatter and content from markdown."""
    # Test with valid frontmatter
    content = """---
title: Test Note
tags: [test, markdown]
date: 2023-01-01
---

# Content starts here

This is the body text."""
    
    meta, body = extract_frontmatter_and_body(content)
    
    assert meta["title"] == "Test Note"
    assert meta["tags"] == ["test", "markdown"]
    assert meta["date"] == "2023-01-01"
    assert body.startswith("# Content starts here")
    assert "This is the body text." in body
    
    # Test without frontmatter
    content = "# Just content\nNo frontmatter here."
    meta, body = extract_frontmatter_and_body(content)
    
    assert meta == {}
    assert body == content
    
    # Test with empty frontmatter
    content = """---
---

Content after empty frontmatter."""
    meta, body = extract_frontmatter_and_body(content)
    
    assert meta == {}
    assert body == "\nContent after empty frontmatter."
    
    # Test with incomplete frontmatter (should be treated as content)
    content = """---
title: Incomplete
No closing delimiter

# Content"""
    meta, body = extract_frontmatter_and_body(content)
    
    assert meta == {}
    assert body == content


def test_extract_wikilinks_filtered():
    """Test filtered extraction of wikilinks from markdown content."""
    content = """This has [[link1]] and [[link2|with alias]].
    
    Also [[filtered_link]] that should be filtered out.
    
    Multiple links on one line: [[link3]] and [[link4]]."""
    
    # Test without filtering
    links = extract_wikilinks_filtered(content)
    assert set(links) == {"link1", "link2", "filtered_link", "link3", "link4"}
    
    # Test with filtering
    links = extract_wikilinks_filtered(content, ignore_pattern=r"filter")
    assert set(links) == {"link1", "link2", "link3", "link4"}
    assert "filtered_link" not in links
    
    # Test empty content
    assert extract_wikilinks_filtered("") == []
    
    # Test content with no links
    assert extract_wikilinks_filtered("No links here") == []


def test_calculate_word_count():
    """Test word count calculation in markdown content."""
    # Basic test
    assert calculate_word_count("One two three") == 3
    
    # Test with markdown formatting
    content = """# Heading
    
    This is **bold** and *italic* text with `code`.
    
    - List item 1
    - List item 2
    
    > Blockquote text here
    
    [[Wikilink]] and regular text."""
    
    assert calculate_word_count(content) == 19
    
    # Test with code blocks (should still count words inside)
    content = """Normal text.
    
    ```python
    def function():
        # This is code
        return True
    ```
    
    More text."""
    
    assert calculate_word_count(content) == 10
    
    # Test empty content
    assert calculate_word_count("") == 0


def test_extract_citations():
    """Test extraction of citations from markdown content."""
    content = """This paper @smith2020 discusses important findings.
    
    Multiple citations: @jones2019 and @brown2021 p. 42.
    
    Wikilinked citation: [[Book title|[@citation2022]]]
    
    This has no citations."""
    
    citations = extract_citations(content)
    assert set(citations) == {"smith2020", "jones2019", "brown2021", "citation2022"}
    
    # Test with no citations
    assert extract_citations("No citations here") == []
    
    # Test empty content
    assert extract_citations("") == []
    
    # Test deduplication
    content = "Repeated citations @same2020 and again @same2020."
    citations = extract_citations(content)
    assert citations == ["same2020"]
