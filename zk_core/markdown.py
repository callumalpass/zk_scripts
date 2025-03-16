"""
Markdown processing utilities for ZK Core.

This module provides functions for working with markdown files:
- Extracting frontmatter and content
- Processing wikilinks
- Extracting citations
- Calculating statistics (word count, etc.)
"""

import re
import yaml
import logging
from typing import Dict, List, Tuple, Any, Optional, Set

from zk_core.constants import (
    WIKILINK_RE,
    INLINE_CITATION_RE,
    WIKILINKED_CITATION_RE,
    WIKILINK_ALL_RE,
    CITATION_ALIAS_RE
)

logger = logging.getLogger(__name__)


def json_ready(data: Any) -> Any:
    """
    Prepare data for JSON serialization, handling date objects.
    
    Args:
        data: Data to prepare
        
    Returns:
        Data ready for JSON serialization
    """
    import datetime
    if isinstance(data, datetime.date):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: json_ready(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_ready(item) for item in data]
    else:
        return data


def extract_frontmatter_and_body(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter and markdown body from content.
    
    Args:
        content: Markdown content with optional YAML frontmatter
        
    Returns:
        Tuple of (frontmatter dict, body text)
    """
    meta: Dict[str, Any] = {}
    body = content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            body = parts[2].strip()
            try:
                meta = yaml.safe_load(yaml_content) or {}
                if not isinstance(meta, dict):
                    logger.warning("YAML frontmatter did not parse to a dictionary. Ignoring frontmatter.")
                    meta = {}
                else:
                    meta = json_ready(meta)
            except yaml.YAMLError as e:
                logger.warning(f"YAML parsing error: {e}. Ignoring frontmatter.")
                meta = {}
    else:
        body = body.strip()
    return meta, body


def extract_wikilinks_filtered(body: str) -> List[str]:
    """
    Extract wikilinks from markdown body, filtering out citations.
    
    Args:
        body: Markdown content
        
    Returns:
        List of wikilink targets (deduplicated)
    """
    outgoing_links: List[str] = []
    for match in re.finditer(WIKILINK_ALL_RE, body):
        target = match.group(1)
        alias = match.group(2)
        if alias and CITATION_ALIAS_RE.match(alias.strip()):
            continue
        outgoing_links.append(target)
    seen: Set[str] = set()
    filtered = []
    for link in outgoing_links:
        if link not in seen:
            seen.add(link)
            filtered.append(link)
    return filtered


def extract_citations(body: str) -> List[str]:
    """
    Extract all citation keys from markdown content.
    
    Args:
        body: Markdown content
        
    Returns:
        List of unique citation keys
    """
    inline_citations = INLINE_CITATION_RE.findall(body)
    wikilink_citations = WIKILINKED_CITATION_RE.findall(body)
    return sorted(set(inline_citations) | set(wikilink_citations))


def calculate_word_count(body: str) -> int:
    """
    Calculate the number of words in text.
    
    Args:
        body: Text content
        
    Returns:
        Word count as integer
    """
    return len(body.split())


def extract_title(body: str) -> Optional[str]:
    """
    Extract title from markdown content (first heading).
    
    Args:
        body: Markdown content
        
    Returns:
        Title text or None if no heading found
    """
    # Look for Markdown headings at the start of the document
    heading_match = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
    if heading_match:
        return heading_match.group(1).strip()
    return None


def format_frontmatter(meta: Dict[str, Any]) -> str:
    """
    Format dictionary as YAML frontmatter.
    
    Args:
        meta: Dictionary to format
        
    Returns:
        YAML frontmatter string with --- delimiters
    """
    if not meta:
        return ""
    try:
        yaml_str = yaml.dump(meta, allow_unicode=True, sort_keys=False)
        return f"---\n{yaml_str}---\n\n"
    except Exception as e:
        logger.error(f"Error formatting frontmatter: {e}")
        return ""


def combine_frontmatter_and_body(meta: Dict[str, Any], body: str) -> str:
    """
    Combine frontmatter and body into a markdown document.
    
    Args:
        meta: Frontmatter dictionary
        body: Markdown content
        
    Returns:
        Complete markdown document with frontmatter
    """
    frontmatter = format_frontmatter(meta)
    return f"{frontmatter}{body}"