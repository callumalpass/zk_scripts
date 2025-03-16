"""Utility functions for ZK Core."""

import os
import sys
import json
import yaml
import logging
import datetime
import re
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path

from zk_core.constants import (
    WIKILINK_RE,
    INLINE_CITATION_RE,
    WIKILINKED_CITATION_RE,
    WIKILINK_ALL_RE,
    CITATION_ALIAS_RE
)

logger = logging.getLogger(__name__)

# --- Data Serialization ---

def json_ready(data: Any) -> Any:
    """Prepare data for JSON serialization, handling date objects."""
    if isinstance(data, datetime.date):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: json_ready(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_ready(item) for item in data]
    else:
        return data

def load_json_file(file_path: Path) -> Any:
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error reading JSON file '{file_path}': {e}")
        return None

def save_json_file(file_path: Path, data: Any, indent: int = 2) -> bool:
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except OSError as e:
        logger.error(f"Error writing JSON file '{file_path}': {e}")
        return False

# --- File System Utilities ---

def scandir_recursive(root: str, exclude_patterns: Optional[List[str]] = None, quiet: bool = False) -> List[str]:
    """Recursively scan a directory, skipping entries that contain any exclude pattern."""
    exclude_patterns = exclude_patterns or []
    paths = []
    if not quiet:
        logger.debug(f"Scanning directory: {root}")
    try:
        with os.scandir(root) as it:
            for entry in it:
                full_path = entry.path
                skip = False
                for pattern in exclude_patterns:
                    if pattern in full_path:
                        if not quiet:
                            logger.debug(f"Excluding {full_path} because it matches pattern '{pattern}'")
                        skip = True
                        break
                if skip:
                    continue
                if entry.is_file():
                    if not quiet:
                        logger.debug(f"Found file: {full_path}")
                    paths.append(full_path)
                elif entry.is_dir():
                    if not quiet:
                        logger.debug(f"Entering directory: {full_path}")
                    paths.extend(scandir_recursive(full_path, exclude_patterns, quiet))
    except PermissionError as e:
        logger.warning(f"Permission error accessing directory: {root}. Skipping. Error: {e}")
    except OSError as e:
        logger.error(f"OS error while scanning directory: {root}. Skipping. Error: {e}")
    return paths

# --- Markdown Processing ---

def extract_frontmatter_and_body(content: str) -> Tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and markdown body from a string."""
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
    """Extract wikilinks from a markdown body text, filtering out citations."""
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

def calculate_word_count(body: str) -> int:
    """Calculate the number of words in a given text."""
    return len(body.split())

def extract_citations(body: str) -> List[str]:
    """Extract all citation keys from a markdown body."""
    inline_citations = INLINE_CITATION_RE.findall(body)
    wikilink_citations = WIKILINKED_CITATION_RE.findall(body)
    return sorted(set(inline_citations) | set(wikilink_citations))