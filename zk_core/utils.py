"""Utility functions for ZK Core."""

import os
import sys
import json
import yaml
import logging
import datetime
import re
import random
import string
import subprocess
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path

from zk_core.constants import (
    WIKILINK_RE,
    INLINE_CITATION_RE,
    WIKILINKED_CITATION_RE,
    WIKILINK_ALL_RE,
    CITATION_ALIAS_RE,
    DEFAULT_FILENAME_FORMAT,
    DEFAULT_FILENAME_EXTENSION
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
    """Recursively scan a directory, skipping entries that match any exclude pattern."""
    import fnmatch
    
    exclude_patterns = exclude_patterns or []
    paths = []
    if not quiet:
        logger.debug(f"Scanning directory: {root}")
    try:
        with os.scandir(root) as it:
            for entry in it:
                full_path = entry.path
                relative_path = os.path.relpath(full_path, root)
                skip = False
                for pattern in exclude_patterns:
                    # Use proper pattern matching instead of simple substring check
                    if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(os.path.basename(full_path), pattern):
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

# --- Command Execution ---

def run_command(cmd: List[str], input_data: Optional[str] = None, check: bool = False) -> Tuple[int, str, str]:
    """
    Run a command and return the return code, stdout, and stderr.
    
    Args:
        cmd: Command to run as a list of strings
        input_data: Optional string to pass as stdin
        check: Whether to raise an exception on non-zero return code
        
    Returns:
        Tuple containing (return_code, stdout, stderr)
    """
    try:
        if input_data is not None:
            proc = subprocess.run(
                cmd, 
                input=input_data,
                text=True,
                capture_output=True,
                check=check
            )
        else:
            proc = subprocess.run(
                cmd,
                text=True, 
                capture_output=True,
                check=check
            )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(cmd)}")
        return e.returncode, e.stdout or "", e.stderr or ""
    except Exception as e:
        logger.error(f"Error running command {' '.join(cmd)}: {e}")
        return 1, "", str(e)

# --- Filename Generation ---

def generate_random_string(length: int) -> str:
    """Generate a random string of lowercase letters."""
    return "".join(random.choices(string.ascii_lowercase, k=length))

def generate_filename(format_str: Optional[str] = None, extension: Optional[str] = None) -> str:
    """
    Generate a filename based on the provided format string.
    
    The format string supports:
    - All strftime format codes (https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
    - {random:N} where N is the number of random lowercase letters to generate
    
    Args:
        format_str: The format string to use (from config or default)
        extension: The file extension to use (from config or default)
        
    Returns:
        A generated filename with extension
    """
    now = datetime.datetime.now()
    
    # Use defaults if not provided
    if format_str is None:
        format_str = DEFAULT_FILENAME_FORMAT
    if extension is None:
        extension = DEFAULT_FILENAME_EXTENSION
    
    # Make sure extension starts with a dot
    if not extension.startswith('.'):
        extension = '.' + extension
    
    # First apply datetime formatting
    try:
        filename = now.strftime(format_str)
    except ValueError as e:
        logger.warning(f"Error in strftime format: {e}. Using default format.")
        filename = now.strftime(DEFAULT_FILENAME_FORMAT)
    
    # Then handle {random:N} placeholder
    random_pattern = re.compile(r'\{random:(\d+)\}')
    while True:
        match = random_pattern.search(filename)
        if not match:
            break
        random_length = int(match.group(1))
        random_str = generate_random_string(random_length)
        filename = filename.replace(match.group(0), random_str, 1)
    
    # Add extension
    return filename + extension