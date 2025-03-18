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
    DEFAULT_FILENAME_EXTENSION,
    DEFAULT_NOTES_DIR
)

logger = logging.getLogger(__name__)

# --- Data Serialization ---

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

def get_path_for_script(config: Dict[str, Any], section_key: str, script_key: str, default_path: str) -> str:
    """
    Get path for an external script from config with section-specific fallback.
    
    Args:
        config: Configuration dictionary
        section_key: Section name in config (e.g., "bibview")
        script_key: Script key name (e.g., "open_doc_script")
        default_path: Default path if not found in config
        
    Returns:
        Resolved path to the script
    """
    from zk_core.config import get_config_value, resolve_path
    
    # Try section-specific config first
    section = get_config_value(config, section_key, {})
    if isinstance(section, dict) and script_key in section:
        path = section[script_key]
    else:
        # Try global config
        path = get_config_value(config, f"{section_key}.{script_key}", default_path)
    
    return resolve_path(path)

def get_socket_path(config: Dict[str, Any], args: Any = None, section_key: Optional[str] = None) -> str:
    """
    Get socket path with consistent precedence:
    1. Command line args (if provided)
    2. Global config socket_path
    3. Section-specific config (if section_key provided)
    4. Environment variable
    5. Default constant value
    
    Args:
        config: Configuration dictionary
        args: Command line args (should have socket_path attribute)
        section_key: Optional section name for backward compatibility
        
    Returns:
        Resolved socket path
    """
    import os
    from zk_core.config import get_config_value, resolve_path
    from zk_core.constants import DEFAULT_NVIM_SOCKET
    
    # Command line argument has highest precedence
    if args and hasattr(args, 'socket_path') and args.socket_path:
        socket_path = args.socket_path
    else:
        # Try global config
        socket_path = get_config_value(config, "socket_path", None)
        
        # Try section-specific config if provided and global not found
        if not socket_path and section_key:
            socket_path = get_config_value(config, f"{section_key}.socket_path", None)
            
        # Fall back to environment variable or default
        if not socket_path:
            socket_path = os.getenv("NVIM_SOCKET", DEFAULT_NVIM_SOCKET)
    
    return resolve_path(socket_path)

def get_index_file_path(config: Dict[str, Any], notes_dir: Optional[str] = None, args: Any = None) -> str:
    """
    Get index file path with consistent precedence:
    1. Command line args (if provided)
    2. Configuration
    3. Default path based on notes_dir
    
    Args:
        config: Configuration dictionary
        notes_dir: Notes directory path (if already resolved)
        args: Command line args (should have index_file attribute)
        
    Returns:
        Resolved index file path
    """
    import os
    from zk_core.config import get_config_value, resolve_path, get_notes_dir
    from zk_core.constants import DEFAULT_INDEX_FILENAME
    
    # Command line argument has highest precedence
    if args and hasattr(args, 'index_file') and args.index_file:
        index_file = args.index_file
    else:
        # Try configuration
        index_file = get_config_value(config, "zk_index.index_file", DEFAULT_INDEX_FILENAME)
    
    # If notes_dir not provided, get it from config
    if notes_dir is None:
        notes_dir = get_notes_dir(config)
    
    # If index_file doesn't include a path, join it with notes_dir
    if not os.path.dirname(index_file):
        index_file = os.path.join(notes_dir, index_file)
    
    return resolve_path(index_file)

def scandir_recursive(
    root: str, 
    exclude_patterns: Optional[List[str]] = None, 
    quiet: bool = False,
    dir_mtimes: Optional[Dict[str, float]] = None,
    notes_dir: Optional[str] = None,
    skipped_dirs: Optional[List[str]] = None
) -> List[str]:
    """
    Recursively scan a directory, skipping entries that match any exclude pattern.
    NOTE: We no longer skip directories based on mtimes - this was causing issues
    with file change detection. We'll just be collecting all files for processing.
    
    Args:
        root: Root directory to scan
        exclude_patterns: List of glob patterns to exclude
        quiet: Whether to suppress debug logging
        dir_mtimes: Dictionary of directory paths to modification times from previous run (unused)
        notes_dir: Base notes directory for calculating relative paths in dir_mtimes (unused)
        skipped_dirs: Optional list to track which directories were skipped (unused)
    """
    import fnmatch
    
    exclude_patterns = exclude_patterns or []
    paths = []
    
    # We'll still track directories for future optimizations
    if skipped_dirs is None:
        skipped_dirs = []
    
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
                    paths.extend(scandir_recursive(
                        full_path, 
                        exclude_patterns, 
                        quiet,
                        dir_mtimes,
                        notes_dir,
                        skipped_dirs
                    ))
    except PermissionError as e:
        logger.warning(f"Permission error accessing directory: {root}. Skipping. Error: {e}")
    except OSError as e:
        logger.error(f"OS error while scanning directory: {root}. Skipping. Error: {e}")
    return paths

# --- Markdown Processing ---

# Import markdown functions for backward compatibility
from zk_core.markdown import (
    extract_frontmatter_and_body,
    extract_wikilinks_filtered,
    calculate_word_count,
    extract_citations,
    json_ready
)

# --- Command Execution ---

# Import run_command as alias for backward compatibility
from zk_core.commands import run_command

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