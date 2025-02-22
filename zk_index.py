#!/usr/bin/env python3
"""
Incremental Fast ZK indexer with Backlinks

• Uses os.scandir recursively to list markdown files quickly.
• Skips whole directories that match any exclude pattern.
• Reads each file at once and splits on the YAML front matter separator.
• Uses a compiled regex for wikilink extraction.
• Parallels file processing using ProcessPoolExecutor.
• Converts YAML dates to ISO strings for JSON output.
• Incremental indexing by checking file modification times.
• Handles deleted notes by removing them from the index.
• Indexes the body content, file size, and word count.
• Processes backlinks: after gathering outgoing links from each note,
  computes for each note which other notes refer to it.

Command line arguments:
  --notes-dir <path>     : Override notes directory from config or default.
  --index-file <path>    : Override index file path from config or default.
  --config-file <path>   : Specify a different config file path.
  --full-reindex, -f     : Force full reindex, ignoring mod times and index state.
  --exclude-patterns <str>: Override exclude patterns (space-separated, e.g., "templates/ .zk/").
  --verbose, -v          : Increase verbosity of output.
  --workers <int>        : Number of worker processes for indexing.
"""

import os
import re
import sys
import json
import yaml
import argparse
import logging
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional

CONFIG_FILE = os.path.expanduser("~/.config/zk_scripts/config.yaml")
DEFAULT_WORKERS = 8  # Default number of worker processes

# Precompiled regex for wikilinks – matches [[target]] optionally with an alias.
WIKILINK_RE = re.compile(r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging


def resolve_config_value(config: Dict[str, Any], key_path: str, default_value: Any) -> Any:
    """Resolves a configuration value from a dotted key path."""
    keys = key_path.split('.')
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value
    return current


def load_config(config_file: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_file):
        logger.warning(f"Config file '{config_file}' not found. Using defaults.")
        return {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError as e:
        logger.warning(f"YAML error in '{config_file}': {e}. Using defaults.")
        return {}


def json_ready(data: Any) -> Any:
    """Recursively converts datetime.date objects to ISO strings for JSON serialization."""
    if isinstance(data, date):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: json_ready(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_ready(item) for item in data]
    else:
        return data


def scandir_recursive(root: str, exclude_patterns: Optional[List[str]] = None) -> List[str]:
    """Recursively finds files under a directory, skipping paths that match any exclude patterns."""
    exclude_patterns = exclude_patterns or []
    paths = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                if any(pattern in entry.path for pattern in exclude_patterns):
                    continue
                if entry.is_file():
                    paths.append(entry.path)
                elif entry.is_dir():
                    paths.extend(scandir_recursive(entry.path, exclude_patterns))
    except PermissionError as e:
        logger.warning(f"Permission error accessing directory: {root}. Skipping. Error: {e}")
    except OSError as e:
        logger.error(f"OS error while scanning directory: {root}. Skipping. Error: {e}")
    return paths


def extract_frontmatter_and_body(content: str) -> Tuple[Dict[str, Any], str]:
    """Extracts YAML front matter and body from the markdown content."""
    meta: Dict[str, Any] = {}
    body = content
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            body = parts[2].strip()  # also strip the body
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


def extract_wikilinks(body: str) -> List[str]:
    """Extracts unique wikilinks from the markdown body."""
    outgoing_links: List[str] = []
    seen_links = set()
    for link in WIKILINK_RE.findall(body):
        link = link.replace("\\", "")
        if link not in seen_links:
            seen_links.add(link)
            outgoing_links.append(link)
    return outgoing_links


def calculate_word_count(body: str) -> int:
    """Calculates the number of words in a given string."""
    return len(body.split())


def process_markdown_file(filepath: str, fd_exclude_patterns: List[str], notes_dir: str) -> Optional[Dict[str, Any]]:
    """Processes a single markdown file: extracts metadata, body, word count, outgoing links and file info."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None

    rel_path = os.path.relpath(filepath, notes_dir)
    name_noext = os.path.splitext(rel_path)[0]

    meta, body = extract_frontmatter_and_body(content)
    outgoing_links = extract_wikilinks(body)
    word_count = calculate_word_count(body)
    file_size = os.path.getsize(filepath)

    result: Dict[str, Any] = {
        "filename": name_noext,
        "outgoing_links": outgoing_links,
        "body": body,
        "word_count": word_count,
        "file_size": file_size,
    }
    result.update(meta)
    return result


def load_index_state(state_file: str) -> Dict[str, float]:
    """Loads the previous index state from a JSON file (mapping filepath to modification timestamp)."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding index state from '{state_file}': {e}. Starting fresh.")
        except FileNotFoundError:
            logger.info(f"Index state file '{state_file}' not found. Starting fresh.")
        except OSError as e:
            logger.error(f"OS error reading index state file '{state_file}': {e}. Starting fresh.")
        return {}
    return {}


def save_index_state(state_file: str, index_state: Dict[str, float]) -> None:
    """Saves the current index state (filepath to modification timestamp) to a JSON file."""
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(index_state, f, indent=2)
    except OSError as e:
        logger.error(f"Error writing index state file '{state_file}': {e}")


def load_existing_index(index_file: str) -> Dict[str, Dict[str, Any]]:
    """Loads the existing index from the index file (mapping note names to their info)."""
    existing_index: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                existing_index = {item['filename']: item for item in json.load(f)}
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
            logger.warning(f"Error loading existing index from '{index_file}': {e}. Starting with empty index.")
            existing_index = {}
    return existing_index


def write_updated_index(index_file: str, updated_index_data: List[Dict[str, Any]]) -> None:
    """Writes the updated index data to the index file and exits on failure."""
    try:
        with open(index_file, 'w', encoding='utf-8') as outf:
            json.dump(updated_index_data, outf, indent=2)
        logger.info(f"Index file updated at: {index_file}")
    except OSError as e:
        logger.error(f"Error writing index file '{index_file}': {e}")
        sys.exit(1)


def remove_index_state_file(state_file: str, verbose: bool) -> None:
    """Removes the state file that tracks file modification times."""
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
            if verbose:
                logger.info(f"Index state file removed: {state_file}")
        except OSError as e:
            logger.warning(f"Could not remove index state file '{state_file}': {e}")
    elif verbose:
        logger.info(f"Index state file not found, skipping removal: {state_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental Fast ZK indexer with Backlinks")
    parser.add_argument("--notes-dir", help="Override notes directory")
    parser.add_argument("--index-file", help="Override index file path")
    parser.add_argument("--config-file", default=CONFIG_FILE, help="Specify config file path")
    parser.add_argument("--full-reindex", "-f", action="store_true", help="Force full reindex, removing index state.")
    parser.add_argument("--exclude-patterns", help="Override exclude patterns (space-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase verbosity of output.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of worker processes (default: {DEFAULT_WORKERS})")

    args = parser.parse_args()

    config = load_config(args.config_file)

    # Defaults – you may want to further parameterize these in your config
    notes_dir_default = "/home/calluma/Dropbox/notes"
    index_file_default = os.path.join(notes_dir_default, "index.json")
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir = args.notes_dir or resolve_config_value(config, "notes_dir", notes_dir_default)
    index_file = args.index_file or resolve_config_value(config, "zk_index.index_file", index_file_default)

    # Get exclusion patterns from the args/config. Expecting something like "-E templates/ -E .zk/"
    fd_exclude_patterns_str = args.exclude_patterns or resolve_config_value(config, "zk_index.fd_exclude_patterns", "-E templates/ -E .zk/")
    fd_exclude_patterns = re.findall(r'-E\s+([^\s]+)', fd_exclude_patterns_str)
    if not fd_exclude_patterns:
        fd_exclude_patterns = fd_exclude_patterns_default

    if not notes_dir:
        logger.error("NOTES_DIR not defined. Please specify --notes-dir or configure it.")
        sys.exit(1)
    if not os.path.isdir(notes_dir):
        logger.error(f"NOTES_DIR '{notes_dir}' is not a valid directory.")
        sys.exit(1)

    state_file = index_file.replace(".json", "_state.json")

    if args.full_reindex:
        remove_index_state_file(state_file, args.verbose)

    previous_index_state: Dict[str, float] = {}
    if not args.full_reindex:
        previous_index_state = load_index_state(state_file)
    current_index_state: Dict[str, float] = {}

    existing_index = load_existing_index(index_file)

    # Get all markdown files under notes_dir (excluding dirs based on fd_exclude_patterns)
    if args.verbose:
        logger.info(f"Scanning files")
    markdown_files = [fp for fp in scandir_recursive(notes_dir, exclude_patterns=fd_exclude_patterns)
                      if fp.lower().endswith(".md")]
    current_filepaths = set(markdown_files)
    previous_filepaths = set(previous_index_state.keys())

    # Handle deleted notes: if a note file existed before but does not anymore, remove it from the index.
    deleted_files = [prev_fp for prev_fp in previous_filepaths if prev_fp not in current_filepaths]
    for deleted in deleted_files:
        key = os.path.splitext(os.path.relpath(deleted, notes_dir))[0]
        if key in existing_index:
            del existing_index[key]
        if args.verbose:
            logger.info(f"Note deleted: {deleted}")

    # Determine files to reprocess: either forced by full reindex or whose modification time is newer.
    files_to_process = []
    for filepath in markdown_files:
        try:
            mod_time = os.path.getmtime(filepath)
            current_index_state[filepath] = mod_time
            if args.full_reindex or filepath not in previous_index_state or previous_index_state.get(filepath, 0) < mod_time:
                files_to_process.append(filepath)
        except OSError as e:
            logger.warning(f"File disappeared or error accessing: {filepath}. Error: {e}")


    if args.verbose:
        logger.info(f"Files to process: {len(files_to_process)}")

    # --- Progress bar for file processing ---
    with tqdm(total=len(files_to_process), desc="Processing files") as file_pbar:
        # Process updated markdown files in parallel.
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_markdown_file, fp, fd_exclude_patterns, notes_dir): fp for fp in files_to_process}
            for future in as_completed(futures):
                filepath = futures[future]
                try:
                    result = future.result()
                    if result:
                        # Update (or add) the note, keyed by its relative filename (without extension).
                        existing_index[result['filename']] = result
                except Exception as e:
                    logger.error(f"Error processing file {filepath} in worker process: {e}")
                file_pbar.update(1) # Update file processing progress bar

    # --- Process backlinks ---
    # Progress bar for backlink calculation
    with tqdm(total=len(existing_index), desc="Calculating backlinks") as backlink_pbar:
        backlink_map: Dict[str, set] = {}
        for note in existing_index.values():
            for target in note.get("outgoing_links", []):
                # Only process targets that exist as a note in the index.
                if target in existing_index:
                    backlink_map.setdefault(target, set()).add(note["filename"])
            backlink_pbar.update(1) # Update backlink progress bar

        # Now update each note with a list of backlinks.
        for key, note in existing_index.items():
            backlinks = sorted(list(backlink_map.get(key, [])))
            note["backlinks"] = backlinks


    updated_index_data = list(existing_index.values())
    write_updated_index(index_file, updated_index_data)
    save_index_state(state_file, current_index_state)



if __name__ == "__main__":
    main()

