#!/usr/bin/env python3
"""
Incremental Fast ZK indexer

• Uses os.scandir recursively to list markdown files quickly.
• Skips whole directories (and their files) that match any exclude pattern.
• Reads each file in one go and splits on the YAML front matter separator.
• Uses a precompiled regex for wikilink extraction.
• Parallels file processing using ProcessPoolExecutor.
• Converts any YAML dates to ISO strings for JSON output.
• Incremental indexing by checking file modification times.
• Handles deleted notes by removing them from the index.

Command line arguments:
  --notes-dir <path>     : Override notes directory from config or default.
  --index-file <path>    : Override index file path from config or default.
  --config-file <path>   : Specify a different config file path.
  --full-reindex, -f     : Force a full reindex, ignoring modification times and removing index state.
  --exclude-patterns <str>: Override exclude patterns from config or default (space-separated, e.g., "templates/ .zk/").
  --verbose, -v          : Increase verbosity of output.
"""

import os
import re
import sys
import json
import yaml
import argparse
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

CONFIG_FILE = os.path.expanduser("~/.config/zk_scripts/config.yaml")

# Precompile wikilink regex (only matching [[target]] without caring about alias)
WIKILINK_RE = re.compile(r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]')

def resolve_config_value(config, key_path, default_value):
    keys = key_path.split('.')
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default_value
    return current

def load_config(config_file):
    if not os.path.exists(config_file):
        print(f"Warning: Config file '{config_file}' not found. Using defaults.", file=sys.stderr)
        return {}
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError as e:
        print(f"Warning: YAML error in '{config_file}': {e}. Using defaults.", file=sys.stderr)
        return {}

def json_ready(data):
    """Convert datetime.date objects to ISO string recursively."""
    if isinstance(data, date):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: json_ready(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_ready(item) for item in data]
    else:
        return data

def scandir_recursive(root, exclude_patterns=None):
    """
    Recursively find files under root using os.scandir.
    If exclude_patterns is provided, then skip directories or files whose paths contain any of these substrings.
    """
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
    except PermissionError:
        pass
    return paths

def process_markdown_file(filepath, fd_exclude_patterns, notes_dir):
    """
    Process one markdown file:
      - Skips file if its path contains any fd_exclude_patterns (defensive check).
      - Reads the entire file.
      - If file begins with '---', then treat the next block as YAML front matter.
      - Extract wikilinks via precompiled regex.
      - Return a dict with a unique identifier (the relative (to notes_dir) filename without extension),
        outgoing_links, and any YAML front matter (converted with json_ready).
    """
    # Defensive check (should have been filtered during scanning)
    for pat in fd_exclude_patterns:
        if pat in filepath:
            return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}", file=sys.stderr)
        return None

    # Use the relative path (without extension) as the unique identifier.
    rel_path = os.path.relpath(filepath, notes_dir)
    name_noext = os.path.splitext(rel_path)[0]

    if content.startswith('---'):
        # Split only into three parts (avoid splitting the rest of the file)
        parts = content.split('---', 2)
        if len(parts) >= 3:
            yaml_content = parts[1].strip()
            body = parts[2]
            try:
                meta = yaml.safe_load(yaml_content) or {}
                if isinstance(meta, dict):
                    meta = json_ready(meta)
                else:
                    meta = {}
            except yaml.YAMLError as e:
                print(f"YAML error in file {filepath}: {e}", file=sys.stderr)
                meta = {}
        else:
            body = content
            meta = {}
    else:
        body = content
        meta = {}

    outgoing_links = []
    seen = set()
    for link in WIKILINK_RE.findall(body):
        link = link.replace("\\", "")
        if link not in seen:
            seen.add(link)
            outgoing_links.append(link)

    result = {"filename": name_noext, "outgoing_links": outgoing_links}
    result.update(meta)
    return result

def load_index_state(state_file):
    """Load the previous index state from a JSON file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Error loading index state from '{state_file}': {e}. Starting fresh.", file=sys.stderr)
            return {}
    return {}

def save_index_state(state_file, index_state):
    """Save the current index state to a JSON file."""
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(index_state, f, indent=2)
    except Exception as e:
        print(f"Error writing index state file '{state_file}': {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Incremental Fast ZK indexer")
    parser.add_argument("--notes-dir", help="Override notes directory")
    parser.add_argument("--index-file", help="Override index file path")
    parser.add_argument("--config-file", default=CONFIG_FILE, help="Specify config file path")
    parser.add_argument("--full-reindex", "-f", action="store_true", help="Force full reindex, removing index state.")
    parser.add_argument("--exclude-patterns", help="Override exclude patterns (space-separated)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase verbosity")

    args = parser.parse_args()

    config = load_config(args.config_file)

    # Default values if not set in config or command line.
    notes_dir_default = "/home/calluma/Dropbox/notes"  # Adjust as needed
    index_file_default = os.path.join(notes_dir_default, "index.json")
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir = args.notes_dir or resolve_config_value(config, "notes_dir", notes_dir_default)
    index_file = args.index_file or resolve_config_value(config, "zk_index.index_file", index_file_default)

    fd_exclude_patterns_str = args.exclude_patterns or resolve_config_value(config, "zk_index.fd_exclude_patterns", "-E templates/ -E .zk/")
    fd_exclude_patterns = re.findall(r'-E\s+([^\s]+)', fd_exclude_patterns_str)
    if not fd_exclude_patterns:
        fd_exclude_patterns = fd_exclude_patterns_default

    if not notes_dir:
        print("Error: NOTES_DIR not defined. Please specify --notes-dir or configure it.", file=sys.stderr)
        sys.exit(1)

    state_file = index_file.replace(".json", "_state.json")

    if args.full_reindex:
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                if args.verbose:
                    print(f"Index state file removed: {state_file}")
            except OSError as e:
                print(f"Warning: Could not remove index state file '{state_file}': {e}", file=sys.stderr)
        else:
            if args.verbose:
                print(f"Index state file not found, skipping removal: {state_file}")

    previous_index_state = {}
    if not args.full_reindex:  # Only load state if not full reindex
        previous_index_state = load_index_state(state_file)
    current_index_state = {}  # To be populated and saved at the end

    # Load existing full index (filename-based) if available.
    existing_index = {}
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                existing_index = {item['filename']: item for item in json.load(f)}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Error loading existing index from '{index_file}': {e}. Starting fresh.", file=sys.stderr)
            existing_index = {}

    # Locate all markdown files, filtering out those in excluded directories.
    markdown_files = [fp for fp in scandir_recursive(notes_dir, exclude_patterns=fd_exclude_patterns)
                      if fp.lower().endswith(".md")]
    current_filepaths = set(markdown_files)
    previous_filepaths = set(previous_index_state.keys())

    # Identify files that have been deleted since last run.
    # (Compute the same key as in process_markdown_file: the relative file path without extension.)
    deleted_files = []
    for prev_fp in previous_filepaths:
        # Reconstruct an absolute file path from previous state if possible.
        # Since our state keys are full filepaths, we simply check if they are still present.
        if prev_fp not in current_filepaths:
            deleted_files.append(prev_fp)
    for deleted in deleted_files:
        # Compute the key as relative file path without extension.
        key = os.path.splitext(os.path.relpath(deleted, notes_dir))[0]
        if key in existing_index:
            del existing_index[key]
        if args.verbose:
            print(f"Note deleted: {deleted}")

    # Decide which files need processing, comparing modification times (unless full reindex).
    files_to_process = []
    for filepath in markdown_files:
        try:
            mod_time = os.path.getmtime(filepath)
            current_index_state[filepath] = mod_time  # update mod time in current state
            if args.full_reindex or filepath not in previous_index_state or previous_index_state.get(filepath, 0) < mod_time:
                files_to_process.append(filepath)
        except OSError:
            print(f"Warning: File disappeared: {filepath}", file=sys.stderr)

    if args.verbose:
        print(f"Files to process: {len(files_to_process)}")

    # Process files in parallel.
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Pass notes_dir to allow relative path computation.
        futures = {executor.submit(process_markdown_file, fp, fd_exclude_patterns, notes_dir): fp for fp in files_to_process}
        for future in as_completed(futures):
            result = future.result()
            if result:
                existing_index[result['filename']] = result

    # Write out the updated index.
    updated_index_data = list(existing_index.values())
    try:
        with open(index_file, 'w', encoding='utf-8') as outf:
            json.dump(updated_index_data, outf, indent=2)
        print(f"Index file updated at: {index_file}")
    except Exception as e:
        print(f"Error writing index file '{index_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Save state only in non-full reindex mode.
    if not args.full_reindex:
        save_index_state(state_file, current_index_state)
    elif args.verbose:
        print("Skipping saving state file because of full reindex.")

if __name__ == "__main__":
    main()

