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
"""

import os
import re
import sys
import json
import yaml
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
            print(f"Warning: '{key_path}' missing in config; using default: {default_value}", file=sys.stderr)
            return default_value
    return current

def load_config(config_file, script_config_section):
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
        return {k: json_ready(v) for k,v in data.items()}
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
                # Check if the entry's full path should be excluded
                if any(pattern in entry.path for pattern in exclude_patterns):
                    # Skip this file or directory completely.
                    continue
                if entry.is_file():
                    paths.append(entry.path)
                elif entry.is_dir():
                    paths.extend(scandir_recursive(entry.path, exclude_patterns))
    except PermissionError:
        pass
    return paths

def process_markdown_file(filepath, fd_exclude_patterns):
    """
    Process one markdown file:
      - Skips file if its path contains any fd_exclude_patterns (defensive check).
      - Reads the entire file.
      - If file begins with '---', then treat the next block as YAML front matter.
      - Extract wikilinks via precompiled regex.
      - Return a dict with filename (without extension), outgoing_links,
        and any YAML front matter (converted with json_ready).
    """
    # Defensive check (should have been filtered during scanning)
    for pat in fd_exclude_patterns:
        if pat in filepath:
            return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None

    name_noext = os.path.splitext(os.path.basename(filepath))[0]

    # Only attempt YAML splitting if the file actually starts with front matter.
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
            except yaml.YAMLError:
                meta = {}
        else:
            body = content
            meta = {}
    else:
        body = content
        meta = {}

    # Extract wikilinks; using a set here to avoid duplicates.
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
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_index_state(state_file, index_state):
    """Save the current index state to a JSON file."""
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(index_state, f, indent=2)
    except Exception as e:
        print(f"Error writing index state file: {e}", file=sys.stderr)

def main():
    config = load_config(CONFIG_FILE, "zk_index")

    # Default values if not set in config.
    notes_dir_default = "/home/calluma/Dropbox/notes"
    index_file_default = os.path.join(notes_dir_default, "index.json")
    # Patterns to exclude. They will be applied both in file scanning and in file processing.
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir = resolve_config_value(config, "notes_dir", notes_dir_default)
    index_file = resolve_config_value(config, "zk_index.index_file", index_file_default)
    fd_exclude_patterns_str = resolve_config_value(config, "zk_index.fd_exclude_patterns", "-E templates/ -E .zk/")
    fd_exclude_patterns =  re.findall(r'-E\s+([^\s]+)', fd_exclude_patterns_str)
    if not fd_exclude_patterns:
        fd_exclude_patterns = fd_exclude_patterns_default

    if not notes_dir:
        print("Error: NOTES_DIR not defined.", file=sys.stderr)
        sys.exit(1)

    state_file = index_file.replace(".json", "_state.json")
    previous_index_state = load_index_state(state_file)
    current_index_state = {}  # To be populated and saved at the end

    # Load existing full index (filename-based) if available.
    existing_index = {}
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                existing_index = {item['filename']: item for item in json.load(f)}
        except (json.JSONDecodeError, FileNotFoundError):
            existing_index = {}

    # Locate all markdown files, filtering out those in excluded directories.
    markdown_files = [fp for fp in scandir_recursive(notes_dir, exclude_patterns=fd_exclude_patterns)
                      if fp.lower().endswith(".md")]
    current_filepaths = set(markdown_files)
    previous_filepaths = set(previous_index_state.keys())

    # Identify files that have been deleted since last run.
    deleted_files = previous_filepaths - current_filepaths
    for deleted in deleted_files:
        fname = os.path.splitext(os.path.basename(deleted))[0]
        if fname in existing_index:
            del existing_index[fname]
        print(f"Note deleted: {deleted}")

    # Decide which files need processing, comparing modification times.
    files_to_process = []
    for filepath in markdown_files:
        try:
            mod_time = os.path.getmtime(filepath)
            current_index_state[filepath] = mod_time  # update mod time in current state
            if filepath not in previous_index_state or previous_index_state[filepath] < mod_time:
                files_to_process.append(filepath)
        except OSError:
            # If the file vanishes between scanning and checking mod time, ignore it.
            print(f"Warning: File disappeared: {filepath}", file=sys.stderr)

    # Process files in parallel.
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_markdown_file, fp, fd_exclude_patterns): fp for fp in files_to_process}
        for future in as_completed(futures):
            filepath = futures[future]
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
        print(f"Error writing index file: {e}", file=sys.stderr)
        sys.exit(1)

    save_index_state(state_file, current_index_state)


if __name__ == "__main__":
    main()

