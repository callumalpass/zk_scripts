#!/usr/bin/env python3
"""
Fast ZK indexer

• Uses os.scandir recursively to list markdown files quickly.
• Reads each file in one go and splits on the YAML front matter separator.
• Uses a precompiled regex for wikilink extraction.
• Parallels file processing using ProcessPoolExecutor.
• Converts any YAML dates to ISO strings for JSON output.
"""

import os
import re
import sys
import json
import yaml
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed

CONFIG_FILE = os.path.expanduser("~/.config/zk_scripts/config.yaml")

# Precompile wikilink regex (only matching [[target]] where optional alias is ignored)
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
        return {k: json_ready(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_ready(item) for item in data]
    else:
        return data

def scandir_recursive(root):
    """Recursively find files under root using os.scandir."""
    paths = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                try:
                    if entry.is_file():
                        paths.append(entry.path)
                    elif entry.is_dir():
                        paths.extend(scandir_recursive(entry.path))
                except PermissionError:
                    continue
    except PermissionError:
        pass
    return paths

def process_markdown_file(filepath, fd_exclude_patterns):
    """
    Process one markdown file:
      - Skip file if its path matches any fd_exclude_patterns substring.
      - Read the entire file at once.
      - If file begins with '---', treat the next block as YAML front matter.
      - Extract wikilinks via precompiled regex.
      - Return a dict with filename (without extension), outgoing_links,
        and any YAML front matter (converted with json_ready).
    """
    for pattern in fd_exclude_patterns:
        if pattern in filepath:
            return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return None

    name_noext = os.path.splitext(os.path.basename(filepath))[0]

    got_yaml = False
    yaml_content = ""
    if content.startswith('---'):
        # Split on first two occurrences of '---'
        parts = content.split('---', 2)
        if len(parts) >= 3:
            # parts[0] is empty if file truly starts with ---
            yaml_content = parts[1].strip()
            got_yaml = True
            body = parts[2]
        else:
            body = content
    else:
        body = content

    # Extract wikilinks. Use a set to avoid duplicates while preserving order.
    outgoing_links = []
    seen = set()
    for link in WIKILINK_RE.findall(body):
        clean_link = link.replace("\\", "")
        if clean_link not in seen:
            seen.add(clean_link)
            outgoing_links.append(clean_link)

    result = {"filename": name_noext, "outgoing_links": outgoing_links}
    if got_yaml and yaml_content:
        try:
            ydict = yaml.safe_load(yaml_content) or {}
            if isinstance(ydict, dict):
                result.update(json_ready(ydict))
        except yaml.YAMLError:
            pass
    return result

def main():
    config = load_config(CONFIG_FILE, "zk_index")

    # Default values if not set in config.
    notes_dir_default = "/home/calluma/Dropbox/notes"
    index_file_default = os.path.join(notes_dir_default, "index.json")
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir = resolve_config_value(config, "notes_dir", notes_dir_default)
    index_file = resolve_config_value(config, "zk_index.index_file", index_file_default)
    fd_exclude_patterns_str = resolve_config_value(config, "zk_index.fd_exclude_patterns", "-E templates/ -E .zk/")
    # Expected format: "-E templates/ -E .zk/" so we extract the patterns.
    fd_exclude_patterns =  re.findall(r'-E\s+([^\s]+)', fd_exclude_patterns_str)
    if not fd_exclude_patterns:
        fd_exclude_patterns = fd_exclude_patterns_default

    if not notes_dir:
        print("Error: NOTES_DIR not defined.", file=sys.stderr)
        sys.exit(1)

    # Locate all Markdown files.
    all_files = scandir_recursive(notes_dir)
    markdown_files = [fp for fp in all_files if fp.lower().endswith(".md")]

    index_data = []
    # Use ProcessPoolExecutor; adjust max_workers to match your hardware.
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_markdown_file, fp, fd_exclude_patterns): fp for fp in markdown_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                index_data.append(result)

    try:
        with open(index_file, 'w', encoding='utf-8') as outf:
            json.dump(index_data, outf, indent=2)
        print(f"Index file updated at: {index_file}")
    except Exception as e:
        print(f"Error writing index file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

