#!/usr/bin/env python3
"""
Note Management Script (py_zk.py)

Processes a JSON index file of notes, computes backlinks, filters notes,
and outputs results in various formats (plain text, CSV, JSON, table).

Features:
  • Load notes from a JSON index file.
  • Compute backlinks to show notes linking to each note.
  • Filter notes by tags (AND/OR logic), filename, date range, backlinks, outgoing links,
    or any arbitrary field.
  • Output in plain text, CSV, JSON, or table formats.
  • Customize output with format strings (with support for literal curly braces via doubling).
  • Provides index information (--info flag).
  • List orphan notes (no incoming or outgoing links) (--orphans flag).
  • Detect and list dangling links (outgoing links to non-existent notes) (--dangling-links flag).

Usage Examples:
  List unique tags:
      py_zk.py --index-file index.json --unique-tags

  Filter notes by tag (OR logic) and output in table format:
      py_zk.py --index-file index.json --filter-tag project active --tag-mode or --output-format table

  Filter notes by tag (AND logic) and output in table format:
      py_zk.py --index-file index.json --filter-tag project active --tag-mode and --output-format table

  Filter notes modified between two dates in CSV format:
      py_zk.py --index-file index.json --date-start 2023-10-01 --date-end 2023-10-31 --output-format csv

  Use a custom format string with literal curly braces:
      py_zk.py --index-file index.json --format-string "{{Filename}}: {filename} - {title}"

  Filter notes that are backlinked from a given note:
      py_zk.py --index-file index.json --filter-backlink TargetNote

  Filter notes that link to a target note (outgoing links):
      py_zk.py --index-file index.json --filter-outgoing-link TargetNote

  List orphan notes:
      py_zk.py --index-file index.json --orphans --output-format table

  List dangling links:
      py_zk.py --index-file index.json --dangling-links --output-format table

For detailed options, run: py_zk.py --help
"""

import os
import sys
import json
import argparse
import logging
import csv
import datetime
import re
import yaml  # pip install pyyaml
from tabulate import tabulate   # pip install tabulate
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from io import StringIO

# ---------------- Global Settings ----------------

# ANSI Color Codes for terminal output.
COLOR_CODES = {
    'reset': "\033[0m",
    'bold': "\033[1m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'magenta': "\033[35m",
    'cyan': "\033[36m",
    'white': "\033[37m",
}

# A rotation of colors for plain output formatting.
PLAIN_OUTPUT_COLORS = ['green', 'cyan', 'magenta', 'blue', 'yellow', 'white']


def colorize(text: str, color: Optional[str]) -> str:
    """Return text wrapped in ANSI escape codes for the given color (if any)."""
    if color and color in COLOR_CODES:
        return f"{COLOR_CODES[color]}{text}{COLOR_CODES['reset']}"
    return text


# ---------------- Data Structures ----------------
@dataclass(frozen=True)
class Note:
    filename: str
    title: str = ""
    tags: List[str] = field(default_factory=list)
    dateModified: str = ""
    aliases: List[str] = field(default_factory=list)
    givenName: str = ""
    familyName: str = ""
    outgoing_links: List[str] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    word_count: int = 0  # New: word count
    file_size: int = 0   # New: file size in bytes
    _extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        standard_fields = {
            'filename', 'title', 'tags', 'dateModified', 'aliases',
            'givenName', 'familyName', 'outgoing_links', 'word_count', 'file_size'
        }
        extra_fields = {k: v for k, v in data.items() if k not in standard_fields}
        return cls(
            filename=data.get('filename', ''),
            title=data.get('title', '') or "",
            tags=data.get('tags', []) if isinstance(data.get('tags', []), list) else [],
            dateModified=data.get('dateModified', ''),
            aliases=data.get('aliases', []) if isinstance(data.get('aliases', []), list) else [],
            givenName=data.get('givenName', ''),
            familyName=data.get('familyName', '') or "",
            outgoing_links=data.get('outgoing_links', []) if isinstance(data.get('outgoing_links', []), list) else [],
            word_count=data.get('word_count', 0) if isinstance(data.get('word_count', 0), int) else 0,
            file_size=data.get('file_size', 0) if isinstance(data.get('file_size', 0), int) else 0,
            backlinks=[],  # to be computed later
            _extra_fields=extra_fields
        )

    def get_field(self, field_name: str) -> Any:
        """Retrieve any field value, including standard and extra fields."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self._extra_fields.get(field_name, "")


def compute_backlinks(notes: List[Note]) -> Dict[str, List[str]]:
    """
    Compute a mapping from each target note's filename to a list of filenames that link to it.
    """
    backlinks: Dict[str, List[str]] = {}
    for note in notes:
        for target in note.outgoing_links:
            backlinks.setdefault(target, []).append(note.filename)
    return backlinks


def add_backlinks_to_notes(notes: List[Note]) -> List[Note]:
    """
    Return a new list of Note objects, where each Note's backlinks field is updated.
    """
    backlinks_map = compute_backlinks(notes)
    updated_notes = []
    for note in notes:
        b_links = backlinks_map.get(note.filename, [])
        updated_notes.append(Note(
            filename=note.filename,
            title=note.title,
            tags=note.tags,
            dateModified=note.dateModified,
            aliases=note.aliases,
            givenName=note.givenName,
            familyName=note.familyName,
            outgoing_links=note.outgoing_links,
            backlinks=b_links,
            word_count=note.word_count,
            file_size=note.file_size,
            _extra_fields=note._extra_fields
        ))
    return updated_notes


# ---------------- Configuration and Loading ----------------
def load_config(config_file: Optional[str]) -> Dict[str, Any]:
    """Load settings from a YAML configuration file."""
    config = {}
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            logging.info(f"Loaded configuration from: {config_file}")
        except FileNotFoundError:
            logging.warning(f"Configuration file not found: {config_file}. Using defaults.")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML configuration file '{config_file}': {e}")
    return config


def load_index_data(index_file: str) -> List[Note]:
    """Load the JSON data from the index file and convert each entry into a Note object."""
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [Note.from_dict(item) for item in data]
    except FileNotFoundError:
        logging.error(f"Index file '{index_file}' not found. Please check the path.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in '{index_file}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while loading '{index_file}': {e}")
        sys.exit(1)


# ---------------- Filtering Functions ----------------
def filter_by_tag(notes: List[Note], tags: List[str], tag_mode: str = 'and',
                  exclude_tags: Optional[List[str]] = None) -> List[Note]:
    """
    Filter notes by tags, with 'and' or 'or' logic.
      - ‘and’ mode: note must contain all supplied tags.
      - ‘or’ mode: note must contain at least one of the tags.
      Optionally, exclude notes with any of the exclude_tags.
    """
    tags_set = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()
    if tag_mode not in ['and', 'or']:
        tag_mode = 'and'  # default to 'and' if invalid mode is given
    filtered = []
    for note in notes:
        note_tags = set(note.tags)
        if tag_mode == 'or':
            if not tags_set.intersection(note_tags):
                continue
        else:
            if not tags_set.issubset(note_tags):
                continue
        if exclude_set and exclude_set.intersection(note_tags):
            continue
        filtered.append(note)
    return filtered


def filter_by_filenames_stdin(notes: List[Note]) -> List[Note]:
    """Filter notes by filenames provided via standard input (one filename per line)."""
    filenames = [line.strip() for line in sys.stdin if line.strip()]
    if not filenames:
        print("Usage: ls <filename list> | py_zk.py --stdin", file=sys.stderr)
        sys.exit(1)
    return [note for note in notes if note.filename in filenames]


def filter_by_filename_contains(notes: List[Note], substring: str) -> List[Note]:
    """Filter notes whose filename contains the given substring."""
    return [note for note in notes if substring in note.filename]


def parse_iso_datetime(dt_str: str) -> Optional[datetime.datetime]:
    """
    Parse an ISO formatted datetime string (e.g. "2023-10-12T14:23:45" or "2023-10-12T14:23:45Z").
    If a trailing "Z" is present, it is removed. Make the datetime object offset-naive.
    Returns None if parsing fails.
    """
    if not dt_str:
        return None
    try:
        # Remove trailing 'Z' if present.
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1]
        # fromisoformat expects a string in ISO 8601 format.
        dt = datetime.datetime.fromisoformat(dt_str)
        # Make the datetime object offset-naive
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None,
                         end_date_str: Optional[str] = None) -> List[Note]:
    """
    Filter notes by dateModified value.
    The dates must be provided as "YYYY-MM-DD" strings.
    """
    filtered = []
    start_date = end_date = None
    try:
        if start_date_str:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        logging.error("Invalid date format provided (expected YYYY-MM-DD).")
        return []
    for note in notes:
        if note.dateModified:
            note_dt = parse_iso_datetime(note.dateModified)
            if not note_dt:
                logging.warning(f"Could not parse dateModified for '{note.filename}' from '{note.dateModified}'. Skipping for date filtering.")
                continue
            note_date = note_dt.date()
            if start_date and note_date < start_date:
                continue
            if end_date and note_date > end_date:
                continue
            filtered.append(note)
    return filtered


def filter_by_field_value(notes: List[Note], field: str, value: str) -> List[Note]:
    """Filter notes where a given field’s value exactly equals the supplied value."""
    return [note for note in notes if str(note.get_field(field)) == value]


def filter_by_outgoing_link(notes: List[Note], target_filename: str) -> List[Note]:
    """Filter notes that link to a note with the given filename (in outgoing_links)."""
    return [note for note in notes if target_filename in note.outgoing_links]

def filter_orphan_notes(notes: List[Note]) -> List[Note]:
    """Filter notes that are orphan notes (no outgoing links and no backlinks)."""
    return [note for note in notes if not note.outgoing_links and not note.backlinks]

def find_dangling_links(notes: List[Note]) -> Dict[str, List[str]]:
    """
    Detect dangling links in notes. A dangling link is an outgoing link that does not
    correspond to any filename in the index.

    Returns a dictionary where keys are filenames of notes with dangling links,
    and values are lists of dangling link target filenames.
    """
    indexed_filenames = {note.filename for note in notes}
    dangling_links_map: Dict[str, List[str]] = {}
    for note in notes:
        dangling = []
        for target in note.outgoing_links:
            if target not in indexed_filenames:
                dangling.append(target)
        if dangling:
            dangling_links_map[note.filename] = dangling
    return dangling_links_map


# ---------------- Index Info ----------------
def get_index_info(index_file: str) -> Dict[str, Any]:
    """
    Gather detailed information about the zk index.
    """
    info = {}
    try:
        notes = load_index_data(index_file)
        info['note_count'] = len(notes)
        tags = set()
        total_word_count = 0
        total_file_size = 0
        notes_with_frontmatter = 0
        notes_with_backlinks = 0
        notes_with_outgoing_links = 0
        min_date = None
        max_date = None
        dangling_links_count = 0
        dangling_links_map = find_dangling_links(notes)
        dangling_links_count = sum(len(links) for links in dangling_links_map.values())


        for note in notes:
            tags.update(note.tags)
            total_word_count += note.word_count
            total_file_size += note.file_size
            if note._extra_fields:
                notes_with_frontmatter += 1
            if note.backlinks:
                notes_with_backlinks += 1
            if note.outgoing_links:
                notes_with_outgoing_links += 1
            if note.dateModified:
                dt = parse_iso_datetime(note.dateModified)
                if dt:
                    note_date = dt.date()
                    if min_date is None or note_date < min_date:
                        min_date = note_date
                    if max_date is None or note_date > max_date:
                        max_date = note_date

        info['unique_tag_count'] = len(tags)
        info['index_file_size_bytes'] = os.path.getsize(index_file)
        info['total_word_count'] = total_word_count
        info['average_word_count'] = total_word_count / len(notes) if notes else 0
        info['total_file_size_bytes'] = total_file_size
        info['average_file_size_bytes'] = total_file_size / len(notes) if notes else 0
        info['notes_with_frontmatter_count'] = notes_with_frontmatter
        info['notes_with_backlinks_count'] = notes_with_backlinks
        info['notes_with_outgoing_links_count'] = notes_with_outgoing_links
        info['date_range'] = f"{min_date.isoformat()} to {max_date.isoformat()}" if min_date and max_date else "N/A"
        info['dangling_links_count'] = dangling_links_count

    except Exception as e:
        logging.error(f"Error gathering index info: {e}")
        return {}
    return info


def list_default(notes: List[Note]) -> List[Note]:
    """Return unfiltered list of notes (default behavior)."""
    return notes


# ---------------- Formatting Functions ----------------
def get_field_value(note: Note, field_name: str) -> str:
    """Retrieve and format the value of a field from a note."""
    value = note.get_field(field_name)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def parse_format_string(format_string: str) -> List[Union[str, Dict[str, str]]]:
    """
    Parses a custom format string into literal text and placeholders.
    Placeholders are defined via {field}. Literal braces can be escaped by doubling.
    For example: "{{Filename}}" will become a literal "{Filename}".
    """
    # First, replace doubled braces with a temporary marker
    format_string = format_string.replace("{{", "__LBRACE__").replace("}}", "__RBRACE__")
    pattern = re.compile(r'(?<!{){([^}]+)}(?!})')
    parts = []
    last_index = 0
    for match in pattern.finditer(format_string):
        if match.start() > last_index:
            parts.append(format_string[last_index:match.start()])
        field_name = match.group(1)
        parts.append({'field': field_name})
        last_index = match.end()
    if last_index < len(format_string):
        parts.append(format_string[last_index:])
    # Restore literal braces in the literal parts
    for i, part in enumerate(parts):
        if isinstance(part, str):
            part = part.replace("__LBRACE__", "{").replace("__RBRACE__", "}")
            parts[i] = part
    return parts


def format_plain(notes: List[Note], fields: List[str], separator: str = '::',
                 format_string: Optional[str] = None, use_color: bool = False) -> List[str]:
    """
    Return a list of plain text lines formatted from note data.
    If a custom format_string is provided, it is used; otherwise, the field values are joined using the separator.
    """
    lines = []
    color_cycle = PLAIN_OUTPUT_COLORS if use_color else [None] * len(PLAIN_OUTPUT_COLORS)
    if format_string:
        try:
            parsed_format = parse_format_string(format_string)
        except ValueError as e:
            logging.error(f"Error parsing format string: {e}")
            return []
        for note in notes:
            parts = []
            for part in parsed_format:
                if isinstance(part, dict) and 'field' in part:
                    field_name = part['field']
                    value = get_field_value(note, field_name)
                    try:
                        index = fields.index(field_name)
                    except ValueError:
                        index = 0
                    if use_color:
                        value = colorize(value, color_cycle[index % len(color_cycle)])
                    parts.append(value)
                else:
                    parts.append(part)
            line = "".join(parts)
            if use_color:
                line = colorize(line, 'yellow')
            lines.append(line)
    else:
        for note in notes:
            parts = []
            for idx, field in enumerate(fields):
                value = get_field_value(note, field)
                if use_color:
                    value = colorize(value, color_cycle[idx % len(color_cycle)])
                parts.append(value)
            line = separator.join(parts)
            if use_color:
                line = colorize(line, 'yellow')
            lines.append(line)
    return lines


def format_csv(notes: List[Note], fields: List[str]) -> str:
    """Return a CSV-formatted string from the notes data."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(fields)
    for note in notes:
        row = []
        for field in fields:
            row.append(get_field_value(note, field))
        writer.writerow(row)
    return output.getvalue()


def format_table(notes: List[Note], fields: List[str], use_color: bool = False) -> str:
    """Return a table-formatted string using tabulate."""
    table_data = []
    for note in notes:
        row = []
        for field in fields:
            row.append(get_field_value(note, field))
        table_data.append(row)
    headers = fields
    if use_color:
        headers = [colorize(h, 'cyan') for h in fields]
    return tabulate(table_data, headers=headers, tablefmt="grid")


def format_json(notes: List[Note]) -> str:
    """Return notes data as a JSON-formatted string."""
    notes_list = []
    for note in notes:
        note_dict = note.__dict__.copy()
        note_dict.update(note_dict.get('_extra_fields', {}))
        if '_extra_fields' in note_dict:
            del note_dict['_extra_fields']
        notes_list.append(note_dict)
    return json.dumps(notes_list, indent=2, ensure_ascii=False)

def format_dangling_links_output(dangling_links_map: Dict[str, List[str]], output_format: str = 'plain', use_color: bool = False) -> str:
    """Format dangling link detection output."""
    if output_format == 'json':
        return json.dumps(dangling_links_map, indent=2, ensure_ascii=False)
    elif output_format == 'csv':
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['filename', 'dangling_links'])
        for filename, dangling_links in dangling_links_map.items():
            writer.writerow([filename, ', '.join(dangling_links)])
        return output.getvalue()
    elif output_format == 'table':
        table_data = []
        for filename, dangling_links in dangling_links_map.items():
            table_data.append([filename, ', '.join(dangling_links)])
        headers = ["Filename", "Dangling Links"]
        if use_color:
            headers = [colorize(h, 'cyan') for h in headers]
        return tabulate(table_data, headers=headers, tablefmt="grid")
    elif output_format == 'plain':
        lines = []
        for filename, dangling_links in dangling_links_map.items():
            if use_color:
                lines.append(colorize(f"Note: {filename}", 'yellow'))
            else:
                lines.append(f"Note: {filename}")
            for link in dangling_links:
                if use_color:
                    lines.append(f"  - {colorize(link, 'red')}")
                else:
                    lines.append(f"  - {link}")
        return "\n".join(lines)
    else:
        return ""


def format_output(notes: List[Note], output_format: str = 'plain', fields: Optional[List[str]] = None,
                  separator: str = '::', format_string: Optional[str] = None, use_color: bool = False) -> str:
    """
    Format notes into the requested output format.
    Default fields vary by output_format.
    """
    default_fields_map = {
        'plain': ['filename', 'title', 'tags'],
        'csv': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'table': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'json': ['filename', 'title', 'tags', 'dateModified', 'aliases', 'givenName', 'familyName',
                 'outgoing_links', 'backlinks', 'word_count', 'file_size'],
    }
    effective_fields = fields if fields else default_fields_map.get(output_format, ['filename', 'title', 'tags'])
    if output_format == 'json':
        return format_json(notes)
    elif output_format == 'csv':
        return format_csv(notes, effective_fields)
    elif output_format == 'table':
        return format_table(notes, effective_fields, use_color)
    elif output_format == 'plain':
        lines = format_plain(notes, effective_fields, separator, format_string, use_color)
        return "\n".join(lines)
    else:
        return ""


def sort_data(notes: List[Note], sort_by: str = 'dateModified') -> List[Note]:
    """Sort the notes based on the specified field. For dateModified, we use proper datetime parsing."""
    if sort_by == 'filename':
        return sorted(notes, key=lambda n: n.filename)
    elif sort_by == 'title':
        return sorted(notes, key=lambda n: n.title or "")
    elif sort_by == 'dateModified':
        def note_date(note: Note):
            dt = parse_iso_datetime(note.dateModified)
            return dt if dt else datetime.datetime.min
        return sorted(notes, key=note_date, reverse=True)
    elif sort_by == 'word_count':
        return sorted(notes, key=lambda n: n.word_count, reverse=True)
    elif sort_by == 'file_size':
        return sorted(notes, key=lambda n: n.file_size, reverse=True)
    else:
        return notes


def merge_config_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration options from a YAML file with command-line arguments.
    Command-line options take precedence.
    """
    args_dict = vars(args)
    merged = args_dict.copy()
    for key, value in config.items():
        if merged.get(key) is None:
            merged[key] = value
    return argparse.Namespace(**merged)


# ---------------- Main Program ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Process index.json data for note management.",
        epilog="For more details see the script's header comments.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Configuration ---
    parser.add_argument('--config-file', type=str, help='Path to a YAML configuration file for settings.')
    parser.add_argument('--index-file', '-i', type=str, required=True, help='Path to the index.json file containing note data.')

    # --- Output Options ---
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-format', '-o', type=str, default='plain', choices=['plain', 'csv', 'json', 'table'],
                              help='Format for output: plain, csv, json, table (default: plain).')
    output_group.add_argument('--output-file', type=str, help='File to write output to (default: stdout).')
    output_group.add_argument('--fields', type=str, nargs='+', help='Fields to include in output (from index.json). Defaults vary by format.')
    output_group.add_argument('--separator', type=str, default='::', help='Separator for plain text output (default: "::").')
    output_group.add_argument('--format-string', type=str, help='Custom format string for plain text output. Use placeholders like "{filename} - {title}". Escape literal braces by doubling (e.g., "{{" or "}}").')
    output_group.add_argument('--color', type=str, default='auto', choices=['always', 'auto', 'never'],
                              help='Colorize output: always, auto (if terminal), never (default: auto).')

    # --- Filtering Options ---
    filter_group = parser.add_argument_group('Filtering Options')
    # Options that are mutually exclusive with --info and --unique-tags:
    filter_exclusive = filter_group.add_mutually_exclusive_group()
    filter_exclusive.add_argument('--info', action='store_true', help='Show information about the index and exit.')
    filter_exclusive.add_argument('--unique-tags', action='store_true', help='List all unique tags found in the notes.')
    filter_exclusive.add_argument('--orphans', '--list-orphans', action='store_true', help='List notes with no outgoing and no incoming links (orphan notes).')
    filter_exclusive.add_argument('--dangling-links', '--list-dangling', action='store_true', help='List notes with dangling outgoing links (links to non-indexed notes).')


    # The following filtering options can be combined.
    filter_group.add_argument('--filter-tag', type=str, nargs='+', help='Filter notes by tags.')
    filter_group.add_argument('--tag-mode', type=str, default='and', choices=['and', 'or'], help='Tag filter mode: \'and\' (default, all tags must be present) or \'or\' (any tag present).')
    filter_group.add_argument('--exclude-tag', type=str, nargs='+', help='Exclude notes that have these tags.')
    filter_group.add_argument('--stdin', action='store_true', help='Filter notes by filenames provided via standard input (one per line).')
    filter_group.add_argument('--filename-contains', type=str, help='Filter notes with filenames containing this substring.')
    filter_group.add_argument('--filter-backlink', type=str, help='Filter notes that are backlinked from a note with the given filename.')
    filter_group.add_argument('--filter-outgoing-link', type=str, help='Filter notes that link to a note with the given filename (via outgoing_links).')
    filter_group.add_argument('--date-start', type=str, help='Filter notes modified on or after this date (YYYY-MM-DD).')
    parser.add_argument('--date-end', type=str, help='Filter notes modified on or before this date (YYYY-MM-DD).')
    filter_group.add_argument('--filter-field', type=str, nargs=2, metavar=('FIELD', 'VALUE'),
                              help='Filter notes where FIELD exactly matches VALUE.')

    # --- Sorting Option ---
    parser.add_argument('--sort-by', '-s', type=str, default='dateModified',
                        choices=['dateModified', 'filename', 'title', 'word_count', 'file_size'],
                        help='Field to sort output by (default: dateModified).')

    args = parser.parse_args()

    # Merge configuration from a file, if provided.
    config = load_config(args.config_file)
    args = merge_config_args(config, args)

    # Determine whether to use color in output.
    use_color = args.color == 'always' or (args.color == 'auto' and sys.stdout.isatty())

    # Load index data.
    notes = load_index_data(args.index_file)

    # If --info or --unique-tags is provided, perform that action and exit.
    if args.info:
        info = get_index_info(args.index_file)
        if info:
            print("ZK Index Information:")
            print(f"  Number of notes: {info.get('note_count', 'N/A')}")
            print(f"  Unique tag count: {info.get('unique_tag_count', 'N/A')}")
            file_size_kb = info.get('index_file_size_bytes', 0) / 1024
            print(f"  Index file size: {file_size_kb:.2f} KB")
            print(f"  Total word count: {info.get('total_word_count', 'N/A')}")
            print(f"  Average word count: {info.get('average_word_count', 'N/A'):.0f}")
            avg_file_size_kb = info.get('average_file_size_bytes', 0) / 1024
            print(f"  Average file size per note: {avg_file_size_kb:.2f} KB")
            print(f"  Notes with YAML frontmatter: {info.get('notes_with_frontmatter_count', 'N/A')}")
            print(f"  Notes with backlinks: {info.get('notes_with_backlinks_count', 'N/A')}")
            print(f"  Notes with outgoing links: {info.get('notes_with_outgoing_links_count', 'N/A')}")
            print(f"  Date range of notes: {info.get('date_range', 'N/A')}")
            print(f"  Dangling links count: {info.get('dangling_links_count', 'N/A')}")
        sys.exit(0)
    elif args.unique_tags:
        unique_tags = sorted({tag for note in notes for tag in note.tags})
        for tag in unique_tags:
            print(tag)
        sys.exit(0)
    elif args.orphans:
        # Compute backlinks (needed for orphan detection)
        notes = add_backlinks_to_notes(notes)
        notes = filter_orphan_notes(notes)
        output_str = format_output(notes, args.output_format, args.fields,
                                   args.separator, args.format_string, use_color)
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(output_str)
            except OSError as e:
                logging.error(f"Error writing to output file '{args.output_file}': {e}")
                sys.exit(1)
        else:
            print(output_str)
        sys.exit(0)
    elif args.dangling_links:
        dangling_links_map = find_dangling_links(notes)
        output_str = format_dangling_links_output(dangling_links_map, args.output_format, use_color)
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(output_str)
            except OSError as e:
                logging.error(f"Error writing to output file '{args.output_file}': {e}")
                sys.exit(1)
        else:
            print(output_str)
        sys.exit(0)


    # Compute backlinks and update each Note
    notes = add_backlinks_to_notes(notes)

    # Apply filtering options in order (options can combine)
    if args.filter_tag:
        notes = filter_by_tag(notes, args.filter_tag, tag_mode=args.tag_mode, exclude_tags=args.exclude_tag)
    elif args.exclude_tag and not args.filter_tag:
        notes = filter_by_tag(notes, [], exclude_tags=args.exclude_tag)

    if args.stdin:
        notes = filter_by_filenames_stdin(notes)
    if args.filename_contains:
        notes = filter_by_filename_contains(notes, args.filename_contains)
    if args.filter_backlink:
        notes = [note for note in notes if args.filter_backlink in note.backlinks]
    if args.filter_outgoing_link:
        notes = filter_by_outgoing_link(notes, args.filter_outgoing_link)
    if args.date_start or args.date_end:
        notes = filter_by_date_range(notes, args.date_start, args.date_end)
    if args.filter_field:
        field_name, field_value = args.filter_field
        notes = filter_by_field_value(notes, field_name, field_value)
    if not (args.filter_tag or args.exclude_tag or args.stdin or args.filename_contains or
            args.filter_backlink or args.filter_outgoing_link or args.date_start or args.date_end or args.filter_field or args.orphans or args.dangling_links):
        notes = list_default(notes)

    # Sort notes.
    notes = sort_data(notes, args.sort_by)

    # Format output.
    output_str = format_output(notes, args.output_format, args.fields,
                               args.separator, args.format_string, use_color)

    # Write output: either to a file or stdout.
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(output_str)
        except OSError as e:
            logging.error(f"Error writing to output file '{args.output_file}': {e}")
            sys.exit(1)
    else:
        print(output_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    main()

