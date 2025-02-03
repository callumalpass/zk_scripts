#!/usr/bin/env python3
"""
Note Management Script (py_zk.py)

Processes a JSON index file of notes, computes backlinks, filters notes,
and outputs results in various formats (plain text, CSV, JSON, table).

Features:
  • Load notes from a JSON index file.
  • Compute backlinks to show notes linking to each note.
  • Filter notes by tags, filename, date range, backlinks, or any field.
  • Output in plain text, CSV, JSON, or table formats.
  • Customize output with format strings and color.

Usage:
  py_zk.py --index-file <index.json> [OPTIONS]

Examples:
  List unique tags:
    py_zk.py --index-file index.json --unique-tags

  List notes tagged 'project' AND 'active' in table format:
    py_zk.py --index-file index.json --filter-tag project active --output-format table

  List notes modified in October 2023 in CSV format:
    py_zk.py --index-file index.json --date-start 2023-10-01 --date-end 2023-10-31 --output-format csv

  List notes with filenames from stdin:
    ls -t *.md | py_zk.py --index-file index.json --stdin

  Format output with a custom string:
    py_zk.py --index-file index.json --format-string "{filename} - Title: {title} - Tags: {tags}"

  List notes linking to 'TargetNote' in plain text:
    py_zk.py --index-file index.json --filter-backlink TargetNote

  List notes by author "Jane Doe" in JSON format:
    py_zk.py --index-file index.json --filter-field author "Jane Doe" --output-format json

  List notes with 'tutorial' tag, excluding those with 'draft' tag in table format:
    py_zk.py --index-file index.json --filter-tag tutorial --exclude-tag draft --output-format table

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
import yaml   # pip install pyyaml
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
    _extra_fields: Dict[str, Any] = field(default_factory=dict) # Store any other fields

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        standard_fields = {
            'filename', 'title', 'tags', 'dateModified', 'aliases',
            'givenName', 'familyName', 'outgoing_links'
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
            backlinks=[],  # Backlinks will be computed later.
            _extra_fields=extra_fields
        )

    def get_field(self, field_name: str) -> Any:
        """Retrieve any field value, including standard and extra fields."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self._extra_fields.get(field_name, "") # Default to empty string if not found


def compute_backlinks(notes: List[Note]) -> Dict[str, List[str]]:
    """
    Compute backlinks mapping from each target note's filename to the list of filenames that link to it.
    """
    backlinks = {}
    for note in notes:
        for target in note.outgoing_links:
            backlinks.setdefault(target, []).append(note.filename)
    return backlinks


def add_backlinks_to_notes(notes: List[Note]) -> List[Note]:
    """
    Return a new list of Note objects where each note's backlinks field is updated using computed backlinks.
    """
    backlinks_map = compute_backlinks(notes)
    updated_notes = []
    for note in notes:
        b_links = backlinks_map.get(note.filename, [])
        # Create a new Note with updated backlinks (since Note is frozen).
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
    Filters notes by tags.
      - For tag_mode 'and', the note must contain all specified tags.
      - For tag_mode 'or', it must contain at least one.
      - If exclude_tags is provided, notes with any of those tags are omitted.
    """
    tags_set = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()
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
    """Filter notes by filenames provided via standard input."""
    filenames = [line.strip() for line in sys.stdin if line.strip()]
    if not filenames:
        print("Usage: ls <filename_list> | script.py --stdin", file=sys.stderr)
        sys.exit(1)
    return [note for note in notes if note.filename in filenames]


def filter_by_filename_contains(notes: List[Note], substring: str) -> List[Note]:
    """Filter notes whose filename contains the given substring."""
    return [note for note in notes if substring in note.filename]


def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None,
                         end_date_str: Optional[str] = None) -> List[Note]:
    """
    Filters notes by dateModified.
    Dates must be provided as YYYY-MM-DD strings.
    """
    filtered = []
    start_date = end_date = None
    try:
        if start_date_str:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        return []
    for note in notes:
        if note.dateModified:
            try:
                note_date = datetime.datetime.strptime(note.dateModified, '%Y%m%d%H%M%S').date()
            except ValueError:
                logging.warning(f"Could not parse dateModified for '{note.filename}' with value '{note.dateModified}'. Skipping.")
                continue
            if start_date and note_date < start_date:
                continue
            if end_date and note_date > end_date:
                continue
            filtered.append(note)
    return filtered

def filter_by_field_value(notes: List[Note], field: str, value: str) -> List[Note]:
    """Filter notes where a given field's value matches a string."""
    return [note for note in notes if str(note.get_field(field)) == value]


def list_default(notes: List[Note]) -> List[Note]:
    """Return notes without any filtering (default behavior)."""
    return notes


# ---------------- Formatting Functions ----------------

def get_field_value(note: Note, field_name: str) -> str:
    """Retrieve and format the value of a field from a note."""
    value = note.get_field(field_name)
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def parse_format_string(format_string: str) -> List[Union[str, Dict[str, str]]]:
    """
    Parses a custom format string into literal text and placeholders.
    Placeholders are defined via {field} and extracted using regex.
    """
    parts = []
    pattern = re.compile(r'(?<!{){([^}]+)}(?!})')
    last_index = 0
    for match in pattern.finditer(format_string):
        if match.start() > last_index:
            parts.append(format_string[last_index:match.start()])
        field_name = match.group(1)
        parts.append({'field': field_name})
        last_index = match.end()
    if last_index < len(format_string):
        parts.append(format_string[last_index:])
    return parts


def format_plain(notes: List[Note], fields: List[str], separator: str = '::',
                 format_string: Optional[str] = None, use_color: bool = False) -> List[str]:
    """
    Returns a list of plain text lines formatted from note data.
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
            line = "".join(parts) # No separator when using format string directly in parts
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
        note_dict.update(note_dict.get('_extra_fields', {})) # Merge extra fields, handle if _extra_fields is missing
        if '_extra_fields' in note_dict: # Check if it exists before deleting
            del note_dict['_extra_fields'] # Remove the _extra_fields key after merging
        notes_list.append(note_dict)
    return json.dumps(notes_list, indent=2, ensure_ascii=False)


def format_output(notes: List[Note], output_format: str = 'plain', fields: Optional[List[str]] = None,
                  separator: str = '::', format_string: Optional[str] = None, use_color: bool = False) -> str:
    """
    Format notes into the requested output format.
    Default fields are chosen based on the output_format if not specified.
    """
    default_fields_map = {
        'plain': ['filename', 'title', 'tags'],
        'csv': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'table': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'json': ['filename', 'title', 'tags', 'dateModified', 'aliases', 'givenName', 'familyName', 'outgoing_links', 'backlinks'], # Standard fields as default for json
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
    """Sort the notes based on the specified field."""
    if sort_by == 'filename':
        return sorted(notes, key=lambda n: n.filename)
    elif sort_by == 'title':
        return sorted(notes, key=lambda n: n.title or "")
    elif sort_by == 'dateModified':
        return sorted(notes, key=lambda n: n.dateModified, reverse=True)
    else:
        return notes # if sort_by is not recognized, return unsorted


def merge_config_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration options from a YAML file with command-line arguments.
    If a command-line option is None, use the config value.
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
        epilog="For more detailed usage and examples, see the script's header comments.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Configuration ---
    parser.add_argument('--config-file', type=str, help='Path to a YAML configuration file for settings.')
    parser.add_argument('--index-file', type=str, required=True, help='Path to the index.json file containing note data.')

    # --- Output Options ---
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output-format', type=str, default='plain', choices=['plain', 'csv', 'json', 'table'],
                             help='Format for output: plain, csv, json, table (default: plain).')
    output_group.add_argument('--output-file', type=str, help='File to write output to (default: stdout).')
    output_group.add_argument('--fields', type=str, nargs='+', help='Fields to include in output (any field from index.json). '
                             'Defaults vary by output format.')
    output_group.add_argument('--separator', type=str, default='::', help='Separator for plain text output (default: "::").')
    output_group.add_argument('--format-string', type=str, help='Custom format string for plain text output. '
                             'Use placeholders like "{filename} - {title}".')
    output_group.add_argument('--color', type=str, default='auto', choices=['always', 'auto', 'never'],
                             help='Colorize output: always, auto (if terminal), never (default: auto).')

    # --- Filtering Options ---
    filter_group = parser.add_argument_group('Filtering Options')
    filter_group_exclusive = filter_group.add_mutually_exclusive_group() # for mutually exclusive filters
    filter_group_exclusive.add_argument('--unique-tags', action='store_true', help='List all unique tags found in the notes.')
    filter_group_exclusive.add_argument('--filter-tag', type=str, nargs='+', help='Filter notes by tags (AND logic by default). '
                                  'Use with --exclude-tag to exclude tags.')
    filter_group.add_argument('--exclude-tag', type=str, nargs='+', help='Exclude notes that have these tags. Used with --filter-tag.')
    filter_group_exclusive.add_argument('--stdin', action='store_true', help='Filter notes by filenames provided via standard input (one filename per line).')
    filter_group_exclusive.add_argument('--filename-contains', type=str, help='Filter notes by filenames containing this substring.')
    filter_group_exclusive.add_argument('--filter-backlink', type=str, help='Filter notes that are backlinked from a note with this filename.')
    filter_group_exclusive.add_argument('--date-start', type=str, help='Filter notes modified on or after this date (YYYY-MM-DD).')
    filter_group_exclusive.add_argument('--date-end', type=str, help='Filter notes modified on or before this date (YYYY-MM-DD).')
    filter_group_exclusive.add_argument('--filter-field', type=str, nargs=2, metavar=('FIELD', 'VALUE'),
                                  help='Filter notes where FIELD exactly matches VALUE.')
    filter_group_exclusive.add_argument('--default', action='store_true', help='List all notes (default operation if no filter is specified).')


    # --- Sorting Option ---
    parser.add_argument('--sort-by', type=str, default='dateModified', choices=['dateModified', 'filename', 'title'],
                        help='Field to sort output by: dateModified, filename, title (default: dateModified).')


    args = parser.parse_args()

    # Merge configuration file options (if any) with command-line arguments.
    config = load_config(args.config_file)
    args = merge_config_args(config, args)

    # Determine whether to use color in output.
    use_color = args.color == 'always' or (args.color == 'auto' and sys.stdout.isatty())

    # Load index data.
    notes = load_index_data(args.index_file)

    # Compute backlinks and update each Note.
    notes = add_backlinks_to_notes(notes)

    # Apply filtering options.
    if args.unique_tags:
        unique_tags = sorted({tag for note in notes for tag in note.tags})
        for tag in unique_tags:
            print(tag)
        sys.exit(0)
    elif args.filter_tag:
        notes = filter_by_tag(notes, args.filter_tag, tag_mode='and', exclude_tags=args.exclude_tag)
    elif args.exclude_tag and not args.filter_tag:
        notes = filter_by_tag(notes, [], exclude_tags=args.exclude_tag)
    elif args.stdin:
        notes = filter_by_filenames_stdin(notes)
    elif args.filename_contains:
        notes = filter_by_filename_contains(notes, args.filename_contains)
    elif args.filter_backlink:
        notes = [note for note in notes if args.filter_backlink in note.backlinks]
    elif args.date_start or args.date_end:
        notes = filter_by_date_range(notes, args.date_start, args.date_end)
    elif args.filter_field:
        field_name, field_value = args.filter_field
        notes = filter_by_field_value(notes, field_name, field_value)
    elif args.default or not any([
            args.unique_tags, args.filter_tag,
            args.stdin, args.filename_contains, args.date_start, args.date_end,
            args.exclude_tag, args.filter_backlink, args.filter_field
    ]):
        notes = list_default(notes)
    else:
        parser.print_help()
        sys.exit(1)

    # Sort notes.
    notes = sort_data(notes, args.sort_by)

    # Format output.
    output_str = format_output(notes, args.output_format, args.fields,
                               args.separator, args.format_string, use_color)

    # Write output (to file or stdout).
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
    else:
        print(output_str)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    main()

