#!/usr/bin/env python3
"""
Modernized Note Management Script (with Hierarchical Tags)

This tool processes a JSON index file of notes and supports:
  • Loading notes (with extra frontmatter fields, word count, file size).
  • Computing backlinks (notes linking to one another).
  • Filtering by tags (AND/OR, exclude, hierarchical), filename (via stdin or substring),
    backlinks, outgoing links, date ranges, arbitrary field value, and word count.
  • Outputting in various formats: plain text (with custom format strings),
    CSV, JSON, or a pretty table.
  • Displaying index information (summary statistics).
  • Listing unique tags, orphan notes (no incoming/outgoing links) and dangling links.
  • Loading default settings from a YAML config file.

Hierarchical Tags: Tags can be hierarchical, using '/' as a separator (e.g., 'project/active', 'topic/programming').
When filtering by a parent tag (e.g., 'project'), notes with any child tags (like 'project/active') will also be included.

Usage examples:

  List notes (filter by tag “project” using OR logic and table output):
      python modern_py_zk.py list -i index.json --mode notes --filter-tag project --tag-mode or --output-format table

  List orphan notes:
      python modern_py_zk.py list -i index.json --mode orphans

  List dangling links (CSV output, written to file):
      python modern_py_zk.py list -i index.json --mode dangling-links --output-format csv --output-file out.csv

  List unique tags:
      python modern_py_zk.py list -i index.json --mode unique-tags

For more details run:
    python modern_py_zk.py --help
"""

from __future__ import annotations
import json
import csv
import datetime
import re
import logging
import sys
from pathlib import Path
from io import StringIO
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

import yaml                     # pip install pyyaml
from tabulate import tabulate   # pip install tabulate
import typer
import click

app = typer.Typer(help="Modern Note Management tool (ZK style) built with Typer.")

# ---------------- Global Settings ----------------

COLOR_CODES: Dict[str, str] = {
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
    if color and color in COLOR_CODES:
        return f"{COLOR_CODES[color]}{text}{COLOR_CODES['reset']}"
    return text

# ---------------- Helper for Config Defaults ----------------
def merge_config_option(ctx: typer.Context, cli_value: Optional[Any], key: str, default: Any) -> Any:
    """
    If a command-line option is not provided (is None), look up a default in the loaded
    configuration (if any). Otherwise return the CLI value.
    """
    config = ctx.obj.get("config", {}) if ctx.obj else {}
    return cli_value if cli_value is not None else config.get(key, default)

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
    word_count: int = 0
    file_size: int = 0
    _extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Note:
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
            backlinks=[],
            _extra_fields=extra_fields
        )

    def get_field(self, field_name: str) -> Any:
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self._extra_fields.get(field_name, "")

# ---------------- Loading and Config ----------------

def load_config(config_file: Optional[Path]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if config_file and config_file.exists():
        try:
            config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
            logging.info(f"Loaded configuration from {config_file}")
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config '{config_file}': {e}")
    return config

def load_index_data(index_file: Path) -> List[Note]:
    try:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        return [Note.from_dict(item) for item in data]
    except FileNotFoundError:
        logging.error(f"Index file '{index_file}' not found.")
        raise typer.Exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in '{index_file}': {e}")
        raise typer.Exit(1)
    except Exception as e:
        logging.error(f"Error reading '{index_file}': {e}")
        raise typer.Exit(1)

# ---------------- Backlink Computation ----------------

def compute_backlinks(notes: List[Note]) -> Dict[str, List[str]]:
    backlinks: Dict[str, List[str]] = {}
    for note in notes:
        for target in note.outgoing_links:
            backlinks.setdefault(target, []).append(note.filename)
    return backlinks

def add_backlinks_to_notes(notes: List[Note]) -> List[Note]:
    backlinks_map = compute_backlinks(notes)
    updated = []
    for note in notes:
        b_links = backlinks_map.get(note.filename, [])
        updated.append(Note(
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
    return updated

# ---------------- Filtering Functions ----------------

def filter_by_tag(notes: List[Note], tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
    filtered_notes = []
    filter_tags = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()

    for note in notes:
        note_tags = set(note.tags)
        include_note = False

        if tag_mode == 'or':
            for filter_tag in filter_tags:
                for note_tag in note_tags:
                    if note_tag.startswith(filter_tag) or filter_tag == note_tag:  # Hierarchical tag matching
                        include_note = True
                        break
                if include_note:
                    break
        elif tag_mode == 'and':
            include_note = True
            for filter_tag in filter_tags:
                tag_match_found = False
                for note_tag in note_tags:
                    if note_tag.startswith(filter_tag) or filter_tag == note_tag:
                        tag_match_found = True
                        break
                if not tag_match_found:
                    include_note = False
                    break
        else:
            logging.error(f"Invalid tag_mode: '{tag_mode}'. Using 'and' mode.")
            include_note = True

        if include_note:
            if exclude_set and exclude_set.intersection(note_tags):
                continue
            filtered_notes.append(note)

    return filtered_notes

def filter_by_filenames_stdin(notes: List[Note]) -> List[Note]:
    filenames = [line.strip() for line in sys.stdin if line.strip()]
    if not filenames:
        typer.echo("Usage: ls <filename list> | script --stdin", err=True)
        raise typer.Exit(1)
    return [note for note in notes if note.filename in filenames]

def filter_by_filename_contains(notes: List[Note], substring: str) -> List[Note]:
    return [note for note in notes if substring in note.filename]

def parse_iso_datetime(dt_str: str) -> Optional[datetime.datetime]:
    if not dt_str:
        return None
    try:
        if dt_str.endswith('Z'):
            dt_str = dt_str[:-1]
        dt = datetime.datetime.fromisoformat(dt_str)
        return dt.replace(tzinfo=None)
    except ValueError:
        return None

def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> List[Note]:
    filtered: List[Note] = []
    start_date = end_date = None
    try:
        if start_date_str:
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        logging.error("Invalid date format (expected YYYY-MM-DD).")
        return []
    for note in notes:
        if note.dateModified:
            note_dt = parse_iso_datetime(note.dateModified)
            if not note_dt:
                logging.warning(f"Could not parse dateModified for '{note.filename}'")
                continue
            note_date = note_dt.date()
            if start_date and note_date < start_date:
                continue
            if end_date and note_date > end_date:
                continue
            filtered.append(note)
    return filtered

def filter_by_word_count_range(notes: List[Note], min_word_count: Optional[int] = None, max_word_count: Optional[int] = None) -> List[Note]:
    filtered = []
    for note in notes:
        if min_word_count is not None and note.word_count < min_word_count:
            continue
        if max_word_count is not None and note.word_count > max_word_count:
            continue
        filtered.append(note)
    return filtered

def filter_by_field_value(notes: List[Note], field: str, value: str) -> List[Note]:
    return [note for note in notes if str(note.get_field(field)) == value]

def filter_by_outgoing_link(notes: List[Note], target_filename: str) -> List[Note]:
    return [note for note in notes if target_filename in note.outgoing_links]

def filter_orphan_notes(notes: List[Note]) -> List[Note]:
    return [note for note in notes if not note.outgoing_links and not note.backlinks]

def find_dangling_links(notes: List[Note]) -> Dict[str, List[str]]:
    indexed = {note.filename for note in notes}
    dangling: Dict[str, List[str]] = {}
    for note in notes:
        missing = [target for target in note.outgoing_links if target not in indexed]
        if missing:
            dangling[note.filename] = missing
    return dangling

# ---------------- Output Formatting ----------------

def get_field_value(note: Note, field_name: str) -> str:
    value = note.get_field(field_name)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)

def parse_format_string(fmt_string: str) -> List[Union[str, Dict[str, str]]]:
    fmt_string = fmt_string.replace("{{", "__LBRACE__").replace("}}", "__RBRACE__")
    pattern = re.compile(r'(?<!{){([^}]+)}(?!})')
    parts: List[Union[str, Dict[str, str]]] = []
    last_index = 0
    for match in pattern.finditer(fmt_string):
        if match.start() > last_index:
            parts.append(fmt_string[last_index:match.start()])
        parts.append({'field': match.group(1)})
        last_index = match.end()
    if last_index < len(fmt_string):
        parts.append(fmt_string[last_index:])
    for i, part in enumerate(parts):
        if isinstance(part, str):
            parts[i] = part.replace("__LBRACE__", "{").replace("__RBRACE__", "}")
    return parts

def format_plain(notes: List[Note], fields: List[str], separator: str = '::',
                 format_string: Optional[str] = None, use_color: bool = False) -> List[str]:
    lines = []
    color_cycle = PLAIN_OUTPUT_COLORS if use_color else [None] * len(PLAIN_OUTPUT_COLORS)
    if format_string:
        parsed = parse_format_string(format_string)
        for note in notes:
            parts = []
            for part in parsed:
                if isinstance(part, dict) and 'field' in part:
                    try:
                        f_index = fields.index(part['field'])
                    except ValueError:
                        f_index = 0
                    value = get_field_value(note, part['field'])
                    if use_color:
                        value = colorize(value, color_cycle[f_index % len(color_cycle)])
                    parts.append(value)
                else:
                    parts.append(part)
            line = "".join(parts)
            if use_color:
                line = colorize(line, 'yellow')
            lines.append(line)
    else:
        for note in notes:
            vals = []
            for idx, field in enumerate(fields):
                value = get_field_value(note, field)
                if use_color:
                    value = colorize(value, color_cycle[idx % len(color_cycle)])
                vals.append(value)
            line = separator.join(vals)
            if use_color:
                line = colorize(line, 'yellow')
            lines.append(line)
    return lines

def format_csv(notes: List[Note], fields: List[str]) -> str:
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(fields)
    for note in notes:
        writer.writerow([get_field_value(note, field) for field in fields])
    return out.getvalue()

def format_table(notes: List[Note], fields: List[str], use_color: bool = False) -> str:
    table_data = [[get_field_value(note, field) for field in fields] for note in notes]
    headers = fields.copy()
    if use_color:
        headers = [colorize(h, 'cyan') for h in headers]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def format_json(notes: List[Note]) -> str:
    notes_list = []
    for note in notes:
        note_dict = asdict(note)
        note_dict.update(note_dict.pop('_extra_fields', {}))
        notes_list.append(note_dict)
    return json.dumps(notes_list, indent=2, ensure_ascii=False)

def format_dangling_links_output(d_map: Dict[str, List[str]], output_format: str = 'plain', use_color: bool = False) -> str:
    match output_format:
        case 'json':
            return json.dumps(d_map, indent=2, ensure_ascii=False)
        case 'csv':
            out = StringIO()
            writer = csv.writer(out)
            writer.writerow(['filename', 'dangling_links'])
            for filename, links in d_map.items():
                writer.writerow([filename, ', '.join(links)])
            return out.getvalue()
        case 'table':
            table_data = [[filename, ', '.join(links)] for filename, links in d_map.items()]
            headers = ["Filename", "Dangling Links"]
            if use_color:
                headers = [colorize(h, 'cyan') for h in headers]
            return tabulate(table_data, headers=headers, tablefmt="grid")
        case _:
            lines = []
            for filename, links in d_map.items():
                line = f"Note: {filename}" if not use_color else colorize(f"Note: {filename}", 'yellow')
                lines.append(line)
                for link in links:
                    sub_line = f"  - {link}" if not use_color else f"  - {colorize(link, 'red')}"
                    lines.append(sub_line)
            return "\n".join(lines)

def format_output(notes: List[Note], output_format: str = 'plain', fields: Optional[List[str]] = None,
                  separator: str = '::', format_string: Optional[str] = None, use_color: bool = False) -> str:
    default_fields_map: Dict[str, List[str]] = {
        'plain': ['filename', 'title', 'tags'],
        'csv': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'table': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'json': ['filename', 'title', 'tags', 'dateModified', 'aliases', 'givenName', 'familyName',
                 'outgoing_links', 'backlinks', 'word_count', 'file_size']
    }
    effective_fields = fields if fields else default_fields_map.get(output_format, ['filename', 'title', 'tags'])
    match output_format:
        case 'json':
            return format_json(notes)
        case 'csv':
            return format_csv(notes, effective_fields)
        case 'table':
            return format_table(notes, effective_fields, use_color)
        case _:
            lines = format_plain(notes, effective_fields, separator, format_string, use_color)
            return "\n".join(lines)

def sort_notes(notes: List[Note], sort_by: str = 'dateModified') -> List[Note]:
    if sort_by == 'filename':
        return sorted(notes, key=lambda n: n.filename)
    elif sort_by == 'title':
        return sorted(notes, key=lambda n: n.title)
    elif sort_by == 'dateModified':
        def note_date(n: Note) -> datetime.datetime:
            dt = parse_iso_datetime(n.dateModified)
            return dt if dt else datetime.datetime.min
        return sorted(notes, key=note_date, reverse=True)
    elif sort_by == 'word_count':
        return sorted(notes, key=lambda n: n.word_count, reverse=True)
    elif sort_by == 'file_size':
        return sorted(notes, key=lambda n: n.file_size, reverse=True)
    else:
        return notes

# ---------------- Index Info ----------------

def get_index_info(index_file: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
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
                    d = dt.date()
                    if min_date is None or d < min_date:
                        min_date = d
                    if max_date is None or d > max_date:
                        max_date = d

        info['unique_tag_count'] = len(tags)
        info['index_file_size_bytes'] = index_file.stat().st_size
        info['total_word_count'] = total_word_count
        info['average_word_count'] = total_word_count / len(notes) if notes else 0
        info['total_file_size_bytes'] = total_file_size
        info['average_file_size_bytes'] = total_file_size / len(notes) if notes else 0
        info['notes_with_frontmatter_count'] = notes_with_frontmatter
        info['notes_with_backlinks_count'] = notes_with_backlinks
        info['notes_with_outgoing_links_count'] = notes_with_outgoing_links
        info['date_range'] = f"{min_date} to {max_date}" if min_date and max_date else "N/A"
        info['dangling_links_count'] = dangling_links_count
    except Exception as e:
        logging.error(f"Error gathering index info: {e}")
    return info

# ---------------- Utility ----------------

def load_and_prepare_notes(index_file: Path) -> List[Note]:
    notes = load_index_data(index_file)
    return add_backlinks_to_notes(notes)

# ---------------- Global Callback ----------------
@app.callback()
def main(ctx: typer.Context,
         config_file: Optional[Path] = typer.Option(None, "--config-file", help="Path to a YAML configuration file.")):
    """
    Global options for the note management tool.
    Configuration file settings are loaded if provided; they supply default values for options.
    """
    ctx.obj = {}
    ctx.obj["config"] = load_config(config_file)

# ---------------- Commands ----------------

@app.command(name="info")
def info(
    ctx: typer.Context,
    index_file: Path = typer.Option(..., "-i", help="Path to index JSON file."),
    color: Optional[str] = typer.Option(None, help="Colorize output: always, auto, never."),
):
    """
    Display detailed information about the index file.
    """
    info_data = get_index_info(index_file)
    if info_data:
        lines = [
            "ZK Index Information:",
            f"  Number of notes: {info_data.get('note_count', 'N/A')}",
            f"  Unique tag count: {info_data.get('unique_tag_count', 'N/A')}",
            f"  Index file size: {(info_data.get('index_file_size_bytes', 0) / 1024):.2f} KB",
            f"  Total word count: {info_data.get('total_word_count', 'N/A')}",
            f"  Average word count: {info_data.get('average_word_count', 'N/A'):.0f}",
            f"  Average file size per note: {(info_data.get('average_file_size_bytes', 0) / 1024):.2f} KB",
            f"  Notes with frontmatter: {info_data.get('notes_with_frontmatter_count', 'N/A')}",
            f"  Notes with backlinks: {info_data.get('notes_with_backlinks_count', 'N/A')}",
            f"  Notes with outgoing links: {info_data.get('notes_with_outgoing_links_count', 'N/A')}",
            f"  Date range of notes: {info_data.get('date_range', 'N/A')}",
            f"  Dangling links count: {info_data.get('dangling_links_count', 'N/A')}"
        ]
        output = "\n".join(lines)
        use_color = (color == "always") or (color == "auto" and sys.stdout.isatty())
        if use_color:
            # Force color by writing directly to sys.stdout.
            sys.stdout.write(output + "\n")
        else:
            typer.echo(output)

@app.command(name="list")
def list_(
    ctx: typer.Context,
    index_file: Path = typer.Option(..., "-i", help="Path to index JSON file."),
    mode: str = typer.Option("notes", "--mode", help="Display mode: notes, unique-tags, orphans, dangling-links."),
    # Common display options:
    output_format: Optional[str] = typer.Option(None, "-o", help="Output format: plain, csv, json, table."),
    fields: Optional[List[str]] = typer.Option(None, help="Fields to include in output."),
    separator: Optional[str] = typer.Option(None, help="Separator for plain text output (default: '::')."),
    format_string: Optional[str] = typer.Option(None, help="Custom format string for plain text output."),
    color: Optional[str] = typer.Option(None, help="Colorize output: always, auto, never."),
    sort_by: Optional[str] = typer.Option(None, "-s", help="Field to sort by (dateModified, filename, title, word_count, file_size)."),
    output_file: Optional[Path] = typer.Option(None, "--output-file", help="Write output to file."),
    # Filtering options (only apply when mode is 'notes')
    filter_tag: Optional[List[str]] = typer.Option(None, help="Tags to filter on (hierarchical tags supported)."),
    tag_mode: Optional[str] = typer.Option(None, help="Tag filter mode: 'and' (default) or 'or'."),
    exclude_tag: Optional[List[str]] = typer.Option(None, help="Exclude notes that contain these tags."),
    stdin: bool = typer.Option(False, help="Filter by filenames from stdin (one per line)."),
    filename_contains: Optional[str] = typer.Option(None, help="Only include notes whose filename contains this substring."),
    filter_backlink: Optional[str] = typer.Option(None, help="Only include notes backlinked from this filename."),
    filter_outgoing_link: Optional[str] = typer.Option(None, help="Only include notes linking to this filename."),
    date_start: Optional[str] = typer.Option(None, help="Start date (YYYY-MM-DD)."),
    date_end: Optional[str] = typer.Option(None, help="End date (YYYY-MM-DD)."),
    filter_field: Optional[List[str]] = typer.Option(None, help="FIELD VALUE pair for literal filtering (e.g., familyName Doe)."),
    min_word_count: Optional[int] = typer.Option(None, help="Minimum word count."),
    max_word_count: Optional[int] = typer.Option(None, help="Maximum word count.")
):
    """
    List notes or note metadata based on various criteria.

    The --mode option selects what to list:
      • notes (default): List all notes (or those filtered via the extra options below)
      • unique-tags: List all unique tags
      • orphans: List notes with no incoming or outgoing links
      • dangling-links: Display notes with outgoing links pointing to non-existent notes
    """
    # Merge with config defaults
    output_format = merge_config_option(ctx, output_format, "output_format", "plain")
    separator = merge_config_option(ctx, separator, "separator", "::")
    color = merge_config_option(ctx, color, "color", "auto")
    sort_by = merge_config_option(ctx, sort_by, "sort_by", "dateModified")
    tag_mode = merge_config_option(ctx, tag_mode, "tag_mode", "and")
    use_color = (color == "always") or (color == "auto" and sys.stdout.isatty())

    mode = mode.lower()
    allowed_modes = ("notes", "unique-tags", "orphans", "dangling-links")
    if mode not in allowed_modes:
        typer.echo(f"Invalid mode '{mode}'. Allowed values: {', '.join(allowed_modes)}", err=True)
        raise typer.Exit(1)

    if mode == "unique-tags":
        notes = load_index_data(index_file)
        tags = sorted({tag for note in notes for tag in note.tags})
        output = "\n".join(tags)
    elif mode == "orphans":
        notes = load_and_prepare_notes(index_file)
        orphan_notes = filter_orphan_notes(notes)
        output = format_output(orphan_notes, output_format, fields, separator, format_string, use_color)
    elif mode == "dangling-links":
        notes = load_index_data(index_file)
        d_map = find_dangling_links(notes)
        output = format_dangling_links_output(d_map, output_format, use_color)
    else:  # mode == "notes"
        notes = load_and_prepare_notes(index_file)
        if filter_tag:
            notes = filter_by_tag(notes, filter_tag, tag_mode, exclude_tag)
        elif exclude_tag and not filter_tag:
            notes = filter_by_tag(notes, [], exclude_tags=exclude_tag)
        if stdin:
            notes = filter_by_filenames_stdin(notes)
        if filename_contains:
            notes = filter_by_filename_contains(notes, filename_contains)
        if filter_backlink:
            notes = [n for n in notes if filter_backlink in n.backlinks]
        if filter_outgoing_link:
            notes = filter_by_outgoing_link(notes, filter_outgoing_link)
        if date_start or date_end:
            notes = filter_by_date_range(notes, date_start, date_end)
        if filter_field and len(filter_field) >= 2:
            field_name = filter_field[0]
            field_value = filter_field[1]
            notes = filter_by_field_value(notes, field_name, field_value)
        if min_word_count is not None or max_word_count is not None:
            notes = filter_by_word_count_range(notes, min_word_count, max_word_count)
        notes = sort_notes(notes, sort_by)
        output = format_output(notes, output_format, fields, separator, format_string, use_color)

    if output_file:
        output_file.write_text(output, encoding="utf-8")
    else:
        # When forcing color, simply write directly to sys.stdout.
        if use_color and output_format not in ('json', 'csv'):
            sys.stdout.write(output + "\n")
        else:
            typer.echo(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    app()

