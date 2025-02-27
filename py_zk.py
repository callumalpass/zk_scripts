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
  • Searching for similar notes using OpenAI embeddings (loaded from a separate embeddings file).

Hierarchical Tags: Tags can be hierarchical, using '/' as a separator (e.g., 'project/active', 'topic/programming').
When filtering by a parent tag (e.g., 'project'), notes with any child tags (like 'project/active') will also be included.

Usage examples:

  List notes (filter by tag “project” using OR logic and table output):
      python py_zk.py list -i index.json --mode notes --filter-tag project --tag-mode or --output-format table

  List orphan notes:
      python py_zk.py list -i index.json --mode orphans

  List untagged orphan notes:
      python py_zk.py list -i index.json --mode orphans --untagged-orphans

  List dangling links (CSV output, written to file):
      python py_zk.py list -i index.json --mode dangling-links --output-format csv --output-file out.csv

  List unique tags:
      python py_zk.py list -i index.json --mode unique-tags

  Get detailed index information:
      python py_zk.py info -i index.json

  Search for similar notes via embeddings (requires an embeddings file alongside index.json):
      python py_zk.py search-embeddings -i index.json --query-file note.md --k 5

For more details run:
    python py_zk.py --help
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
from collections import Counter
import os

import yaml                     # pip install pyyaml
from tabulate import tabulate   # pip install tabulate
import typer
import numpy as np
import openai
import time

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
            backlinks=data.get('backlinks', []) if isinstance(data.get('backlinks', []), list) else [],
            word_count=data.get('word_count', 0) if isinstance(data.get('word_count', 0), int) else 0,
            file_size=data.get('file_size', 0) if isinstance(data.get('file_size', 0), int) else 0,
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
        logging.exception(f"Unexpected error reading '{index_file}': {e}")
        raise typer.Exit(1)

def get_cached_notes(ctx: typer.Context, index_file: Path) -> List[Note]:
    """
    Cache and return notes from the index file using the Typer context.
    """
    if 'notes_cache' not in ctx.obj:
        ctx.obj['notes_cache'] = {}
    cache_key = str(index_file.resolve())
    if cache_key not in ctx.obj['notes_cache']:
        ctx.obj['notes_cache'][cache_key] = load_index_data(index_file)
    return ctx.obj['notes_cache'][cache_key]

# ---------------- Graph Generation ----------------

def generate_graph_data(notes: List[Note]) -> Dict[str, Any]:
    nodes = [{"id": note.filename, "title": note.title or note.filename} for note in notes]
    links = []
    existing_files = {note.filename for note in notes}
    for note in notes:
        for target in note.outgoing_links:
            if target in existing_files:
                links.append({"source": note.filename, "target": target})
    return {"nodes": nodes, "links": links}

def generate_graph_html(notes: List[Note], output_file: Path):
    graph_data = generate_graph_data(notes)
    template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Note Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; }
        svg { width: 100vw; height: 100vh; }
        .links line { stroke: #999; }
        .nodes circle { fill: steelblue; }
        .labels text { font-size: 10px; }
    </style>
</head>
<body>
    <svg></svg>
    <script>
        const data = /* GRAPH_DATA_PLACEHOLDER */;
        const svg = d3.select("svg");
        const width = window.innerWidth;
        const height = window.innerHeight;
        const g = svg.append("g");
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id))
            .force("charge", d3.forceManyBody().strength(-50))
            .force("center", d3.forceCenter(width / 2, height / 2));
        const link = g.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(data.links)
            .enter().append("line");
        const node = g.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("r", 5);
        const label = g.append("g")
            .attr("class", "labels")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .text(d => d.title)
            .attr("dx", 8)
            .attr("dy", 3);
        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });
        svg.call(d3.zoom().on("zoom", (event) => {
            g.attr("transform", event.transform);
        }));
    </script>
</body>
</html>
    """
    html_content = template.replace('/* GRAPH_DATA_PLACEHOLDER */', json.dumps(graph_data))
    try:
        output_file.write_text(html_content, encoding="utf-8")
    except Exception as e:
        logging.exception(f"Failed to write graph HTML to '{output_file}': {e}")
        raise typer.Exit(1)

# ---------------- Filtering Functions ----------------

def filter_by_tag(notes: List[Note], tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
    """
    For each note, check if its tags match the given filter.
    Hierarchical matching: a note tag matches if it is exactly equal to a filter tag or starts with 'filter_tag/'.
    """
    filtered_notes = []
    filter_tags = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()

    for note in notes:
        note_tags = set(note.tags)
        if tag_mode == 'or':
            include = any(
                any(nt == ft or nt.startswith(ft + '/') for nt in note_tags)
                for ft in filter_tags
            )
        elif tag_mode == 'and':
            include = all(
                any(nt == ft or nt.startswith(ft + '/') for nt in note_tags)
                for ft in filter_tags
            )
        else:
            logging.error(f"Invalid tag_mode: '{tag_mode}'. Defaulting to 'and'.")
            include = all(
                any(nt == ft or nt.startswith(ft + '/') for nt in note_tags)
                for ft in filter_tags
            )
        if include and not exclude_set.intersection(note_tags):
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
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except ValueError:
        return None

def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> List[Note]:
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
    except ValueError:
        logging.error("Invalid date format (expected YYYY-MM-DD).")
        return []
    filtered = []
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
    return [
        note for note in notes
        if (min_word_count is None or note.word_count >= min_word_count)
        and (max_word_count is None or note.word_count <= max_word_count)
    ]

def filter_by_field_value(notes: List[Note], field: str, value: str) -> List[Note]:
    return [note for note in notes if str(note.get_field(field)) == value]

def filter_by_outgoing_link(notes: List[Note], target_filename: str) -> List[Note]:
    return [note for note in notes if target_filename in note.outgoing_links]

def filter_orphan_notes(notes: List[Note]) -> List[Note]:
    return [note for note in notes if not note.outgoing_links and not note.backlinks]

def filter_untagged_orphan_notes(notes: List[Note]) -> List[Note]:
    """Filters notes to return only orphan notes that also have no tags."""
    return [note for note in filter_orphan_notes(notes) if not note.tags]

def find_dangling_links(notes: List[Note]) -> Dict[str, List[str]]:
    indexed = {note.filename for note in notes}
    dangling: Dict[str, List[str]] = {}
    for note in notes:
        missing = []
        for target in note.outgoing_links:
            base_target = target.split('#')[0]
            if not (base_target.startswith("biblib/") and base_target.lower().endswith(".pdf")):
                if target not in indexed:
                    missing.append(target)
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
    sort_options = {
        'filename': (lambda n: n.filename, False),
        'title': (lambda n: n.title, False),
        'dateModified': (lambda n: parse_iso_datetime(n.dateModified) or datetime.datetime.min, True),
        'word_count': (lambda n: n.word_count, True),
        'file_size': (lambda n: n.file_size, True)
    }
    if sort_by in sort_options:
        key_func, reverse = sort_options[sort_by]
        return sorted(notes, key=key_func, reverse=reverse)
    else:
        logging.warning(f"Sort field '{sort_by}' not recognized. Returning unsorted notes.")
        return notes

# ---------------- Enhanced Index Info ----------------

def get_index_info(index_file: Path, notes: List[Note]) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info['note_count'] = len(notes)
        tags = set()
        total_word_count = 0
        total_file_size = 0
        notes_with_frontmatter = 0
        notes_with_backlinks = 0
        notes_with_outgoing_links = 0
        notes_with_tags = 0
        notes_with_body = 0
        word_counts = []
        file_sizes = []
        tag_counter = Counter()
        orphan_count = 0      # notes with neither outgoing links nor backlinks
        untagged_orphan_count = 0
        dates = []

        for note in notes:
            # Tags and tag frequency
            if note.tags:
                notes_with_tags += 1
                tag_counter.update(note.tags)
                tags.update(note.tags)
            # Body field: check whether the note has a non-empty 'body' field.
            if str(note.get_field("body")).strip():
                notes_with_body += 1
            # Frontmatter (any extra fields)
            if note._extra_fields:
                notes_with_frontmatter += 1
            # Backlinks and outgoing links
            if note.backlinks:
                notes_with_backlinks += 1
            if note.outgoing_links:
                notes_with_outgoing_links += 1
            # Word count & file size
            total_word_count += note.word_count
            total_file_size += note.file_size
            word_counts.append(note.word_count)
            file_sizes.append(note.file_size)
            # Date range
            if note.dateModified:
                dt = parse_iso_datetime(note.dateModified)
                if dt:
                    dates.append(dt.date())
            # Orphan check
            if not note.outgoing_links and not note.backlinks:
                orphan_count += 1
                if not note.tags:
                    untagged_orphan_count += 1

        # Calculate averages and medians
        avg_word_count = total_word_count / len(notes) if notes else 0
        avg_file_size = total_file_size / len(notes) if notes else 0
        median_word_count = float(np.median(word_counts)) if word_counts else 0
        median_file_size = float(np.median(file_sizes)) if file_sizes else 0

        # Determine date range
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            date_range = f"{min_date} to {max_date}"
        else:
            date_range = "N/A"

        # Most common tags (top 5)
        most_common_tags = tag_counter.most_common(5)

        # Compile all info
        info.update({
            'unique_tag_count': len(tags),
            'index_file_size_bytes': index_file.stat().st_size if index_file.exists() else 0,
            'total_word_count': total_word_count,
            'average_word_count': avg_word_count,
            'median_word_count': median_word_count,
            'total_file_size_bytes': total_file_size,
            'average_file_size_bytes': avg_file_size,
            'median_file_size_bytes': median_file_size,
            'notes_with_frontmatter_count': notes_with_frontmatter,
            'notes_with_backlinks_count': notes_with_backlinks,
            'notes_with_outgoing_links_count': notes_with_outgoing_links,
            'notes_with_tags_count': notes_with_tags,
            'notes_without_tags_count': len(notes) - notes_with_tags,
            'notes_with_body_count': notes_with_body,
            'notes_without_body_count': len(notes) - notes_with_body,
            'date_range': date_range,
            'dangling_links_count': sum(len(links) for links in find_dangling_links(notes).values()),
            'orphan_notes_count': orphan_count,
            'untagged_orphan_notes_count': untagged_orphan_count,
            'most_common_tags': most_common_tags,
        })
    except Exception as e:
        logging.exception(f"Error gathering index info: {e}")
    return info


# ---------------- Embeddings Search Helpers ----------------

def get_embedding(text: str, model: str = "text-embedding-3-small", max_retries: int = 5) -> List[float]:
    # Ensure the API key is set
    if not openai.api_key:
        openai.api_key = os.getenv("OPEN_AI_KEY")
    for attempt in range(max_retries):
        try:
            result = openai.embeddings.create(input=text, model=model)
            return result.data[0].embedding
        except Exception as e:
            typer.echo(f"Error fetching embedding (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    raise Exception("Failed to fetch embedding after multiple attempts.")

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10))

# ---------------- Embeddings Search Command ----------------

@app.command("search-embeddings")
def search_embeddings(
    ctx: typer.Context,
    index_file: Path = typer.Option(..., "-i", help="Path to index JSON file."),
    query_file: Optional[Path] = typer.Argument(None, help="Path to the note file to search similar ones for."),
    embeddings_file: Optional[Path] = typer.Option(None, "--embeddings", help="Path to embeddings JSON file. Default: {index_file.parent}/embeddings.json"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model", help="OpenAI embedding model to use."),
    k: int = typer.Option(5, "--k", help="Number of similar notes to return."),
    query: Optional[str] = typer.Option(None, "--query", help="Instead of providing a query file, semantically search for this given query string.")
):
    """
    Search for similar notes using OpenAI embeddings.

    This command loads the index and then loads the embeddings file (if available).
    The query is determined in one of two ways:
      • If a --query string is supplied, that string will be embedded.
      • Otherwise, the note provided via the query_file argument is used.

    The top k most similar notes (by cosine similarity) are then displayed using the default format:

        {filename}::{title}::{similarity}

    """
    # Determine default for embeddings file if not set.
    if not embeddings_file:
        embeddings_file = index_file.parent / "embeddings.json"
    # Load index data (using cache)
    notes = get_cached_notes(ctx, index_file)
    if not notes:
        raise typer.Exit("No notes loaded from index.")

    # Build a mapping: note filename ➔ Note object
    notes_dict: Dict[str, Note] = {n.filename: n for n in notes}

    # If a query string is provided, use it. Otherwise, use the query_file.
    if query is not None:
        typer.echo("---Embedding supplied query string...---")
        query_text = query
        # When searching by a query string, there is no “query note” from the index.
        query_id = "__QUERY_STRING__"
    else:
        if query_file is None:
            raise typer.Exit("Either supply a query file or use the --query option with a search string.")
        # Determine query note by matching its basename (without extension)
        query_basename = query_file.stem
        query_note: Optional[Note] = None
        for note in notes:
            if Path(note.filename).stem == query_basename:
                query_note = note
                break
        if not query_note:
            raise typer.Exit(f"Query note matching '{query_file}' not found in index.")
        query_text = str(query_note.get_field("body"))
        if not query_text:
            raise typer.Exit(f"Query note '{query_note.filename}' has no 'body' field to compute an embedding.")
        query_id = query_note.filename

    # Get or compute the query embedding.
    # (Note: Even if a query note did not already have an embedding stored, we embed the provided text.)
    query_embedding = get_embedding(query_text, model=embedding_model)

    # Load embeddings mapping from file
    if not embeddings_file.exists():
        raise typer.Exit(f"Embeddings file '{embeddings_file}' not found. Cannot perform search.")
    try:
        with embeddings_file.open("r", encoding="utf-8") as ef:
            embeddings_map: Dict[str, List[float]] = json.load(ef)
    except Exception as e:
        raise typer.Exit(f"Error reading embeddings file '{embeddings_file}': {e}")

    # Build lists of embeddings and corresponding note IDs.
    available_ids = []
    embeddings_list = []
    for note in notes:
        nid = note.filename
        if nid in embeddings_map:
            embeddings_list.append(embeddings_map[nid])
            available_ids.append(nid)
    if not embeddings_list:
        raise typer.Exit("No embeddings found in the embeddings file to compare against.")
    embeddings_array = np.array(embeddings_list, dtype="float32")
    # Normalize embeddings.
    def normalize(vec: np.ndarray) -> np.ndarray:
        return vec / (np.linalg.norm(vec) + 1e-10)
    embeddings_normalized = np.array([normalize(np.array(vec, dtype="float32")) for vec in embeddings_list])
    query_vector = normalize(np.array(query_embedding, dtype="float32"))
    # Compute cosine similarities.
    similarities = np.dot(embeddings_normalized, query_vector)
    # Find top k indices; if the query is from the index, we can exclude it.
    try:
        query_idx = available_ids.index(query_id)
    except ValueError:
        query_idx = None
    sorted_indices = np.argsort(similarities)[::-1]
    top_k = []
    for idx in sorted_indices:
        if query_idx is not None and idx == query_idx:
            continue
        top_k.append(idx)
        if len(top_k) >= k:
            break
    # Prepare results for display.
    results = []
    for idx in top_k:
        nid = available_ids[idx]
        sim_score = similarities[idx]
        note = notes_dict.get(nid)
        if note:
            results.append({"filename": nid, "title": note.title or nid, "similarity": sim_score})
    if not results:
        typer.echo("No similar notes found.")
        raise typer.Exit()
    # Output using a default plain format: "{filename}::{title}::{similarity}"
    default_format = "{filename}::{title}::{similarity:.4f}"
    lines = []
    for res in results:
        line = default_format.format(**res)
        lines.append(line)
    output = "\n".join(lines)
    typer.echo(output)

# ---------------- Global Callback ----------------

@app.callback()
def main(ctx: typer.Context,
         config_file: Optional[Path] = typer.Option(None, "--config-file", help="Path to a YAML configuration file.")):
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
    Display detailed information about the index file, including extra statistics such as median word count,
    file size, note body coverage, orphan counts, and tag distributions.
    """
    notes = get_cached_notes(ctx, index_file)
    info_data = get_index_info(index_file, notes)

    lines = [
        "Enhanced ZK Index Information:",
        f"  Number of notes: {info_data.get('note_count', 'N/A')}",
        f"  Index file size: {(info_data.get('index_file_size_bytes', 0) / 1024):.2f} KB",
        "",
        "Content Statistics:",
        f"  Total word count: {info_data.get('total_word_count', 'N/A')}",
        f"  Average word count: {info_data.get('average_word_count', 0):.2f}",
        f"  Median word count: {info_data.get('median_word_count', 0):.2f}",
        f"  Total file size (bytes): {info_data.get('total_file_size_bytes', 'N/A')}",
        f"  Average file size per note: {(info_data.get('average_file_size_bytes', 0) / 1024):.2f} KB",
        f"  Median file size: {(info_data.get('median_file_size_bytes', 0) / 1024):.2f} KB",
        "",
        "Note Features:",
        f"  Notes with frontmatter: {info_data.get('notes_with_frontmatter_count', 'N/A')}",
        f"  Notes with backlinks: {info_data.get('notes_with_backlinks_count', 'N/A')}",
        f"  Notes with outgoing links: {info_data.get('notes_with_outgoing_links_count', 'N/A')}",
        f"  Orphan notes (no links): {info_data.get('orphan_notes_count', 'N/A')}",
        f"  Untagged orphan notes: {info_data.get('untagged_orphan_notes_count', 'N/A')}",
        "",
        "Tag Statistics:",
        f"  Unique tags: {info_data.get('unique_tag_count', 'N/A')}",
        f"  Notes with tags: {info_data.get('notes_with_tags_count', 'N/A')} " +
            f"({(info_data.get('notes_with_tags_count', 0) / info_data.get('note_count', 1) * 100):.2f}%)",
        f"  Notes without tags: {info_data.get('notes_without_tags_count', 'N/A')} " +
            f"({(info_data.get('notes_without_tags_count', 0) / info_data.get('note_count', 1) * 100):.2f}%)",
        f"  Most common tags: {', '.join([f'{tag} ({count})' for tag, count in info_data.get('most_common_tags', [])]) or 'N/A'}",
        "",
        "Additional Details:",
        f"  Notes with body (for embeddings): {info_data.get('notes_with_body_count', 'N/A')}",
        f"  Notes without body: {info_data.get('notes_without_body_count', 'N/A')}",
        f"  Date range: {info_data.get('date_range', 'N/A')}",
        f"  Dangling links count: {info_data.get('dangling_links_count', 'N/A')}",
    ]

    output = "\n".join(lines)
    use_color = (color == "always") or (color == "auto" and sys.stdout.isatty())
    if use_color:
        sys.stdout.write(colorize(output, 'yellow') + "\n")
    else:
        typer.echo(output)



@app.command(name="list")
def list_(
    ctx: typer.Context,
    index_file: Path = typer.Option(..., "-i", help="Path to index JSON file."),
    mode: str = typer.Option("notes", "--mode", help="Display mode: notes, unique-tags, orphans, dangling-links."),
    output_format: Optional[str] = typer.Option(None, "-o", help="Output format: plain, csv, json, table."),
    fields: Optional[List[str]] = typer.Option(None, help="Fields to include in output."),
    separator: Optional[str] = typer.Option(None, help="Separator for plain text output (default: '::')."),
    format_string: Optional[str] = typer.Option(None, help="Custom format string for plain text output."),
    color: Optional[str] = typer.Option(None, help="Colorize output: always, auto, never."),
    sort_by: Optional[str] = typer.Option(None, "-s", help="Field to sort by (dateModified, filename, title, word_count, file_size)."),
    output_file: Optional[Path] = typer.Option(None, "--output-file", help="Write output to file."),
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
    max_word_count: Optional[int] = typer.Option(None, help="Maximum word count."),
    untagged_orphans: bool = typer.Option(False, "--untagged-orphans", help="List only orphan notes that have no tags.")
):
    """
    List notes or note metadata based on various criteria.
    """
    output_format = merge_config_option(ctx, output_format, "output_format", "plain")
    separator = merge_config_option(ctx, separator, "separator", "::")
    color = merge_config_option(ctx, color, "color", "auto")
    sort_by = merge_config_option(ctx, sort_by, "sort_by", "dateModified")
    tag_mode = merge_config_option(ctx, tag_mode, "tag_mode", "and")
    use_color = (color == "always") or (color == "auto" and sys.stdout.isatty())

    mode = mode.lower()
    allowed_modes = ("notes", "unique-tags", "orphans", "dangling-links", "untagged-orphans")
    if mode not in allowed_modes:
        typer.echo(f"Invalid mode '{mode}'. Allowed values: {', '.join(allowed_modes)}", err=True)
        raise typer.Exit(1)

    if mode == "unique-tags":
        notes = get_cached_notes(ctx, index_file)
        tags = sorted({tag for note in notes for tag in note.tags})
        output = "\n".join(tags)
    elif mode == "orphans":
        notes = get_cached_notes(ctx, index_file)
        orphan_notes = filter_orphan_notes(notes)
        output = format_output(orphan_notes, output_format, fields, separator, format_string, use_color)
    elif mode == "untagged-orphans":
        notes = get_cached_notes(ctx, index_file)
        untagged_orphan_notes = filter_untagged_orphan_notes(notes)
        output = format_output(untagged_orphan_notes, output_format, fields, separator, format_string, use_color)
    elif mode == "dangling-links":
        notes = get_cached_notes(ctx, index_file)
        d_map = find_dangling_links(notes)
        output = format_dangling_links_output(d_map, output_format, use_color)
    else:  # mode == "notes"
        notes = get_cached_notes(ctx, index_file)
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
        try:
            output_file.write_text(output, encoding="utf-8")
        except Exception as e:
            logging.exception(f"Failed to write output to '{output_file}': {e}")
            raise typer.Exit(1)
    else:
        if use_color and output_format not in ('json', 'csv'):
            sys.stdout.write(output + "\n")
        else:
            typer.echo(output)

@app.command(name="graph")
def graph(
    ctx: typer.Context,
    index_file: Path = typer.Option(..., "-i", help="Path to index JSON file."),
    output_file: Path = typer.Option("graph.html", "--output", help="Output HTML file.")
):
    """
    Generate an interactive graph visualization of note connections.
    """
    notes = get_cached_notes(ctx, index_file)
    generate_graph_html(notes, output_file)
    typer.echo(f"Graph generated at {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    app()

