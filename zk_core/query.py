"""
Query module for Zettelkasten notes.

This module provides functionality to query and filter notes from a Zettelkasten index.
"""

import sys
import os
import json
import logging
import datetime
import re
import csv
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import typer
import yaml
import openai
import numpy as np
from tabulate import tabulate

from zk_core.models import Note, IndexInfo
from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR, DEFAULT_INDEX_FILENAME, DEFAULT_NUM_WORKERS
from zk_core.utils import load_json_file, save_json_file
from zk_core.analytics import get_index_info, find_dangling_links, parse_iso_datetime

# Try to import orjson for faster JSON processing
try:
    import orjson
except ImportError:
    orjson = None

app = typer.Typer(help="Query and filter Zettelkasten notes.")

# ANSI Colors for formatting
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

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def colorize(text: str, color: Optional[str]) -> str:
    """Apply ANSI color to text if color is specified."""
    if color and color in COLOR_CODES:
        return f"{COLOR_CODES[color]}{text}{COLOR_CODES['reset']}"
    return text

def merge_config_option(ctx: typer.Context, cli_value: Optional[Any], key: str, default: Any) -> Any:
    """Get configuration value with priority: CLI > config file > default."""
    config = ctx.obj.get("config", {}) if ctx.obj else {}
    return cli_value if cli_value is not None else config.get(key, default)

def _fast_json_loads(json_bytes: bytes) -> Any:
    """Load JSON data quickly using orjson if available."""
    if orjson:
        return orjson.loads(json_bytes)
    else:
        return json.loads(json_bytes.decode("utf-8"))

def _fast_json_dumps(data: Any) -> str:
    """Dump data to JSON string quickly using orjson if available."""
    if orjson:
        # orjson.dumps returns bytes
        return orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
    else:
        return json.dumps(data, indent=2, ensure_ascii=False)

def load_index_data(index_file: Path) -> List[Note]:
    """Load note data from index file."""
    try:
        file_bytes = index_file.read_bytes()
        data = _fast_json_loads(file_bytes)
        return [Note.from_dict(item) for item in data]
    except FileNotFoundError:
        logger.error(f"Index file '{index_file}' not found.")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error reading '{index_file}': {e}")
        raise typer.Exit(1)

def get_cached_notes(ctx: typer.Context, index_file: Path) -> List[Note]:
    """Cache and return notes from the index file using Typer context."""
    if 'notes_cache' not in ctx.obj:
        ctx.obj['notes_cache'] = {}
    cache_key = str(index_file.resolve())
    if cache_key not in ctx.obj['notes_cache']:
        start_time = time.time()
        ctx.obj['notes_cache'][cache_key] = load_index_data(index_file)
        load_time = time.time() - start_time
        logger.debug(f"Loaded {len(ctx.obj['notes_cache'][cache_key])} notes in {load_time:.2f} seconds")
        
        # Create an index of filename to note for quick lookups
        if 'notes_indices' not in ctx.obj:
            ctx.obj['notes_indices'] = {}
        ctx.obj['notes_indices'][cache_key] = {note.filename: note for note in ctx.obj['notes_cache'][cache_key]}
        
        # Create tag index for faster filtering
        if 'tag_indices' not in ctx.obj:
            ctx.obj['tag_indices'] = {}
        tag_index = {}
        for note in ctx.obj['notes_cache'][cache_key]:
            for tag in note.tags:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(note)
        ctx.obj['tag_indices'][cache_key] = tag_index
    
    return ctx.obj['notes_cache'][cache_key]

# --- Filtering Functions ---

def filter_by_tag_optimized(ctx: typer.Context, index_file: Path, tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
    """Filter notes by tag inclusion and exclusion using the tag index for better performance."""
    # If no tags and no exclusions, return all notes
    if not tags and not exclude_tags:
        return get_cached_notes(ctx, index_file)
    
    # Fall back to regular filtering if tag index is not available
    if 'tag_indices' not in ctx.obj or str(index_file.resolve()) not in ctx.obj['tag_indices']:
        logger.debug("Tag index not available, falling back to regular filtering")
        return filter_by_tag(get_cached_notes(ctx, index_file), tags, tag_mode, exclude_tags)
    
    tag_index = ctx.obj['tag_indices'][str(index_file.resolve())]
    notes = get_cached_notes(ctx, index_file)
    
    # Use set operations for better performance
    filter_tags = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()
    
    # Special case: only exclusions, no inclusion filters
    if not filter_tags and exclude_set:
        all_notes = set(notes)
        for ex_tag in exclude_set:
            # Collect all notes with excluded tag (including hierarchical)
            excluded_notes = set()
            # Direct matches
            if ex_tag in tag_index:
                excluded_notes.update(tag_index[ex_tag])
            
            # Hierarchical matches (notes with tags that start with ex_tag/)
            ex_prefix = f"{ex_tag}/"
            for indexed_tag, notes_with_tag in tag_index.items():
                if indexed_tag.startswith(ex_prefix):
                    excluded_notes.update(notes_with_tag)
            
            # Remove excluded notes from result set
            all_notes.difference_update(excluded_notes)
        
        return list(all_notes)
    
    # Get candidates based on tag mode
    if tag_mode == 'or':
        # Notes that have any of the tags (including hierarchical)
        candidates = set()
        for tag in filter_tags:
            # Direct tag matches
            if tag in tag_index:
                candidates.update(tag_index[tag])
                
            # Hierarchical matches (include notes with tags that start with tag/)
            tag_prefix = f"{tag}/"
            for indexed_tag, notes_with_tag in tag_index.items():
                if indexed_tag.startswith(tag_prefix):
                    candidates.update(notes_with_tag)
    else:  # 'and' mode
        if not filter_tags:  # Empty filter tags with 'and' mode
            candidates = set(notes)
        else:
            # For each filter tag, collect matching notes
            tag_matches_by_filter = []
            for tag in filter_tags:
                tag_matches = set()
                # Direct tag matches
                if tag in tag_index:
                    tag_matches.update(tag_index[tag])
                
                # Hierarchical matches (include notes with tags that start with tag/)
                tag_prefix = f"{tag}/"
                for indexed_tag, notes_with_tag in tag_index.items():
                    if indexed_tag.startswith(tag_prefix):
                        tag_matches.update(notes_with_tag)
                
                tag_matches_by_filter.append(tag_matches)
            
            # In AND mode, notes must match ALL filter criteria
            # Start with all matches for the first filter
            candidates = tag_matches_by_filter[0] if tag_matches_by_filter else set()
            # Intersect with matches for each additional filter
            for matches in tag_matches_by_filter[1:]:
                candidates.intersection_update(matches)
    
    # Exclude tags if necessary
    if exclude_set:
        filtered_notes = []
        for note in candidates:
            exclude_this_note = False
            note_tags = note.tags
            # Check if any of note's tags match exclude patterns
            for ex_tag in exclude_set:
                # Direct match
                if ex_tag in note_tags:
                    exclude_this_note = True
                    break
                # Hierarchical match (note tag starts with ex_tag/)
                ex_prefix = f"{ex_tag}/"
                for note_tag in note_tags:
                    if note_tag.startswith(ex_prefix):
                        exclude_this_note = True
                        break
                if exclude_this_note:
                    break
            
            if not exclude_this_note:
                filtered_notes.append(note)
        return filtered_notes
    else:
        return list(candidates)

def filter_by_tag(notes: List[Note], tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
    """Filter notes by tag inclusion and exclusion."""
    if not tags and not exclude_tags:
        return notes
        
    filtered_notes = []
    filter_tags = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()
    
    for note in notes:
        note_tags = set(note.tags)
        
        # Skip early if we're excluding tags and any match
        if exclude_set and exclude_set.intersection(note_tags):
            continue
        
        # Check tag inclusion based on mode
        if not filter_tags:  # No tags to filter on, only exclusions
            filtered_notes.append(note)
            continue
            
        if tag_mode == 'or':
            include = any(any(nt == ft or nt.startswith(ft + '/') for nt in note_tags) for ft in filter_tags)
        else:  # default to 'and'
            include = all(any(nt == ft or nt.startswith(ft + '/') for nt in note_tags) for ft in filter_tags)
        
        if include:
            filtered_notes.append(note)
    
    return filtered_notes

def filter_by_filenames_stdin(notes: List[Note]) -> List[Note]:
    """Filter notes by filenames from standard input."""
    filenames = [line.strip() for line in sys.stdin if line.strip()]
    if not filenames:
        typer.echo("Usage: ls <filename list> | script --stdin", err=True)
        raise typer.Exit(1)
        
    # Use set for O(1) lookups
    filename_set = set(filenames)
    return [note for note in notes if note.filename in filename_set]

def filter_by_filename_contains(notes: List[Note], substring: str) -> List[Note]:
    """Filter notes by substring in filename."""
    return [note for note in notes if substring.lower() in note.filename.lower()]

def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> List[Note]:
    """Filter notes by date range."""
    if not start_date_str and not end_date_str:
        return notes
        
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
    except ValueError:
        logger.error("Invalid date format (expected YYYY-MM-DD).")
        return []
    
    # Use ThreadPoolExecutor for better performance with date parsing
    filtered = []
    with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
        # Create futures for parsing dates
        futures = {}
        for i, note in enumerate(notes):
            if note.dateModified:
                futures[executor.submit(parse_iso_datetime, note.dateModified)] = (i, note)
        
        # Process results as they complete
        for future in as_completed(futures):
            i, note = futures[future]
            note_dt = future.result()
            if not note_dt:
                logger.debug(f"Could not parse dateModified for '{note.filename}'")
                continue
                
            note_date = note_dt.date()
            if start_date and note_date < start_date:
                continue
            if end_date and note_date > end_date:
                continue
            filtered.append(note)
    
    return filtered

def filter_by_word_count_range(notes: List[Note], min_word_count: Optional[int] = None, max_word_count: Optional[int] = None) -> List[Note]:
    """Filter notes by word count range."""
    if min_word_count is None and max_word_count is None:
        return notes
        
    return [
        note for note in notes
        if (min_word_count is None or note.word_count >= min_word_count)
        and (max_word_count is None or note.word_count <= max_word_count)
    ]

def filter_by_field_value(notes: List[Note], field: str, value: str) -> List[Note]:
    """Filter notes by field value."""
    return [note for note in notes if str(note.get_field(field)) == value]

def filter_by_outgoing_link(notes: List[Note], target_filename: str) -> List[Note]:
    """Filter notes by outgoing link to target filename."""
    return [note for note in notes if target_filename in note.outgoing_links]

def filter_orphan_notes(notes: List[Note]) -> List[Note]:
    """Filter to orphan notes (no links in or out)."""
    return [note for note in notes if not note.outgoing_links and not note.backlinks]

def filter_untagged_orphan_notes(notes: List[Note]) -> List[Note]:
    """Filter to orphan notes with no tags."""
    return [note for note in filter_orphan_notes(notes) if not note.tags]

# --- Output Formatting ---

def get_field_value(note: Note, field_name: str) -> str:
    """Get a field value as a string, handling lists."""
    value = note.get_field(field_name)
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)

def parse_format_string(fmt_string: str) -> List[Union[str, Dict[str, str]]]:
    """Parse a format string with placeholders like {field}."""
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
    """Format notes as plain text."""
    lines = []
    color_cycle = PLAIN_OUTPUT_COLORS if use_color else [None] * len(PLAIN_OUTPUT_COLORS)
    
    # For large result sets, use parallel processing
    if len(notes) > 1000 and not format_string:
        # Pre-compute field indices for the ThreadPoolExecutor
        field_indices = {field: idx for idx, field in enumerate(fields)}
        
        # Function to format a single note
        def format_note(note: Note) -> str:
            vals = []
            for field in fields:
                value = get_field_value(note, field)
                if use_color:
                    idx = field_indices.get(field, 0)
                    value = colorize(value, color_cycle[idx % len(color_cycle)])
                vals.append(value)
            line = separator.join(vals)
            if use_color:
                line = colorize(line, 'yellow')
            return line
        
        # Use ThreadPoolExecutor to parallelize formatting
        with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
            lines = list(executor.map(format_note, notes))
        return lines
    
    # Regular processing for smaller result sets or custom format strings
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
    """Format notes as CSV."""
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(fields)
    
    # For large result sets, use bulk processing
    if len(notes) > 1000:
        # Pre-compute all field values in bulk
        all_values = []
        with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
            # Create a function to process one note
            def process_note(note: Note) -> List[str]:
                return [get_field_value(note, field) for field in fields]
            
            # Process all notes in parallel
            all_values = list(executor.map(process_note, notes))
        
        # Write all rows at once
        writer.writerows(all_values)
    else:
        # Regular processing for smaller result sets
        for note in notes:
            writer.writerow([get_field_value(note, field) for field in fields])
    
    return out.getvalue()

def format_table(notes: List[Note], fields: List[str], use_color: bool = False) -> str:
    """Format notes as a table."""
    # For large result sets, use parallel processing to extract field values
    if len(notes) > 1000:
        with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
            # Function to process a single note
            def get_note_values(note: Note) -> List[str]:
                return [get_field_value(note, field) for field in fields]
                
            # Process all notes in parallel
            table_data = list(executor.map(get_note_values, notes))
    else:
        table_data = [[get_field_value(note, field) for field in fields] for note in notes]
    
    headers = fields.copy()
    if use_color:
        headers = [colorize(h, 'cyan') for h in headers]
    
    return tabulate(table_data, headers=headers, tablefmt="grid")

def format_json(notes: List[Note]) -> str:
    """Format notes as JSON."""
    # For large result sets, use chunked processing to avoid memory issues
    if len(notes) > 1000:
        chunk_size = 500
        chunks = [notes[i:i + chunk_size] for i in range(0, len(notes), chunk_size)]
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
            # Function to process a chunk of notes
            def process_chunk(chunk: List[Note]) -> List[Dict[str, Any]]:
                return [note.to_dict() for note in chunk]
                
            # Process all chunks and combine results
            results = list(executor.map(process_chunk, chunks))
            
        # Flatten the list of chunks
        notes_list = [note_dict for chunk_result in results for note_dict in chunk_result]
    else:
        notes_list = [note.to_dict() for note in notes]
    
    return _fast_json_dumps(notes_list)

def format_dangling_links_output(d_map: Dict[str, List[str]], output_format: str = 'plain', use_color: bool = False) -> str:
    """Format dangling links output."""
    match output_format:
        case 'json':
            return _fast_json_dumps(d_map)
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
    """Format notes output in various formats."""
    default_fields_map: Dict[str, List[str]] = {
        'plain': ['filename', 'title', 'tags'],
        'csv': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'table': ['filename', 'title', 'tags', 'outgoing_links', 'backlinks'],
        'json': ['filename', 'title', 'tags', 'dateModified', 'dateCreated', 'aliases', 'givenName', 'familyName',
                 'outgoing_links', 'backlinks', 'word_count', 'file_size']
    }
    effective_fields = fields if fields is not None else default_fields_map.get(output_format, ['filename', 'title', 'tags'])
    
    # Log performance data for large result sets
    if len(notes) > 1000:
        logger.debug(f"Formatting {len(notes)} notes as {output_format}")
        start_time = time.time()
    
    result = ""
    match output_format:
        case 'json':
            result = format_json(notes)
        case 'csv':
            result = format_csv(notes, effective_fields)
        case 'table':
            result = format_table(notes, effective_fields, use_color)
        case _:
            lines = format_plain(notes, effective_fields, separator, format_string, use_color)
            result = "\n".join(lines)
    
    # Log performance data
    if len(notes) > 1000:
        logger.debug(f"Formatted {len(notes)} notes in {time.time() - start_time:.2f} seconds")
    
    return result

# --- Sorting ---

def sort_notes(notes: List[Note], sort_by: str = 'dateModified') -> List[Note]:
    """Sort notes by a field."""
    if not notes or len(notes) <= 1:
        return notes
        
    sort_options = {
        'filename': (lambda n: n.filename.lower(), False),  # Case-insensitive sorting
        'title': (lambda n: (n.title or "").lower(), False),  # Case-insensitive and None-safe
        'dateModified': (lambda n: parse_iso_datetime(n.dateModified) or datetime.datetime.min, True),
        'word_count': (lambda n: n.word_count, True),
        'file_size': (lambda n: n.file_size, True),
        'dateCreated': (lambda n: parse_iso_datetime(n.dateCreated) or datetime.datetime.min, True)
    }
    
    if sort_by in sort_options:
        key_func, reverse = sort_options[sort_by]
        
        # Pre-compute sort keys for better performance, especially for date parsing
        start_time = time.time()
        
        # For date fields, use ThreadPoolExecutor to parallelize date parsing
        if sort_by in ['dateModified', 'dateCreated'] and len(notes) > 1000:
            with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
                # Function to compute sort key for a note
                def compute_sort_key(i_note: tuple) -> tuple:
                    i, note = i_note
                    return (i, key_func(note))
                
                # Compute sort keys in parallel
                sort_keys = list(executor.map(compute_sort_key, enumerate(notes)))
                
                # Sort by the computed keys
                sort_keys.sort(key=lambda x: x[1], reverse=reverse)
                
                # Reorder notes based on sorted indices
                sorted_notes = [notes[i] for i, _ in sort_keys]
                
                logger.debug(f"Sorted {len(notes)} notes by {sort_by} in {time.time() - start_time:.2f} seconds using parallelization")
                return sorted_notes
        else:
            # Regular sorting for other fields or smaller datasets
            result = sorted(notes, key=key_func, reverse=reverse)
            if len(notes) > 1000:
                logger.debug(f"Sorted {len(notes)} notes by {sort_by} in {time.time() - start_time:.2f} seconds")
            return result
    else:
        logger.warning(f"Sort field '{sort_by}' not recognized. Returning unsorted notes.")
        return notes

# --- Embeddings Search ---

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10))

def get_embedding(text: str, model: str = "text-embedding-3-small", max_retries: int = 5) -> List[float]:
    """Get an embedding from the OpenAI API."""
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

# --- Commands ---

@app.callback()
def main(ctx: typer.Context,
         config_file: Optional[Path] = typer.Option(None, "--config-file", help="Path to a YAML configuration file.")):
    """Initialize the Typer context with configuration."""
    ctx.obj = {}
    ctx.obj["config"] = load_config(config_file)

@app.command(name="info")
def info(
    ctx: typer.Context,
    index_file: Optional[Path] = typer.Option(None, "-i", help="Path to index JSON file."),
    color: Optional[str] = typer.Option(None, help="Colorize output: always, auto, never."),
    focus: Optional[str] = typer.Option(None, "--focus", 
                                        help="Focus on specific info section: network, content, dates, writing, notable, advanced, all"),
):
    """
    Display detailed information about the index file.
    
    Use --focus to show specific information categories:
    - network: Show network and connectivity metrics
    - content: Show content statistics (word count, tags, etc.)
    - dates: Show temporal analysis
    - writing: Show writing productivity patterns
    - notable: Show most notable notes (longest, shortest, newest, etc.)
    - advanced: Show advanced analytics (tag clusters, bridge notes, etc.)
    - all: Show all information (default)
    """
    config = ctx.obj.get("config", {})
    
    # Get default index from config if not specified
    if index_file is None:
        default_index = get_config_value(config, "query.default_index", DEFAULT_INDEX_FILENAME)
        notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
        notes_dir = resolve_path(notes_dir)
        index_file = Path(os.path.join(notes_dir, default_index))
        logger.debug(f"Using default index file from config: {index_file}")
    
    # Measure performance
    start_time = time.time()
    
    # Get notes and generate analytics
    notes = get_cached_notes(ctx, index_file)
    logger.debug(f"Loaded {len(notes)} notes in {time.time() - start_time:.2f} seconds")
    
    # Calculate analytics
    analytics_start = time.time()
    info_data = get_index_info(notes)
    logger.debug(f"Generated analytics in {time.time() - analytics_start:.2f} seconds")
    
    note_count = info_data.note_count
    focus = focus.lower() if focus else 'all'

    # Basic info section - always included
    lines = [
        "Enhanced ZK Index Information:",
        f"  Total notes: {note_count}",
        f"  Index file size: {(index_file.stat().st_size / 1024):.2f} KB",
    ]
    
    # Content statistics section
    if focus in ['content', 'all']:
        lines.extend([
            "",
            "Word Count Statistics:",
            f"  Total word count: {info_data.total_word_count:,}",
            f"  Minimum word count: {info_data.min_word_count:,}",
            f"  Maximum word count: {info_data.max_word_count:,}",
            f"  Average word count: {info_data.average_word_count:.2f}",
            f"  Median word count: {info_data.median_word_count:.2f}",
        ])
        
        # Add word count distribution if available
        if hasattr(info_data, 'word_count_distribution') and info_data.word_count_distribution:
            lines.append("  Word count distribution:")
            for size_cat, count in info_data.word_count_distribution.items():
                lines.append(f"    - {size_cat}: {count} notes ({count/note_count*100:.1f}%)")
        
        lines.extend([
            "",
            "File Size Statistics:",
            f"  Total file size: {(info_data.total_file_size_bytes / 1024 / 1024):.2f} MB",
            f"  Minimum file size: {(info_data.min_file_size_bytes / 1024):.2f} KB",
            f"  Maximum file size: {(info_data.max_file_size_bytes / 1024):.2f} KB",
            f"  Average file size: {(info_data.average_file_size_bytes / 1024):.2f} KB",
            f"  Median file size: {(info_data.median_file_size_bytes / 1024):.2f} KB",
            "",
            "Tag Statistics:",
            f"  Unique tags: {info_data.unique_tag_count}",
            f"  Average tags per note: {info_data.average_tags_per_note:.2f}",
            f"  Median tags per note: {info_data.median_tags_per_note:.2f}",
            f"  Most common tags (top 10): " +
                (", ".join([f'{tag} ({count})' for tag, count in info_data.most_common_tags]) or "N/A"),
        ])
        
        # Add tag co-occurrence if available
        if hasattr(info_data, 'tag_co_occurrence') and info_data.tag_co_occurrence:
            lines.append("  Common tag combinations:")
            for pair, count in list(info_data.tag_co_occurrence.items())[:5]:
                lines.append(f"    - {pair}: {count} occurrences")
        
        lines.extend([
            "",
            "Extra Frontmatter Fields:",
            f"  Fields found: " +
                (", ".join([f"{field} ({count})" for field, count in info_data.extra_frontmatter_keys[:10]])
                 if info_data.extra_frontmatter_keys else "N/A"),
            "",
            "Reference and Alias Statistics:",
            f"  Total references: {info_data.total_references}",
            f"  Average references per note: {info_data.average_references:.2f}",
            f"  Total aliases: {info_data.total_aliases}",
            f"  Average aliases per note: {info_data.average_aliases:.2f}",
        ])
        
        # Add citation hubs if available
        if hasattr(info_data, 'citation_hubs') and info_data.citation_hubs:
            lines.append("  Citation-rich notes:")
            for note_filename, ref_count in info_data.citation_hubs:
                lines.append(f"     {note_filename}:: {ref_count} references")
    
    # Network metrics section
    if focus in ['network', 'all']:
        lines.extend([
            "",
            "Network and Connectivity Metrics:",
            f"  Total links between notes: {info_data.total_links}",
            f"  Average outgoing links per note: {info_data.average_outgoing_links:.2f}",
            f"  Median outgoing links per note: {info_data.median_outgoing_links:.2f}",
            f"  Average backlinks per note: {info_data.average_backlinks:.2f}",
            f"  Median backlinks per note: {info_data.median_backlinks:.2f}",
        ])
        
        # Add network density if available
        if hasattr(info_data, 'network_density'):
            lines.append(f"  Network density: {info_data.network_density:.5f} " +
                        f"({info_data.network_density*100:.2f}% of possible connections)")
            
        lines.extend([
            f"  Orphan notes (no connections): {info_data.orphan_notes_count} " +
                f"({(info_data.orphan_notes_count / note_count * 100) if note_count else 0:.2f}%)",
            f"  Untagged orphan notes: {info_data.untagged_orphan_notes_count} " +
                f"({(info_data.untagged_orphan_notes_count / note_count * 100) if note_count else 0:.2f}%)",
            f"  Dangling links count: {info_data.dangling_links_count}",
        ])
        
        # Highly connected notes section
        if info_data.highly_connected_notes:
            lines.append("  Most connected notes (connection hub notes):")
            for note_filename, connection_count in info_data.highly_connected_notes:
                lines.append(f"     {note_filename}:: {connection_count} connections")
        else:
            lines.append("  No highly connected notes found")
            
        # Bridge notes section
        if hasattr(info_data, 'bridge_notes') and info_data.bridge_notes:
            lines.append("  Bridge notes (connect different clusters):")
            for note_filename, connection_count in info_data.bridge_notes:
                lines.append(f"     {note_filename}:: {connection_count} connections")
    
    # Writing productivity analysis section
    if focus in ['writing', 'all']:
        lines.extend([
            "",
            "Writing Productivity Analysis:",
        ])
        
        # Writing velocity metrics
        if hasattr(info_data, 'writing_velocity'):
            lines.append(f"  Average writing velocity: {info_data.writing_velocity:.1f} words/day")
            
        # Most productive periods
        if hasattr(info_data, 'most_productive_periods') and info_data.most_productive_periods:
            lines.append("  Most productive periods (month, word count):")
            for period, count in info_data.most_productive_periods:
                lines.append(f"    - {period}: {count:,} words")
                
        # Word count by month
        if hasattr(info_data, 'word_count_by_month') and info_data.word_count_by_month:
            lines.append("  Recent monthly word counts:")
            for month, count in list(info_data.word_count_by_month.items())[-6:]:  # Last 6 months
                lines.append(f"    - {month}: {count:,} words")
                
        # Growth rate by year
        if hasattr(info_data, 'growth_rate') and info_data.growth_rate:
            lines.append("  Growth rate (notes/day):")
            for year, rate in sorted(info_data.growth_rate.items()):
                lines.append(f"    - {year}: {rate:.2f} notes/day ({rate*365:.1f} notes/year)")
    
    # Notable notes section
    if focus in ['notable', 'all']:
        lines.extend([
            "",
            "Notable Notes:",
        ])
        
        # Longest notes
        if hasattr(info_data, 'longest_notes') and info_data.longest_notes:
            lines.append("  Longest notes (word count):")
            for filename, word_count in info_data.longest_notes:
                lines.append(f"     {filename}:: {word_count:,} words")
                
        # Shortest notes (non-empty)
        if hasattr(info_data, 'shortest_notes') and info_data.shortest_notes:
            lines.append("  Shortest notes (word count):")
            for filename, word_count in info_data.shortest_notes:
                lines.append(f"     {filename}:: {word_count} words")
                
        # Newest notes
        if hasattr(info_data, 'newest_notes') and info_data.newest_notes:
            lines.append("  Most recently created notes:")
            for filename, date_str in info_data.newest_notes:
                lines.append(f"     {filename}:: {date_str}")
                
        # Oldest notes
        if hasattr(info_data, 'oldest_notes') and info_data.oldest_notes:
            lines.append("  Oldest notes:")
            for filename, date_str in info_data.oldest_notes:
                lines.append(f"     {filename}:: {date_str}")
                
        # Untouched notes
        if hasattr(info_data, 'untouched_notes') and info_data.untouched_notes:
            lines.append("  Notes not modified in over a year:")
            for filename, date_str in info_data.untouched_notes:
                lines.append(f"     {filename}:: last modified {date_str}")
    
    # Advanced analytics section
    if focus in ['advanced', 'all']:
        # Tag clusters
        if hasattr(info_data, 'tag_clusters') and info_data.tag_clusters:
            lines.extend([
                "",
                "Advanced Analytics:",
                "  Tag clusters (related tags):",
            ])
            for tag, related_tags in info_data.tag_clusters:
                related_str = ", ".join(related_tags)
                lines.append(f"    - {tag} → {related_str}")
    
    # Temporal analysis section
    if focus in ['dates', 'all']:
        lines.extend([
            "",
            "Temporal Analysis:",
            f"  Date range: {info_data.date_range}",
            "",
            "Notes by Month of Creation:",
        ])
        
        # Show monthly distribution
        notes_by_month = info_data.notes_by_month
        import calendar
        for month in list(calendar.month_name)[1:]:  # Skip empty first item
            lines.append(f"    {month}: {notes_by_month.get(month, 0)}")
        
        lines.append(f"  Peak creation month: {info_data.peak_creation_month}")
        lines.append("")
        lines.append("Notes by Day of the Week:")
        
        # Add line for each day (Monday - Sunday)
        notes_by_day = info_data.notes_by_day_of_week
        for day in list(calendar.day_name):
            lines.append(f"    {day}: {notes_by_day.get(day, 0)}")
        
        lines.extend([
            f"  Peak creation day: {info_data.peak_creation_day}",
            "",
            "Notes by Year:",
        ])
        
        notes_by_year = info_data.notes_by_year
        if notes_by_year:
            for year, count in notes_by_year.items():
                lines.append(f"    {year}: {count} notes")
        else:
            lines.append("    N/A")
            
        # Add content age distribution if available
        if hasattr(info_data, 'content_age') and info_data.content_age:
            lines.append("")
            lines.append("Content Age Distribution:")
            for age_bucket, percentage in info_data.content_age.items():
                lines.append(f"    {age_bucket}: {percentage:.1f}%")
    
    output = "\n".join(lines)
    use_color = (color == "always") or (color == "auto" and sys.stdout.isatty())
    if use_color:
        sys.stdout.write(colorize(output, 'yellow') + "\n")
    else:
        typer.echo(output)

@app.command(name="list")
def list_(
    ctx: typer.Context,
    index_file: Optional[Path] = typer.Option(None, "-i", help="Path to index JSON file."),
    mode: str = typer.Option("notes", "--mode", help="Display mode: notes, unique-tags, orphans, dangling-links."),
    output_format: Optional[str] = typer.Option(None, "-o", help="Output format: plain, csv, json, table."),
    fields: Optional[List[str]] = typer.Option(None, help="Fields to include in output."),
    separator: Optional[str] = typer.Option(None, help="Separator for plain text output (default: '::')."),
    format_string: Optional[str] = typer.Option(None, help="Custom format string for plain text output."),
    color: Optional[str] = typer.Option(None, help="Colorize output: always, auto, never."),
    sort_by: Optional[str] = typer.Option(None, "-s", help="Field to sort by (dateModified, dateCreated, filename, title, word_count, file_size)."),
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
    # Track execution time
    start_time = time.time()
    config = ctx.obj.get("config", {})
    
    # Get default index from config if not specified
    if index_file is None:
        default_index = get_config_value(config, "query.default_index", DEFAULT_INDEX_FILENAME)
        notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
        notes_dir = resolve_path(notes_dir)
        index_file = Path(os.path.join(notes_dir, default_index))
        logger.debug(f"Using default index file from config: {index_file}")
    
    # Use default fields from config if not specified
    if fields is None:
        fields = get_config_value(config, "query.default_fields", ["filename", "title", "tags"])
        logger.debug(f"Using default fields from config: {fields}")
    
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
    
    # Load notes
    notes_load_start = time.time()
    notes = get_cached_notes(ctx, index_file)
    logger.debug(f"Loaded {len(notes)} notes in {time.time() - notes_load_start:.2f} seconds")
    
    # Process based on mode
    process_start = time.time()
    if mode == "unique-tags":
        # Use a set comprehension for better performance
        all_tags = set()
        for note in notes:
            all_tags.update(note.tags)
        tags = sorted(all_tags)
        output = "\n".join(tags)
        logger.debug(f"Found {len(tags)} unique tags in {time.time() - process_start:.2f} seconds")
    elif mode == "orphans":
        orphan_notes = filter_orphan_notes(notes)
        output = format_output(orphan_notes, output_format, fields, separator, format_string, use_color)
        logger.debug(f"Found {len(orphan_notes)} orphan notes in {time.time() - process_start:.2f} seconds")
    elif mode == "untagged-orphans":
        untagged_orphan_notes = filter_untagged_orphan_notes(notes)
        output = format_output(untagged_orphan_notes, output_format, fields, separator, format_string, use_color)
        logger.debug(f"Found {len(untagged_orphan_notes)} untagged orphan notes in {time.time() - process_start:.2f} seconds")
    elif mode == "dangling-links":
        d_map = find_dangling_links(notes)
        output = format_dangling_links_output(d_map, output_format, use_color)
        total_dangling = sum(len(links) for links in d_map.values())
        logger.debug(f"Found {total_dangling} dangling links in {len(d_map)} notes in {time.time() - process_start:.2f} seconds")
    else:
        # Default notes listing with filtering
        filter_start = time.time()
        
        # Always use optimized tag filtering (it handles no tags case efficiently)
        notes = filter_by_tag_optimized(ctx, index_file, 
                                     filter_tag or [], 
                                     tag_mode, 
                                     exclude_tag)
        logger.debug(f"Tag filtering completed in {time.time() - filter_start:.2f} seconds, {len(notes)} notes remaining")
        
        # Continue with other filters
        if stdin:
            stdin_start = time.time()
            notes = filter_by_filenames_stdin(notes)
            logger.debug(f"Filtered by stdin in {time.time() - stdin_start:.2f} seconds, {len(notes)} notes remaining")
            
        if filename_contains:
            filename_start = time.time()
            notes = filter_by_filename_contains(notes, filename_contains)
            logger.debug(f"Filtered by filename contains '{filename_contains}' in {time.time() - filename_start:.2f} seconds, {len(notes)} notes remaining")
            
        if filter_backlink:
            backlink_start = time.time()
            notes = [n for n in notes if filter_backlink in n.backlinks]
            logger.debug(f"Filtered by backlink '{filter_backlink}' in {time.time() - backlink_start:.2f} seconds, {len(notes)} notes remaining")
            
        if filter_outgoing_link:
            outlink_start = time.time()
            notes = filter_by_outgoing_link(notes, filter_outgoing_link)
            logger.debug(f"Filtered by outgoing link '{filter_outgoing_link}' in {time.time() - outlink_start:.2f} seconds, {len(notes)} notes remaining")
            
        if date_start or date_end:
            date_start_time = time.time()
            notes = filter_by_date_range(notes, date_start, date_end)
            logger.debug(f"Filtered by date range in {time.time() - date_start_time:.2f} seconds, {len(notes)} notes remaining")
            
        if filter_field and len(filter_field) >= 2:
            field_start = time.time()
            field_name = filter_field[0]
            field_value = filter_field[1]
            notes = filter_by_field_value(notes, field_name, field_value)
            logger.debug(f"Filtered by field {field_name}={field_value} in {time.time() - field_start:.2f} seconds, {len(notes)} notes remaining")
            
        if min_word_count is not None or max_word_count is not None:
            wc_start = time.time()
            notes = filter_by_word_count_range(notes, min_word_count, max_word_count)
            logger.debug(f"Filtered by word count range in {time.time() - wc_start:.2f} seconds, {len(notes)} notes remaining")
        
        # Sorting
        sort_start = time.time()
        notes = sort_notes(notes, sort_by)
        logger.debug(f"Sorted {len(notes)} notes by {sort_by} in {time.time() - sort_start:.2f} seconds")
        
        # Output formatting
        format_start = time.time()
        output = format_output(notes, output_format, fields, separator, format_string, use_color)
        logger.debug(f"Formatted output in {time.time() - format_start:.2f} seconds")
        
    if output_file:
        try:
            output_file.write_text(output, encoding="utf-8")
        except Exception as e:
            logger.exception(f"Failed to write output to '{output_file}': {e}")
            raise typer.Exit(1)
    else:
        if use_color and output_format not in ('json', 'csv'):
            sys.stdout.write(output + "\n")
        else:
            typer.echo(output)

@app.command(name="search-embeddings")
def search_embeddings(
    ctx: typer.Context,
    index_file: Optional[Path] = typer.Option(None, "-i", help="Path to index JSON file."),
    query_file: Optional[Path] = typer.Argument(None, help="Path to the note file to search similar ones for."),
    embeddings_file: Optional[Path] = typer.Option(None, "--embeddings", help="Path to embeddings JSON file. Default: {index_file.parent}/embeddings.json"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model", help="OpenAI embedding model to use."),
    k: int = typer.Option(5, "--k", help="Number of similar notes to return."),
    query: Optional[str] = typer.Option(None, "--query", help="Instead of providing a query file, semantically search for this given query string.")
):
    """
    Search for similar notes using OpenAI embeddings.
    """
    config = ctx.obj.get("config", {})
    
    # Get default index from config if not specified
    if index_file is None:
        default_index = get_config_value(config, "query.default_index", DEFAULT_INDEX_FILENAME)
        notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
        notes_dir = resolve_path(notes_dir)
        index_file = Path(os.path.join(notes_dir, default_index))
        logger.debug(f"Using default index file from config: {index_file}")
        
    # Determine default for embeddings file if not set.
    if not embeddings_file:
        embeddings_file = index_file.parent / "embeddings.json"
        
    # Load index data (using cache)
    start_time = time.time()
    notes = get_cached_notes(ctx, index_file)
    logger.debug(f"Loaded {len(notes)} notes in {time.time() - start_time:.2f} seconds")
    
    if not notes:
        raise typer.Exit("No notes loaded from index.")

    # Build a mapping: note filename → Note object
    notes_dict: Dict[str, Note] = {n.filename: n for n in notes}

    # Load embeddings
    emb_start_time = time.time()
    try:
        # Use orjson for faster parsing if available
        if embeddings_file.exists():
            if orjson:
                embeddings_map = orjson.loads(embeddings_file.read_bytes())
            else:
                with embeddings_file.open("r", encoding="utf-8") as ef:
                    embeddings_map = json.load(ef)
            logger.debug(f"Loaded embeddings in {time.time() - emb_start_time:.2f} seconds")
        else:
            raise typer.Exit(f"Embeddings file '{embeddings_file}' not found. Cannot perform search.")
    except Exception as e:
        raise typer.Exit(f"Error reading embeddings file '{embeddings_file}': {e}")

    # If a query string is provided, use it. Otherwise, use the query_file.
    query_start_time = time.time()
    if query is not None:
        typer.echo("---Embedding supplied query string...---")
        query_text = query
        # When searching by a query string, there is no "query note" from the index.
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
    query_embedding = get_embedding(query_text, model=embedding_model)
    logger.debug(f"Generated query embedding in {time.time() - query_start_time:.2f} seconds")

    # Process embeddings and prepare for similarity computation
    similarity_start_time = time.time()
    available_ids = []
    embeddings_list = []
    for note in notes:
        nid = note.filename
        if nid in embeddings_map:
            embeddings_list.append(embeddings_map[nid])
            available_ids.append(nid)
    
    if not embeddings_list:
        raise typer.Exit("No embeddings found in the embeddings file to compare against.")
        
    # Use numpy for efficient vector operations
    embeddings_array = np.array(embeddings_list, dtype="float32")
    
    # Normalize embeddings for faster cosine similarity computation
    def normalize(vec: np.ndarray) -> np.ndarray:
        return vec / (np.linalg.norm(vec) + 1e-10)
    
    # Normalize all embeddings (in parallel for large datasets)
    if len(embeddings_list) > 1000:
        with ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS) as executor:
            def normalize_embedding(vec):
                return normalize(np.array(vec, dtype="float32"))
            embeddings_normalized = list(executor.map(normalize_embedding, embeddings_list))
            embeddings_normalized = np.array(embeddings_normalized)
    else:
        embeddings_normalized = np.array([normalize(np.array(vec, dtype="float32")) for vec in embeddings_list])
    
    # Normalize query vector
    query_vector = normalize(np.array(query_embedding, dtype="float32"))
    
    # Compute cosine similarities using efficient matrix operation
    similarities = np.dot(embeddings_normalized, query_vector)
    logger.debug(f"Computed similarities in {time.time() - similarity_start_time:.2f} seconds")
    
    # Find top k indices; if the query is from the index, exclude it.
    try:
        query_idx = available_ids.index(query_id)
    except ValueError:
        query_idx = None
        
    results_start_time = time.time()
    # Get indices for top k similarities (using argpartition for better performance)
    if len(similarities) > k * 10:  # For larger datasets
        # Use argpartition which is faster than argsort for finding top-k
        if query_idx is not None:
            # Temporarily set similarity to -1 to exclude from results
            similarities[query_idx] = -1
        
        # Get indices of top k+1 results (in case we need to exclude query)
        top_indices = np.argpartition(similarities, -(k+1))[-(k+1):]
        # Sort just these top indices by similarity
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        # Take only k results
        top_k = top_indices[:k]
    else:
        # For smaller datasets, use the original sort-based approach
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
    logger.debug(f"Prepared results in {time.time() - results_start_time:.2f} seconds")
            
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

if __name__ == "__main__":
    app()
