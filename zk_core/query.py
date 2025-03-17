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
import calendar
import csv
import numpy as np
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from collections import Counter
import time

import typer
import yaml
import openai
from tabulate import tabulate

from zk_core.models import Note, IndexInfo
from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR, DEFAULT_INDEX_FILENAME
from zk_core.utils import load_json_file, save_json_file

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

def parse_iso_datetime(dt_str: str) -> Optional[datetime.datetime]:
    """Parse ISO format datetime string."""
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
        ctx.obj['notes_cache'][cache_key] = load_index_data(index_file)
    return ctx.obj['notes_cache'][cache_key]

# --- Filtering Functions ---

def filter_by_tag(notes: List[Note], tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
    """Filter notes by tag inclusion and exclusion."""
    filtered_notes = []
    filter_tags = set(tags)
    exclude_set = set(exclude_tags) if exclude_tags else set()
    for note in notes:
        note_tags = set(note.tags)
        if tag_mode == 'or':
            include = any(any(nt == ft or nt.startswith(ft + '/') for nt in note_tags) for ft in filter_tags)
        elif tag_mode == 'and':
            include = all(any(nt == ft or nt.startswith(ft + '/') for nt in note_tags) for ft in filter_tags)
        else:
            logger.error(f"Invalid tag_mode: '{tag_mode}'. Defaulting to 'and'.")
            include = all(any(nt == ft or nt.startswith(ft + '/') for nt in note_tags) for ft in filter_tags)
        if include and not exclude_set.intersection(note_tags):
            filtered_notes.append(note)
    return filtered_notes

def filter_by_filenames_stdin(notes: List[Note]) -> List[Note]:
    """Filter notes by filenames from standard input."""
    filenames = [line.strip() for line in sys.stdin if line.strip()]
    if not filenames:
        typer.echo("Usage: ls <filename list> | script --stdin", err=True)
        raise typer.Exit(1)
    return [note for note in notes if note.filename in filenames]

def filter_by_filename_contains(notes: List[Note], substring: str) -> List[Note]:
    """Filter notes by substring in filename."""
    return [note for note in notes if substring in note.filename]

def filter_by_date_range(notes: List[Note], start_date_str: Optional[str] = None, end_date_str: Optional[str] = None) -> List[Note]:
    """Filter notes by date range."""
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date() if start_date_str else None
        end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date() if end_date_str else None
    except ValueError:
        logger.error("Invalid date format (expected YYYY-MM-DD).")
        return []
    filtered = []
    for note in notes:
        # Default filtering by dateModified:
        if note.dateModified:
            note_dt = parse_iso_datetime(note.dateModified)
            if not note_dt:
                logger.warning(f"Could not parse dateModified for '{note.filename}'")
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

def find_dangling_links(notes: List[Note]) -> Dict[str, List[str]]:
    """Find links to non-existent notes."""
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
    for note in notes:
        writer.writerow([get_field_value(note, field) for field in fields])
    return out.getvalue()

def format_table(notes: List[Note], fields: List[str], use_color: bool = False) -> str:
    """Format notes as a table."""
    table_data = [[get_field_value(note, field) for field in fields] for note in notes]
    headers = fields.copy()
    if use_color:
        headers = [colorize(h, 'cyan') for h in headers]
    return tabulate(table_data, headers=headers, tablefmt="grid")

def format_json(notes: List[Note]) -> str:
    """Format notes as JSON."""
    notes_list = []
    for note in notes:
        notes_list.append(note.to_dict())
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

# --- Sorting ---

def sort_notes(notes: List[Note], sort_by: str = 'dateModified') -> List[Note]:
    """Sort notes by a field."""
    sort_options = {
        'filename': (lambda n: n.filename, False),
        'title': (lambda n: n.title, False),
        'dateModified': (lambda n: parse_iso_datetime(n.dateModified) or datetime.datetime.min, True),
        'word_count': (lambda n: n.word_count, True),
        'file_size': (lambda n: n.file_size, True),
        'dateCreated': (lambda n: parse_iso_datetime(n.dateCreated) or datetime.datetime.min, True)
    }
    if sort_by in sort_options:
        key_func, reverse = sort_options[sort_by]
        
        # Month names for readable output
        month_names = list(calendar.month_name)[1:]  # Skip empty first item
        today = datetime.datetime.now().date()
        
        # Word count by month
        word_count_by_month = Counter()

        for note in notes:
            # Collect metadata for sorting later
            note_meta = {
                "filename": note.filename,
                "word_count": note.word_count,
                "outgoing_links": len(note.outgoing_links),
                "backlinks": len(note.backlinks),
                "references": len(note.references),
                "created": None,
                "modified": None,
                "tags": len(note.tags),
                "connections": len(note.outgoing_links) + len(note.backlinks)
            }
            
            # Count tags per note
            tag_count = len(note.tags)
            tag_counts.append(tag_count)
            all_tags.extend(note.tags)
            tag_counter.update(note.tags)
            
            # Track tag co-occurrences
            if len(note.tags) >= 2:
                for i, tag1 in enumerate(note.tags):
                    for tag2 in note.tags[i+1:]:
                        tag_pairs.append((tag1, tag2))
            
            # Extra frontmatter keys
            for key in note._extra_fields.keys():
                extra_fields_counter[key] += 1
                
            # Word count and file size
            total_word_count += note.word_count
            total_file_size += note.file_size
            word_counts.append(note.word_count)
            file_sizes.append(note.file_size)
            
            # Word count distribution
            wc = note.word_count
            if wc <= 100:
                word_count_dist["tiny (1-100)"] += 1
            elif wc <= 250:
                word_count_dist["small (101-250)"] += 1
            elif wc <= 500:
                word_count_dist["medium (251-500)"] += 1
            elif wc <= 1000:
                word_count_dist["large (501-1000)"] += 1
            else:
                word_count_dist["huge (1001+)"] += 1
                
            # Modification date
            if note.dateModified:
                dt = parse_iso_datetime(note.dateModified)
                if dt:
                    dates.append(dt.date())
                    note_meta["modified"] = dt.date()
                else:
                    logger.warning(f"Could not parse dateModified for '{note.filename}'")
            
            # Creation date
            dt_created = parse_iso_datetime(note.dateCreated)
            if dt_created:
                creation_dates.append(dt_created.date())
                note_meta["created"] = dt_created.date()
                
                # Add words to month counter
                month_key = f"{dt_created.year}-{dt_created.month:02d}"
                word_count_by_month[month_key] += note.word_count
            elif note.dateModified:
                dt_mod = parse_iso_datetime(note.dateModified)
                if dt_mod:
                    note_meta["created"] = dt_mod.date()
                    
            # For orphan metrics
            if not note.outgoing_links and not note.backlinks:
                orphan_count += 1
                if not note.tags:
                    untagged_orphan_count += 1
            
            # Track network metrics
            outgoing_count = len(note.outgoing_links)
            backlink_count = len(note.backlinks)
            total_links += outgoing_count
            outgoing_links_counts.append(outgoing_count)
            backlinks_counts.append(backlink_count)
            
            # Store connections for network analysis
            note_connections[note.filename] = note.outgoing_links
            
            # Track highly connected notes
            total_connections = outgoing_count + backlink_count
            if total_connections > 10:  # arbitrary threshold for "highly connected"
                highly_connected_notes.append((note.filename, total_connections))
            
            # Track references and aliases
            ref_count = len(note.references)
            reference_counts.append(ref_count)
            alias_counts.append(len(note.aliases))
            
            # Find citation-rich notes
            if ref_count > 5:  # arbitrary threshold
                citation_rich_notes.append((note.filename, ref_count))

            # Determine creation date for statistics
            dt_created = parse_iso_datetime(note.dateCreated)
            if not dt_created:
                dt_created = parse_iso_datetime(note.dateModified)
            if dt_created:
                creation_day_counter[dt_created.strftime("%A")] += 1
                creation_year_counter[dt_created.year] += 1
                creation_month_counter[dt_created.month] += 1
            
            # Add to metadata collection
            notes_with_metadata.append(note_meta)
            
        # Calculate statistics
        info.total_word_count = total_word_count
        info.average_word_count = total_word_count / note_count if note_count else 0
        info.median_word_count = float(np.median(word_counts)) if word_counts else 0
        info.min_word_count = min(word_counts) if word_counts else 0
        info.max_word_count = max(word_counts) if word_counts else 0
        
        info.total_file_size_bytes = total_file_size
        info.average_file_size_bytes = total_file_size / note_count if note_count else 0
        info.median_file_size_bytes = float(np.median(file_sizes)) if file_sizes else 0
        info.min_file_size_bytes = min(file_sizes) if file_sizes else 0
        info.max_file_size_bytes = max(file_sizes) if file_sizes else 0
        
        info.average_tags_per_note = sum(tag_counts) / note_count if note_count else 0
        info.median_tags_per_note = float(np.median(tag_counts)) if tag_counts else 0
        
        info.orphan_notes_count = orphan_count
        info.untagged_orphan_notes_count = untagged_orphan_count
        
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            info.date_range = f"{min_date} to {max_date}"
            
        info.dangling_links_count = sum(len(links) for links in find_dangling_links(notes).values())
        info.unique_tag_count = len(tag_counter.keys())
        info.most_common_tags = tag_counter.most_common(10)
        
        info.extra_frontmatter_keys = list(extra_fields_counter.most_common())
        
        # Day of week distribution
        days_order = list(calendar.day_name)
        info.notes_by_day_of_week = {day: creation_day_counter.get(day, 0) for day in days_order}
        
        # Peak creation day
        if creation_day_counter:
            peak_day, peak_count = creation_day_counter.most_common(1)[0]
            info.peak_creation_day = f"{peak_day} ({peak_count} notes)"
            
        # Distribution by year
        info.notes_by_year = dict(sorted(creation_year_counter.items()))
        
        # Network metrics
        info.total_links = total_links
        info.average_outgoing_links = sum(outgoing_links_counts) / note_count if note_count else 0
        info.median_outgoing_links = float(np.median(outgoing_links_counts)) if outgoing_links_counts else 0
        info.average_backlinks = sum(backlinks_counts) / note_count if note_count else 0
        info.median_backlinks = float(np.median(backlinks_counts)) if backlinks_counts else 0
        info.highly_connected_notes = sorted(highly_connected_notes, key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate network density (ratio of actual connections to possible connections)
        possible_connections = note_count * (note_count - 1)
        if possible_connections > 0:
            info.network_density = total_links / possible_connections
        
        # Identify bridge notes - approximate with notes having both incoming and outgoing links to different sets of notes
        bridge_candidates = []
        for note in notes:
            # Check if this note connects different parts of the network
            if note.outgoing_links and note.backlinks:
                # Calculate the overlap between outgoing links and backlinks
                outgoing_set = set(note.outgoing_links)
                backlink_set = set(note.backlinks)
                # If there's little overlap and sufficient links, consider it a bridge
                overlap = len(outgoing_set.intersection(backlink_set))
                if overlap < 2 and len(outgoing_set) + len(backlink_set) > 5:
                    bridge_candidates.append((note.filename, len(outgoing_set) + len(backlink_set)))
        info.bridge_notes = sorted(bridge_candidates, key=lambda x: x[1], reverse=True)[:5]
        
        # Analysis of tag co-occurrence
        tag_co_occurrence = Counter(tag_pairs)
        info.tag_co_occurrence = {f"{tag1} & {tag2}": count 
                                 for (tag1, tag2), count in tag_co_occurrence.most_common(10)}
        
        # Tag clusters (simplified approximation based on co-occurrence)
        common_pairs = tag_co_occurrence.most_common(20)
        if common_pairs:
            freq_tags = set()
            for (tag1, tag2), _ in common_pairs:
                freq_tags.add(tag1)
                freq_tags.add(tag2)
            top_tags = list(freq_tags)[:10]  # Use top 10 tags for clusters
            tag_clusters = []
            for i, tag in enumerate(top_tags):
                related = []
                for pair, count in common_pairs:
                    if tag in pair:
                        other = pair[0] if pair[1] == tag else pair[1]
                        related.append(other)
                if related:
                    tag_clusters.append((tag, related[:3]))  # Top 3 related tags
            info.tag_clusters = tag_clusters[:5]  # Top 5 clusters
        
        # Reference and alias stats
        info.total_references = sum(reference_counts)
        info.average_references = info.total_references / note_count if note_count else 0
        info.total_aliases = sum(alias_counts)
        info.average_aliases = info.total_aliases / note_count if note_count else 0
        info.citation_hubs = sorted(citation_rich_notes, key=lambda x: x[1], reverse=True)[:5]
        
        # Monthly creation patterns
        info.notes_by_month = {month_names[month-1]: creation_month_counter.get(month, 0) 
                              for month in range(1, 13)}
        if creation_month_counter:
            peak_month = max(creation_month_counter.items(), key=lambda x: x[1])
            info.peak_creation_month = f"{month_names[peak_month[0]-1]} ({peak_month[1]} notes)"
        
        # Writing velocity metrics
        if creation_dates:
            # Calculate days covered
            first_date = min(creation_dates)
            last_date = max(creation_dates)
            days_covered = (last_date - first_date).days + 1
            if days_covered > 0:
                info.writing_velocity = total_word_count / days_covered
                
            # Calculate growth rate per year
            for year, count in creation_year_counter.items():
                year_days = 365 if year % 4 != 0 else 366
                info.growth_rate[str(year)] = count / year_days
                
        # Word count by month (formatted)
        sorted_months = sorted(word_count_by_month.items())
        info.word_count_by_month = {month_key: count for month_key, count in sorted_months[-12:]}  # Last 12 months
        
        # Identify most productive periods
        if sorted_months:
            productive_periods = sorted(word_count_by_month.items(), key=lambda x: x[1], reverse=True)
            info.most_productive_periods = [(period, count) for period, count in productive_periods[:3]]
            
        # Word count distribution
        info.word_count_distribution = word_count_dist
        
        # Find longest and shortest notes
        if notes_with_metadata:
            info.longest_notes = [(n["filename"], n["word_count"]) 
                                for n in sorted(notes_with_metadata, 
                                                key=lambda x: x["word_count"], 
                                                reverse=True)[:5]]
            
            info.shortest_notes = [(n["filename"], n["word_count"]) 
                                 for n in sorted(notes_with_metadata, 
                                                 key=lambda x: x["word_count"])[:5] 
                                 if n["word_count"] > 0]  # Exclude empty notes
            
            # Find newest and oldest notes
            dated_notes = [n for n in notes_with_metadata if n["created"]]
            if dated_notes:
                info.newest_notes = [(n["filename"], str(n["created"])) 
                                    for n in sorted(dated_notes, 
                                                   key=lambda x: x["created"], 
                                                   reverse=True)[:5]]
                
                info.oldest_notes = [(n["filename"], str(n["created"])) 
                                    for n in sorted(dated_notes, 
                                                   key=lambda x: x["created"])[:5]]
                
                # Find untouched notes (not modified in a long time)
                untouched_threshold = today - datetime.timedelta(days=365)  # 1 year
                untouched = [(n["filename"], str(n["modified"])) 
                            for n in notes_with_metadata 
                            if n["modified"] and n["modified"] < untouched_threshold]
                info.untouched_notes = sorted(untouched, key=lambda x: x[1])[:5]
        
        # Calculate content age distribution (percentage of content by age)
        if creation_dates:
            age_buckets = {
                "< 1 month old": 0,
                "1-3 months old": 0,
                "3-6 months old": 0,
                "6-12 months old": 0,
                "1-2 years old": 0,
                "> 2 years old": 0
            }
            
            for dt in creation_dates:
                age_days = (today - dt).days
                if age_days < 30:
                    age_buckets["< 1 month old"] += 1
                elif age_days < 90:
                    age_buckets["1-3 months old"] += 1
                elif age_days < 180:
                    age_buckets["3-6 months old"] += 1
                elif age_days < 365:
                    age_buckets["6-12 months old"] += 1
                elif age_days < 730:
                    age_buckets["1-2 years old"] += 1
                else:
                    age_buckets["> 2 years old"] += 1
                    
            # Convert to percentages
            info.content_age = {k: (v / len(creation_dates) * 100) for k, v in age_buckets.items()}
    
    except Exception as e:
        logger.exception(f"Error gathering index info: {e}")
        
    return info

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
    
    notes = get_cached_notes(ctx, index_file)
    info_data = get_index_info(index_file, notes)
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
                lines.append(f"    - {note_filename}: {ref_count} references")
    
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
                lines.append(f"    - {note_filename}: {connection_count} connections")
        else:
            lines.append("  No highly connected notes found")
            
        # Bridge notes section
        if hasattr(info_data, 'bridge_notes') and info_data.bridge_notes:
            lines.append("  Bridge notes (connect different clusters):")
            for note_filename, connection_count in info_data.bridge_notes:
                lines.append(f"    - {note_filename}: {connection_count} connections")
    
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
                lines.append(f"    - {filename}: {word_count:,} words")
                
        # Shortest notes (non-empty)
        if hasattr(info_data, 'shortest_notes') and info_data.shortest_notes:
            lines.append("  Shortest notes (word count):")
            for filename, word_count in info_data.shortest_notes:
                lines.append(f"    - {filename}: {word_count} words")
                
        # Newest notes
        if hasattr(info_data, 'newest_notes') and info_data.newest_notes:
            lines.append("  Most recently created notes:")
            for filename, date_str in info_data.newest_notes:
                lines.append(f"    - {filename}: {date_str}")
                
        # Oldest notes
        if hasattr(info_data, 'oldest_notes') and info_data.oldest_notes:
            lines.append("  Oldest notes:")
            for filename, date_str in info_data.oldest_notes:
                lines.append(f"    - {filename}: {date_str}")
                
        # Untouched notes
        if hasattr(info_data, 'untouched_notes') and info_data.untouched_notes:
            lines.append("  Notes not modified in over a year:")
            for filename, date_str in info_data.untouched_notes:
                lines.append(f"    - {filename}: last modified {date_str}")
    
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
    else:
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
    notes = get_cached_notes(ctx, index_file)
    if not notes:
        raise typer.Exit("No notes loaded from index.")

    # Build a mapping: note filename → Note object
    notes_dict: Dict[str, Note] = {n.filename: n for n in notes}

    # If a query string is provided, use it. Otherwise, use the query_file.
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
    
    # Find top k indices; if the query is from the index, exclude it.
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

if __name__ == "__main__":
    app()