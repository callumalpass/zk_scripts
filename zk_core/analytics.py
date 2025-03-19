"""
Analytics module for Zettelkasten notes.

This module provides functionality to analyze and generate statistics about Zettelkasten notes.
"""

import datetime
import logging
import calendar
import numpy as np
from typing import List, Dict, Any, Optional, Set, Counter as CounterType
from collections import Counter

from zk_core.models import Note, IndexInfo

logger = logging.getLogger(__name__)


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


def get_index_info(notes: List[Note]) -> IndexInfo:
    """Get detailed information about the index."""
    info = IndexInfo()
    try:
        note_count = len(notes)
        info.note_count = note_count

        # Collect tag and word/file size info
        all_tags: List[str] = []
        tag_counter = Counter()
        total_word_count = 0
        total_file_size = 0
        word_counts = []
        file_sizes = []
        tag_counts = []  # track number of tags per note
        tag_pairs = []  # track co-occurring tags
        orphan_count = 0
        untagged_orphan_count = 0
        dates = []
        creation_dates = []
        extra_fields_counter = Counter()

        # Counters for creation dates
        creation_day_counter = Counter()
        creation_year_counter = Counter()
        creation_month_counter = Counter()
        
        # Network and connectivity metrics
        total_links = 0
        outgoing_links_counts = []
        backlinks_counts = []
        highly_connected_notes = []
        citation_rich_notes = []
        
        # Reference and alias tracking
        reference_counts = []
        alias_counts = []
        
        # Word count distribution buckets
        word_count_dist = {
            "tiny (1-100)": 0,
            "small (101-250)": 0,
            "medium (251-500)": 0,
            "large (501-1000)": 0,
            "huge (1001+)": 0
        }
        
        # Collect nodes and edges for network analysis
        note_connections: Dict[str, List[str]] = {}  # Maps note filename to list of connected note filenames
        
        # Tracking for best/worst notes
        notes_with_metadata = []
        
        # Month names for readable output
        month_names = list(calendar.month_name)[1:]  # Skip empty first item
        today = datetime.datetime.now().date()
        
        # Word count by month
        word_count_by_month: CounterType[str] = Counter()

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
            min_date = min(creation_dates)
            max_date = max(creation_dates)
            info.date_range = f"{min_date} to {max_date}"
            
        # Calculate dangling links
        dangling_links_count = 0
        indexed = {note.filename for note in notes}
        for note in notes:
            for target in note.outgoing_links:
                base_target = target.split('#')[0]
                if not (base_target.startswith("biblib/") and base_target.lower().endswith(".pdf")):
                    if target not in indexed:
                        dangling_links_count += 1
                        
        info.dangling_links_count = dangling_links_count
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
        info.highly_connected_notes = sorted(highly_connected_notes, key=lambda x: x[1], reverse=True)[:10]
        
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
        info.bridge_notes = sorted(bridge_candidates, key=lambda x: x[1], reverse=True)[:10]
        
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
            info.tag_clusters = tag_clusters[:10]  # Top 5 clusters
        
        # Reference and alias stats
        info.total_references = sum(reference_counts)
        info.average_references = info.total_references / note_count if note_count else 0
        info.total_aliases = sum(alias_counts)
        info.average_aliases = info.total_aliases / note_count if note_count else 0
        info.citation_hubs = sorted(citation_rich_notes, key=lambda x: x[1], reverse=True)[:10]
        
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
                                                reverse=True)[:10]]
            
            info.shortest_notes = [(n["filename"], n["word_count"]) 
                                 for n in sorted(notes_with_metadata, 
                                                 key=lambda x: x["word_count"])[:10] 
                                 if n["word_count"] > 0]  # Exclude empty notes
            
            # Find newest and oldest notes
            dated_notes = [n for n in notes_with_metadata if n["created"]]
            if dated_notes:
                info.newest_notes = [(n["filename"], str(n["created"])) 
                                    for n in sorted(dated_notes, 
                                                  key=lambda x: x["created"], 
                                                  reverse=True)[:10]]
                
                info.oldest_notes = [(n["filename"], str(n["created"])) 
                                    for n in sorted(dated_notes, 
                                                  key=lambda x: x["created"])[:10]]
                
                # Find untouched notes (not modified in a long time)
                untouched_threshold = today - datetime.timedelta(days=365)  # 1 year
                untouched = [(n["filename"], str(n["modified"])) 
                            for n in notes_with_metadata 
                            if n["modified"] and n["modified"] < untouched_threshold]
                info.untouched_notes = sorted(untouched, key=lambda x: x[1])[:10]
        
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
