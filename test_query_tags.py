"""
Test script for the query module tag filtering functionality.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from zk_core.query import filter_by_tag
from zk_core.models import Note

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create sample notes with tags
def create_sample_notes():
    notes = []
    
    # Note 1: Programming/Python
    notes.append(Note.from_dict({
        "filename": "note1.md",
        "title": "Python Basics",
        "tags": ["programming/python", "tutorial"],
        "outgoing_links": [],
        "backlinks": []
    }))
    
    # Note 2: Programming/JavaScript
    notes.append(Note.from_dict({
        "filename": "note2.md",
        "title": "JavaScript Basics",
        "tags": ["programming/javascript", "tutorial"],
        "outgoing_links": [],
        "backlinks": []
    }))
    
    # Note 3: Programming (parent tag)
    notes.append(Note.from_dict({
        "filename": "note3.md",
        "title": "Programming Overview",
        "tags": ["programming", "overview"],
        "outgoing_links": [],
        "backlinks": []
    }))
    
    # Note 4: No programming tags
    notes.append(Note.from_dict({
        "filename": "note4.md",
        "title": "Design Patterns",
        "tags": ["design", "architecture"],
        "outgoing_links": [],
        "backlinks": []
    }))
    
    return notes

def test_regular_filtering():
    """Test the regular filter_by_tag function."""
    notes = create_sample_notes()
    
    # Test AND mode (default)
    filtered = filter_by_tag(notes, ["programming", "tutorial"], "and")
    logger.info(f"AND mode (programming + tutorial): {[n.filename for n in filtered]}")
    
    # Test OR mode
    filtered = filter_by_tag(notes, ["programming", "design"], "or")
    logger.info(f"OR mode (programming OR design): {[n.filename for n in filtered]}")
    
    # Test with exclusions
    filtered = filter_by_tag(notes, ["tutorial"], "and", ["programming/javascript"])
    logger.info(f"Exclude 'programming/javascript': {[n.filename for n in filtered]}")
    
    # Test hierarchical tags in filter_by_tag
    filtered = filter_by_tag(notes, ["programming"], "and")
    logger.info(f"Hierarchical 'programming': {[n.filename for n in filtered]}")
    
    # Test with no filtering
    filtered = filter_by_tag(notes, [], "and")
    logger.info(f"No tags: {[n.filename for n in filtered]}")
    
    # Test with empty tag list but exclusions
    filtered = filter_by_tag(notes, [], "and", ["tutorial"])
    logger.info(f"No tags but exclude 'tutorial': {[n.filename for n in filtered]}")

def test_simulated_optimized_filtering():
    """Simulate the optimized filter_by_tag_optimized function."""
    notes = create_sample_notes()
    
    # In the actual code, Notes are already hashable
    # For our testing, we'll use filenames as a proxy for simplicity
    tag_index = {}
    for note in notes:
        for tag in note.tags:
            if tag not in tag_index:
                tag_index[tag] = []
            tag_index[tag].append(note.filename)
    
    # Create a mapping of filenames to notes for easy lookup
    notes_by_filename = {note.filename: note for note in notes}
    
    # This simulates the filter_by_tag_optimized function, but without the Typer context
    def simulated_filter_by_tag_optimized(notes: List[Note], tags: List[str], tag_mode: str = 'and', exclude_tags: Optional[List[str]] = None) -> List[Note]:
        """A simplified version of filter_by_tag_optimized to test the logic."""
        # If no tags and no exclusions, return all notes
        if not tags and not exclude_tags:
            return notes
        
        # Get filenames for set operations
        note_filenames = set(note.filename for note in notes)
        
        # Use set operations for better performance
        filter_tags = set(tags)
        exclude_set = set(exclude_tags) if exclude_tags else set()
        
        # Special case: only exclusions, no inclusion filters
        if not filter_tags and exclude_set:
            all_filenames = note_filenames.copy()
            for ex_tag in exclude_set:
                # Collect all notes with excluded tag (including hierarchical)
                excluded_filenames = set()
                # Direct matches
                if ex_tag in tag_index:
                    excluded_filenames.update(tag_index[ex_tag])
                
                # Hierarchical matches (notes with tags that start with ex_tag/)
                ex_prefix = f"{ex_tag}/"
                for indexed_tag, filenames_with_tag in tag_index.items():
                    if indexed_tag.startswith(ex_prefix):
                        excluded_filenames.update(filenames_with_tag)
                
                # Remove excluded notes from result set
                all_filenames.difference_update(excluded_filenames)
            
            # Convert filenames back to notes
            return [notes_by_filename[filename] for filename in all_filenames]
        
        # Get candidates based on tag mode
        if tag_mode == 'or':
            # Notes that have any of the tags (including hierarchical)
            candidate_filenames = set()
            for tag in filter_tags:
                # Direct tag matches
                if tag in tag_index:
                    candidate_filenames.update(tag_index[tag])
                    
                # Hierarchical matches (include notes with tags that start with tag/)
                tag_prefix = f"{tag}/"
                for indexed_tag, filenames_with_tag in tag_index.items():
                    if indexed_tag.startswith(tag_prefix):
                        candidate_filenames.update(filenames_with_tag)
        else:  # 'and' mode
            if not filter_tags:  # Empty filter tags with 'and' mode
                candidate_filenames = note_filenames.copy()
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
                    for indexed_tag, filenames_with_tag in tag_index.items():
                        if indexed_tag.startswith(tag_prefix):
                            tag_matches.update(filenames_with_tag)
                    
                    tag_matches_by_filter.append(tag_matches)
                
                # In AND mode, notes must match ALL filter criteria
                # Start with all matches for the first filter
                candidate_filenames = tag_matches_by_filter[0] if tag_matches_by_filter else set()
                # Intersect with matches for each additional filter
                for matches in tag_matches_by_filter[1:]:
                    candidate_filenames.intersection_update(matches)
        
        # Exclude tags if necessary
        if exclude_set:
            filtered_notes = []
            for filename in candidate_filenames:
                note = notes_by_filename[filename]
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
            # Convert filenames back to notes
            return [notes_by_filename[filename] for filename in candidate_filenames]
    
    # Test AND mode (default)
    filtered = simulated_filter_by_tag_optimized(notes, ["programming", "tutorial"], "and")
    logger.info(f"SIMULATED AND mode (programming + tutorial): {[n.filename for n in filtered]}")
    
    # Test OR mode
    filtered = simulated_filter_by_tag_optimized(notes, ["programming", "design"], "or")
    logger.info(f"SIMULATED OR mode (programming OR design): {[n.filename for n in filtered]}")
    
    # Test with exclusions
    filtered = simulated_filter_by_tag_optimized(notes, ["tutorial"], "and", ["programming/javascript"])
    logger.info(f"SIMULATED Exclude 'programming/javascript': {[n.filename for n in filtered]}")
    
    # Test with hierarchical tags
    filtered = simulated_filter_by_tag_optimized(notes, ["programming"], "and")
    logger.info(f"SIMULATED Hierarchical 'programming': {[n.filename for n in filtered]}")
    
    # Test with no filtering
    filtered = simulated_filter_by_tag_optimized(notes, [], "and")
    logger.info(f"SIMULATED No tags: {[n.filename for n in filtered]}")
    
    # Test with empty tag list but exclusions
    filtered = simulated_filter_by_tag_optimized(notes, [], "and", ["tutorial"])
    logger.info(f"SIMULATED No tags but exclude 'tutorial': {[n.filename for n in filtered]}")

def run_benchmark():
    """Run a simple benchmark to compare performance."""
    import time
    
    # Create a larger set of notes for benchmarking
    large_notes = []
    for i in range(1000):
        # Use a distribution of tags to simulate real data
        tags = []
        if i % 3 == 0:
            tags.append("programming")
        if i % 5 == 0:
            tags.append("tutorial")
        if i % 7 == 0:
            tags.append("programming/python")
        if i % 11 == 0:
            tags.append("programming/javascript")
        if i % 13 == 0:
            tags.append("design")
        if i % 17 == 0:
            tags.append("architecture")
        
        large_notes.append(Note.from_dict({
            "filename": f"note{i}.md",
            "title": f"Note {i}",
            "tags": tags,
            "outgoing_links": [],
            "backlinks": []
        }))
    
    logger.info(f"Created {len(large_notes)} notes for benchmarking")
    
    # --- Benchmark Regular Filtering ---
    
    # Test AND mode with both tags
    start_time = time.time()
    filtered = filter_by_tag(large_notes, ["programming", "tutorial"], "and")
    regular_time_and = time.time() - start_time
    logger.info(f"Regular filtering AND mode: {len(filtered)} matches in {regular_time_and:.6f} seconds")
    
    # Test OR mode with both tags
    start_time = time.time()
    filtered = filter_by_tag(large_notes, ["programming", "design"], "or")
    regular_time_or = time.time() - start_time
    logger.info(f"Regular filtering OR mode: {len(filtered)} matches in {regular_time_or:.6f} seconds")
    
    # Test hierarchical tags
    start_time = time.time()
    filtered = filter_by_tag(large_notes, ["programming"], "and")
    regular_time_hierarchical = time.time() - start_time
    logger.info(f"Regular filtering hierarchical: {len(filtered)} matches in {regular_time_hierarchical:.6f} seconds")
    
    # Test with exclusions
    start_time = time.time()
    filtered = filter_by_tag(large_notes, ["tutorial"], "and", ["programming/javascript"])
    regular_time_exclusions = time.time() - start_time
    logger.info(f"Regular filtering with exclusions: {len(filtered)} matches in {regular_time_exclusions:.6f} seconds")
    
    # --- Benchmark Simulated Optimized Filtering ---
    
    # Build tag index
    tag_index = {}
    for note in large_notes:
        for tag in note.tags:
            if tag not in tag_index:
                tag_index[tag] = []
            tag_index[tag].append(note.filename)
    
    notes_by_filename = {note.filename: note for note in large_notes}
    
    def simulated_filter_by_tag_optimized(notes, tags, tag_mode='and', exclude_tags=None):
        if not tags and not exclude_tags:
            return notes
        
        note_filenames = set(note.filename for note in notes)
        filter_tags = set(tags)
        exclude_set = set(exclude_tags) if exclude_tags else set()
        
        if not filter_tags and exclude_set:
            all_filenames = note_filenames.copy()
            for ex_tag in exclude_set:
                excluded_filenames = set()
                if ex_tag in tag_index:
                    excluded_filenames.update(tag_index[ex_tag])
                
                ex_prefix = f"{ex_tag}/"
                for indexed_tag, filenames_with_tag in tag_index.items():
                    if indexed_tag.startswith(ex_prefix):
                        excluded_filenames.update(filenames_with_tag)
                
                all_filenames.difference_update(excluded_filenames)
            
            return [notes_by_filename[filename] for filename in all_filenames]
        
        if tag_mode == 'or':
            candidate_filenames = set()
            for tag in filter_tags:
                if tag in tag_index:
                    candidate_filenames.update(tag_index[tag])
                    
                tag_prefix = f"{tag}/"
                for indexed_tag, filenames_with_tag in tag_index.items():
                    if indexed_tag.startswith(tag_prefix):
                        candidate_filenames.update(filenames_with_tag)
        else:
            if not filter_tags:
                candidate_filenames = note_filenames.copy()
            else:
                tag_matches_by_filter = []
                for tag in filter_tags:
                    tag_matches = set()
                    if tag in tag_index:
                        tag_matches.update(tag_index[tag])
                    
                    tag_prefix = f"{tag}/"
                    for indexed_tag, filenames_with_tag in tag_index.items():
                        if indexed_tag.startswith(tag_prefix):
                            tag_matches.update(filenames_with_tag)
                    
                    tag_matches_by_filter.append(tag_matches)
                
                candidate_filenames = tag_matches_by_filter[0] if tag_matches_by_filter else set()
                for matches in tag_matches_by_filter[1:]:
                    candidate_filenames.intersection_update(matches)
        
        if exclude_set:
            filtered_notes = []
            for filename in candidate_filenames:
                note = notes_by_filename[filename]
                exclude_this_note = False
                note_tags = note.tags
                for ex_tag in exclude_set:
                    if ex_tag in note_tags:
                        exclude_this_note = True
                        break
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
            return [notes_by_filename[filename] for filename in candidate_filenames]
    
    # Test AND mode with both tags
    start_time = time.time()
    filtered = simulated_filter_by_tag_optimized(large_notes, ["programming", "tutorial"], "and")
    optimized_time_and = time.time() - start_time
    logger.info(f"Optimized filtering AND mode: {len(filtered)} matches in {optimized_time_and:.6f} seconds")
    
    # Test OR mode with both tags
    start_time = time.time()
    filtered = simulated_filter_by_tag_optimized(large_notes, ["programming", "design"], "or")
    optimized_time_or = time.time() - start_time
    logger.info(f"Optimized filtering OR mode: {len(filtered)} matches in {optimized_time_or:.6f} seconds")
    
    # Test hierarchical tags
    start_time = time.time()
    filtered = simulated_filter_by_tag_optimized(large_notes, ["programming"], "and")
    optimized_time_hierarchical = time.time() - start_time
    logger.info(f"Optimized filtering hierarchical: {len(filtered)} matches in {optimized_time_hierarchical:.6f} seconds")
    
    # Test with exclusions
    start_time = time.time()
    filtered = simulated_filter_by_tag_optimized(large_notes, ["tutorial"], "and", ["programming/javascript"])
    optimized_time_exclusions = time.time() - start_time
    logger.info(f"Optimized filtering with exclusions: {len(filtered)} matches in {optimized_time_exclusions:.6f} seconds")
    
    # Log performance improvements
    logger.info("\nPerformance Improvement:")
    logger.info(f"AND mode: {regular_time_and/optimized_time_and:.2f}x faster")
    logger.info(f"OR mode: {regular_time_or/optimized_time_or:.2f}x faster")
    logger.info(f"Hierarchical: {regular_time_hierarchical/optimized_time_hierarchical:.2f}x faster")
    logger.info(f"With exclusions: {regular_time_exclusions/optimized_time_exclusions:.2f}x faster")

if __name__ == "__main__":
    logger.info("Testing regular tag filtering...")
    test_regular_filtering()
    
    logger.info("\nTesting simulated optimized tag filtering...")
    test_simulated_optimized_filtering()
    
    logger.info("\nRunning performance benchmark...")
    run_benchmark()