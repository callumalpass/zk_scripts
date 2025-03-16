"""
Incremental Fast ZK indexer with Backlinks, Citation Extraction,
and optional Embedding Generation.
"""

import os
import re
import sys
import json
import argparse
import logging
import time
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Additional import for embedding generation
import openai

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.utils import (
    json_ready, scandir_recursive, extract_frontmatter_and_body,
    extract_wikilinks_filtered, calculate_word_count, extract_citations,
    load_json_file, save_json_file
)

# Default worker processes
DEFAULT_WORKERS = 64

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_embedding(text: str, model: str = "text-embedding-3-small", max_retries: int = 5, quiet: bool = False) -> List[float]:
    """Get OpenAI embedding for text."""
    for attempt in range(max_retries):
        try:
            result = openai.embeddings.create(input=text, model=model)
            return result.data[0].embedding
        except Exception as e:
            if not quiet:
                logger.debug(f"Error fetching embedding (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    raise Exception("Failed to fetch embedding after multiple attempts.")


def process_markdown_file(filepath: str,
                          fd_exclude_patterns: List[str],
                          notes_dir: str,
                          generate_embeddings: bool = False,
                          embedding_model: str = "text-embedding-3-small",
                          quiet: bool = False) -> Optional[Dict[str, Any]]:
    """Process a markdown file and extract data for indexing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return None
    note_id = os.path.splitext(os.path.relpath(filepath, notes_dir))[0]
    meta, body = extract_frontmatter_and_body(content)
    references = extract_citations(body)
    outgoing_links = extract_wikilinks_filtered(body)
    word_count = calculate_word_count(body)
    file_size = os.path.getsize(filepath)
    result: Dict[str, Any] = {
        "filename": note_id,
        "outgoing_links": outgoing_links,
        "body": body,
        "word_count": word_count,
        "file_size": file_size,
        "references": references,
    }
    result.update(meta)
    # DATE-CREATED CHANGE:
    if "dateModified" in result and "dateCreated" not in result:
        result["dateCreated"] = result["dateModified"]
    if generate_embeddings:
        if not openai.api_key:
            openai.api_key = os.getenv("OPEN_AI_KEY")
        try:
            embedding = get_embedding(body, model=embedding_model, quiet=quiet)
            result["embedding"] = embedding
        except Exception as e:
            logger.error(f"Failed to fetch embedding for {filepath}: {e}")
    return result


def load_index_state(state_file: str, quiet: bool = False) -> Dict[str, float]:
    """Load index state from file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                if not quiet:
                    logger.debug(f"Loaded state from {state_file}: {state}")
                return state
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading state file '{state_file}': {e}. Starting fresh.")
    return {}


def save_index_state(state_file: str, index_state: Dict[str, float], quiet: bool = False) -> None:
    """Save index state to file."""
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(index_state, f, indent=2)
        if not quiet:
            logger.debug(f"Saved state to {state_file}: {index_state}")
    except OSError as e:
        logger.error(f"Error writing state file '{state_file}': {e}")


def load_existing_index(index_file: str, quiet: bool = False) -> Dict[str, Dict[str, Any]]:
    """Load existing index from file."""
    existing_index: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                for item in json.load(f):
                    existing_index[item['filename']] = item
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error loading index file '{index_file}': {e}. Starting with empty index.")
    return existing_index


def write_updated_index(index_file: str, updated_index_data: List[Dict[str, Any]], quiet: bool = False) -> None:
    """Write updated index to file."""
    try:
        with open(index_file, 'w', encoding='utf-8') as outf:
            json.dump(updated_index_data, outf, indent=2)
        if not quiet:
            logger.info(f"Index file updated at: {index_file}")
    except OSError as e:
        logger.error(f"Error writing index file '{index_file}': {e}")
        sys.exit(1)


def load_embeddings_file(embeddings_file: str, quiet: bool = False) -> Dict[str, Any]:
    """Load embeddings from file."""
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading embeddings file '{embeddings_file}': {e}. Using empty embeddings.")
    return {}


def write_embeddings_file(embeddings_file: str, embeddings_data: Dict[str, Any], quiet: bool = False) -> None:
    """Write embeddings to file."""
    try:
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2)
        if not quiet:
            logger.info(f"Embeddings file updated at: {embeddings_file}")
    except OSError as e:
        logger.error(f"Error writing embeddings file '{embeddings_file}': {e}")
        sys.exit(1)


def remove_index_state_file(state_file: str, verbose: bool, quiet: bool = False) -> None:
    """Remove index state file."""
    if os.path.exists(state_file):
        try:
            os.remove(state_file)
            if verbose and not quiet:
                logger.info(f"Removed state file: {state_file}")
        except OSError as e:
            logger.warning(f"Could not remove state file '{state_file}': {e}")
    elif verbose and not quiet:
        logger.info(f"State file not found, skipping removal: {state_file}")


def main() -> None:
    """Main function for the indexer."""
    parser = argparse.ArgumentParser(
        description="Incremental Fast ZK indexer with Backlinks, Citation Extraction, optional Embedding Generation "
                    "and now full support for both dateModified and dateCreated metadata keys.")
    parser.add_argument("--notes-dir", help="Override notes directory")
    parser.add_argument("--index-file", help="Override index file path")
    parser.add_argument("--config-file", help="Specify config file path")
    parser.add_argument("--full-reindex", "-f", action="store_true", help="Force full reindex, removing state.")
    parser.add_argument("--exclude-patterns", help="Override exclude patterns (as a string; e.g., '-E tmp/ -E templates/')")
    parser.add_argument("--no-exclude", action="store_true", help="Disable all exclude patterns (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Increase verbosity of output.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run quietly, suppressing most output.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                        help=f"Number of worker processes (default: {DEFAULT_WORKERS})")
    parser.add_argument("--generate-embeddings", action="store_true", 
                        help="Generate OpenAI embeddings for each note body")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", 
                        help="OpenAI embedding model to use (default: text-embedding-3-small)")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.CRITICAL)

    config = load_config(args.config_file)

    notes_dir_default = os.path.expanduser("~/notes")
    index_file_default = os.path.join(notes_dir_default, "index.json")
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir = args.notes_dir or get_config_value(config, "notes_dir", notes_dir_default)
    notes_dir = resolve_path(notes_dir)
    
    index_file = args.index_file or get_config_value(config, "zk_index.index_file", index_file_default)
    index_file = resolve_path(index_file)
    
    embeddings_file = os.path.join(os.path.dirname(index_file), "embeddings.json") if args.generate_embeddings else None

    if args.no_exclude:
        exclude_patterns: List[str] = []
    elif args.exclude_patterns:
        matches = re.findall(r'-E\s+([^\s]+)', args.exclude_patterns)
        if matches:
            exclude_patterns = matches
        else:
            exclude_patterns = args.exclude_patterns.split()
    else:
        config_val = get_config_value(config, "zk_index.fd_exclude_patterns", fd_exclude_patterns_default)
        if isinstance(config_val, str):
            matches = re.findall(r'-E\s+([^\s]+)', config_val)
            exclude_patterns = matches if matches else config_val.split()
        else:
            exclude_patterns = config_val
    logger.debug(f"Using exclude patterns: {exclude_patterns}")

    state_file = index_file.replace(".json", "_state.json")
    embedding_state_file = embeddings_file.replace(".json", "_state.json") if embeddings_file else None

    if args.full_reindex:
        remove_index_state_file(state_file, args.verbose, args.quiet)
        if embedding_state_file:
            remove_index_state_file(embedding_state_file, args.verbose, args.quiet)

    previous_index_state: Dict[str, float] = {}
    if not args.full_reindex:
        previous_index_state = load_index_state(state_file, args.quiet)
    current_index_state: Dict[str, float] = {}

    previous_embeddings_state: Dict[str, float] = {}
    if args.generate_embeddings and embedding_state_file:
        previous_embeddings_state = load_index_state(embedding_state_file, args.quiet)

    existing_index = load_existing_index(index_file, args.quiet)
    existing_embeddings: Dict[str, Any] = {}
    if args.generate_embeddings and embeddings_file:
        existing_embeddings = load_embeddings_file(embeddings_file, args.quiet)

    logger.debug("Scanning files")
    markdown_files = [fp for fp in scandir_recursive(notes_dir, exclude_patterns=exclude_patterns, quiet=args.quiet)
                      if fp.lower().endswith(".md")]
    logger.info(f"Found {len(markdown_files)} markdown files.")

    current_note_ids = set()
    for fp in markdown_files:
        note_id = os.path.splitext(os.path.relpath(fp, notes_dir))[0]
        current_note_ids.add(note_id)
        try:
            mod_time = os.path.getmtime(fp)
            current_index_state[note_id] = mod_time
        except OSError as e:
            logger.warning(f"Error accessing file {fp}: {e}")

    previous_note_ids = set(previous_index_state.keys())
    deleted_note_ids = previous_note_ids - current_note_ids
    for note_id in deleted_note_ids:
        if note_id in existing_index:
            del existing_index[note_id]
        if args.generate_embeddings and note_id in existing_embeddings:
            del existing_embeddings[note_id]
        logger.info(f"Note deleted: {note_id}")

    files_to_process = []
    for fp in markdown_files:
        note_id = os.path.splitext(os.path.relpath(fp, notes_dir))[0]
        try:
            mod_time = os.path.getmtime(fp)
            if args.full_reindex or note_id not in previous_index_state or previous_index_state.get(note_id, 0) < mod_time:
                files_to_process.append(fp)
        except OSError as e:
            logger.warning(f"Error accessing file {fp}: {e}")

    logger.info(f"Files to process: {len(files_to_process)}")

    new_embeddings: Dict[str, Any] = {}
    pbar_iterator = files_to_process if args.quiet else tqdm(files_to_process, desc="Processing files")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_markdown_file,
                                     fp,
                                     fd_exclude_patterns=exclude_patterns,
                                     notes_dir=notes_dir,
                                     generate_embeddings=args.generate_embeddings,
                                     embedding_model=args.embedding_model,
                                     quiet=args.quiet): fp
                   for fp in pbar_iterator}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                result = future.result()
                if result:
                    note_id = result["filename"]
                    if args.generate_embeddings and "embedding" in result:
                        new_embeddings[note_id] = result.pop("embedding")
                    existing_index[note_id] = result
                else:
                    logger.debug(f"No result returned for file: {fp}")
            except Exception as e:
                logger.error(f"Error processing file {fp}: {e}")
            if not args.quiet and isinstance(pbar_iterator, tqdm):
                pbar_iterator.update(1)

    bk_pbar_iterator = existing_index.values() if args.quiet else tqdm(existing_index.values(), desc="Calculating backlinks")
    backlink_map: Dict[str, set] = {}
    for note in bk_pbar_iterator:
        for target in note.get("outgoing_links", []):
            if target in existing_index:
                backlink_map.setdefault(target, set()).add(note["filename"])
        if not args.quiet and isinstance(bk_pbar_iterator, tqdm):
            bk_pbar_iterator.update(1)
    for note_id, note in existing_index.items():
        note["backlinks"] = sorted(list(backlink_map.get(note_id, [])))

    updated_index_data = list(existing_index.values())
    write_updated_index(index_file, updated_index_data, args.quiet)
    save_index_state(state_file, current_index_state, args.quiet)
    if args.generate_embeddings and embeddings_file:
        existing_embeddings.update(new_embeddings)
        for key in list(existing_embeddings.keys()):
            if key not in existing_index:
                del existing_embeddings[key]
        write_embeddings_file(embeddings_file, existing_embeddings, args.quiet)
        if embedding_state_file:
            save_index_state(embedding_state_file, current_index_state, args.quiet)


if __name__ == "__main__":
    main()