"""
Incremental Fast ZK indexer with Backlinks, Citation Extraction,
and optional Embedding Generation.

Usage:
  zk-index run                     Build/update the index
  zk-index run --full-reindex      Force a complete reindex
  zk-index run --generate-embeddings  Generate OpenAI embeddings
  
  zk-index test-api                Test OpenAI API connection
  zk-index validate-embeddings     Check embeddings for all notes
  zk-index regenerate-embeddings   Regenerate all embeddings
  
Environment Variables:
  OPEN_AI_KEY                      Required for embedding generation
"""

import os
import re
import sys
import json
import logging
import time
from datetime import date
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Additional import for embedding generation
import openai
import typer

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.utils import (
    json_ready, scandir_recursive, extract_frontmatter_and_body,
    extract_wikilinks_filtered, calculate_word_count, extract_citations,
    load_json_file, save_json_file
)

# Import constants
from zk_core.constants import (
    DEFAULT_NUM_WORKERS as DEFAULT_WORKERS,
    DEFAULT_NOTES_DIR,
    DEFAULT_INDEX_FILENAME
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer(help="Incremental Fast ZK indexer with Backlinks, Citation Extraction, and Embedding Generation.")


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


def load_index_state(state_file: str, quiet: bool = False) -> Dict[str, Any]:
    """Load index state from file."""
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                if not quiet:
                    logger.debug(f"Loaded state from {state_file}: {state}")
                # Handle legacy format which was just a dict of file mtimes
                if state and isinstance(next(iter(state.values())), (int, float)):
                    return {"files": state, "dirs": {}}
                return state
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error reading state file '{state_file}': {e}. Starting fresh.")
    return {"files": {}, "dirs": {}}


def save_index_state(state_file: str, index_state: Dict[str, Any], quiet: bool = False) -> None:
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


@app.callback()
def callback(ctx: typer.Context,
          config_file: Optional[Path] = typer.Option(None, "--config-file", help="Path to a YAML configuration file.")):
    """Initialize the Typer context with configuration."""
    ctx.obj = {}
    ctx.obj["config"] = load_config(config_file)


@app.command(name="run")
def run_indexer(
    ctx: typer.Context,
    notes_dir: Optional[str] = typer.Option(None, "--notes-dir", help="Override notes directory"),
    index_file: Optional[str] = typer.Option(None, "--index-file", help="Override index file path"),
    full_reindex: bool = typer.Option(False, "--full-reindex", "-f", help="Force full reindex, removing state."),
    exclude_patterns: Optional[str] = typer.Option(None, "--exclude-patterns", help="Override exclude patterns (as a string; e.g., '-E tmp/ -E templates/')"),
    no_exclude: bool = typer.Option(False, "--no-exclude", help="Disable all exclude patterns (for testing)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Increase verbosity of output."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Run quietly, suppressing most output."),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", help=f"Number of worker processes (default: {DEFAULT_WORKERS})"),
    generate_embeddings: bool = typer.Option(False, "--generate-embeddings", help="Generate OpenAI embeddings for each note body"),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model", help="OpenAI embedding model to use (default: text-embedding-3-small)"),
) -> None:
    """Run the Zettelkasten indexer to create or update the index file."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    elif quiet:
        logger.setLevel(logging.CRITICAL)

    config = ctx.obj.get("config", {})

    notes_dir_default = DEFAULT_NOTES_DIR
    index_file_default = os.path.join(notes_dir_default, DEFAULT_INDEX_FILENAME)
    fd_exclude_patterns_default = ["templates/", ".zk/"]

    notes_dir_path = notes_dir or get_config_value(config, "notes_dir", notes_dir_default)
    notes_dir_path = resolve_path(notes_dir_path)
    
    index_file_path = index_file or get_config_value(config, "zk_index.index_file", index_file_default)
    index_file_path = resolve_path(index_file_path)
    
    embeddings_file = os.path.join(os.path.dirname(index_file_path), "embeddings.json") if generate_embeddings else None
    
    # Check for OpenAI API key if generating embeddings
    if generate_embeddings:
        api_key = os.getenv("OPEN_AI_KEY")
        if not api_key:
            typer.echo("Error: OPEN_AI_KEY environment variable not set but --generate-embeddings was specified!")
            typer.echo("Set it with: export OPEN_AI_KEY=your_api_key")
            raise typer.Exit(1)
        openai.api_key = api_key
        typer.echo(f"Will generate embeddings using model: {embedding_model}")

    if no_exclude:
        exclude_patterns_list: List[str] = []
    elif exclude_patterns:
        matches = re.findall(r'-E\s+([^\s]+)', exclude_patterns)
        if matches:
            exclude_patterns_list = matches
        else:
            exclude_patterns_list = exclude_patterns.split()
    else:
        config_val = get_config_value(config, "zk_index.exclude_patterns", fd_exclude_patterns_default)
        if isinstance(config_val, str):
            matches = re.findall(r'-E\s+([^\s]+)', config_val)
            exclude_patterns_list = matches if matches else config_val.split()
        else:
            exclude_patterns_list = config_val
    logger.debug(f"Using exclude patterns: {exclude_patterns_list}")

    # Use XDG cache directory for state files
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "zk_scripts")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a unique and stable state file name based on the notes directory path
    # Use a more stable identifier than hash() which can vary between Python runs
    notes_dir_id = notes_dir_path.replace("/", "_").replace(" ", "_").replace(".", "_")
    if not quiet:
        logger.debug(f"Using notes directory ID: {notes_dir_id}")
    state_file = os.path.join(cache_dir, f"index_state_{notes_dir_id}.json")
    embedding_state_file = os.path.join(cache_dir, f"embedding_state_{notes_dir_id}.json") if embeddings_file else None
    
    if not quiet:
        logger.debug(f"Using state file: {state_file}")
    
    if full_reindex:
        remove_index_state_file(state_file, verbose, quiet)
        if embedding_state_file:
            remove_index_state_file(embedding_state_file, verbose, quiet)

    previous_index_state: Dict[str, Any] = {}
    if not full_reindex:
        previous_index_state = load_index_state(state_file, quiet)
    current_index_state: Dict[str, Any] = {}

    previous_embeddings_state: Dict[str, Any] = {}
    if generate_embeddings and embedding_state_file:
        previous_embeddings_state = load_index_state(embedding_state_file, quiet)

    existing_index = load_existing_index(index_file_path, quiet)
    existing_embeddings: Dict[str, Any] = {}
    if generate_embeddings and embeddings_file:
        existing_embeddings = load_embeddings_file(embeddings_file, quiet)

    logger.debug("Scanning files")
    
    # Initialize new index state structure
    current_index_state = {"files": {}, "dirs": {}}
    
    # Get directory modification times from previous run
    dir_mtimes = previous_index_state.get("dirs", {})
    
    # Keep track of which directories were skipped (unchanged)
    skipped_dirs = []
    
    # Use enhanced scandir_recursive with directory mtimes for efficiency
    markdown_files = [fp for fp in scandir_recursive(
        notes_dir_path, 
        exclude_patterns=exclude_patterns_list, 
        quiet=quiet,
        dir_mtimes=dir_mtimes,
        notes_dir=notes_dir_path,
        skipped_dirs=skipped_dirs
    ) if fp.lower().endswith(".md")]
    
    if not quiet and skipped_dirs:
        logger.info(f"Skipped {len(skipped_dirs)} unchanged directories")
    
    logger.info(f"Found {len(markdown_files)} markdown files.")

    # Add the root directory to the list of directories to track
    try:
        root_mtime = os.path.getmtime(notes_dir_path)
        current_index_state["dirs"]["."] = root_mtime  # Use "." to represent the root directory
    except OSError as e:
        logger.warning(f"Error getting mtime for root directory {notes_dir_path}: {e}")
        
    # Map to track directories with updated files
    updated_dirs = set()
    current_note_ids = set()
    
    # Add files from scanned directories
    for fp in markdown_files:
        note_id = os.path.splitext(os.path.relpath(fp, notes_dir_path))[0]
        current_note_ids.add(note_id)
        try:
            mod_time = os.path.getmtime(fp)
            current_index_state["files"][note_id] = mod_time
            
            # Track which directory this file is in
            dir_path = os.path.dirname(os.path.relpath(fp, notes_dir_path))
            updated_dirs.add(dir_path)
        except OSError as e:
            logger.warning(f"Error accessing file {fp}: {e}")
    
    # We're no longer relying on directory skipping since it was causing issues
    # with file change detection. This means we're checking every individual file again.
    # Let's keep this section empty for now. If needed, we can add more robust directory 
    # caching in the future, but the current implementation will scan all files
    # and rely on file-level modification times, which is more reliable.
    
    # Process directory modification times for caching
    for dir_path in updated_dirs:
        if not dir_path:  # Skip empty dir path (files in root are already tracked)
            continue
            
        abs_dir_path = os.path.join(notes_dir_path, dir_path)
        try:
            if os.path.exists(abs_dir_path):
                dir_mtime = os.path.getmtime(abs_dir_path)
                current_index_state["dirs"][dir_path] = dir_mtime
        except OSError as e:
            logger.warning(f"Error getting mtime for directory {abs_dir_path}: {e}")
    
    # Still record directory mtimes for potential future optimizations
    # even though we're not currently using them for skipping
    
    previous_note_ids = set(previous_index_state.get("files", {}).keys())
    deleted_note_ids = previous_note_ids - current_note_ids
    for note_id in deleted_note_ids:
        if note_id in existing_index:
            del existing_index[note_id]
        if generate_embeddings and note_id in existing_embeddings:
            del existing_embeddings[note_id]
        logger.info(f"Note deleted: {note_id}")

    files_to_process = []
    
    # Check modified files from the scan
    for fp in markdown_files:
        note_id = os.path.splitext(os.path.relpath(fp, notes_dir_path))[0]
        try:
            mod_time = os.path.getmtime(fp)
            prev_files = previous_index_state.get("files", {})
            if full_reindex or note_id not in prev_files or prev_files.get(note_id, 0) < mod_time:
                files_to_process.append(fp)
        except OSError as e:
            logger.warning(f"Error accessing file {fp}: {e}")
            
    # Check all existing files to see if any weren't found in the scan but need updating
    prev_files = previous_index_state.get("files", {})
    for note_id, prev_mod_time in prev_files.items():
        # Skip files we've already decided to process
        if any(os.path.splitext(os.path.relpath(fp, notes_dir_path))[0] == note_id for fp in files_to_process):
            continue
            
        # Try to find the file on disk
        file_path = os.path.join(notes_dir_path, note_id + ".md")
        if os.path.exists(file_path):
            try:
                curr_mod_time = os.path.getmtime(file_path)
                if curr_mod_time > prev_mod_time:
                    files_to_process.append(file_path)
            except OSError as e:
                logger.warning(f"Error accessing file {file_path}: {e}")

    logger.info(f"Files to process: {len(files_to_process)}")

    new_embeddings: Dict[str, Any] = {}
    pbar_iterator = files_to_process if quiet else tqdm(files_to_process, desc="Processing files")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_markdown_file,
                                     fp,
                                     fd_exclude_patterns=exclude_patterns_list,
                                     notes_dir=notes_dir_path,
                                     generate_embeddings=generate_embeddings,
                                     embedding_model=embedding_model,
                                     quiet=quiet): fp
                   for fp in pbar_iterator}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                result = future.result()
                if result:
                    note_id = result["filename"]
                    if generate_embeddings and "embedding" in result:
                        new_embeddings[note_id] = result.pop("embedding")
                    existing_index[note_id] = result
                else:
                    logger.debug(f"No result returned for file: {fp}")
            except Exception as e:
                logger.error(f"Error processing file {fp}: {e}")
            if not quiet and isinstance(pbar_iterator, tqdm):
                pbar_iterator.update(1)

    bk_pbar_iterator = existing_index.values() if quiet else tqdm(existing_index.values(), desc="Calculating backlinks")
    backlink_map: Dict[str, set] = {}
    for note in bk_pbar_iterator:
        for target in note.get("outgoing_links", []):
            if target in existing_index:
                backlink_map.setdefault(target, set()).add(note["filename"])
        if not quiet and isinstance(bk_pbar_iterator, tqdm):
            bk_pbar_iterator.update(1)
    for note_id, note in existing_index.items():
        note["backlinks"] = sorted(list(backlink_map.get(note_id, [])))

    updated_index_data = list(existing_index.values())
    write_updated_index(index_file_path, updated_index_data, quiet)
    save_index_state(state_file, current_index_state, quiet)
    
    # Update the embeddings file if needed
    if generate_embeddings and embeddings_file:
        existing_embeddings.update(new_embeddings)
        # Remove any embeddings for notes that no longer exist
        for key in list(existing_embeddings.keys()):
            if key not in existing_index:
                del existing_embeddings[key]
        write_embeddings_file(embeddings_file, existing_embeddings, quiet)
        if embedding_state_file:
            # For embeddings state, use the same structure as the index state
            save_index_state(embedding_state_file, current_index_state, quiet)
    
    # Print summary
    typer.echo("\n--- Summary ---")
    typer.echo(f"✅ Processed {len(files_to_process)} of {len(markdown_files)} total files")
    typer.echo(f"✅ Index updated with {len(existing_index)} notes at: {index_file_path}")
    if deleted_note_ids:
        typer.echo(f"ℹ️ Removed {len(deleted_note_ids)} deleted notes from index")
    
    if generate_embeddings and embeddings_file:
        typer.echo(f"✅ Generated {len(new_embeddings)} new embeddings")
        typer.echo(f"✅ Embeddings file updated with {len(existing_embeddings)} total embeddings at: {embeddings_file}")


@app.command(name="test-api")
def test_api(
    ctx: typer.Context,
    model: str = typer.Option("text-embedding-3-small", "--model", help="OpenAI embedding model to use."),
):
    """Test the OpenAI API key by generating an embedding for a simple text."""
    api_key = os.getenv("OPEN_AI_KEY")
    
    if not api_key:
        typer.echo("Error: OPEN_AI_KEY environment variable not set!")
        typer.echo("Set it with: export OPEN_AI_KEY=your_api_key")
        raise typer.Exit(1)
    
    openai.api_key = api_key
    test_text = "This is a test of the OpenAI embeddings API."
    
    typer.echo(f"Testing OpenAI API with model: {model}")
    typer.echo(f"Test text: '{test_text}'")
    
    try:
        start_time = time.time()
        embedding = get_embedding(test_text, model=model)
        end_time = time.time()
        
        embedding_length = len(embedding)
        embedding_sample = embedding[:5]
        
        typer.echo(f"✅ Success! Generated embedding with {embedding_length} dimensions in {end_time - start_time:.2f} seconds.")
        typer.echo(f"First 5 values: {embedding_sample}")
        
    except Exception as e:
        typer.echo(f"❌ Error: Failed to generate embedding: {e}")
        raise typer.Exit(1)


@app.command(name="regenerate-embeddings")
def regenerate_embeddings(
    ctx: typer.Context,
    index_file: Optional[str] = typer.Option(None, "--index-file", help="Override index file path"),
    embeddings_file: Optional[str] = typer.Option(None, "--embeddings-file", help="Path to embeddings JSON file"),
    model: str = typer.Option("text-embedding-3-small", "--model", help="OpenAI embedding model to use."),
    batch_size: int = typer.Option(50, "--batch-size", help="Number of embeddings to process in each batch."),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", help=f"Number of worker processes (default: {DEFAULT_WORKERS})"),
):
    """Regenerate embeddings for all notes in the index."""
    config = ctx.obj.get("config", {})
    
    notes_dir_default = DEFAULT_NOTES_DIR
    index_file_default = os.path.join(notes_dir_default, DEFAULT_INDEX_FILENAME)
    
    index_file_path = index_file or get_config_value(config, "zk_index.index_file", index_file_default)
    index_file_path = resolve_path(index_file_path)
    
    emb_file = embeddings_file
    if not emb_file:
        emb_file = os.path.join(os.path.dirname(index_file_path), "embeddings.json")
    emb_file = resolve_path(emb_file)
    
    # Check for OpenAI API key
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        typer.echo("Error: OPEN_AI_KEY environment variable not set!")
        typer.echo("Set it with: export OPEN_AI_KEY=your_api_key")
        raise typer.Exit(1)
    openai.api_key = api_key
    
    if not os.path.exists(index_file_path):
        typer.echo(f"Error: Index file '{index_file_path}' not found!")
        raise typer.Exit(1)
    
    # Load existing data
    existing_index = load_existing_index(index_file_path, quiet=False)
    note_ids = list(existing_index.keys())
    typer.echo(f"Found {len(note_ids)} notes in the index.")
    
    # Load existing embeddings if the file exists
    existing_embeddings = {}
    if os.path.exists(emb_file):
        try:
            with open(emb_file, 'r', encoding='utf-8') as f:
                existing_embeddings = json.load(f)
            typer.echo(f"Loaded existing embeddings file with {len(existing_embeddings)} entries.")
        except Exception as e:
            typer.echo(f"Warning: Could not load existing embeddings file: {e}")
            typer.echo("Starting with empty embeddings.")
    
    # Process notes in batches
    typer.echo(f"Will process {len(note_ids)} notes in batches of {batch_size} using model {model}")
    new_embeddings = {}
    
    with typer.progressbar(length=len(note_ids), label="Generating embeddings") as progress:
        for i in range(0, len(note_ids), batch_size):
            batch = note_ids[i:i+batch_size]
            batch_items = []
            
            for note_id in batch:
                note_data = existing_index[note_id]
                if "body" in note_data:
                    batch_items.append((note_id, note_data["body"]))
            
            # Process batch with parallel workers
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(get_embedding, text, model=model): note_id 
                          for note_id, text in batch_items}
                
                for future in as_completed(futures):
                    note_id = futures[future]
                    try:
                        embedding = future.result()
                        new_embeddings[note_id] = embedding
                    except Exception as e:
                        typer.echo(f"Error generating embedding for {note_id}: {e}")
            
            # Update progress bar
            progress.update(len(batch))
    
    # Write updated embeddings to file
    with open(emb_file, 'w', encoding='utf-8') as f:
        json.dump(new_embeddings, f, indent=2)
    
    typer.echo(f"\n✅ Successfully regenerated {len(new_embeddings)} embeddings")
    typer.echo(f"✅ Embeddings saved to: {emb_file}")


@app.command(name="validate-embeddings")
def validate_embeddings(
    ctx: typer.Context,
    index_file: Optional[str] = typer.Option(None, "--index-file", help="Override index file path"),
    embeddings_file: Optional[str] = typer.Option(None, "--embeddings-file", help="Path to embeddings JSON file"),
):
    """Validate that embeddings exist for all notes in the index."""
    config = ctx.obj.get("config", {})
    
    notes_dir_default = DEFAULT_NOTES_DIR
    index_file_default = os.path.join(notes_dir_default, DEFAULT_INDEX_FILENAME)
    
    index_file_path = index_file or get_config_value(config, "zk_index.index_file", index_file_default)
    index_file_path = resolve_path(index_file_path)
    
    emb_file = embeddings_file
    if not emb_file:
        emb_file = os.path.join(os.path.dirname(index_file_path), "embeddings.json")
    emb_file = resolve_path(emb_file)
    
    if not os.path.exists(index_file_path):
        typer.echo(f"Error: Index file '{index_file_path}' not found!")
        raise typer.Exit(1)
        
    if not os.path.exists(emb_file):
        typer.echo(f"Error: Embeddings file '{emb_file}' not found!")
        raise typer.Exit(1)
    
    # Load existing data
    existing_index = load_existing_index(index_file_path, quiet=False)
    note_ids = set(existing_index.keys())
    typer.echo(f"Found {len(note_ids)} notes in the index.")
    
    try:
        with open(emb_file, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        emb_ids = set(embeddings.keys())
        typer.echo(f"Found {len(emb_ids)} notes in the embeddings file.")
        
        # Check for notes without embeddings
        missing_embeddings = note_ids - emb_ids
        if missing_embeddings:
            typer.echo(f"Found {len(missing_embeddings)} notes missing embeddings:")
            for note_id in sorted(list(missing_embeddings)[:10]):  # Show first 10
                typer.echo(f"  - {note_id}")
            if len(missing_embeddings) > 10:
                typer.echo(f"  ...and {len(missing_embeddings) - 10} more")
        else:
            typer.echo("✅ All notes have embeddings!")
            
        # Check for embeddings without notes
        orphaned_embeddings = emb_ids - note_ids
        if orphaned_embeddings:
            typer.echo(f"Found {len(orphaned_embeddings)} orphaned embeddings (embeddings without notes):")
            for emb_id in sorted(list(orphaned_embeddings)[:10]):  # Show first 10
                typer.echo(f"  - {emb_id}")
            if len(orphaned_embeddings) > 10:
                typer.echo(f"  ...and {len(orphaned_embeddings) - 10} more")
        else:
            typer.echo("✅ No orphaned embeddings!")
            
    except Exception as e:
        typer.echo(f"Error: Could not validate embeddings: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()