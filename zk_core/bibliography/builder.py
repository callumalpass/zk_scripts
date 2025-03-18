"""
Bibliography builder module.

This module generates bibliography files from notes with the 'literature_note' tag:
- Creates citation key lists
- Generates bibliography JSON files
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR
from zk_core.utils import get_index_file_path, get_path_for_script
from zk_core.commands import CommandExecutor

logger = logging.getLogger(__name__)


def generate_citation_keys(biblib_dir: str, notes_dir: str, index_file: str = None) -> bool:
    """
    Generate citation key files from literature notes in the index.
    
    Args:
        biblib_dir: Path to the bibliography library directory
        notes_dir: Path to the notes directory
        index_file: Path to the index file (if not provided, uses default from config)
        
    Returns:
        True if successful, False otherwise
    """
    citekeylist_file = "citekeylist"
    
    try:
        # First try to get citation keys from literature notes in the index
        citation_keys = []
        
        if index_file and os.path.isfile(index_file):
            try:
                with open(index_file, "r") as f:
                    index_data = json.load(f)
                
                # Ensure index_data is a list
                if index_data is None:
                    logger.warning("Index data is None")
                    # Will fall back to directory scan
                elif not isinstance(index_data, list):
                    # If it's a dict with a notes key, try to extract notes
                    if isinstance(index_data, dict) and 'notes' in index_data:
                        # Handle the legacy format where notes are in a dict
                        notes_list = []
                        for filename, note_data in index_data['notes'].items():
                            # Convert dict entry to a list item with filename
                            note_item = dict(note_data)
                            note_item['filename'] = filename
                            notes_list.append(note_item)
                        index_data = notes_list
                    else:
                        logger.warning(f"Index data is not a list or recognized format: {type(index_data)}")
                        # Will fall back to directory scan
                
                # Extract filenames from literature notes
                if isinstance(index_data, list):
                    for note in index_data:
                        if not isinstance(note, dict):
                            continue
                            
                        if ('tags' in note and note['tags'] and
                            any(tag.startswith('literature_note') for tag in note.get('tags', [])) and
                            'filename' in note):
                            # Remove '@' prefix if it exists
                            cite_key = note['filename']
                            if cite_key.startswith('@'):
                                cite_key = cite_key[1:]
                            citation_keys.append(cite_key)
                    
                    logger.info(f"Found {len(citation_keys)} citation keys from literature notes")
                
            except Exception as e:
                logger.warning(f"Could not extract citation keys from index: {e}")
        
        # If no keys found from index (or index not available), fall back to directory scan
        if not citation_keys:
            logger.info("No citation keys found from index, falling back to directory scan")
            for item in os.scandir(biblib_dir):
                if item.is_dir() and not item.name.startswith('.'):
                    citation_keys.append(item.name)
            
            logger.info(f"Found {len(citation_keys)} citation keys from directory scan")
        
        # Sort the citation keys
        citation_keys = list(set(citation_keys))  # Remove duplicates
        citation_keys.sort()
        
        # Handle empty citation keys list
        if not citation_keys:
            logger.warning("No citation keys found from any source")
            # Create empty files to avoid errors
            citation_keys = []
        
        # Format as string
        raw_keys = "\n".join(citation_keys)
        
        # Save raw output to biblib directory
        with open(os.path.join(biblib_dir, citekeylist_file), "w") as f:
            f.write(raw_keys)
        logger.info(f"Wrote citation keys to {os.path.join(biblib_dir, citekeylist_file)}")
        
        # Create a formatted copy in notes directory with @ prefixes
        with open(os.path.join(notes_dir, citekeylist_file + ".md"), "w") as f:
            for key in citation_keys:
                f.write(f"@{key}\n")
        logger.info(f"Wrote formatted citation keys to {os.path.join(notes_dir, citekeylist_file + '.md')}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating citation keys: {e}")
        return False


def generate_bibliography(index_file: str, output_paths: List[str]) -> bool:
    """
    Generate bibliography JSON file from the index file.
    
    Args:
        index_file: Path to the index file
        output_paths: List of paths where to save the bibliography JSON
        
    Returns:
        True if successful, False otherwise
    """
    # Check that index file exists
    if not os.path.isfile(index_file):
        logger.error(f"Index file not found: {index_file}")
        return False
    
    try:
        # Load index data
        with open(index_file, "r") as f:
            index_data = json.load(f)
        
        # Ensure index_data is a list
        if index_data is None:
            logger.error("Index data is None")
            return False
        
        if not isinstance(index_data, list):
            # If it's a dict with a notes key, try to extract notes
            if isinstance(index_data, dict) and 'notes' in index_data:
                # Handle the legacy format where notes are in a dict
                notes_list = []
                for filename, note_data in index_data['notes'].items():
                    # Convert dict entry to a list item with filename
                    note_item = dict(note_data)
                    note_item['filename'] = filename
                    notes_list.append(note_item)
                index_data = notes_list
            else:
                logger.error(f"Index data is not a list or recognized format: {type(index_data)}")
                return False
        
        # Filter notes with 'literature_note' tag
        literature_notes = []
        for note in index_data:
            if not isinstance(note, dict):
                continue
                
            if 'tags' in note and note['tags']:
                if any(tag.startswith('literature_note') for tag in note.get('tags', [])):
                    literature_notes.append(note)
        
        logger.info(f"Found {len(literature_notes)} literature notes")
        
        # Convert to JSON
        bibliography_json = json.dumps(literature_notes, indent=2)
        
        # Write to all destination paths
        success = True
        for output_path in output_paths:
            try:
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, "w") as dest:
                    dest.write(bibliography_json)
                logger.info(f"Wrote bibliography JSON to {output_path}")
            except Exception as e:
                logger.error(f"Error writing bibliography JSON to {output_path}: {e}")
                success = False
        
        return success
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing index file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing bibliography: {e}")
        return False


def get_bibliography_config(config: Dict[str, Any], args: Optional[Any] = None) -> Dict[str, str]:
    """
    Retrieve bibliography configuration with proper defaults.
    
    Args:
        config: Configuration dictionary
        args: Optional command line arguments
        
    Returns:
        Dictionary of bibliography configuration values
    """
    # Get notes directory
    notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
    notes_dir = resolve_path(notes_dir)
    
    # Get biblib directory
    biblib_dir = get_config_value(config, "bibview.library", os.path.join(notes_dir, "biblib"))
    biblib_dir = resolve_path(biblib_dir)
    
    # Get index file
    index_file = get_index_file_path(config, notes_dir, args)
    
    # Get mybin directory for scripts
    mybin_dir = get_config_value(config, "mybin_dir", os.path.expanduser("~/mybin"))
    mybin_dir = resolve_path(mybin_dir)
    
    # No longer need getbibkeys_script
    
    # Get bibliography output files
    bibliography_file = get_config_value(
        config,
        "bibview.bibliography_json", 
        os.path.join(biblib_dir, "bibliography.json")
    )
    bibliography_file = resolve_path(bibliography_file)
    
    # Optional additional output location
    dropbox_bibliography = get_config_value(
        config,
        "bibview.dropbox_bibliography_json", 
        os.path.join(os.path.expanduser("~/Dropbox"), "bibliography.json")
    )
    dropbox_bibliography = resolve_path(dropbox_bibliography)
    
    return {
        "notes_dir": notes_dir,
        "biblib_dir": biblib_dir,
        "index_file": index_file,
        "bibliography_file": bibliography_file,
        "dropbox_bibliography": dropbox_bibliography
    }


def run_build(config: Optional[Dict[str, Any]] = None, args: Optional[Any] = None) -> int:
    """
    Main function to run bibliography building process.
    
    Args:
        config: Optional configuration dictionary (loaded if not provided)
        args: Optional command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Get configuration values
    bib_config = get_bibliography_config(config, args)
    notes_dir = bib_config["notes_dir"]
    biblib_dir = bib_config["biblib_dir"]
    index_file = bib_config["index_file"]
    bibliography_file = bib_config["bibliography_file"]
    dropbox_bibliography = bib_config["dropbox_bibliography"]
    
    # List of all output paths
    output_paths = [bibliography_file]
    
    # Add dropbox path if different from main path
    if dropbox_bibliography and dropbox_bibliography != bibliography_file:
        output_paths.append(dropbox_bibliography)
    
    # Check required directories exist
    if not os.path.isdir(notes_dir):
        logger.error(f"Notes directory doesn't exist: {notes_dir}")
        return 1
    
    # Create biblib directory if it doesn't exist
    if not os.path.isdir(biblib_dir):
        logger.warning(f"Bibliography library directory doesn't exist: {biblib_dir}")
        try:
            os.makedirs(biblib_dir, exist_ok=True)
            logger.info(f"Created bibliography library directory: {biblib_dir}")
        except Exception as e:
            logger.error(f"Error creating directory {biblib_dir}: {e}")
            return 1
    
    # Check if index file exists
    if not os.path.isfile(index_file):
        logger.error(f"Index file not found: {index_file}")
        return 1
    
    # Determine which operations to perform based on args
    generate_keys = True
    generate_bib = True
    
    if args:
        if hasattr(args, 'keys_only') and args.keys_only:
            generate_bib = False
            logger.info("Only generating citation keys (--keys-only specified)")
        
        if hasattr(args, 'bib_only') and args.bib_only:
            generate_keys = False
            logger.info("Only generating bibliography (--bib-only specified)")
            
        # Check for conflict
        if hasattr(args, 'keys_only') and hasattr(args, 'bib_only') and args.keys_only and args.bib_only:
            logger.warning("Both --keys-only and --bib-only specified, doing nothing.")
            return 0
    
    success = True
    
    # Generate citation keys if needed
    if generate_keys:
        logger.info("Generating citation keys...")
        if not generate_citation_keys(biblib_dir, notes_dir, index_file):
            logger.error("Failed to generate citation keys")
            success = False
    
    # Generate bibliography JSON if needed
    if generate_bib:
        logger.info("Generating bibliography JSON...")
        if not generate_bibliography(index_file, output_paths):
            logger.error("Failed to generate bibliography JSON")
            success = False
    
    if success:
        logger.info("Bibliography generation completed successfully.")
        return 0
    else:
        logger.error("Bibliography generation completed with errors.")
        return 1


def cli():
    """Command-line interface using Typer."""
    import typer
    from enum import Enum
    from pathlib import Path
    from typing import Optional as OptionalType
    
    # Create a CLI app with help and version information
    app = typer.Typer(
        help="Build citation key list and bibliography JSON from literature notes",
        add_completion=False,
    )
    
    # Define verbosity option as an enum
    class Verbosity(str, Enum):
        QUIET = "quiet"
        NORMAL = "normal"
        VERBOSE = "verbose"
    
    # Define the main command
    @app.command()
    def build(
        # General options
        verbosity: Verbosity = typer.Option(
            Verbosity.NORMAL,
            "--verbosity", "-v",
            help="Set output verbosity level",
        ),
        config: OptionalType[Path] = typer.Option(
            None,
            "--config", "-c",
            help="Path to config file (default: ~/.config/zk_scripts/config.yaml)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
        
        # Input/output options
        index_file: OptionalType[Path] = typer.Option(
            None,
            "--index-file", "-i",
            help="Path to index file (overrides config)",
        ),
        notes_dir: OptionalType[Path] = typer.Option(
            None,
            "--notes-dir", "-n",
            help="Path to notes directory (overrides config)",
            exists=False,
            file_okay=False,
            dir_okay=True,
        ),
        biblib_dir: OptionalType[Path] = typer.Option(
            None,
            "--biblib-dir", "-b",
            help="Path to bibliography library directory (overrides config)",
            exists=False,
            file_okay=False,
            dir_okay=True,
        ),
        output: OptionalType[Path] = typer.Option(
            None,
            "--output", "-o",
            help="Path to bibliography output file (overrides config)",
        ),
        
        # Processing options
        keys_only: bool = typer.Option(
            False,
            "--keys-only",
            help="Only generate citation keys, skip bibliography generation",
        ),
        bib_only: bool = typer.Option(
            False,
            "--bib-only",
            help="Only generate bibliography JSON, skip citation key generation",
        ),
    ) -> None:
        """
        Build citation key list and bibliography JSON from literature notes.
        
        This tool extracts citation keys from literature notes in the index and
        generates a bibliography JSON file for use with viewing tools.
        """
        # Set up logging based on verbosity
        if verbosity == Verbosity.VERBOSE:
            logging.basicConfig(level=logging.DEBUG)
        elif verbosity == Verbosity.QUIET:
            logging.basicConfig(level=logging.ERROR)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Create args object to match the expected interface in run_build
        class Args:
            pass
        
        args = Args()
        args.keys_only = keys_only
        args.bib_only = bib_only
        
        # Convert Path objects to strings for backward compatibility
        if index_file:
            args.index_file = str(index_file)
        if notes_dir:
            args.notes_dir = str(notes_dir)
        if biblib_dir:
            args.biblib_dir = str(biblib_dir)
        if output:
            args.output = str(output)
        if config:
            args.config = str(config)
        
        # Load config or use default
        config_dict = None
        if hasattr(args, 'config') and args.config:
            from zk_core.config import load_config_from_file
            config_dict = load_config_from_file(args.config)
        else:
            from zk_core.config import load_config
            config_dict = load_config()
        
        # Override config with CLI arguments
        if hasattr(args, 'notes_dir') or hasattr(args, 'biblib_dir') or hasattr(args, 'index_file') or hasattr(args, 'output'):
            # Create a copy of the config to avoid modifying the original
            config_copy = dict(config_dict)
            
            if hasattr(args, 'notes_dir') and args.notes_dir:
                config_copy["notes_dir"] = args.notes_dir
            
            if hasattr(args, 'biblib_dir') and args.biblib_dir:
                if "bibview" not in config_copy:
                    config_copy["bibview"] = {}
                config_copy["bibview"]["library"] = args.biblib_dir
            
            if hasattr(args, 'index_file') and args.index_file:
                if "zk_index" not in config_copy:
                    config_copy["zk_index"] = {}
                config_copy["zk_index"]["index_file"] = args.index_file
            
            if hasattr(args, 'output') and args.output:
                if "bibview" not in config_copy:
                    config_copy["bibview"] = {}
                config_copy["bibview"]["bibliography_json"] = args.output
            
            config_dict = config_copy
        
        # Run the build process
        exit_code = run_build(config_dict, args)
        raise typer.Exit(exit_code)
    
    return app()


def main() -> None:
    """Entry point for command-line use."""
    try:
        import typer
        cli()
    except ImportError:
        typer_missing()


def typer_missing():
    """Fallback when typer is not installed."""
    import argparse
    import sys
    
    print("WARNING: Typer package not found, falling back to basic argparse implementation.")
    print("For a better CLI experience, install typer: pip install typer")
    
    parser = argparse.ArgumentParser(
        description="Build citation key list and bibliography JSON from literature notes",
    )
    
    # General options
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true", 
        help="Suppress all output except errors"
    )
    parser.add_argument(
        "-c", "--config", 
        help="Path to config file (default: ~/.config/zk_scripts/config.yaml)"
    )
    
    # Input/output options
    parser.add_argument(
        "-i", "--index-file", 
        help="Path to index file (overrides config)"
    )
    parser.add_argument(
        "-n", "--notes-dir", 
        help="Path to notes directory (overrides config)"
    )
    parser.add_argument(
        "-b", "--biblib-dir", 
        help="Path to bibliography library directory (overrides config)"
    )
    parser.add_argument(
        "-o", "--output", 
        help="Path to bibliography output file (overrides config)"
    )
    
    # Processing options
    parser.add_argument(
        "--keys-only", 
        action="store_true", 
        help="Only generate citation keys, skip bibliography generation"
    )
    parser.add_argument(
        "--bib-only", 
        action="store_true", 
        help="Only generate bibliography JSON, skip citation key generation"
    )
    
    args = parser.parse_args()
    
    # Set up logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Load config or use default
    config = None
    if args.config:
        from zk_core.config import load_config_from_file
        config = load_config_from_file(args.config)
    else:
        from zk_core.config import load_config
        config = load_config()
    
    # Override config with CLI arguments
    if args.notes_dir or args.biblib_dir or args.index_file or args.output:
        # Create a copy of the config to avoid modifying the original
        config_copy = dict(config)
        
        if args.notes_dir:
            config_copy["notes_dir"] = args.notes_dir
        
        if args.biblib_dir:
            if "bibview" not in config_copy:
                config_copy["bibview"] = {}
            config_copy["bibview"]["library"] = args.biblib_dir
        
        if args.index_file:
            if "zk_index" not in config_copy:
                config_copy["zk_index"] = {}
            config_copy["zk_index"]["index_file"] = args.index_file
        
        if args.output:
            if "bibview" not in config_copy:
                config_copy["bibview"] = {}
            config_copy["bibview"]["bibliography_json"] = args.output
        
        config = config_copy
    
    # Run the build process
    exit_code = run_build(config, args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()