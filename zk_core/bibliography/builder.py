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


def generate_citation_keys(getbibkeys_script: str, biblib_dir: str, notes_dir: str) -> bool:
    """
    Generate citation key files using the getbibkeys script.
    
    Args:
        getbibkeys_script: Path to the getbibkeys script
        biblib_dir: Path to the bibliography library directory
        notes_dir: Path to the notes directory
        
    Returns:
        True if successful, False otherwise
    """
    citekeylist_file = "citekeylist"
    
    # Check if script exists and is executable
    if not os.path.isfile(getbibkeys_script) or not os.access(getbibkeys_script, os.X_OK):
        logger.error(f"The getbibkeys script is not executable or not found: {getbibkeys_script}")
        return False
    
    # Run script to generate citation keys
    rc, stdout, stderr = CommandExecutor.run([getbibkeys_script])
    if rc != 0:
        logger.error(f"Error running getbibkeys script: {stderr}")
        return False
    
    # Save raw output to biblib directory
    try:
        with open(os.path.join(biblib_dir, citekeylist_file), "w") as f:
            f.write(stdout)
        logger.info(f"Wrote citation keys to {os.path.join(biblib_dir, citekeylist_file)}")
    except Exception as e:
        logger.error(f"Error writing to {os.path.join(biblib_dir, citekeylist_file)}: {e}")
        return False
    
    # Create a formatted copy in notes directory with @ prefixes
    try:
        with open(os.path.join(notes_dir, citekeylist_file + ".md"), "w") as f:
            for line in stdout.splitlines():
                f.write(f"@{line}\n")
        logger.info(f"Wrote formatted citation keys to {os.path.join(notes_dir, citekeylist_file + '.md')}")
    except Exception as e:
        logger.error(f"Error writing to {os.path.join(notes_dir, citekeylist_file + '.md')}: {e}")
        return False
    
    return True


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
        # Filter notes with 'literature_note' tag using jq
        jq_cmd = ["jq", '[ .[] | select(.tags[]? | startswith("literature_note"))]', index_file]
        rc, stdout, stderr = CommandExecutor.run(jq_cmd)
        
        if rc != 0:
            logger.error(f"Error filtering literature notes with jq: {stderr}")
            return False
        
        # Create a temporary file with the filtered data
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(stdout)
            tmp.flush()
            
            # Verify JSON is valid by loading it
            try:
                with open(tmp_path, "r") as src:
                    json_data = json.load(src)
                logger.debug(f"Generated bibliography with {len(json_data)} entries")
            except json.JSONDecodeError as e:
                logger.error(f"Generated JSON is invalid: {e}")
                os.unlink(tmp_path)
                return False
            
            # Copy to all destination paths
            success = True
            for output_path in output_paths:
                try:
                    # Create parent directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(tmp_path, "r") as src:
                        content = src.read()
                        
                        with open(output_path, "w") as dest:
                            dest.write(content)
                        logger.info(f"Wrote bibliography JSON to {output_path}")
                except Exception as e:
                    logger.error(f"Error copying bibliography JSON to {output_path}: {e}")
                    success = False
            
            # Cleanup temp file
            os.unlink(tmp_path)
            return success
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
    
    # Get script paths
    getbibkeys_script = get_path_for_script(
        config, 
        "bibview", 
        "getbibkeys_script", 
        os.path.join(mybin_dir, "getbibkeys.sh")
    )
    
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
        "getbibkeys_script": getbibkeys_script,
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
    getbibkeys_script = bib_config["getbibkeys_script"]
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
    
    # Generate citation keys
    if not generate_citation_keys(getbibkeys_script, biblib_dir, notes_dir):
        logger.error("Failed to generate citation keys")
        return 1
    
    # Generate bibliography JSON
    if not generate_bibliography(index_file, output_paths):
        logger.error("Failed to generate bibliography JSON")
        return 1
    
    logger.info("Bibliography generation completed successfully.")
    return 0


def main() -> None:
    """Entry point for command-line use."""
    exit_code = run_build()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()