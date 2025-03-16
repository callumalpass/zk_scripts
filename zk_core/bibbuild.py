"""
Bibliography building module.

This module generates bibliography files from notes with the 'literature_note' tag.
It creates:
- A list of citation keys
- A bibliography JSON file
"""

import os
import sys
import subprocess
import json
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.utils import run_command, save_json_file, load_json_file

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
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
    rc, stdout, stderr = run_command([getbibkeys_script])
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
        rc, stdout, stderr = run_command(jq_cmd)
        
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


def main() -> None:
    """Main function."""
    # Load configuration
    config = load_config()
    
    # Get configuration values with defaults
    notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    notes_dir = resolve_path(notes_dir)
    
    biblib_dir = get_config_value(config, "bibview.library", os.path.join(notes_dir, "biblib"))
    biblib_dir = resolve_path(biblib_dir)
    
    index_file = get_config_value(config, "zk_index.index_file", os.path.join(notes_dir, "index.json"))
    index_file = resolve_path(index_file)
    
    mybin_dir = get_config_value(config, "mybin_dir", os.path.expanduser("~/mybin"))
    mybin_dir = resolve_path(mybin_dir)
    
    getbibkeys_script = get_config_value(config, "bibview.getbibkeys_script", 
                                         os.path.join(mybin_dir, "getbibkeys.sh"))
    getbibkeys_script = resolve_path(getbibkeys_script)
    
    # Get bibliography output files
    bibliography_file = get_config_value(config, "bibview.bibliography_json", 
                                        os.path.join(biblib_dir, "bibliography.json"))
    bibliography_file = resolve_path(bibliography_file)
    
    # Optional additional output locations
    dropbox_bibliography = get_config_value(config, "bibview.dropbox_bibliography_json", 
                                           os.path.join(os.path.expanduser("~/Dropbox"), "bibliography.json"))
    dropbox_bibliography = resolve_path(dropbox_bibliography)
    
    # List of all output paths
    output_paths = [bibliography_file]
    
    # Add dropbox path if different from main path
    if dropbox_bibliography and dropbox_bibliography != bibliography_file:
        output_paths.append(dropbox_bibliography)
    
    # Check required directories exist
    if not os.path.isdir(notes_dir):
        logger.error(f"Notes directory doesn't exist: {notes_dir}")
        sys.exit(1)
    
    # Create biblib directory if it doesn't exist
    if not os.path.isdir(biblib_dir):
        logger.warning(f"Bibliography library directory doesn't exist: {biblib_dir}")
        try:
            os.makedirs(biblib_dir, exist_ok=True)
            logger.info(f"Created bibliography library directory: {biblib_dir}")
        except Exception as e:
            logger.error(f"Error creating directory {biblib_dir}: {e}")
            sys.exit(1)
    
    # Generate citation keys
    if not generate_citation_keys(getbibkeys_script, biblib_dir, notes_dir):
        logger.error("Failed to generate citation keys")
        sys.exit(1)
    
    # Generate bibliography JSON
    if not generate_bibliography(index_file, output_paths):
        logger.error("Failed to generate bibliography JSON")
        sys.exit(1)
    
    logger.info("Bibliography generation completed successfully.")


if __name__ == "__main__":
    main()