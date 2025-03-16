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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return return code, stdout, and stderr."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        logger.error(f"Error running command {' '.join(cmd)}: {e}")
        return 1, "", str(e)


def main() -> None:
    """Main function."""
    # Load configuration
    config = load_config()
    
    # Get configuration values
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
    
    citekeylist_file = "citekeylist"
    bibliography_file = "bibliography.json"
    
    # Check required directories exist
    if not os.path.isdir(biblib_dir):
        logger.error(f"Bibliography library directory doesn't exist: {biblib_dir}")
        sys.exit(1)
    
    if not os.path.isdir(notes_dir):
        logger.error(f"Notes directory doesn't exist: {notes_dir}")
        sys.exit(1)
    
    # Run getbibkeys script to generate citation key list
    if os.path.isfile(getbibkeys_script) and os.access(getbibkeys_script, os.X_OK):
        # Create citekeylist in biblib directory
        rc, stdout, stderr = run_command([getbibkeys_script])
        if rc != 0:
            logger.error(f"Error running getbibkeys script: {stderr}")
            sys.exit(1)
        
        # Save the output to biblib directory
        try:
            with open(os.path.join(biblib_dir, citekeylist_file), "w") as f:
                f.write(stdout)
            logger.info(f"Wrote citation keys to {os.path.join(biblib_dir, citekeylist_file)}")
        except Exception as e:
            logger.error(f"Error writing to {os.path.join(biblib_dir, citekeylist_file)}: {e}")
            sys.exit(1)
        
        # Create a copy in notes directory with @ prefixes for each line
        try:
            with open(os.path.join(notes_dir, citekeylist_file + ".md"), "w") as f:
                for line in stdout.splitlines():
                    f.write(f"@{line}\n")
            logger.info(f"Wrote formatted citation keys to {os.path.join(notes_dir, citekeylist_file + '.md')}")
        except Exception as e:
            logger.error(f"Error writing to {os.path.join(notes_dir, citekeylist_file + '.md')}: {e}")
            sys.exit(1)
    else:
        logger.error(f"The getbibkeys script is not executable or not found: {getbibkeys_script}")
        sys.exit(1)
    
    # Generate bibliography JSON from index
    if not os.path.isfile(index_file):
        logger.error(f"Index file not found: {index_file}")
        sys.exit(1)
    
    try:
        # Create a temporary file for processing
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False) as tmp:
            tmp_path = tmp.name
            
            # Use jq to filter notes with 'literature_note' tag
            jq_cmd = ["jq", '[ .[] | select(.tags[]? | startswith("literature_note"))]', index_file]
            rc, stdout, stderr = run_command(jq_cmd)
            
            if rc != 0:
                logger.error(f"Error filtering literature notes with jq: {stderr}")
                os.unlink(tmp_path)
                sys.exit(1)
            
            # Write the filtered data to temp file
            tmp.write(stdout)
            tmp.flush()
            
            # Copy to destination files
            biblib_json = os.path.join(biblib_dir, bibliography_file)
            dropbox_json = os.path.join(os.path.expanduser("~/Dropbox"), bibliography_file)
            
            try:
                with open(tmp_path, "r") as src:
                    content = src.read()
                    
                    with open(biblib_json, "w") as dest:
                        dest.write(content)
                    logger.info(f"Wrote bibliography JSON to {biblib_json}")
                    
                    with open(dropbox_json, "w") as dest:
                        dest.write(content)
                    logger.info(f"Wrote bibliography JSON to {dropbox_json}")
            except Exception as e:
                logger.error(f"Error copying bibliography JSON: {e}")
                os.unlink(tmp_path)
                sys.exit(1)
            
            # Cleanup
            os.unlink(tmp_path)
    except Exception as e:
        logger.error(f"Error processing bibliography: {e}")
        sys.exit(1)
    
    logger.info("Bibliography generation completed successfully.")


if __name__ == "__main__":
    main()