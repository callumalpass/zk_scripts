"""
Person search module.

This module provides functionality for searching person notes and inserting them as links.
It uses the query module to find person notes and the FZF interface to display and select them.
"""

import os
import sys
import subprocess
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.query import app as query_app

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], input_text: Optional[str] = None) -> tuple[int, str, str]:
    """
    Run a command and return return code, stdout, and stderr.
    
    Args:
        cmd: Command to run as a list of strings
        input_text: Optional text to pass to the command's stdin
        
    Returns:
        Tuple of (return code, stdout, stderr)
    """
    try:
        proc = subprocess.run(
            cmd, 
            input=input_text,
            capture_output=True, 
            text=True
        )
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        logger.error(f"Error running command {' '.join(cmd)}: {e}")
        return 1, "", str(e)


def main() -> None:
    """Main entry point for the person search module."""
    parser = argparse.ArgumentParser(description="Person search utility")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Get configuration values
    notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    notes_dir = resolve_path(notes_dir)
    
    # Get personSearch specific configuration
    person_search_config = get_config_value(config, "personSearch", {})
    bat_command = get_config_value(person_search_config, "bat_command", "bat")
    
    # Log configuration if in debug mode
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Notes directory: {notes_dir}")
        logger.debug(f"bat command: {bat_command}")
    
    # Change to notes directory
    try:
        os.chdir(notes_dir)
        logger.debug(f"Changed to directory: {notes_dir}")
    except Exception as e:
        logger.error(f"Unable to change directory to {notes_dir}: {e}")
        sys.exit(1)
    
    # Create a temporary file to capture query results
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as temp_file:
            # Prepare arguments for query list command
            from typer.testing import CliRunner
            runner = CliRunner()
            query_args = [
                "list",
                "--mode", "notes",
                "-i", "index.json",
                "--filter-tag", "person",
                "--fields", "filename",
                "--fields", "aliases",
                "--fields", "givenName",
                "--fields", "familyName",
                "--color", "always"
            ]
            
            if args.debug:
                logger.debug(f"Running query with args: {' '.join(query_args)}")
            
            # Run the query and capture output
            result = runner.invoke(query_app, query_args)
            temp_file.write(result.stdout)
            temp_file_path = temp_file.name
        
        # Re-open the file to pipe its contents to fzf
        p_list = subprocess.Popen(['cat', temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    except Exception as e:
        logger.error(f"Error running query command: {e}")
        sys.exit(1)
    
    # Build fzf arguments for interactive selection
    fzf_cmd = [
        "fzf",
        "--bind", f"ctrl-e:execute[nvim {notes_dir}/{{1}}.md]",
        "--bind", "one:accept",
        "--delimiter", "::",
        "--with-nth", "2,3,4",
        "--tiebreak", "begin,index",
        "--info", "right",
        "--ellipsis", "",
        "--preview-label", "",
        "--preview", f"{bat_command} {notes_dir}/{{1}}.md",
        "--preview-window", "wrap:50%:<40(up)",
        "--ansi"
    ]
    
    if args.debug:
        logger.debug(f"Running fzf command: {' '.join(fzf_cmd)}")
    
    try:
        # Pipe the py_zk list output into fzf
        fzf_result = subprocess.run(fzf_cmd, stdin=p_list.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p_list.stdout.close()
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)
    
    selection = fzf_result.stdout.strip()
    if args.debug:
        logger.debug(f"fzf selection: {selection}")
    
    if not selection:
        logger.debug("No selection made")
        sys.exit(0)
    
    # Parse the selection (fields separated by "::")
    parts = selection.split("::")
    filename = parts[0].strip()
    
    # Use query module to fetch additional fields for the selected note
    try:
        from typer.testing import CliRunner
        runner = CliRunner()
        fields_args = [
            "list",
            "--mode", "notes",
            "-i", "index.json",
            "--stdin",
            "--fields", "aliases",
            "--fields", "givenName"
        ]
        
        if args.debug:
            logger.debug(f"Running fields command with args: {' '.join(fields_args)}")
        
        # Run the query with stdin input
        result = runner.invoke(query_app, fields_args, input=filename, catch_exceptions=False)
        return_code = 0 if result.exit_code == 0 else 1
        stdout = result.stdout
        
        if return_code != 0:
            logger.error(f"Error getting fields: Exit code {result.exit_code}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error running fields command: {e}")
        sys.exit(1)
    
    fields_output = stdout.strip()
    fields_parts = fields_output.split("::")
    aliases = fields_parts[0].strip() if len(fields_parts) > 0 else ""
    given_name = fields_parts[1].strip() if len(fields_parts) > 1 else ""
    
    # Build the wikilink using aliases if available; otherwise use the given name
    if aliases:
        transform = f"[[{filename}|{aliases}]]"
    else:
        transform = f"[[{filename}|{given_name}]]"
    
    if args.debug:
        logger.debug(f"Final wikilink: {transform}")
    
    # Send the processed link to the active tmux pane
    try:
        subprocess.run(["tmux", "send-keys", transform], check=True)
        logger.debug("Sent link to tmux")
    except Exception as e:
        logger.error(f"Error sending keys to tmux: {e}")
        sys.exit(1)
    
    logger.debug("Person search completed successfully")


if __name__ == "__main__":
    main()