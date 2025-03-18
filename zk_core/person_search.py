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
from typing import List, Dict, Any, Optional, Tuple

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR
from zk_core.query import app as query_app
from zk_core.fzf_manager import FzfManager, FzfBinding

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_person_search_fzf_manager(notes_dir: str, bat_command: str) -> FzfManager:
    """
    Create and configure a FzfManager with bindings for person search.
    
    Args:
        notes_dir: Path to the notes directory
        bat_command: Command to use for previewing files
        
    Returns:
        Configured FzfManager instance
    """
    # Create a new FzfManager instance
    manager = FzfManager()
    
    # Add basic bindings
    manager.add_bindings([
        FzfBinding(
            key="ctrl-e",
            command=f"ctrl-e:execute[nvim {notes_dir}/{{1}}.md]",
            description="Edit person note in nvim",
            category="Editing"
        ),
        FzfBinding(
            key="one",
            command="one:accept",
            description="Accept single selection",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-?",
            command="alt-?:toggle-preview",
            description="Toggle preview window",
            category="Navigation"
        ),
    ])
    
    # Add help binding
    manager.add_help_binding("zk-person-search")
    
    return manager


from zk_core.commands import CommandExecutor
from zk_core.fzf_utils import FzfHelper


def main() -> None:
    """Main entry point for the person search module."""
    try:
        import typer
        from typing import Optional as OptionalType
        from pathlib import Path
        
        app = typer.Typer(help="Search for person notes and insert as links")
        
        @app.command()
        def search(
            debug: bool = typer.Option(
                False,
                "--debug", "-d",
                help="Enable debug output"
            ),
            list_hotkeys: bool = typer.Option(
                False,
                "--list-hotkeys", "-l",
                help="Print a list of available fzf hotkeys and their functions, then exit"
            ),
            config_file: OptionalType[Path] = typer.Option(
                None,
                "--config", "-c",
                help="Path to config file (default: ~/.config/zk_scripts/config.yaml)",
                exists=False,
                file_okay=True,
                dir_okay=False
            )
        ) -> None:
            """
            Search for person notes and insert selected entries as wikilinks.
            
            This tool helps you search through notes tagged with 'person' and
            insert properly formatted wikilinks with aliases into other notes.
            """
            # Create args object for backward compatibility
            class Args:
                pass
                
            args = Args()
            args.debug = debug
            args.list_hotkeys = list_hotkeys
            
            if config_file:
                args.config = str(config_file)
                
            # Call the run function
            run_person_search(args)
            
        # Run the app
        app()
            
    except ImportError:
        # Fall back to argparse if typer is not available
        parser = argparse.ArgumentParser(description="Person search utility")
        parser.add_argument("--debug", "-d", action="store_true", help="Enable debug output")
        parser.add_argument("--list-hotkeys", "-l", action="store_true", 
                          help="Print a list of available fzf hotkeys and their functions, then exit")
        parser.add_argument("--config", "-c", help="Path to config file")
        args = parser.parse_args()
        
        print("WARNING: Typer package not found, falling back to basic argparse implementation.")
        print("For a better CLI experience, install typer: pip install typer")
        
        # Call the run function
        run_person_search(args)


def run_person_search(args) -> None:
    """Run the person search with the given arguments."""
    # Load configuration
    config_file = getattr(args, 'config', None)
    
    if config_file:
        from zk_core.config import load_config_from_file
        config = load_config_from_file(config_file)
    else:
        config = load_config()
    
    # Get configuration values
    notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
    notes_dir = resolve_path(notes_dir)
    
    # Get personSearch specific configuration
    person_search_config = get_config_value(config, "personSearch", {})
    bat_command = get_config_value(person_search_config, "bat_command", "bat")
    
    # Log configuration if in debug mode
    if hasattr(args, 'debug') and args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Notes directory: {notes_dir}")
        logger.debug(f"bat command: {bat_command}")
    
    # Build the FzfManager with person search bindings
    fzf_manager = build_person_search_fzf_manager(notes_dir, bat_command)
    
    # If --list-hotkeys was specified, print the hotkey help and exit
    if hasattr(args, 'list_hotkeys') and args.list_hotkeys:
        FzfHelper.print_hotkeys(fzf_manager, "PERSON SEARCH KEYBOARD SHORTCUTS")
        sys.exit(0)
    
    # Change to notes directory
    try:
        os.chdir(notes_dir)
        if hasattr(args, 'debug') and args.debug:
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
            
            if hasattr(args, 'debug') and args.debug:
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
    
    # Prepare additional fzf arguments specific to person search
    additional_args = [
        "--delimiter", "::",
        "--with-nth", "2,3,4",
        "--tiebreak", "begin,index",
        "--info", "right",
        "--ellipsis", "",
        "--preview-label", "",
        "--preview", f"{bat_command} {notes_dir}/{{1}}.md",
        "--preview-window", "wrap:50%:<40(up)",
    ]
    
    # Get complete fzf arguments
    fzf_args = fzf_manager.get_fzf_args(additional_args)
    
    if hasattr(args, 'debug') and args.debug:
        logger.debug(f"Running fzf command: {' '.join(fzf_args)}")
    
    try:
        # Pipe the query results into fzf
        fzf_result = subprocess.run(fzf_args, stdin=p_list.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p_list.stdout.close()
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)
    
    selection = fzf_result.stdout.strip()
    if hasattr(args, 'debug') and args.debug:
        logger.debug(f"fzf selection: {selection}")
    
    if not selection:
        if hasattr(args, 'debug') and args.debug:
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
        
        if hasattr(args, 'debug') and args.debug:
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
    
    if hasattr(args, 'debug') and args.debug:
        logger.debug(f"Final wikilink: {transform}")
    
    # Send the processed link to the active tmux pane
    try:
        subprocess.run(["tmux", "send-keys", transform], check=True)
        if hasattr(args, 'debug') and args.debug:
            logger.debug("Sent link to tmux")
    except Exception as e:
        logger.error(f"Error sending keys to tmux: {e}")
        sys.exit(1)
    
    if hasattr(args, 'debug') and args.debug:
        logger.debug("Person search completed successfully")


if __name__ == "__main__":
    main()