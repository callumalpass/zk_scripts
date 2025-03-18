"""
Wikilink generator module.

This module provides functionality for searching notes with configurable criteria
and inserting them as wikilinks with customizable formatting.
"""

import os
import sys
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR
from zk_core.query import app as query_app
from zk_core.fzf_manager import FzfManager, FzfBinding
from zk_core.commands import CommandExecutor
from zk_core.fzf_utils import FzfHelper

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class WikilinkConfig:
    """Configuration for a wikilink generator profile."""
    
    def __init__(
        self,
        name: str,
        filter_tags: List[str] = None,
        search_fields: List[str] = None,
        display_fields: List[str] = None,
        alias_fields: List[str] = None,
        preview_config: Dict[str, Any] = None,
        fzf_config: Dict[str, Any] = None
    ):
        """
        Initialize a new wikilink generator configuration.
        
        Args:
            name: Name of this configuration profile
            filter_tags: Tags to filter notes by (e.g., ["person", "concept"])
            search_fields: Fields to include for searching in FZF
            display_fields: Fields to display in FZF (defaults to search_fields)
            alias_fields: Fields to use for wikilink aliases (in priority order)
            preview_config: Configuration for the preview window
            fzf_config: Additional FZF configuration options
        """
        self.name = name
        self.filter_tags = filter_tags or []
        self.search_fields = search_fields or ["filename"]
        # Ensure display_fields is a separate list, not just a reference to search_fields
        self.display_fields = list(display_fields) if display_fields else list(self.search_fields)
        self.alias_fields = alias_fields or ["aliases", "title"]
        
        # Default preview config
        self.preview_config = {
            "command": "bat",
            "window": "wrap:50%:<40(up)",
            "label": ""
        }
        if preview_config:
            self.preview_config.update(preview_config)
            
        # Default FZF config
        self.fzf_config = {
            "delimiter": "::",
            "tiebreak": "begin,index",
            "info": "right",
            "ellipsis": ""
        }
        if fzf_config:
            self.fzf_config.update(fzf_config)


def build_wikilink_fzf_manager(config: WikilinkConfig, notes_dir: str) -> FzfManager:
    """
    Create and configure a FzfManager with bindings for wikilink generation.
    
    Args:
        config: WikilinkConfig instance with configuration
        notes_dir: Path to the notes directory
        
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
            description=f"Edit note in nvim",
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
    manager.add_help_binding(f"zk-wikilink --profile {config.name}")
    
    return manager


def create_wikilink_from_selection(
    selection: str,
    config: WikilinkConfig,
    notes_dir: str,
    debug: bool = False
) -> str:
    """
    Create a wikilink from the FZF selection.
    
    Args:
        selection: The raw selection from FZF
        config: The wikilink configuration to use
        notes_dir: Path to the notes directory
        debug: Enable debug output
        
    Returns:
        A formatted wikilink string
    """
    # Parse the selection (fields separated by "::")
    parts = selection.split("::")
    filename = parts[0].strip()
    
    if debug:
        logger.debug(f"Selected filename: {filename}")
    
    # Get additional fields for the selected note
    try:
        from typer.testing import CliRunner
        runner = CliRunner()
        
        # Create a list of all fields we might need
        all_fields = list(set(config.alias_fields))
        
        fields_args = [
            "list",
            "--mode", "notes",
            "-i", "index.json",
            "--stdin"
        ]
        
        # Add all fields we want to retrieve
        for field in all_fields:
            fields_args.extend(["--fields", field])
        
        if debug:
            logger.debug(f"Running fields command with args: {' '.join(fields_args)}")
        
        # Run the query with stdin input
        result = runner.invoke(query_app, fields_args, input=filename, catch_exceptions=False)
        
        if result.exit_code != 0:
            logger.error(f"Error getting fields: Exit code {result.exit_code}")
            return f"[[{filename}]]"  # Fallback to basic wikilink
            
        fields_output = result.stdout.strip()
        
    except Exception as e:
        logger.error(f"Error running fields command: {e}")
        return f"[[{filename}]]"  # Fallback to basic wikilink
    
    # Parse the fields output
    fields_parts = fields_output.split("::")
    field_values = {}
    
    # Make sure we're correctly associating field names with their values
    # The output of the query command will have values in the same order as the fields we requested
    for i, field in enumerate(all_fields):
        if i < len(fields_parts):
            # Only populate if the value is not empty
            if fields_parts[i].strip():
                field_values[field] = fields_parts[i].strip()
            else:
                field_values[field] = ""
        else:
            field_values[field] = ""
            
    if debug:
        logger.debug(f"Parsed field values for alias selection: {field_values}")
    
    if debug:
        logger.debug(f"Retrieved fields: {field_values}")
    
    # Find the best alias based on the priority order in config.alias_fields
    alias = None
    if debug:
        logger.debug(f"Looking for alias using fields (in priority order): {config.alias_fields}")
    
    for field in config.alias_fields:
        if field in field_values and field_values[field]:
            alias = field_values[field]
            if debug:
                logger.debug(f"Using '{field}' for alias: {alias}")
            break
            
    if debug and not alias:
        logger.debug("Could not find any suitable alias, using basic wikilink")
    
    # Build the wikilink
    if alias:
        return f"[[{filename}|{alias}]]"
    else:
        return f"[[{filename}]]"


def run_wikilink_generator(
    config: WikilinkConfig,
    notes_dir: str,
    debug: bool = False
) -> None:
    """
    Run the wikilink generator with the given configuration.
    
    Args:
        config: WikilinkConfig instance with configuration
        notes_dir: Path to the notes directory
        debug: Enable debug logging
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Running wikilink generator with profile: {config.name}")
        logger.debug(f"Notes directory: {notes_dir}")
    
    # Build the FzfManager with appropriate bindings
    fzf_manager = build_wikilink_fzf_manager(config, notes_dir)
    
    # Change to notes directory
    try:
        os.chdir(notes_dir)
        if debug:
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
                "--color", "always"
            ]
            
            # Add tag filters
            for tag in config.filter_tags:
                query_args.extend(["--filter-tag", tag])
                
            # We need to ensure filename is always the first field in query results
            # even if it's not in the display fields
            all_query_fields = []
            
            # Make sure filename is first in the query fields
            if "filename" not in config.search_fields and "filename" not in config.display_fields:
                all_query_fields.append("filename")
            elif "filename" in config.search_fields:
                all_query_fields.append("filename")
            
            # Add remaining search fields and display fields
            for field in config.search_fields:
                if field != "filename" and field not in all_query_fields:
                    all_query_fields.append(field)
                    
            for field in config.display_fields:
                if field not in all_query_fields:
                    all_query_fields.append(field)
            
            if debug:
                logger.debug(f"Final query fields: {all_query_fields}")
            
            # Add all needed fields to query
            for field in all_query_fields:
                query_args.extend(["--fields", field])
            
            if debug:
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
    
    # Prepare additional fzf arguments based on the config
    additional_args = [
        "--delimiter", config.fzf_config["delimiter"],
        "--tiebreak", config.fzf_config["tiebreak"],
        "--info", config.fzf_config["info"],
        "--ellipsis", config.fzf_config["ellipsis"],
    ]
    
    # We should use the same field ordering as used in the query
    all_query_fields = []
    
    # Make sure filename is first in the query fields
    if "filename" not in config.search_fields and "filename" not in config.display_fields:
        all_query_fields.append("filename")
    elif "filename" in config.search_fields:
        all_query_fields.append("filename")
    
    # Add remaining search fields and display fields
    for field in config.search_fields:
        if field != "filename" and field not in all_query_fields:
            all_query_fields.append(field)
            
    for field in config.display_fields:
        if field not in all_query_fields:
            all_query_fields.append(field)
    
    # Calculate with-nth parameter based on display fields
    display_indices = []
    
    # For each display field, find its position in the query results (1-based for FZF)
    for display_field in config.display_fields:
        if display_field in all_query_fields:
            # Add 1 because FZF is 1-indexed
            index = all_query_fields.index(display_field) + 1
            display_indices.append(str(index))
    
    # Make sure we're getting proper string representation with FZF
    display_indices_str = ",".join(display_indices)
    
    if debug:
        logger.debug(f"All query fields: {all_query_fields}")
        logger.debug(f"Display fields: {config.display_fields}")
        logger.debug(f"Display indices: {display_indices_str}")
    
    additional_args.extend([
        "--with-nth", display_indices_str,
        "--preview-label", config.preview_config["label"],
        "--preview", f"{config.preview_config['command']} {notes_dir}/{{1}}.md",
        "--preview-window", config.preview_config["window"],
    ])
    
    # Get complete fzf arguments
    fzf_args = fzf_manager.get_fzf_args(additional_args)
    
    if debug:
        logger.debug(f"Running fzf command: {' '.join(fzf_args)}")
    
    try:
        # Pipe the query results into fzf
        fzf_result = subprocess.run(fzf_args, stdin=p_list.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p_list.stdout.close()
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)
    
    selection = fzf_result.stdout.strip()
    if debug:
        logger.debug(f"fzf selection: {selection}")
    
    if not selection:
        if debug:
            logger.debug("No selection made")
        sys.exit(0)
    
    # Create the wikilink from the selection
    wikilink = create_wikilink_from_selection(selection, config, notes_dir, debug)
    
    if debug:
        logger.debug(f"Final wikilink: {wikilink}")
    
    # Send the processed link to the active tmux pane
    try:
        subprocess.run(["tmux", "send-keys", wikilink], check=True)
        if debug:
            logger.debug("Sent link to tmux")
    except Exception as e:
        logger.error(f"Error sending keys to tmux: {e}")
        sys.exit(1)
    
    if debug:
        logger.debug(f"Wikilink generator ({config.name} profile) completed successfully")


def main():
    """Main entry point for the wikilink generator module."""
    try:
        import typer
        from typing import Optional as OptionalType
        from pathlib import Path
        
        app = typer.Typer(help="Generate and insert wikilinks from configurable note searches")
        
        @app.command()
        def generate(
            profile: str = typer.Option(
                "default",
                "--profile", "-p",
                help="Configuration profile to use"
            ),
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
            Search for notes and insert selected entries as wikilinks.
            
            Use --profile to specify which wikilink configuration to use.
            """
            # Load configuration
            if config_file:
                from zk_core.config import load_config_from_file
                config = load_config_from_file(str(config_file))
            else:
                config = load_config()
            
            # Get configuration values
            notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
            notes_dir = resolve_path(notes_dir)
            
            # Get wikilink specific configuration
            wikilink_configs = get_config_value(config, "wikilink", {})
            profile_config = get_config_value(wikilink_configs, profile, None)
            
            if not profile_config:
                # If profile doesn't exist, create a default one
                if profile == "default":
                    logger.info("No default wikilink profile found, using built-in defaults")
                    profile_config = {
                        "filter_tags": [],
                        "search_fields": ["filename", "title"],
                        "display_fields": ["filename", "title"],
                        "alias_fields": ["title", "aliases"],
                    }
                else:
                    logger.error(f"Wikilink profile '{profile}' not found in configuration")
                    sys.exit(1)
            
            # Convert the config dict to a WikilinkConfig object
            wikilink_config = WikilinkConfig(
                name=profile,
                filter_tags=profile_config.get("filter_tags", []),
                search_fields=profile_config.get("search_fields", ["filename", "title"]),
                # Explicitly use the display_fields or default to search_fields
                display_fields=profile_config.get("display_fields", profile_config.get("search_fields", ["filename", "title"])),
                alias_fields=profile_config.get("alias_fields", ["title", "aliases"]),
                preview_config=profile_config.get("preview", None),
                fzf_config=profile_config.get("fzf", None)
            )
            
            # If --list-hotkeys was specified, print the hotkey help and exit
            if list_hotkeys:
                # Build the FzfManager just to get the hotkeys
                fzf_manager = build_wikilink_fzf_manager(wikilink_config, notes_dir)
                FzfHelper.print_hotkeys(fzf_manager, f"WIKILINK GENERATOR ({profile}) KEYBOARD SHORTCUTS")
                sys.exit(0)
                
            # Run the wikilink generator
            run_wikilink_generator(wikilink_config, notes_dir, debug)
            
        # Define the list-profiles command
        @app.command()
        def list_profiles(
            config_file: OptionalType[Path] = typer.Option(
                None,
                "--config", "-c",
                help="Path to config file (default: ~/.config/zk_scripts/config.yaml)",
                exists=False,
                file_okay=True,
                dir_okay=False
            )
        ) -> None:
            """List available wikilink generator profiles in the configuration."""
            # Load configuration
            if config_file:
                from zk_core.config import load_config_from_file
                config = load_config_from_file(str(config_file))
            else:
                config = load_config()
            
            # Get wikilink specific configuration
            wikilink_configs = get_config_value(config, "wikilink", {})
            
            # Print the available profiles
            print("\033[1;36mAvailable Wikilink Generator Profiles:\033[0m")
            
            if not wikilink_configs:
                print("  No profiles defined. Add profiles to your config.yaml file.")
                print("\n  Example configuration:")
                print("""
  wikilink:
    person:
      filter_tags: ["person"]
      search_fields: ["filename", "aliases", "givenName", "familyName"]
      alias_fields: ["aliases", "givenName"]
    book:
      filter_tags: ["book"]
      search_fields: ["filename", "title", "author"]
      alias_fields: ["title"]
                """)
                return
            
            for profile_name, profile_config in wikilink_configs.items():
                filter_tags = ", ".join(profile_config.get("filter_tags", []) or ["none"])
                fields = ", ".join(profile_config.get("search_fields", ["filename"]))
                alias_fields = ", ".join(profile_config.get("alias_fields", ["title", "aliases"]))
                
                print(f"\n\033[1;33m{profile_name}:\033[0m")
                print(f"  \033[1;32mFilter tags:\033[0m {filter_tags}")
                print(f"  \033[1;32mSearch fields:\033[0m {fields}")
                print(f"  \033[1;32mAlias fields:\033[0m {alias_fields}")
        
        # Run the app
        app()
            
    except ImportError:
        # Fall back to basic implementation if typer is not available
        print("ERROR: Typer package not found. Please install typer: pip install typer")
        sys.exit(1)


if __name__ == "__main__":
    main()