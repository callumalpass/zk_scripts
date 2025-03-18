"""
Bibliography viewer module.

This module is a thin wrapper around the bibliography viewer functionality in the
zk_core.bibliography.viewer module.

It provides an interactive interface for viewing bibliography entries with custom
keybindings and formatting.
"""

import sys
import logging
from enum import Enum
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the bibview script."""
    try:
        import typer
        from zk_core.bibliography.viewer import run_viewer
        
        # Create a new Typer app
        app = typer.Typer(help="Interactive bibliography viewer with FZF interface")
        
        # Define sort options enum
        class SortOrder(str, Enum):
            YEAR = "year"
            DATE_MODIFIED = "dateModified"
        
        # Define the main command
        @app.command()
        def view(
            sort: SortOrder = typer.Option(
                SortOrder.DATE_MODIFIED,
                "--sort",
                "-s",
                help="Sort bibliography entries by year or modification date"
            ),
            debug: bool = typer.Option(
                False,
                "--debug",
                "-d", 
                help="Enable debug output"
            ),
            list_hotkeys: bool = typer.Option(
                False,
                "--list-hotkeys",
                "-l",
                help="Print a list of available FZF hotkeys and their functions, then exit"
            ),
            config_file: Optional[str] = typer.Option(
                None,
                "--config",
                "-c",
                help="Path to config file (default: ~/.config/zk_scripts/config.yaml)"
            )
        ) -> None:
            """
            View and interact with your bibliography entries using FZF.
            
            This interactive viewer lets you browse, search, and open bibliography entries
            with a variety of keyboard shortcuts and integrations with external tools.
            """
            # Create an args object compatible with the run_viewer function
            class Args:
                pass
                
            args = Args()
            args.sort = sort
            args.debug = debug
            args.list_hotkeys = list_hotkeys
            
            if config_file:
                args.config = config_file
                
            # Enable debug logging if debug flag is set
            if debug:
                logger.setLevel(logging.DEBUG)
                
            # Call the viewer with our args
            exit_code = run_viewer(args=args)
            
            # Return the appropriate exit code
            raise typer.Exit(exit_code)
        
        # Run the app
        app()
        
    except ImportError:
        # Fallback to argparse if typer is not available
        import argparse
        from zk_core.bibliography.viewer import run_viewer
        
        print("WARNING: Typer package not found, falling back to basic argparse implementation.")
        print("For a better CLI experience, install typer: pip install typer")
        
        parser = argparse.ArgumentParser(description="Bibliography viewer")
        parser.add_argument(
            "--sort", 
            choices=["year", "dateModified"], 
            default="dateModified",
            help="Sort bibliography by 'year' or 'dateModified'"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug output")
        parser.add_argument("--list-hotkeys", action="store_true", 
                           help="Print a list of available fzf hotkeys and their functions, then exit")
        parser.add_argument("--config", help="Path to config file")
        args = parser.parse_args()
        
        # Enable debug logging if debug flag is set
        if args.debug:
            logger.setLevel(logging.DEBUG)
            
        # Call the viewer with our args
        exit_code = run_viewer(args=args)
        
        # Exit with the appropriate exit code
        sys.exit(exit_code)

if __name__ == "__main__":
    main()