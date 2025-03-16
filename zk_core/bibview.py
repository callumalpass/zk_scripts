"""
Bibliography viewer module.

This module is a thin wrapper around the bibliography viewer functionality in the
zk_core.bibliography.viewer module.

It provides an interactive interface for viewing bibliography entries with custom
keybindings and formatting.
"""

import sys
import logging
import argparse
from zk_core.bibliography.viewer import run_viewer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the bibview script."""
    # Parse command line arguments to match the previous interface
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
    args = parser.parse_args()
    
    # Call the viewer's run_viewer function with the parsed args
    exit_code = run_viewer(args=args)
    
    # Exit with the appropriate exit code
    sys.exit(exit_code)

if __name__ == "__main__":
    main()