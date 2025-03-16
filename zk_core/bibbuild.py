"""
Bibliography building module.

This module is a thin wrapper around the bibliography builder functionality in the
zk_core.bibliography.builder module.

It creates:
- A list of citation keys
- A bibliography JSON file
"""

import sys
import logging
from zk_core.bibliography.builder import run_build

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """Main entry point for the script."""
    # Call the builder's run_build function
    exit_code = run_build()
    
    # Exit with the appropriate exit code
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
