"""
FZF utilities for ZK Core.

This module provides shared utilities for working with FZF:
- Standard FZF binding creation
- Hotkey printing with consistent formatting
- Command building for FZF
- Additional helper functions for working with FZF
"""

import os
import sys
import logging
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from pathlib import Path

from zk_core.fzf_manager import FzfManager, FzfBinding
from zk_core.commands import CommandExecutor

logger = logging.getLogger(__name__)

# ANSI Colors for UI formatting
COLORS = {
    'reset': "\033[0m",
    'bold': "\033[1m",
    'italic': "\033[3m",
    'underline': "\033[4m",
    'red': "\033[31m",
    'green': "\033[32m",
    'yellow': "\033[33m",
    'blue': "\033[34m",
    'magenta': "\033[35m",
    'cyan': "\033[36m",
    'white': "\033[37m",
}

def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    if color in COLORS:
        return f"{COLORS[color]}{text}{COLORS['reset']}"
    return text

class FzfHelper:
    """Helper class for working with FZF interfaces."""
    
    @staticmethod
    def print_hotkeys(
        fzf_manager: FzfManager, 
        title: str = "KEYBOARD SHORTCUTS", 
        categories: Optional[Dict[str, List[str]]] = None,
        sort_categories: bool = True
    ) -> None:
        """
        Print formatted hotkeys for FZF with custom categories.
        
        Args:
            fzf_manager: The FzfManager with bindings to display
            title: Title for the hotkey display
            categories: Optional dictionary of category name to list of keys to display
            sort_categories: Whether to sort categories alphabetically
        """
        print(colorize(f"=== {title} ===", 'cyan') + colorize("", 'bold'))
        
        key_to_binding = {b.key: b for b in fzf_manager.bindings}
        
        if categories:
            # Use provided custom categories
            category_items = categories.items()
            if sort_categories:
                category_items = sorted(category_items)
                
            for category_name, keys in category_items:
                print(f"\n{colorize(category_name, 'yellow')}")
                for key in keys:
                    if key in key_to_binding:
                        binding = key_to_binding[key]
                        print(f"  {colorize(binding.key, 'green'):<12} : {binding.desc}")
            
            # Print any remaining keys in "Other Commands" category
            all_categorized_keys = []
            for keys in categories.values():
                all_categorized_keys.extend(keys)
            
            other_keys = [k for k in key_to_binding.keys() if k not in all_categorized_keys]
            if other_keys:
                print(f"\n{colorize('Other Commands', 'yellow')}")
                for key in other_keys:
                    binding = key_to_binding[key]
                    print(f"  {colorize(binding.key, 'green'):<12} : {binding.desc}")
        else:
            # Group by category in the manager
            categories_dict: Dict[str, List[FzfBinding]] = {}
            for binding in fzf_manager.bindings:
                cat = binding.category or "Other"
                if cat not in categories_dict:
                    categories_dict[cat] = []
                categories_dict[cat].append(binding)
            
            # Print grouped by category
            category_names = list(categories_dict.keys())
            if sort_categories:
                category_names.sort()
                
            for category in category_names:
                bindings = categories_dict[category]
                print(f"\n{colorize(category, 'yellow')}")
                for binding in sorted(bindings, key=lambda b: b.key):
                    print(f"  {colorize(binding.key, 'green'):<12} : {binding.desc}")
        
        print(f"\n{colorize('Press q to exit this help screen', 'cyan')}")
    
    @staticmethod
    def run_fzf(
        fzf_manager: FzfManager,
        input_data: str,
        additional_args: Optional[List[str]] = None,
        extra_bindings: Optional[List[FzfBinding]] = None,
        suppress_errors: bool = False
    ) -> subprocess.CompletedProcess:
        """
        Run FZF with the given input and configuration.
        
        Args:
            fzf_manager: FzfManager with bindings
            input_data: Input data to pipe to FZF
            additional_args: Additional arguments to pass to FZF
            extra_bindings: Additional bindings to add just for this run
            suppress_errors: Whether to suppress error logging
            
        Returns:
            CompletedProcess instance from subprocess
        """
        if extra_bindings:
            for binding in extra_bindings:
                fzf_manager.add_binding(binding)
                
        fzf_args = fzf_manager.get_fzf_args(additional_args)
        
        try:
            # Create a process to generate the input
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(input_data)
                tmp.flush()
                
                # Run FZF with the input
                cmd = ["cat", tmp_path]
                p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                fzf_result = subprocess.run(fzf_args, stdin=p1.stdout, capture_output=True, text=True)
                p1.stdout.close()  # Allow p1 to receive a SIGPIPE if fzf exits
                
                # Clean up
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                    
                return fzf_result
        except Exception as e:
            if not suppress_errors:
                logger.error(f"Error running fzf: {e}")
            raise
    
    @staticmethod
    def create_standard_bindings(
        fzf_manager: FzfManager,
        socket_path: str,
        notes_dir: str,
        index_file: str,
        additional_bindings: Optional[List[FzfBinding]] = None
    ) -> FzfManager:
        """
        Add standard ZK bindings to an FZF manager.
        
        Args:
            fzf_manager: The FZF manager to add bindings to
            socket_path: Path to the Neovim socket for editor integration
            notes_dir: Path to the notes directory
            index_file: Path to the index file
            additional_bindings: Optional list of additional bindings
            
        Returns:
            The updated FZF manager
        """
        # Standard navigation and viewing bindings
        fzf_manager.add_bindings([
            FzfBinding(
                key="Enter",
                command=f"Enter:execute[nvim --server {socket_path} --remote {notes_dir}/{{+1}}.md]+abort",
                description="Open the selected note in nvim (via socket).",
                category="Navigation"
            ),
            FzfBinding(
                key="alt-?",
                command="alt-?:toggle-preview",
                description="Toggle fzf preview window on/off.",
                category="Navigation"
            ),
            FzfBinding(
                key="?",
                command=f"?:reload(zk-query info -i {index_file} )",
                description="Display additional info for the selected note.",
                category="Navigation"
            ),
        ])
        
        # Add help binding
        fzf_manager.add_help_binding("zk-fzf")
        
        # Add any additional bindings
        if additional_bindings:
            fzf_manager.add_bindings(additional_bindings)
            
        return fzf_manager

    @staticmethod
    def build_standard_preview_command(
        index_file: str, 
        bat_theme: str = "default",
        show_backlinks: bool = True
    ) -> str:
        """
        Build a standard preview command for FZF.
        
        Args:
            index_file: Path to the index file
            bat_theme: Theme for bat
            show_backlinks: Whether to show backlinks in the preview
            
        Returns:
            FZF preview command string
        """
        backlinks_part = f"echo \"Backlinks:\"; zk-query list -i {index_file} --filter-outgoing-link {{1}} --color always; " if show_backlinks else ""
        return (
            f"{backlinks_part}bat --theme=\"{bat_theme}\" --color=always --decorations=never {{1}}.md "
            f"-H {{2}} 2> /dev/null || bat {{1}}.md"
        )