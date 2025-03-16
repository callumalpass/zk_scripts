"""
A shared module for managing fzf integrations across different tools.

This module provides common functionality for handling fzf bindings,
generating help menus, and configuring fzf instances with consistent options.
"""

import os
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class FzfBinding:
    """A class representing a single fzf keybinding."""
    
    def __init__(self, key: str, command: str, description: str, category: str = "Other"):
        """
        Initialize a new fzf binding.
        
        Args:
            key: The key or key combination (e.g., "alt-h", "ctrl-e")
            command: The fzf command string 
            description: A human-readable description of what the binding does
            category: Category for organizing bindings in help menus
        """
        self.key = key
        self.fzf_cmd = command
        self.desc = description
        self.category = category

class FzfManager:
    """A class for managing fzf bindings and configuration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize a new FzfManager instance.
        
        Args:
            config: Optional configuration dictionary
        """
        self.bindings = []
        self.config = config or {}
        self.categories = {
            "Navigation": [],
            "Filtering": [],
            "Editing": [],
            "Other": []
        }
        
    def add_binding(self, binding: FzfBinding) -> None:
        """
        Add a binding to the manager.
        
        Args:
            binding: The FzfBinding to add
        """
        self.bindings.append(binding)
        
        # Also add to category tracking
        if binding.category not in self.categories:
            self.categories[binding.category] = []
        self.categories[binding.category].append(binding)
        
    def add_bindings(self, bindings: List[FzfBinding]) -> None:
        """
        Add multiple bindings at once.
        
        Args:
            bindings: List of FzfBinding objects to add
        """
        for binding in bindings:
            self.add_binding(binding)
    
    def get_binding_strings(self) -> List[str]:
        """
        Get a list of fzf binding strings suitable for passing to fzf.
        
        Returns:
            List of strings formatted for fzf's --bind parameter
        """
        binding_strings = []
        for binding in self.bindings:
            binding_strings.append(binding.fzf_cmd)
        return binding_strings
    
    def get_hotkeys_info(self) -> List[Tuple[str, str, str]]:
        """
        Get a list of hotkeys and their descriptions.
        
        Returns:
            List of tuples containing (key, description, category)
        """
        return [(b.key, b.desc, b.category) for b in self.bindings]
    
    def get_fzf_args(self, additional_args: Optional[List[str]] = None) -> List[str]:
        """
        Build the complete fzf arguments list.
        
        Args:
            additional_args: Additional arguments to add to the fzf command
            
        Returns:
            List of strings to pass to subprocess.run
        """
        # Start with basic fzf arguments
        fzf_args = ["fzf", "--ansi"]
        
        # Add all bindings
        for binding in self.bindings:
            fzf_args.extend(["--bind", binding.fzf_cmd])
        
        # Add any additional arguments
        if additional_args:
            fzf_args.extend(additional_args)
            
        return fzf_args
    
    def print_help(self, custom_categories: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Print a formatted list of hotkeys and their descriptions.
        
        Args:
            custom_categories: Optional custom categorization of keys
        """
        print("\033[1;36m=== FZF KEYBOARD SHORTCUTS ===\033[0m")
        
        # Create a mapping of keys to bindings for easier lookup
        key_to_binding = {b.key: b for b in self.bindings}
        
        # If custom categories are provided, use them
        if custom_categories:
            for category_name, keys in custom_categories.items():
                print(f"\n\033[1;33m{category_name}:\033[0m")
                for key in keys:
                    if key in key_to_binding:
                        binding = key_to_binding[key]
                        print(f"  \033[1;32m{binding.key:<12}\033[0m : {binding.desc}")
        # Otherwise use the categories from the bindings
        else:
            for category_name, bindings in self.categories.items():
                if bindings:  # Only print categories with bindings
                    print(f"\n\033[1;33m{category_name}:\033[0m")
                    for binding in bindings:
                        print(f"  \033[1;32m{binding.key:<12}\033[0m : {binding.desc}")
        
        print("\n\033[1;36mPress q to exit this help screen\033[0m")
    
    def run_fzf(self, input_data: Optional[str] = None, 
                additional_args: Optional[List[str]] = None) -> subprocess.CompletedProcess:
        """
        Run fzf with the configured bindings and arguments.
        
        Args:
            input_data: Optional string data to pipe to fzf's stdin
            additional_args: Additional arguments to pass to fzf
            
        Returns:
            The completed process object from subprocess.run
        """
        fzf_args = self.get_fzf_args(additional_args)
        
        try:
            if input_data:
                result = subprocess.run(fzf_args, input=input_data, text=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(fzf_args, text=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result
        except Exception as e:
            logger.error(f"Error running fzf: {e}")
            raise
    
    def add_help_binding(self, script_name: str) -> None:
        """
        Add a standard help binding that shows all available hotkeys.
        
        Args:
            script_name: The name of the script to execute with --list-hotkeys
        """
        help_binding = FzfBinding(
            key="alt-h",
            command=f"alt-h:execute({script_name} --list-hotkeys | less -R)",
            description="Show this hotkeys help (prints the list of hotkeys).",
            category="Navigation"
        )
        self.add_binding(help_binding)