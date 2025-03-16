#!/usr/bin/env python3
"""Test script to demonstrate automatic config file creation."""

import os
import sys
from pathlib import Path

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zk_core.config import load_config
from zk_core.constants import DEFAULT_CONFIG_PATH

def main():
    """Test loading configuration with automatic creation."""
    # Test using the default path
    print(f"Testing config loading with default path: {DEFAULT_CONFIG_PATH}")
    config = load_config()
    print(f"Config loaded successfully with {len(config)} sections")
    
    # Test using a custom path
    custom_path = os.path.expanduser("~/test_zk_config.yaml")
    print(f"\nTesting config loading with custom path: {custom_path}")
    
    # Remove test config if it exists
    if os.path.exists(custom_path):
        os.remove(custom_path)
        print(f"Removed existing test config file: {custom_path}")
    
    # Load with custom path (should create the file)
    config = load_config(custom_path)
    print(f"Config loaded successfully with {len(config)} sections")
    
    # Check if the file was created
    if os.path.exists(custom_path):
        print(f"Test config file created successfully: {custom_path}")
        
        # Print the file size
        file_size = os.path.getsize(custom_path)
        print(f"Config file size: {file_size} bytes")
        
        # Get a file listing
        with open(custom_path, 'r', encoding='utf-8') as f:
            first_ten_lines = [next(f) for _ in range(10)]
        print("\nFirst 10 lines of config file:")
        for i, line in enumerate(first_ten_lines, 1):
            print(f"{i}: {line.rstrip()}")
    else:
        print("Error: Test config file was not created")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()