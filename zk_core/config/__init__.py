"""Configuration management for ZK Core."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator

from zk_core.constants import (
    DEFAULT_CONFIG_PATH, DEFAULT_NOTES_DIR, DEFAULT_INDEX_FILENAME, 
    DEFAULT_LOG_LEVEL, DEFAULT_FILENAME_FORMAT, DEFAULT_FILENAME_EXTENSION,
    DEFAULT_NVIM_SOCKET
)

logger = logging.getLogger(__name__)

class IndexConfig(BaseModel):
    """Index configuration model."""
    index_file: str = Field(default=DEFAULT_INDEX_FILENAME, description="Name of the index file")
    exclude_patterns: List[str] = Field(default_factory=lambda: [".git", ".obsidian", "node_modules"], 
                                      description="Directories to exclude")
    excluded_files: List[str] = Field(default_factory=lambda: ["README.md"], 
                                    description="Files to exclude from indexing")

class QueryConfig(BaseModel):
    """Query configuration model."""
    default_index: str = Field(default=DEFAULT_INDEX_FILENAME, description="Default index file")
    default_fields: List[str] = Field(default_factory=lambda: ["filename", "title", "tags"], 
                                    description="Default fields to display")

class LoggingConfig(BaseModel):
    """Logging configuration model."""
    level: str = Field(default=DEFAULT_LOG_LEVEL, description="Logging level")
    file: Optional[str] = Field(default=None, description="Log file path")
    
    @validator('level')
    def validate_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            logger.warning(f"Invalid logging level: {v}. Using default: {DEFAULT_LOG_LEVEL}")
            return DEFAULT_LOG_LEVEL
        return v.upper()

class FilenameConfig(BaseModel):
    """Filename formatting configuration model."""
    format: str = Field(
        default=DEFAULT_FILENAME_FORMAT, 
        description="Format string for generated filenames. Supports strftime format codes and {random:N} for random letters"
    )
    extension: str = Field(
        default=DEFAULT_FILENAME_EXTENSION, 
        description="File extension for generated files"
    )
    
    @validator('format')
    def validate_format(cls, v: str) -> str:
        """Validate filename format."""
        if not v:
            logger.warning(f"Invalid filename format: {v}. Using default: {DEFAULT_FILENAME_FORMAT}")
            return DEFAULT_FILENAME_FORMAT
        return v
    
    @validator('extension')
    def validate_extension(cls, v: str) -> str:
        """Validate filename extension."""
        if not v.startswith('.'):
            v = '.' + v
        return v

class ZKConfig(BaseModel):
    """Main configuration model."""
    notes_dir: str = Field(default=DEFAULT_NOTES_DIR, description="Path to notes directory")
    socket_path: str = Field(default=DEFAULT_NVIM_SOCKET, description="Path to Neovim socket")
    zk_index: IndexConfig = Field(default_factory=IndexConfig, description="Index configuration")
    query: QueryConfig = Field(default_factory=QueryConfig, description="Query configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    filename: FilenameConfig = Field(default_factory=FilenameConfig, description="Filename formatting configuration")
    
    # Additional configuration sections can be added here
    
    @validator('notes_dir', 'socket_path')
    def resolve_paths(cls, v: str) -> str:
        """Resolve paths with environment variables and user home."""
        return resolve_path(v)

def create_default_config(config_file: Path) -> Dict[str, Any]:
    """
    Create a default configuration file.
    
    Args:
        config_file: Path to the config file to create.
        
    Returns:
        Dict with default configuration values.
    """
    # Create default configuration
    default_config = ZKConfig()
    config_dict = default_config.dict()
    
    # Add additional configuration sections that aren't in the core ZKConfig
    config_dict.update({
        "fzf_interface": {
            "bat_command": "bat",
            "fzf_args": "--height=80% --layout=reverse --info=inline",
            "diary_subdir": "",
            "bat_theme": "default"
        },
        "working_mem": {
            "file": os.path.join(DEFAULT_NOTES_DIR, "workingMem.md"),
            "template_path": os.path.join(DEFAULT_NOTES_DIR, "templates/working_mem.md"),
            "editor": "nvim",
            "tag": "working_mem"
        },
        "backlinks": {
            "notes_dir": DEFAULT_NOTES_DIR,
            "bat_theme": "Dracula"
        },
        "bibview": {
            "bibliography_json": "~/bibliography.json",
            "bibhist": "~/.bibhist",
            "library": "~/biblib",
            "notes_dir_for_zk": DEFAULT_NOTES_DIR,
            "bat_theme": "Dracula",
            "bibview_open_doc_script": "~/bin/open_doc.sh",
            "llm_path": "~/bin/llm",
            "zk_script": "~/bin/zk",
            "link_zathura_tmp_script": "~/bin/link_zathura.sh",
            "obsidian_socket": "/tmp/obsidiansocket",
            "getbibkeys_script": "~/bin/getbibkeys.sh"
        },
        "personSearch": {
            "notes_dir": DEFAULT_NOTES_DIR,
            "bat_command": "bat"
        }
    })
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    # Create friendly config file with description
    config_description = """# ZK Scripts Configuration
# 
# This configuration file was automatically generated with default values.
# You can modify these values to customize the behavior of zk_scripts.
#
# Main configuration sections:
#
# notes_dir: Path to your notes directory (default: ~/notes)
#
# zk_index:
#   - Settings for the indexing system (excluded directories, index filename)
#
# query:
#   - Settings for the query tool (default fields, output format)
#
# filename:
#   - Settings for generated filenames (format, extension)
#
# fzf_interface:
#   - Settings for the fuzzy finder interface
#
# working_mem:
#   - Settings for the working memory feature
#
# backlinks:
#   - Settings for the backlinks viewer
#
# bibview:
#   - Settings for bibliography integration
#
# personSearch:
#   - Settings for person search functionality
#
# logging:
#   - Logging settings (level, file path)
#
"""
    
    # Write configuration to file with description
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_description)
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created default configuration file at {config_file}")
    return config_dict

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from the YAML config file.
    If the config file doesn't exist, create it with default values.
    
    Args:
        config_path: Path to the config file. If None, default is used.
        
    Returns:
        Dict with configuration values.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    config: Dict[str, Any] = {}
    
    try:
        # Resolve the path to expand ~ to home directory
        resolved_path = resolve_path(path)
        config_file = Path(resolved_path)
        
        if config_file.exists():
            # Load existing configuration
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
                
            # Validate configuration
            try:
                # First validate the basic structure
                validated_config = ZKConfig(**raw_config)
                # Convert back to dictionary for backward compatibility
                config = validated_config.dict()
                
                # Preserve any additional sections not covered by ZKConfig
                for key, value in raw_config.items():
                    if key not in config:
                        config[key] = value
                
                logger.debug(f"Loaded and validated configuration from {resolved_path}")
            except Exception as validation_error:
                logger.error(f"Configuration validation error: {validation_error}")
                logger.warning("Using default configuration with provided values where valid")
                # Use as much of the config as possible
                config = raw_config
        else:
            # Create default configuration file
            logger.warning(f"Config file '{resolved_path}' not found. Creating with defaults.")
            config = create_default_config(config_file)
    except Exception as e:
        logger.error(f"Error loading config file '{path}': {e}")
        # Return minimal default config
        config = ZKConfig().dict()
    
    return config

def resolve_path(path: str) -> str:
    """Resolve path with environment variables and user home."""
    return os.path.expanduser(os.path.expandvars(path))

def get_notes_dir(config: Dict[str, Any]) -> str:
    """Get notes directory from config or use default."""
    notes_dir = config.get("notes_dir", DEFAULT_NOTES_DIR)
    return resolve_path(notes_dir)

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get configuration value using a dot-separated path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "section.key")
        default: Default value if path not found
        
    Returns:
        Configuration value or default if not found
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
            
    return current