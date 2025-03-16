"""Configuration management for ZK Core."""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field, validator

from zk_core.constants import (
    DEFAULT_CONFIG_PATH, DEFAULT_NOTES_DIR, DEFAULT_INDEX_FILENAME, 
    DEFAULT_LOG_LEVEL, DEFAULT_FILENAME_FORMAT, DEFAULT_FILENAME_EXTENSION
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
    zk_index: IndexConfig = Field(default_factory=IndexConfig, description="Index configuration")
    query: QueryConfig = Field(default_factory=QueryConfig, description="Query configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    filename: FilenameConfig = Field(default_factory=FilenameConfig, description="Filename formatting configuration")
    
    # Additional configuration sections can be added here
    
    @validator('notes_dir')
    def resolve_notes_dir(cls, v: str) -> str:
        """Resolve notes directory path."""
        return resolve_path(v)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from the YAML config file.
    
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
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f) or {}
                
            # Validate configuration
            try:
                # First validate the basic structure
                validated_config = ZKConfig(**raw_config)
                # Convert back to dictionary for backward compatibility
                config = validated_config.dict()
                logger.debug(f"Loaded and validated configuration from {resolved_path}")
            except Exception as validation_error:
                logger.error(f"Configuration validation error: {validation_error}")
                logger.warning("Using default configuration with provided values where valid")
                # Use as much of the config as possible
                config = raw_config
        else:
            logger.warning(f"Config file '{resolved_path}' not found. Using defaults.")
    except Exception as e:
        logger.error(f"Error loading config file '{path}': {e}")
    
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