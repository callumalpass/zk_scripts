"""Constants for the ZK Core package."""

import re

# Regular expression patterns
WIKILINK_RE = re.compile(r'\[\[([^|\]]+)(?:\|[^\]]+)?\]\]')
INLINE_CITATION_RE = re.compile(r'@([a-zA-Z0-9_]+)(?:\s+p\.\s+\d+)?')
WIKILINKED_CITATION_RE = re.compile(r'\[\[.*?\|\[@([a-zA-Z0-9_]+).*?\]\]')
WIKILINK_ALL_RE = re.compile(r'\[\[([^|\]]+)(?:\|([^\]]+))?\]\]')
CITATION_ALIAS_RE = re.compile(r'^\[@([a-zA-Z0-9_]+)')

# File extensions
MARKDOWN_EXTENSIONS = [".md", ".markdown"]

# Default configuration values
DEFAULT_CONFIG_PATH = "~/.config/zk_scripts/config.yaml"
DEFAULT_NOTES_DIR = "~/notes"
DEFAULT_INDEX_FILENAME = "index.json"
DEFAULT_LOG_LEVEL = "INFO"

# Filename format constants
DEFAULT_FILENAME_FORMAT = "%y%m%d{random:3}"
DEFAULT_FILENAME_EXTENSION = ".md"

# Index processing constants
DEFAULT_NUM_WORKERS = 64
MAX_CHUNK_SIZE = 1000
