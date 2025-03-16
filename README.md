# ZK Core

A modular Python package for Zettelkasten-style note management.

## Overview

ZK Core is a collection of tools for managing, searching, and analyzing Zettelkasten-style markdown notes. It provides functionality for note indexing, querying, creating working memory notes, viewing backlinks, managing bibliographies, and more.

Originally a collection of separate scripts, ZK Core has been refactored into a modular, pip-installable Python package that reduces redundancy and improves maintainability.

## Features

- **Note Indexing**: Scan your notes directory and create a searchable index with metadata
- **Advanced Querying**: Search notes by tags, content, and other metadata
- **Interactive Note Navigation**: Browse notes using fuzzy-finder (fzf) interface
- **Backlink Tracking**: View and analyze connections between notes
- **Bibliography Management**: Handle academic citations and references
- **Person Search**: Quickly find and insert person references
- **Workout Logging**: Track exercises, sets, and workout sessions

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/zk_core.git
cd zk_core

# Install the package
pip install .
```

## Configuration

ZK Core uses a YAML configuration file located at `~/.config/zk_scripts/config.yaml`. Create this file with the following structure:

```yaml
# Main configuration
notes_dir: "~/notes"  # Path to your notes directory

# Index configuration
zk_index:
  index_file: "index.json"  # Name of the index file
  exclude_patterns: [".git", ".obsidian", "node_modules"]  # Directories to exclude

# Bibview configuration
bibview:
  bibliography_json: "~/Dropbox/bibliography.json"  # Path to bibliography JSON
  library: "~/biblib"  # Path to bibliography library
  notes_dir_for_zk: "~/notes"  # Path for Zettelkasten notes

# Person search configuration
personSearch:
  notes_dir: "~/notes"  # Path to notes directory
  py_zk: "~/bin/py_zk.py"  # Path to py_zk script
  bat_command: "bat"  # Command for preview

# Add other module-specific configurations as needed
```

## Usage

ZK Core provides several command-line tools:

### Note Indexing

```bash
# Generate an index of your notes
zk-index
```

### Note Querying

```bash
# List all notes with a specific tag
zk-query list --mode notes --filter-tag project

# Search notes containing a keyword
zk-query search "python"

# Get detailed information about a note
zk-query info "note-filename"
```

### Fuzzy Finder Interface

```bash
# Interactive fuzzy-finder interface for notes
zk-fzf
```

### Working Memory

```bash
# Create a new working memory note
zk-working-mem
```

### Backlinks Viewer

```bash
# View backlinks for a note
zk-backlinks "note-filename"
```

### Bibliography Tools

```bash
# Build bibliography files
zk-bib-build

# Interactive bibliography viewer
zk-bib-view
```

### Person Search

```bash
# Search for person notes and insert a link
zk-person-search
```

### Workout Logging

```bash
# Launch the workout logging interface
zk-workout-log

# Export workout history
zk-workout-log --export-history
```

## Development

### Package Structure

```
zk_core/
├── __init__.py
├── config/
│   └── __init__.py
├── utils.py
├── models.py
├── index.py
├── query.py
├── fzf_interface.py
├── working_mem.py
├── backlinks.py
├── bibbuild.py
├── bibview.py
├── person_search.py
└── workout_log.py
```

### Adding New Features

To add new functionality:

1. Create a new module in the `zk_core` directory
2. Add the module to `pyproject.toml` in the `project.scripts` section
3. Ensure it follows the modular pattern:
   - Uses the shared configuration system
   - Reuses utility functions when possible
   - Uses proper type hints
   - Provides good error handling and logging
4. Add tests for the new module

## License

[MIT License](LICENSE)