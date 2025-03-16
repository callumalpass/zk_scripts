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
git clone https://github.com/callumalpass/zk_core.git
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

The indexing script is the foundation of all other functionality in this package. 
It is designed to be as fast as possible---even on slow machines. 
The json index is used by several other scripts for super-fast querying and analysis. 

```bash
# Generate an index of your notes
zk-index

# Force a full reindex 
zk-index --full-reindex

# Generate embeddings along with the index
zk-index --generate-embeddings
```

The indexing script (`zk-index`) is a powerful tool that scans your Zettelkasten directory and creates a searchable index with rich metadata about each note. Key features include:

- **Incremental Indexing**: Only processes files that have changed since the last indexing run, making it efficient for large note collections
- **Metadata Extraction**: Extracts frontmatter, tags, date information, and other metadata from your notes
- **Backlink Tracking**: Automatically identifies and records backlinks between notes
- **Citation Extraction**: Identifies and indexes academic citations within your notes
- **Embedding Generation**: Optionally generates vector embeddings using OpenAI models for semantic search functionality
- **Parallel Processing**: Uses multiple processor cores for faster indexing of large note collections
- **Customization**: Allows exclusion of specific directories and patterns from indexing

The generated index is stored as a JSON file, which serves as a foundation for the various search and query operations.

### Note Querying

```bash
# List all notes with a specific tag
zk-query list --mode notes --filter-tag project

# Search notes containing a keyword
zk-query search "python"

# Get detailed information about a note
zk-query info "note-filename"

# Find notes similar to a specific note using embeddings
zk-query search-embeddings --query-file "path/to/note.md"

# Find notes semantically related to a text query
zk-query search-embeddings --query "artificial intelligence applications"
```

The querying script (`zk-query`) provides comprehensive tools for searching, filtering, and analyzing your notes. It works with the index created by `zk-index` and offers several powerful capabilities:

- **Multiple Query Modes**:
  - `list`: Display notes filtered by various criteria
  - `info`: Show detailed statistics about your note collection
  - `search-embeddings`: Find semantically similar notes using vector embeddings

- **Advanced Filtering Options**:
  - Filter by tags (with hierarchical tag support and AND/OR logic)
  - Filter by date range
  - Filter by content
  - Filter by backlinks or outgoing links
  - Filter by word count range
  - Filter by custom frontmatter fields

- **Specialized Views**:
  - View orphan notes (notes with no connections)
  - Find dangling links (references to non-existent notes)
  - List all unique tags
  - View notes with no tags and no connections

- **Flexible Output Formats**:
  - Plain text with customizable formatting
  - CSV for spreadsheet import
  - JSON for programmatic use
  - Tabular display for terminal viewing

- **Sorting and Organization**:
  - Sort by modification/creation date, word count, or filename
  - Customize which fields are displayed in results

The query system enables powerful workflows for managing and exploring your Zettelkasten, helping you discover connections and insights within your knowledge base.

### Fuzzy Finder Interface

```bash
# Interactive fuzzy-finder interface for notes
zk-fzf
```

The fuzzy finder interface (`zk-fzf`) provides a highly interactive terminal-based environment for browsing and searching your notes using the powerful [fzf](https://github.com/junegunn/fzf) tool. This central tool integrates many features of the ZK Core system:

- **Multi-Modal Search**:
  - Rapid fuzzy filtering as you type
  - Full-text search via ripgrep integration
  - Semantic search using embeddings to find conceptually similar notes
  - Tag-based filtering with dedicated shortcuts

- **Rich Preview Context**:
  - Inline note content preview with syntax highlighting
  - Automatic backlink display for selected notes
  - Preview toggling for different viewing modes

- **Powerful Note Management**:
  - Quick editing in your preferred editor (nvim/Obsidian)
  - Create working memory notes with selected references
  - View orphan notes and untagged content
  - Filter specialized note types (literature, diary, etc.)

- **Advanced Navigation Features**:
  - Extensive keyboard shortcuts for efficient workflows
  - Generate formatted Markdown links from selections
  - Toggle between different view modes (all notes, tags only, etc.)
  - Integrated help system with shortcut documentation

- **System Integration**:
  - Automatic index updates to ensure fresh content
  - Background notelist generation for external tools
  - tmux integration for sending links to other panes

The fuzzy finder serves as a command center for your Zettelkasten, combining searching, browsing, note management, and analysis in a single, keyboard-driven interface optimized for efficiency and discovery.

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

The backlinks viewer (`zk-backlinks`) is a sophisticated tool for exploring the network of connections in your Zettelkasten system. It reveals both explicit references and deeper conceptual relationships between your notes:

- **Comprehensive Backlink Analysis**: Identifies all notes that directly reference the target note, creating a map of explicit connections in your knowledge network
- **Rich Context Display**: Shows the full paragraph surrounding each backlink reference, providing the complete context in which the connection was made
- **Advanced Semantic Similarity**: Leverages vector embeddings to discover conceptually related notes that don't contain explicit links but share thematic connections
- **Semantic Neighborhood Exploration**: Reveals "conceptual neighbors" - notes that explore similar ideas but might use different terminology or approaches
- **Dynamic Filtering**: Sort and filter results by relevance, creation date, or other metadata to focus on the most important connections
- **Interactive Navigation**: Seamlessly open any backlinked or semantically similar note to continue exploring connection threads
- **Multi-dimensional Connections**: Understand both the explicit structure (links) and implicit structure (semantic relationships) of your knowledge base
- **Inspiration Discovery**: Surface unexpected connections between ideas that you might not have consciously made, sparking new insights and creative connections

The backlink viewer transforms your Zettelkasten from a collection of individual notes into a richly interconnected knowledge network, helping you discover relationships between ideas across different domains and contexts.

### Bibliography Tools

```bash
# Build bibliography files
zk-bib-build

# Interactive bibliography viewer
zk-bib-view
```

The bibliography tools provide specialized functionality for managing academic references within your Zettelkasten system:

#### Bibliography Builder (`zk-bib-build`)

This tool generates and updates bibliography files from your notes:

- **Citation Key Extraction**: Generates a list of citation keys from your notes
- **Literature Note Processing**: Identifies notes tagged as "literature_note" and processes them
- **Bibliography JSON Generation**: Creates a structured JSON bibliography file for use with other tools
- **Multiple Output Locations**: Saves bibliography files to configurable locations (biblib directory and Dropbox)
- **Markdown Formatting**: Creates markdown-friendly citation key lists with proper formatting

#### Bibliography Viewer (`zk-bib-view`)

This interactive interface helps you explore and work with your bibliography:

- **Rich Interactive Display**: Shows bibliography entries with color-coded fields and type icons
- **Customizable Sorting**: Sort by year or modification date
- **Document Preview**: View file information and note connections in the preview panel
- **PDF Integration**: Open associated PDFs directly from the interface with multiple viewer options
- **Citation Management**: Copy citation keys or generate new literature notes
- **Reading Tracking**: Start tracking time spent reading specific references
- **Advanced Filtering**: Filter entries using fuzzy search
- **History Navigation**: Browse through previously viewed entries
- **External Tool Integration**: Connect with Obsidian, time tracking, and note-taking tools

The bibliography tools create a seamless workflow between your reference manager, PDFs, and notes, helping you maintain a comprehensive research library that's deeply integrated with your knowledge management system.

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

# Export workout history as JSON
zk-workout-log --export-json

# Create a manual backup
zk-workout-log --backup
```

The workout logging system (`zk-workout-log`) demonstrates the flexibility of the Zettelkasten approach by extending the markdown+YAML structure to track physical exercise. It leverages the same file-based structure and YAML frontmatter technique used in the core note system:

- **Markdown-Based Data Storage**: Stores all workout data as markdown files with YAML frontmatter, consistent with the Zettelkasten philosophy of plain text storage
- **Hierarchical Organization**: Organizes workout data into exercises, templates, and session records, each in their own subdirectory
- **Structured Templates**: Creates reusable workout templates with predefined exercise sets
- **Session Tracking**: Records complete workout sessions with exercises, sets, weights, reps, and notes
- **History Analysis**: Provides filtering and analysis of workout history by date ranges and exercises
- **Automatic Backups**: Creates timestamped ZIP backups of all workout data at configurable intervals
- **Data Export**: Exports workout history in CSV and JSON formats for external analysis
- **Text-Based Interface**: Offers a terminal user interface for managing workout data

This module demonstrates how the structured yet flexible approach of Zettelkasten can be applied beyond traditional note-taking to personal tracking systems. The workout logger takes advantage of the core utilities for YAML processing and file handling while maintaining compatibility with the overall system architecture.

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
