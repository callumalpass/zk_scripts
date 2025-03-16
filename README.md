# ZK Core

A Python toolkit for Zettelkasten note-taking.

## What is it?

ZK Core is a set of Python tools designed to help you manage your markdown-based Zettelkasten notes. It provides a modular collection of utilities for indexing, searching, navigating, and connecting your notes, along with a few extra tools for specialized tasks. 

It evolved from a collection of personal scripts into this organized package.

## Key Features

### Core Functionality:

-   **Note Indexing:** A fast indexer that scans your notes directory, extracting key information and building a quick-access index. It intelligently updates only what's changed.
-   **Query Tools:** Command-line tools for searching your notes by text, tags, and dates.
-   **Interactive Navigation (Fuzzy Finder):** An interactive interface (built on `fzf`) that lets you quickly navigate your notes, view backlinks, and explore connections. This is a central component for daily use.
-   **Backlink Tracking:** Uncovers the links between your notes. It supports semantic similarity analysis (using OpenAI embeddings) to find hidden relationships.

### Additional Tools:

-   **Bibliography Management:** Tools for managing references and integrating them with your notes for more academic workflows.
-   **Person Search:** Quickly find notes related to specific people, useful for journaling or contact management.
-   **Workout Logger:** Track your workouts in markdown. Includes reporting capabilities.

## Installation

```bash
# Clone the repo
git clone https://github.com/callumalpass/zk_core.git
cd zk_core

# Install
pip install .
```

## Configuration

You'll need a configuration file at `~/.config/zk_scripts/config.yaml`. Here's an example:

```yaml
# Your notes directory
notes_dir: "~/notes"

# Indexing settings
zk_index:
  index_file: "index.json"
  exclude_patterns: [".git", ".obsidian", "node_modules"]

# Bibliography settings (for bibview)
bibview:
  bibliography_json: "~/Dropbox/bibliography.json"
  library: "~/biblib"

# Person search settings
personSearch:
  notes_dir: "~/notes"
  bat_command: "bat"  # Optional: for pretty-printed previews
```

Adjust those paths to match your system.

## Usage Examples

### Indexing

The indexer is the engine that powers much of ZK Core. It parses your notes, extracts metadata (like tags and links), and can optionally generate embeddings for semantic search (finding notes based on meaning).

```bash
# Build the index
zk-index run

# Rebuild everything from scratch
zk-index run --full-reindex

# Generate OpenAI embeddings (needs your OPEN_AI_KEY)
zk-index run --generate-embeddings

# Test your OpenAI API key
zk-index test-api

# See if any notes are missing embeddings
zk-index validate-embeddings

# Regenerate all embeddings (e.g., if you switch models)
zk-index regenerate-embeddings
```

The indexer is designed for speed – it only re-processes modified files. To use the semantic search features, set the `OPEN_AI_KEY` environment variable.

### Querying

The `zk-query` command is your tool for finding information in your notes.

```bash
# List notes tagged with 'project'
zk-query list --mode notes --filter-tag project

# Find notes containing "python"
zk-query search "python"

# Get some stats about your note collection
zk-query info

# Find notes similar to a specific file
zk-query search-embeddings --query-file "path/to/note.md"

# Find notes similar to a query string
zk-query search-embeddings --query "language learning techniques"
```

### Interactive Navigation (zk-fzf)

`zk-fzf` launches the interactive fuzzy finder:

```bash
zk-fzf
```

This is the command center for navigating your notes. Quickly search, view backlinks, and explore semantic connections. Press `alt-h` inside the interface for a list of keyboard shortcuts. `alt-2` shows notes that are semantically similar to the current one.

### Working Memory

`zk-working-mem` creates a temporary note to gather related thoughts:

```bash
zk-working-mem
```

### Backlinks

`zk-backlinks` shows you which notes link to a given note:

```bash
zk-backlinks
```

It can also show conceptually related notes via semantic similarity.

### Bibliography Tools

```bash
# Update your bibliography from your notes
zk-bib-build

# Browse your bibliography interactively
zk-bib-view
```

These tools are useful for managing academic references, linking PDFs, and generating bibliographies.


### Workout Log

```bash
# Create a new workout entry for today
zk-workout-log

# Export your workout history
zk-workout-log --export-history
```

Track your workouts in markdown and review your progress.

## Project Structure

```
zk_core/
├── __init__.py
├── config/
│   └── __init__.py
├── utils.py
├── models.py
├── constants.py
├── index.py        # Typer CLI
├── query.py        # Typer CLI
├── fzf_interface.py
├── working_mem.py
├── backlinks.py
├── bibbuild.py
├── bibview.py
├── person_search.py
└── workout_log.py
```

## Extending ZK Core

To add your own tools:

1.  Create a new Python module inside `zk_core`.
2.  Add an entry point to `pyproject.toml`.

