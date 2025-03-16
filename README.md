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

### Querying (zk-query)

The `zk-query` command is a versatile tool for analyzing and filtering your notes collection. It provides a wide range of functionality for finding information, generating statistics, and discovering connections between notes.

It also accepts filenames from `stdin`, and can print information about the provided notes. This can even be used for chaining together query commands. 

#### Key Commands

- **info**: Display detailed statistics about your notes collection
- **list**: Filter and display notes based on various criteria
- **search-embeddings**: Find semantically similar notes using OpenAI embeddings

#### Usage Examples

```bash
# Get comprehensive statistics about your notes collection
zk-query info -i ~/notes/index.json

# List notes tagged with 'project'
zk-query list -i ~/notes/index.json --filter-tag project

# List notes with multiple tags (AND logic)
zk-query list -i ~/notes/index.json --filter-tag project --filter-tag active

# List notes with any of the given tags (OR logic)
zk-query list -i ~/notes/index.json --filter-tag project --filter-tag research --tag-mode or

# List notes excluding certain tags
zk-query list -i ~/notes/index.json --exclude-tag archive

# List notes tagged with 'project' and created within a date range
zk-query list -i ~/notes/index.json --filter-tag project --date-start 2023-01-01 --date-end 2023-12-31

# List notes with a minimum or maximum word count
zk-query list -i ~/notes/index.json --min-word-count 500
zk-query list -i ~/notes/index.json --max-word-count 1000

# Find orphan notes (notes with no incoming or outgoing links)
zk-query list -i ~/notes/index.json --mode orphans

# Find untagged orphan notes
zk-query list -i ~/notes/index.json --mode untagged-orphans

# Find dangling links (references to non-existent notes)
zk-query list -i ~/notes/index.json --mode dangling-links

# Get a list of all unique tags used in your notes
zk-query list -i ~/notes/index.json --mode unique-tags

# Find notes that link to a specific note
zk-query list -i ~/notes/index.json --filter-backlink "specific-note.md"

# Find notes that are linked from a specific note
zk-query list -i ~/notes/index.json --filter-outgoing-link "specific-note.md"

# Find notes containing a substring in their filename
zk-query list -i ~/notes/index.json --filename-contains "project"

# Filter by a specific field value
zk-query list -i ~/notes/index.json --filter-field familyName Smith

# Find semantically similar notes to a specific file
zk-query search-embeddings -i ~/notes/index.json --query-file "path/to/note.md"

# Find notes semantically related to a query string
zk-query search-embeddings -i ~/notes/index.json --query "language learning techniques"

# Control the number of similar notes returned
zk-query search-embeddings -i ~/notes/index.json --query "AI ethics" --k 10
```

#### Output Formatting

Control how `zk-query` displays results:

```bash
# Output as plain text with custom fields (default)
zk-query list -i ~/notes/index.json --fields filename title dateModified

# Output as CSV
zk-query list -i ~/notes/index.json -o csv

# Output as JSON
zk-query list -i ~/notes/index.json -o json

# Output as a formatted table
zk-query list -i ~/notes/index.json -o table

# Use custom field separator for plain output
zk-query list -i ~/notes/index.json --separator "|"

# Use a custom format string
zk-query list -i ~/notes/index.json --format-string "File: {filename} - Title: {title}"

# Sort results
zk-query list -i ~/notes/index.json --sort-by word_count  # Sort by word count
zk-query list -i ~/notes/index.json --sort-by dateCreated  # Sort by creation date
zk-query list -i ~/notes/index.json --sort-by title  # Sort alphabetically by title

# Output to a file
zk-query list -i ~/notes/index.json -o json --output-file notes.json
```

The `info` command provides comprehensive statistics about your notes collection, including word count distributions, tag usage, orphaned notes, creation patterns by day of week, and more.

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

`zk-backlinks` shows you which notes link to the note that you currently have open in `neovim`. You need to have a socket exposed in `neovim`. 

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

These were built out of the conviction that a full-blown reference manager like Zotero is not necessary. With these tools, you can keep the bibliographic information in the frontmatter of markdown files in your Zettelkasten. Since `pandoc` works with csl-json files, it is easy to convert this YAML frontmatter to a csl-valid bibliography file, ready for `pandoc` conversion of academic documents. Check out [biblio-note](https://github.com/callumalpass/biblio-note) for a script that helps to automate the creation of literature note files that can be converted to CSL valid bibliography json files. 


### Workout Log

```bash
# Launch the interactive workout tracker TUI
zk-workout-log

# Export your workout history as CSV
zk-workout-log --export-history

# Export your workout history as JSON
zk-workout-log --export-json
```

The workout log is a comprehensive workout tracking system built around markdown files, integrated with your Zettelkasten. It features:

- **Interactive TUI Interface**: A full-featured terminal user interface with menus for all operations
- **Workout Session Recording**: Log exercises with sets, reps, and weights during your workout
- **Exercise Management**: Create, edit, and organize your personal exercise library
- **Workout Templates**: Save and reuse workout routines for quick session starts
- **History & Statistics**: Review past workouts with detailed breakdowns by exercise
- **Data Analysis**: View your progress with statistics on total sets, reps, and weights
- **Export Capabilities**: Export your workout data to CSV or JSON for external analysis
- **Fully Integrated**: All data is stored as markdown files in your notes directory

#### File Types

The workout log uses three types of markdown files, each with specific YAML frontmatter:

1. **Exercise Files**: Define individual exercises
   ```yaml
   ---
   title: "Bench Press"
   tags: ["exercise"]
   date: "2023-01-01"
   dateCreated: "2023-01-01T10:00"
   dateModified: "2023-01-01T10:00"
   planned_exercise: false
   exercise_equipment: ["barbell"]
   ---
   ```

2. **Workout Session Files**: Record completed workout sessions
   ```yaml
   ---
   zettelid: "230101abc"
   title: "Workout Session on 2023-01-01"
   tags: ["workout"]
   date: "2023-01-01"
   dateCreated: "2023-01-01T18:30"
   dateModified: "2023-01-01T18:30"
   exercises:
     - id: "220915xyz.md"
       title: "Bench Press"
       sets:
         - reps: 10
           weight: 135
         - reps: 8
           weight: 155
     # Additional exercises...
   ---
   ```

3. **Workout Template Files**: Define reusable workout routines
   ```yaml
   ---
   title: "Push Day"
   description: "Chest, shoulders, and triceps"
   date: "2023-01-01"
   dateCreated: "2023-01-01T09:00"
   dateModified: "2023-01-01T09:00"
   exercises:
     - exercise_filename: "220915xyz.md"
       order: 1
     # Additional exercises...
   tags: ["workout_template"]
   ---
   # Push Day

   Chest, shoulders, and triceps
   ```

The system reads from your main index file (created by `zk-index`) for optimal performance but writes new entries directly as markdown files, keeping all your workout data in your Zettelkasten system.

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

