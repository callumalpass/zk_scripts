# ZK Core

A Python toolkit for Zettelkasten note-taking.

## What is it?

ZK Core is a set of Python tools designed around a super fast indexing system to help you manage your markdown-based Zettelkasten notes. It provides a modular collection of utilities for indexing, searching, navigating, and connecting your notes, along with a few extra tools for specialized tasks. 

It evolved from a collection of personal scripts into this organized package.

## Key Features

### Core Functionality:

-   **Note Indexing:** A fast indexer that scans your notes directory, extracting key information and building a quick-access index. It intelligently updates only what's changed.
-   **Query Tools:** Command-line tools for searching your notes by text, tags, and dates.
-   **Interactive Navigation (Fuzzy Finder):** An interactive interface (built on `fzf`) that lets you quickly navigate your notes, view backlinks, and explore connections. This is a central component for daily use.
-   **Backlink Tracking:** Uncovers the links between your notes. It supports semantic similarity analysis (using OpenAI embeddings) to find hidden relationships.

### Additional Tools:

-   **Bibliography Management:** Tools for managing references and integrating them with your notes for more academic workflows.
-   **Wikilink Generator:** Create configurable wikilink insertion tools for any type of note (people, books, concepts, etc.) with customizable field display and alias selection.
-   **Working Memory:** A tool for capturing thoughts and ideas quickly, with smart note creation and organization capabilities.
-   **Workout Logger:** Track your workouts in markdown. Includes reporting capabilities.

### Limitations

The current implementation is somewhat opinionated about how a zettelkasten should be managed. The scripts in this package will work best when you have tags in the frontmatter and `dateCreated` and `dateModified` values. The "title" of your notes should be defined in the frontmatter; it is assumed that you are not using filenames to store titles. 

There is currently no dedicated feature for the *creation* of notes from templates (except of "working memory" notes). This has been avoided under the belief that templates are better managed by other tools; e.g. snippets in `neovim`.

## Installation

```bash
# Clone the repo
git clone https://github.com/callumalpass/zk_core.git
cd zk_core

# Install
pip install .
```

## Configuration

You'll need a configuration file at `~/.config/zk_scripts/config.yaml`.

```yaml
# ZK Core Example Configuration

# Main configuration
notes_dir: "~/notes"  # Path to your notes directory
socket_path: "/tmp/obsidian.sock"  # Path to Neovim socket for editor integration

# Index configuration
zk_index:
  index_file: "index.json"  # Name of the index file
  exclude_patterns: [".git", ".obsidian", "node_modules"]  # Directories to exclude
  excluded_files: ["README.md"]  # Files to exclude from indexing

# Query configuration
query:
  default_index: "index.json"
  default_fields: ["filename", "title", "tags"]

# FZF interface configuration
fzf_interface:
  bat_command: "bat"  # Command for preview
  fzf_args: "--height=80% --layout=reverse --info=inline"

# Working memory configuration
working_mem:
  template_path: "~/notes/templates/working_mem.md"
  editor: "nvim"
  tag: "working_mem"

# Backlinks configuration
backlinks:
  notes_dir: "~/notes"  # This is redundant with the global notes_dir, kept for backward compatibility
  bat_theme: "Dracula"  # Theme for bat preview

# Bibliography configuration
bibview:
  bibliography_json: "~/Dropbox/bibliography.json"  # Path to bibliography JSON
  dropbox_bibliography_json: "~/Dropbox/bibliography.json"  # Optional additional bibliography output path
  bibhist: "~/.bibhist"  # Path to history file
  library: "~/notes/biblib"  # Path to bibliography pdf library
  notes_dir_for_zk: "~/notes"  # Path for Zettelkasten notes
  bat_theme: "Dracula"  # Theme for bat preview
  bibview_open_doc_script: "~/bin/open_doc.sh"  # Script for opening documents

# Wikilink generator configuration
wikilink:
  # Profile for person notes
  person:
    filter_tags: ["person"]
    search_fields: ["filename", "aliases", "givenName", "familyName"]
    display_fields: ["filename", "aliases", "givenName", "familyName"]
    alias_fields: ["aliases", "givenName"]
    preview:
      command: "bat"
      window: "wrap:50%:<40(up)"
    fzf:
      delimiter: "::"
      tiebreak: "begin,index"
  
  # Profile for book notes
  book:
    filter_tags: ["book"]
    search_fields: ["filename", "title", "author"]
    display_fields: ["title", "author", "filename"]
    alias_fields: ["title"]
  
  # Profile for concept notes
  concept:
    filter_tags: ["concept"]
    search_fields: ["filename", "title", "description"]
    alias_fields: ["title"]
  
  # Default profile (searches all notes)
  default:
    filter_tags: []  # No tag filter - all notes
    search_fields: ["filename", "title", "tags"]
    alias_fields: ["title", "aliases"]

# Global logging configuration
logging:
  level: "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  file: "~/.zk_core.log"  # Log file path

# Filename configuration
filename:
  format: "%Y%m%d{random:3}"  # Format for generated filenames; supports strftime and {random:N}
  extension: ".md"  # File extension for generated files
```

### Filename Configuration

The `filename` section allows you to customize how new note filenames are generated. This applies to notes created by the working memory tool, workout logger, and other tools that create new files.

The format supports:

1. **Date/time formatting**: All standard [Python strftime format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes), including:
   - `%Y` - Four-digit year (e.g., 2025)
   - `%y` - Two-digit year (e.g., 25)
   - `%m` - Month as zero-padded decimal (01-12)
   - `%d` - Day of the month as zero-padded decimal (01-31)
   - `%H` - Hour (24-hour clock) as zero-padded decimal (00-23)
   - `%M` - Minute as zero-padded decimal (00-59)

2. **Random characters**: Use `{random:N}` where N is the number of random lowercase letters to generate.

Some example formats:

```yaml
# Default format (YYYYMMDDxxx.md)
format: "%Y%m%d{random:3}" 

# ISO-style date with 4 random characters (2025-03-16_abcd.md)
format: "%Y-%m-%d_{random:4}"

# Two-digit year with timestamp and 6 random characters (250316-1423-abcdef.md)
format: "%y%m%d-%H%M-{random:6}"

# Prefixed notes with date (note-20250316-xyz.md)
format: "note-%Y%m%d-{random:3}"
```

You can also customize the file extension:

```yaml
extension: ".md"  # Default
# or
extension: ".markdown"
```

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
zk-working-mem             # Standard note creation
zk-working-mem -s          # Create a quick scratch note
zk-working-mem -q          # Quick capture mode (append directly to working memory file)
zk-working-mem --nvim      # Use content from current Neovim buffer
```

The Working Memory tool (`zk-working-mem`) is designed to capture fleeting thoughts, ideas, and connections quickly, while integrating them into your Zettelkasten. 

This feature was built after experiencing repeated difficulties with the zettelkasten system. 
First, I often create notes note quite sure how to title them. I'll have an idea I want to work out, and having to come up with a title before I've done the writing impedes, or adds friction, to the writing process.
Second, I think writing is a really important tool for thinking. And the zettelkasten system doesn't necessarily encourage scrappy, exploratory writing. It seems to encourage "evergreen" notes, or at least notes that could turn into something evergreen. I found this stopping me from writing just to explore an idea. With the `zk-working-mem` script, you are encouraged to do scrappy, "scratch" writing. You don't have to come up with a title for your notes, if they don't seem "title-able". Or you can pass the note to an llm, and see what titles it might suggest. After you've created the note, a link is pasted to the 'working-mem' file, a place where you can review what you've been working on recently, so that you can move on from the thought with the confidence that you will be encouraged to soon review it. 

**Key Features:**

- **Quick Capture:** Create notes without interrupting your workflow
- **AI-Assisted Title Generation:** Automatically suggests relevant titles using LLMs (if configured)
- **Tag Integration:** Select from existing tags in your Zettelkasten
- **Working Memory File:** Maintains a linked index of all created notes
- **Multiple Modes:**
  - Standard mode for detailed notes
  - Scratch mode for quick thoughts
  - Journal mode for date-based entries
  - Neovim integration for using existing buffer content

**How It Works:**

1. When you run `zk-working-mem`, it opens your preferred editor (default: nvim) for note writing
2. After saving, it offers several options:
   - Generate a title with an AI assistant
   - Enter a title manually
   - Use a "Scratch" title (timestamped)
   - Use a "Journal" title (date-based)
3. It then allows you to select relevant tags from your existing tag library
4. The note is saved with proper frontmatter into your Zettelkasten
5. A wikilink to the new note is appended to your working memory file

This creates a powerful system for:
- Capturing ideas that arise during work without losing context
- Gradually developing fleeting thoughts into permanent notes
- Maintaining awareness of your thinking process over time
- Ensuring new notes are properly integrated with your knowledge system

### Backlinks

`zk-backlinks` presents a TUI interface with a list of notes that link to the note that you currently have open in `neovim`. You need to have a socket exposed in `neovim`. It can also show a list of notes semantically-similar to the one that you have currently open. 

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

### Wikilink Generator

```bash
# Use different profiles for different note types
zk-wikilink --profile person
zk-wikilink --profile book
zk-wikilink --profile concept
zk-wikilink --profile default  # No tag filter - all notes

# List available profiles and their configurations
zk-wikilink list-profiles

# Show keyboard shortcuts
zk-wikilink --list-hotkeys
```

The wikilink generator provides a configurable system for searching notes and inserting properly formatted wikilinks with appropriate aliases. You can define multiple profiles in your configuration file, each with its own specific settings:

1. **Tag filters** - Which notes to include in the search (e.g., "person", "book", "concept")
2. **Search fields** - Which metadata fields to pass to `fzf` filtering notes
3. **Display fields** - Which fields to *show* `fzf`
4. **Alias fields** - Which fields to use for the alias part of wikilinks in priority order

If you run the generator from within `tmux`, it will use `tmux`'s 'send-keys' function to print the wikilink directly into the screen buffer. If it is run from outside `tmux`, the wikilink will be copied to the clipboard. 

Example configuration:

```yaml
wikilink:
  # Profile for person notes
  person:
    filter_tags: ["person"]
    search_fields: ["filename", "aliases", "givenName", "familyName"]
    display_fields: ["filename", "aliases", "givenName", "familyName"]
    alias_fields: ["aliases", "givenName"]
    
  # Profile for book notes
  book:
    filter_tags: ["book"]
    search_fields: ["filename", "title", "author"]
    display_fields: ["title", "author", "filename"]
    alias_fields: ["title"]
  
  # Default profile (searches all notes)
  default:
    filter_tags: []  # No tag filter - all notes
    search_fields: ["filename", "title", "tags"]
    alias_fields: ["title", "aliases"]
```

This tool allows you to create link-insertion tools for any type of note in your system.


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
├── commands.py
├── markdown.py
├── fzf_utils.py
├── index.py        # Typer CLI
├── query.py        # Typer CLI
├── fzf_interface.py
├── fzf_manager.py
├── working_mem.py
├── backlinks.py
├── bibbuild.py     # Thin wrapper around bibliography.builder
├── bibview.py      # Thin wrapper around bibliography.viewer
├── bibliography/   # Bibliography package
│   ├── __init__.py
│   ├── builder.py
│   └── viewer.py
├── wikilink_generator.py
└── workout_log.py
```

### Core Modules

- **commands.py**: Unified command execution utilities
- **markdown.py**: Markdown processing utilities (frontmatter, wikilinks, citations)
- **fzf_utils.py**: Fuzzy finder interface helpers
- **bibliography/**: Package for bibliography management (building and viewing)

The project has been modularized to improve code organization, reduce duplication, and make it easier to extend with new features.

## Extending ZK Core

To add your own tools:

1.  Create a new Python module inside `zk_core`.
2.  Add an entry point to `pyproject.toml`.

