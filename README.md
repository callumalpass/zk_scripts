# zk-scripts: Enhanced Zettelkasten Workflow with fzf and Python

This repository provides a collection of scripts designed to enhance a Zettelkasten workflow, primarily focused on integration with Neovim, tmux, fzf, and a Python-based note index reader. These scripts are built to streamline note-taking, searching, backlink management, and bibliographic entry handling within a plain text Markdown note system.


## Overview

The scripts are designed to work together to provide a powerful and efficient note-taking environment. Here's a brief overview of each script:

- **`zk_fzf`**:  The primary entry point for searching and interacting with your notes. It uses `fzf` to provide an interactive fuzzy finder over your notes, powered by the `py_zk.py` index. It allows you to open notes in Neovim, perform actions like indexing, tag filtering, and preview backlinks directly from the fzf interface.

- **`poll_backlinks`**: A Python script that runs in a Neovim terminal buffer to provide a real-time, interactive backlink viewer. It polls for changes in the currently open note and displays backlinks with a dynamic UI, including preview snippets, sorting, filtering, and bookmarking features.

- **`py_zk.py`**: The core Python script responsible for indexing your Markdown notes (`index.json`), computing backlinks, and providing flexible querying and output formatting. It's a command-line tool used by other scripts to retrieve and process note data.

- **`personSearch`**:  Another fzf-based script specifically designed for searching and inserting person names from your notes. It's optimized for quickly finding and linking to person notes within your Zettelkasten.

- **`bibview`**: A script that provides an fzf interface for browsing and interacting with bibliographic entries from a JSON bibliography file. It allows you to search, preview, open PDFs, create Zettel notes from entries, and manage your reading list, integrating with tools like `timewarrior`.

- **`zk_index`**: A Bash script that generates the `index.json` file required by `py_zk.py` and `zk_fzf`. It uses `fd`, `gawk`, and `yq` to efficiently scan your notes directory, extract metadata (YAML frontmatter, outgoing links), and convert it into a JSON index.

- **`config_loader.sh`**: A shell script that handles configuration loading for all scripts in this repository. It reads a central `config.yaml` file, resolves configuration values using `yq`, sets environment variables, and ensures consistent configuration across all scripts.

- **`config.yaml` (example)**:  An example configuration file (not provided in the script listing, but implied) that stores settings like note directories, paths to scripts, and tool-specific configurations, allowing for easy customization of the entire workflow.

## Dependencies

Before using these scripts, ensure you have the following software installed and accessible in your `PATH`:

- **Python 3**: (With `pynvim`, `pyyaml`, `tabulate`, and `curses` libraries: `pip install pynvim pyyaml tabulate curses-curses`)
- **fzf**:  Fuzzy finder: [https://github.com/junegunn/fzf](https://github.com/junegunn/fzf)
- **rg (ripgrep)**:  Fast line-oriented search tool: [https://github.com/BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)
- **fd (find-faster)**: A user-friendly alternative to `find`: [https://github.com/sharkdp/fd](https://github.com/sharkdp/fd)
- **gawk (GNU Awk)**:  Pattern scanning and text processing language: [https://www.gnu.org/software/gawk/](https://www.gnu.org/software/gawk/)
- **yq (yq - YAML processor)**: Command-line YAML processor: [https://github.com/mikefarah/yq](https://github.com/mikefarah/yq)
- **jq (jq - command-line JSON processor)**: Lightweight and flexible command-line JSON processor: [https://stedolan.github.io/jq/](https://stedolan.github.io/jq/)
- **bat (bat - A cat(1) clone with wings)**:  Syntax highlighting `cat` clone: [https://github.com/sharkdp/bat](https://github.com/sharkdp/bat)
- **nnn (nnn - The fastest, most feature-packed file manager)**:  Terminal file manager (optional, for `bibview`): [https://github.com/jarun/nnn](https://github.com/jarun/nnn)
- **evince/qpdfview/zathura/other PDF viewer**: For opening PDF documents (used in `bibview`).
- **tmux**:  Terminal multiplexer (optional, for `zk_fzf` and `personSearch` to send links to tmux).
- **Neovim**:  Modern text editor (required for most scripts): [https://neovim.io/](https://neovim.io/)
- **simonw-llm (or similar LLM CLI tool)**: Command-line interface to language models (optional, for advanced `bibview` features): [https://llm.datasette.io/en/stable/](https://llm.datasette.io/en/stable/)

## Configuration

The scripts rely on a central configuration file, `~/.config/zk_scripts/config.yaml` (specified by `CONFIG_FILE` variable in each script), to manage settings. You'll need to create this file and configure it according to your system and preferences.

Here's an example structure for `config.yaml`:

```yaml
# Global settings
notes_dir: "/path/to/your/notes" # e.g., "~/Dropbox/notes"
mybin_dir: "/path/to/your/mybin" # e.g., "~/mybin"

# --- zk_index script settings ---
zk_index:
  index_file: "{{.notes_dir}}/index.json"
  index_yaml_file: "{{.notes_dir}}/index.yaml"
  fd_exclude_patterns: "-E templates/ -E .zk/"

# --- zk_fzf script settings ---
zk_fzf:
  index_file: "{{.notes_dir}}/index.json"
  py_zk: "{{.mybin_dir}}/py_zk.py"
  zk_index_script: "{{.mybin_dir}}/zk_index"
  notes_diary_subdir: "diary"
  bat_theme: "TwoDark"

# --- bibview script settings ---
bibview:
  bibhist: "~/.cache/bibview.history"
  library: "{{.notes_dir}}/biblib"
  bibliography_json: "{{.library}}/bibliography.json"
  bibview_open_doc_script: "{{.mybin_dir}}/bibview.openDocument"
  llm_path: "{{.mybin_dir}}/simonw-llm/venv/bin/llm"
  add_to_reading_list_script: "{{.mybin_dir}}/addToReadingList"
  link_zathura_tmp_script: "{{.mybin_dir}}/linkZathuraTmp"
  obsidian_socket: "/tmp/obsidian.sock"
  notes_dir_for_zk: "{{.notes_dir}}"
  bat_theme: "TwoDark"

# --- personSearch script settings ---
personSearch:
  py_zk: "{{.mybin_dir}}/py_zk.py"
  bat_command: "bat"
  bat_theme: "TwoDark"
```

**Key Configuration Variables:**

- **`notes_dir`**:  Path to your main notes directory.
- **`mybin_dir`**: Path to a directory where you keep your scripts and executables (like `mybin` in the scripts).
- **`index_file`**: Path to the `index.json` file generated by `zk_index`.
- **`index_yaml_file`**: Path to the (optional) YAML version of the index.
- **`fd_exclude_patterns`**:  Patterns for `fd` to exclude when indexing notes.
- **`py_zk`**: Path to the `py_zk.py` script.
- **`zk_index_script`**: Path to the `zk_index` script.
- **`notes_diary_subdir`**: Subdirectory within `notes_dir` for diary notes.
- **`bat_theme`**: Theme for `bat` syntax highlighting.
- **`bibhist`**: Path to the history file for `bibview`.
- **`library`**: Path to your bibliographic library directory.
- **`bibliography_json`**: Path to your `bibliography.json` file.
- **`bibview_open_doc_script`**: Path to a script to open documents in `bibview` (not provided in the scripts, you might need to create this).
- **`llm_path`**: Path to your `llm` command-line tool.
- **`add_to_reading_list_script`**: Path to a script to add to your reading list (not provided in the scripts).
- **`link_zathura_tmp_script`**: Path to a script to link with Zathura (not provided in the scripts).
- **`obsidian_socket`**: Path to your Obsidian socket (or Neovim server socket).
- **`notes_dir_for_zk`**:  Path to your notes directory for the `zk` CLI tool.
- **`bat_command`**: Command to execute `bat`.

**`config_loader.sh`**: This script is crucial for loading these configurations. It uses `yq` to parse the YAML file and sets environment variables that are then used by all other scripts.  Make sure `yq` is installed and in your `PATH`.

## Script Descriptions and Usage

### `zk_fzf`

**Purpose:**  Fuzzy search and interact with your notes using fzf.

**Usage:**  Simply run `./zk_fzf` from your terminal.

**Key Bindings:**

- **`Enter`**: Open the selected note in Neovim.
- **`alt-y`**: Copy the Obsidian-style link of the selected note to the clipboard (via tmux, customize as needed).
- **`alt-a`**: Re-index your notes using `zk_index`.
- **`ctrl-e`**: Edit the selected note in Neovim and reload the note list.
- **`ctrl-alt-r`**: Delete the selected note and reload the note list.
- **`ctrl-alt-d`**: Create a new diary note for tomorrow.
- **`alt-9`**: Reload notes, filtering by unique tags.
- **`alt-1`**: Search for notes linked from a given term (using `rg`).
- **`alt-8`**: Filter notes by a specific tag.
- **`?`**: Show key bindings help overlay.
- **`alt-?`**: Toggle preview window.
- **`alt-j`**: Filter notes by the `diary` tag.
- **`alt-b`**: Filter notes by the `literature_note` tag.
- **`ctrl-s`**: Sort notes by modified date (using `rg`).

**Configuration Variables (loaded from `config.yaml`):**

- `INDEX_FILE`
- `PY_ZK`
- `ZK_INDEX_SCRIPT`
- `NOTES_DIARY_SUBDIR`
- `BAT_THEME`

### `poll_backlinks`

**Purpose:** Real-time backlink viewer for Neovim.

**Usage:** Open a Neovim terminal buffer and run `./poll_backlinks`.

**Key Bindings (within the Neovim terminal buffer running `poll_backlinks`):**

- **`↑/↓/j/k`**: Navigate backlink list.
- **`Enter`**: Open the selected backlink in Neovim.
- **`PgUp/PgDn/u/d`**: Scroll preview window.
- **`t`**: Toggle frontmatter view in preview.
- **`s`**: Cycle sort order of backlinks (filename, line number, title, none).
- **`f`**: Enter filter mode to filter backlinks by keyword.
- **`r`**: Force refresh backlink scan.
- **`m`**: Toggle preview mode (snippet/full file).
- **`b`**: Bookmark the currently selected backlink.
- **`B`**: View bookmarked backlinks.
- **`?`**: Toggle help overlay.
- **`q`**: Quit the backlink viewer.

**Configuration Variables (loaded from `config.yaml`):**

- `THESIS_DIR` (renamed to `NOTES_DIR` in `config_loader.sh` for consistency)
- `POLL_INTERVAL`
- `CONTEXT_LINES`
- `STATUS_BAR_CLEAR_TIME`
- `SEPARATOR_LINE_LENGTH`
- `PREVIEW_SCROLL_LINES`
- `LOG_MAX_BYTES`
- `LOG_BACKUP_COUNT`
- `CONFIG_PATH`
- `LOG_FILE`

### `py_zk.py`

**Purpose:**  Python script for indexing, querying, and formatting note data.

**Usage:**  `./py_zk.py --index-file index.json [OPTIONS]`

**Options:**  Run `./py_zk.py --help` to see all available options, including:

- `--index-file <index.json>`: (Required) Path to the index file.
- `--output-format <format>`: Output format (`plain`, `csv`, `json`, `table`).
- `--output-file <file>`: File to write output to.
- `--fields <field> [<field> ...]`: Fields to include in output.
- `--separator <string>`: Separator for plain text output.
- `--format-string <string>`: Custom format string for plain text output.
- `--color <auto|always|never>`: Colorize output.
- `--unique-tags`: List unique tags.
- `--filter-tag <tag> [<tag> ...]`: Filter notes by tags (AND logic).
- `--exclude-tag <tag> [<tag> ...]`: Exclude notes with these tags.
- `--stdin`: Filter notes by filenames from stdin.
- `--filename-contains <substring>`: Filter by filename substring.
- `--filter-backlink <filename>`: Filter notes backlinked from a file.
- `--date-start <YYYY-MM-DD>`: Filter notes modified after a date.
- `--date-end <YYYY-MM-DD>`: Filter notes modified before a date.
- `--filter-field <FIELD> <VALUE>`: Filter notes where FIELD equals VALUE.
- `--sort-by <dateModified|filename|title>`: Sort output by field.

**Configuration Variables (loaded from `config.yaml` - although it primarily uses command-line arguments):**

- `--config-file` (to load configuration, if used)

### `personSearch`

**Purpose:** Fuzzy search for person notes and insert Obsidian-style links.

**Usage:** `./personSearch`

**Key Bindings:**

- **`ctrl-e`**: Open the selected person note in Neovim.
- **`one` (or `Enter`):** Accept the selection and insert the Obsidian-style link.

**Configuration Variables (loaded from `config.yaml`):**

- `PY_ZK`
- `BAT_COMMAND`
- `BAT_THEME`

### `bibview`

**Purpose:**  Fuzzy search and interact with bibliographic entries.

**Usage:** `./bibview [year]` (optional `year` argument sorts by year instead of modified date).

**Key Bindings:**

- **`ctrl-space`**: Open PDF in evince and start reading timer.
- **`ctrl-z`**: Open PDF in evince and abort fzf (start reading timer).
- **`ctrl-e`**: Generate a Zettel note for the selected bibliography entry (if none exists).
- **`ctrl-v`**: Open PDF in qpdfview in a right split.
- **`ctrl-y`**: Copy the citation key to the clipboard.
- **`ctrl-f`**: Open the bibliography entry's directory in `nnn`.
- **`alt-n`**: Next history item in fzf history.
- **`alt-p`**: Previous history item in fzf history.
- **`alt-y`**: Copy the path of the PDF to the clipboard.
- **`/`**: Toggle preview window.
- **`ctrl-t`**: Start `timewarrior` reading timer for the selected entry.
- **`alt-g`**: Generate a reading critique JSON file using `llm`.
- **`alt-t`**: Translate the PDF to English using `llm` and save as Markdown.
- **`Ctrl-a`**: Search for notes linking to the selected bibliography entry using `rgnotesearch` (not provided, likely a custom script).
- **`ctrl-r`**: Add the selected entry to a reading list (using `addToReadingList` script, not provided).
- **`Ctrl-e`**: Open the bibliography entry's Zettel note (if exists) in Neovim.
- **`alt-z`**: Link the PDF in Zathura using a temporary file (using `linkZathuraTmp` script, not provided).
- **`Ctrl-t`**: Start `timewarrior` reading timer for the selected entry.
- **`?`**: Show key bindings help overlay.
- **`Enter`**: Open PDF in Obsidian and start reading timer.

**Configuration Variables (loaded from `config.yaml`):**

- `BIBHIST`
- `LIBRARY`
- `BIBLIOGRAPHY_JSON`
- `BIBVIEW_OPEN_DOC_SCRIPT`
- `LLM_PATH`
- `ZK_SCRIPT`
- `ADD_TO_READING_LIST_SCRIPT`
- `LINK_ZATHURA_TMP_SCRIPT`
- `OBSIDIAN_SOCKET`
- `NOTES_DIR_FOR_ZK`
- `BAT_THEME`

### `zk_index`

**Purpose:** Generate `index.json` from Markdown notes.

**Usage:** `./zk_index`

**Configuration Variables (loaded from `config.yaml`):**

- `NOTES_DIR`
- `INDEX_FILE`
- `INDEX_YAML_FILE`
- `FD_EXCLUDE_PATTERNS`

### `config_loader.sh`

**Purpose:** Load configuration from `config.yaml` and set environment variables.

**Usage:**  `source ./config_loader.sh <config_file> <script_config_section>` (This script is sourced by other scripts, not run directly).

**Configuration Variables:**

- `CONFIG_FILE` (set in each script, defaults to `~/.config/zk_scripts/config.yaml`)

## Setup Instructions

1. **Clone the repository:** `git clone <repository_url> zk-scripts`
2. **Navigate to the repository directory:** `cd zk-scripts`
3. **Create the configuration directory:** `mkdir -p ~/.config/zk_scripts`
4. **Create `config.yaml`:**  Copy the example `config.yaml` content (provided in the Configuration section above) into `~/.config/zk_scripts/config.yaml` and adjust the paths to match your system.
5. **Make scripts executable:** `chmod +x *.sh *.py`
6. **Ensure dependencies are installed:** Install all the software listed in the Dependencies section.
7. **Run `zk_index` to generate `index.json`:** `./zk_index` (Make sure `NOTES_DIR` in `config.yaml` is correctly pointing to your notes directory before running).
8. **Test `zk_fzf`:** `./zk_fzf` to see if the fuzzy finder works correctly.
9. **Configure Neovim integration:**  Ensure Neovim is configured to connect to the Obsidian socket (if using Obsidian integration) or a Neovim server socket.
10. **Customize scripts and keybindings:** Modify the scripts and `config.yaml` to tailor the workflow to your specific needs.

## Example Usage Scenarios

- **Quick Note Search and Open:** Run `zk_fzf`, type keywords to fuzzy search your notes, and press `Enter` to open the desired note in Neovim.
- **Finding Backlinks:** While editing a note in Neovim, open a terminal buffer and run `poll_backlinks` to see real-time backlinks to the current note.
- **Inserting Person Links:** Use `personSearch` to quickly find and insert links to person notes while writing notes.
- **Managing Bibliographic Entries:** Use `bibview` to search your bibliography, open PDFs, and create Zettel notes from bibliographic entries.
- **Periodic Indexing:** Run `zk_index` regularly (e.g., via cron or systemd timer) to keep your `index.json` up-to-date.
