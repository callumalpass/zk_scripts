#!/bin/bash

# --- Script Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # Get the absolute path of the script's directory
CONFIG_DIR="$HOME/.config/zk_scripts"
CONFIG_FILE="$CONFIG_DIR/config.yaml"
MYBIN_DIR="$HOME/mybin"
SCRIPTS=(
    zk_fzf
    zk_index
    poll_backlinks
    personSearch
    bibview
    bibbuild
    py_zk.py
    config_loader.sh
)
PYTHON_LIBS=(
    pynvim
    pyyaml
    tabulate
    curses-curses
)
DEPENDENCIES=(
    fd
    gawk
    yq
    jq
    bat
    python3
    pip
)

# --- Functions ---

check_dependency() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: Dependency '$1' is not installed or not in your PATH."
        return 1 # Indicate failure
    fi
    return 0 # Indicate success
}

check_python_lib() {
    if ! python3 -m pip show "$1" >/dev/null 2>&1; then
        echo "Error: Python library '$1' is not installed."
        return 1 # Indicate failure
    fi
    return 0 # Indicate success
}

create_default_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        mkdir -p "$CONFIG_DIR"
        echo "Configuration file '$CONFIG_FILE' not found. Creating with default settings."
        default_config_content=$(cat <<EOF
# Global settings
notes_dir: "/path/to/your/notes" # e.g., "~/Dropbox/notes"
mybin_dir: "/path/to/your/mybin" # e.g., "~/mybin"

# --- zk_index script settings ---
zk_index:
  index_file: "/path/to/your/notes/index.json"
  index_yaml_file: "/path/to/your/notes/index.yaml"
  fd_exclude_patterns: "-E templates/ -E .zk/"

# --- zk_fzf script settings ---
zk_fzf:
  index_file: "/path/to/your/notes/index.json"
  py_zk: "/path/to/your/mybin/py_zk.py"
  zk_index_script: "/path/to/your/mybin/zk_index"
  notes_diary_subdir: "diary"
  bat_theme: "TwoDark"

# --- bibview script settings ---
bibview:
  bibhist: "~/.cache/bibview.history"
  library: "/path/to/your/notes/biblib"
  bibliography_json: "/path/to/your/notes/biblib/bibliography.json"
  bibview_open_doc_script: "/path/to/your/mybin/bibview.openDocument"
  llm_path: "/path/to/your/mybin/simonw-llm/venv/bin/llm"
  add_to_reading_list_script: "/path/to/your/mybin/addToReadingList"
  link_zathura_tmp_script: "/path/to/your/mybin/linkZathuraTmp"
  obsidian_socket: "/tmp/obsidian.sock"
  notes_dir_for_zk: "/path/to/your/notes"
  bat_theme: "TwoDark"

# --- personSearch script settings ---
personSearch:
  py_zk: "/path/to/your/mybin/py_zk.py"
  bat_command: "bat"
  bat_theme: "TwoDark"
EOF
        )
        echo "$default_config_content" > "$CONFIG_FILE"
    fi
}

create_symlinks() {
    mkdir -p "$MYBIN_DIR"
    for script in "${SCRIPTS[@]}"; do
        script_path="$SCRIPT_DIR/$script"
        link_path="$MYBIN_DIR/$script"
        if [[ -e "$link_path" ]]; then
            echo "Removing existing symlink: $link_path"
            rm -f "$link_path"
        fi
        echo "Creating symlink: $link_path -> $script_path"
        ln -s "$script_path" "$link_path"
        chmod +x "$link_path" # Ensure symlink is executable
    done
}

check_dependencies_and_python_libs() {
    echo "--- Checking Dependencies ---"
    missing_deps=()
    for dep in "${DEPENDENCIES[@]}"; do
        if ! check_dependency "$dep"; then
            missing_deps+=("$dep")
        fi
    done

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "\n\033[31mError: Some dependencies are missing:\033[0m"
        for dep in "${missing_deps[@]}"; do
            echo "- $dep"
        done
        echo -e "\n\033[33mPlease install these dependencies. For example:\033[0m"
        echo "  - For fzf, rg, fd, bat, jq, yq: Use your system's package manager (apt, pacman, brew etc.)."
        echo "  - For gawk: 'sudo apt install gawk' (Debian/Ubuntu) or equivalent."
        echo "  - For python3 and pip: Ensure Python 3 and pip are installed."
        return 1 # Indicate failure
    fi
    echo -e "\033[32mAll script dependencies are installed.\033[0m"

    echo -e "\n--- Checking Python Libraries ---"
    missing_libs=()
    for lib in "${PYTHON_LIBS[@]}"; do
        if ! check_python_lib "$lib"; then
            missing_libs+=("$lib")
        fi
    done

    if [[ ${#missing_libs[@]} -gt 0 ]]; then
        echo -e "\n\033[31mError: Some Python libraries are missing:\033[0m"
        for lib in "${missing_libs[@]}"; do
            echo "- $lib"
        end

