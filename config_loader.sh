#!/usr/bin/env bash

# Helper function to safely resolve config values with yq and handle errors
resolve_config_value() {
  local key="$1"
  local default_value="$2"
  local config_file="$3"
  local value

  value=$(yq -r ". // {} | ${key} // \"${default_value}\"" "$config_file" 2>/dev/null)
  if [[ -z "$value" ]]; then # Check if yq returned empty or if variable is still empty after yq
    echo "Warning: Could not read config key '${key}' from '$config_file'. Using default value: '$default_value'." >&2
    value="$default_value" # Fallback to default value in script if y fails
  fi
  echo "$value"
}

load_config() {
  local config_yaml="$1"
  local script_config_section="$2" # Pass the script section name as the second argument

  # Default values - Hardcoded sensible defaults
  local notes_dir_default="/home/calluma/Dropbox/notes"
  local mybin_dir_default="/home/calluma/mybin"
  local index_file_default="$notes_dir_default/index.json" # Use variables here
  local index_yaml_file_default="$notes_dir_default/index.yaml" # Use variables here
  local fd_exclude_patterns_default="-E templates/ -E .zk/"
  local py_zk_default="$mybin_dir_default/py_zk.py" # Use variables here
  local zk_index_script_default="$mybin_dir_default/zk_index" # Use variables here
  local notes_diary_subdir_default="diary"
  local bat_theme_default="TwoDark"
  local bibhist_default="/home/calluma/.cache/bibview.history"
  local library_default="$notes_dir_default/biblib" # Use variables here
  local bibliography_json_default="$library_default/bibliography.json" # Use variables here
  local fzf_default_opts_file_default="$mybin_dir_default/fzfDefaultOpts" # Use variables here
  local bibview_open_doc_script_default="$mybin_dir_default/bibview.openDocument" # Use variables here
  local llm_path_default="$mybin_dir_default/simonw-llm/venv/bin/llm" # Use variables here
  local zk_script_default="zk"
  local add_to_reading_list_script_default="$mybin_dir_default/addToReadingList" # Use variables here
  local link_zathura_tmp_script_default="$mybin_dir_default/linkZathuraTmp" # Use variables here
  local obsidian_socket_default="/tmp/obsidian.sock"
  local notes_dir_for_zk_default="$notes_dir_default" # Use variables here
  local bat_command_default="bat"

  # Default config content as a heredoc string - NO MORE TEMPLATE VARIABLES
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

  # Check if config file exists, create with defaults if not
  if [[ ! -f "$config_yaml" ]]; then
    mkdir -p "$(dirname "$config_yaml")"
    echo "Configuration file '$config_yaml' not found. Creating with default settings."
    echo "$default_config_content" > "$config_yaml"
  fi

  # Load global paths first and EXPORT them
  export NOTES_DIR=$(resolve_config_value ".notes_dir" "$notes_dir_default" "$config_yaml")
  export MYBIN_DIR=$(resolve_config_value ".mybin_dir" "$mybin_dir_default" "$config_yaml")

  # Script-specific settings - use script_config_section to target the right section
  if [[ "$script_config_section" == "zk_index" ]]; then
    export INDEX_FILE=$(resolve_config_value ".${script_config_section}.index_file" "$index_file_default" "$config_yaml") # Use default variable
    export INDEX_YAML_FILE=$(resolve_config_value ".${script_config_section}.index_yaml_file" "$index_yaml_file_default" "$config_yaml") # Use default variable
    export FD_EXCLUDE_PATTERNS=$(resolve_config_value ".${script_config_section}.fd_exclude_patterns" "$fd_exclude_patterns_default" "$config_yaml")
  elif [[ "$script_config_section" == "zk_fzf" ]]; then
    export INDEX_FILE=$(resolve_config_value ".${script_config_section}.index_file" "$index_file_default" "$config_yaml") # Use default variable
    export PY_ZK=$(resolve_config_value ".${script_config_section}.py_zk" "$py_zk_default" "$config_yaml") # Use default variable
    export ZK_INDEX_SCRIPT=$(resolve_config_value ".${script_config_section}.zk_index_script" "$zk_index_script_default" "$config_yaml") # Use default variable
    export NOTES_DIARY_SUBDIR=$(resolve_config_value ".${script_config_section}.notes_diary_subdir" "$notes_diary_subdir_default" "$config_yaml")
    export BAT_THEME=$(resolve_config_value ".${script_config_section}.bat_theme" "$bat_theme_default" "$config_yaml")
  elif [[ "$script_config_section" == "bibview" ]]; then
    export BIBHIST=$(resolve_config_value ".${script_config_section}.bibhist" "$bibhist_default" "$config_yaml")
    export LIBRARY=$(resolve_config_value ".${script_config_section}.library" "$library_default" "$config_yaml") # Use default variable
    export BIBLIOGRAPHY_JSON=$(resolve_config_value ".${script_config_section}.bibliography_json" "$bibliography_json_default" "$config_yaml") # Use default variable
    # export FZF_DEFAULT_OPTS_FILE=$(resolve_config_value ".${script_config_section}.fzf_default_opts_file" "$fzf_default_opts_file_default" "$config_yaml")
    export BIBVIEW_OPEN_DOC_SCRIPT=$(resolve_config_value ".${script_config_section}.bibview_open_doc_script" "$bibview_open_doc_script_default" "$config_yaml") # Use default variable
    export LLM_PATH=$(resolve_config_value ".${script_config_section}.llm_path" "$llm_path_default" "$config_yaml") # Use default variable
    export ZK_SCRIPT=$(resolve_config_value ".${script_config_section}.zk_script" "$zk_script_default" "$config_yaml")
    export ADD_TO_READING_LIST_SCRIPT=$(resolve_config_value ".${script_config_section}.add_to_reading_list_script" "$add_to_reading_list_script_default" "$config_yaml") # Use default variable
    export LINK_ZATHURA_TMP_SCRIPT=$(resolve_config_value ".${script_config_section}.link_zathura_tmp_script" "$link_zathura_tmp_script_default" "$config_yaml") # Use default variable
    export OBSIDIAN_SOCKET=$(resolve_config_value ".${script_config_section}.obsidian_socket" "$obsidian_socket_default" "$config_yaml")
    export NOTES_DIR_FOR_ZK=$(resolve_config_value ".${script_config_section}.notes_dir_for_zk" "$notes_dir_for_zk_default" "$config_yaml") # Use default variable
    export BAT_THEME=$(resolve_config_value ".${script_config_section}.bat_theme" "$bat_theme_default" "$config_yaml")
  elif [[ "$script_config_section" == "personSearch" ]]; then
    export PY_ZK=$(resolve_config_value ".${script_config_section}.py_zk" "$py_zk_default" "$config_yaml") # Use default variable
    export BAT_COMMAND=$(resolve_config_value ".${script_config_section}.bat_command" "$bat_command_default" "$config_yaml")
    export BAT_THEME=$(resolve_config_value ".${script_config_section}.bat_theme" "$bat_theme_default" "$config_yaml")
  fi

  # Add mybin to PATH after MYBIN_DIR is loaded and exported
  if [[ -d "$MYBIN_DIR" && ! :"$PATH": =~ :"$MYBIN_DIR": ]]; then
    export PATH="$PATH:$MYBIN_DIR"
  fi


  # Final check and error if NOTES_DIR is still not set after all attempts
  if [[ -z "${NOTES_DIR}" ]]; then
    echo "Error: NOTES_DIR is not defined in config file, environment variables, or defaults." >&2
    exit 1
  fi
}

