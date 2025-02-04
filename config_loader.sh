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

  # Default values
  local notes_dir_default="/home/calluma/Dropbox/notes"
  local mybin_dir_default="/home/calluma/mybin"
  local index_file_default="$NOTES_DIR/index.json"
  local index_yaml_file_default="$NOTES_DIR/index.yaml"
  local fd_exclude_patterns_default="-E templates/ -E .zk/"
  local py_zk_default="$MYBIN_DIR/py_zk.py"
  local zk_index_script_default="$MYBIN_DIR/zk_index"
  local notes_diary_subdir_default="diary"
  local bat_theme_default="TwoDark"
  local bibhist_default="/home/calluma/.cache/bibview.history"
  local library_default="$NOTES_DIR/biblib"
  local bibliography_json_default="$LIBRARY/bibliography.json"
  local fzf_default_opts_file_default="$MYBIN_DIR/fzfDefaultOpts"
  local bibview_open_doc_script_default="$MYBIN_DIR/bibview.openDocument"
  local llm_path_default="$MYBIN_DIR/simonw-llm/venv/bin/llm"
  local zk_script_default="zk"
  local add_to_reading_list_script_default="$MYBIN_DIR/addToReadingList"
  local link_zathura_tmp_script_default="$MYBIN_DIR/linkZathuraTmp"
  local obsidian_socket_default="/tmp/obsidian.sock"
  local notes_dir_for_zk_default="$NOTES_DIR"
  local bat_command_default="bat"

  # Load global paths first and EXPORT them
  export NOTES_DIR=$(resolve_config_value ".notes_dir" "$notes_dir_default" "$config_yaml")
  export MYBIN_DIR=$(resolve_config_value ".mybin_dir" "$mybin_dir_default" "$config_yaml")

  # Script-specific settings - use script_config_section to target the right section
  if [[ "$script_config_section" == "zk_index" ]]; then
    export INDEX_FILE=$(resolve_config_value ".${script_config_section}.index_file" "$notes_dir_default/index.json" "$config_yaml")
    export INDEX_YAML_FILE=$(resolve_config_value ".${script_config_section}.index_yaml_file" "$notes_dir_default/index.yaml" "$config_yaml")
    export FD_EXCLUDE_PATTERNS=$(resolve_config_value ".${script_config_section}.fd_exclude_patterns" "$fd_exclude_patterns_default" "$config_yaml")
  elif [[ "$script_config_section" == "zk_fzf" ]]; then
    export INDEX_FILE=$(resolve_config_value ".${script_config_section}.index_file" "$notes_dir_default/index.json" "$config_yaml")
    export PY_ZK=$(resolve_config_value ".${script_config_section}.py_zk" "$py_zk_default" "$config_yaml")
    export ZK_INDEX_SCRIPT=$(resolve_config_value ".${script_config_section}.zk_index_script" "$zk_index_script_default" "$config_yaml")
    export NOTES_DIARY_SUBDIR=$(resolve_config_value ".${script_config_section}.notes_diary_subdir" "$notes_diary_subdir_default" "$config_yaml")
    export BAT_THEME=$(resolve_config_value ".${script_config_section}.bat_theme" "$bat_theme_default" "$config_yaml")
  elif [[ "$script_config_section" == "bibview" ]]; then
    export BIBHIST=$(resolve_config_value ".${script_config_section}.bibhist" "$bibhist_default" "$config_yaml")
    export LIBRARY=$(resolve_config_value ".${script_config_section}.library" "$notes_dir_default/biblib" "$config_yaml")
    export BIBLIOGRAPHY_JSON=$(resolve_config_value ".${script_config_section}.bibliography_json" "$library_default/bibliography.json" "$config_yaml")
    # export FZF_DEFAULT_OPTS_FILE=$(resolve_config_value ".${script_config_section}.fzf_default_opts_file" "$fzf_default_opts_file_default" "$config_yaml")
    export BIBVIEW_OPEN_DOC_SCRIPT=$(resolve_config_value ".${script_config_section}.bibview_open_doc_script" "$bibview_open_doc_script_default" "$config_yaml")
    export LLM_PATH=$(resolve_config_value ".${script_config_section}.llm_path" "$llm_path_default" "$config_yaml")
    export ZK_SCRIPT=$(resolve_config_value ".${script_config_section}.zk_script" "$zk_script_default" "$config_yaml")
    export ADD_TO_READING_LIST_SCRIPT=$(resolve_config_value ".${script_config_section}.add_to_reading_list_script" "$add_to_reading_list_script_default" "$config_yaml")
    export LINK_ZATHURA_TMP_SCRIPT=$(resolve_config_value ".${script_config_section}.link_zathura_tmp_script" "$link_zathura_tmp_script_default" "$config_yaml")
    export OBSIDIAN_SOCKET=$(resolve_config_value ".${script_config_section}.obsidian_socket" "$obsidian_socket_default" "$config_yaml")
    export NOTES_DIR_FOR_ZK=$(resolve_config_value ".${script_config_section}.notes_dir_for_zk" "$notes_dir_for_zk_default" "$config_yaml")
    export BAT_THEME=$(resolve_config_value ".${script_config_section}.bat_theme" "$bat_theme_default" "$config_yaml")
  elif [[ "$script_config_section" == "personSearch" ]]; then
    export PY_ZK=$(resolve_config_value ".${script_config_section}.py_zk" "$py_zk_default" "$config_yaml")
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

