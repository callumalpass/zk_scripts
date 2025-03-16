"""
Bibliography viewer module.

This module provides an interactive interface for viewing bibliography entries.
It loads bibliographic configuration, processes bibliography JSON file using gojq,
formats the data with colors and icons, and feeds it into fzf with custom keybindings.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.fzf_manager import FzfManager, FzfBinding

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_bibview_fzf_manager(bibhist: str, library: str, llm_path: str = "", 
                             zk_script: str = "", notes_dir_for_zk: str = "",
                             obsidian_socket: str = "", link_zathura_tmp_script: str = "",
                             bat_theme: str = "DEFAULT") -> FzfManager:
    """
    Create and configure an FzfManager with bibview-specific bindings.
    
    Args:
        bibhist: Path to the bibliography history file
        library: Path to the bibliography library directory
        llm_path: Optional path to the LLM script
        zk_script: Optional path to the ZK script
        notes_dir_for_zk: Optional path to the notes directory for ZK
        obsidian_socket: Optional path to the Obsidian socket
        link_zathura_tmp_script: Optional path to the Zathura tmp script
        bat_theme: Theme for bat
        
    Returns:
        Configured FzfManager instance
    """
    # Create a new FzfManager instance
    manager = FzfManager()
    
    # Add basic bindings
    manager.add_bindings([
        FzfBinding(
            key="ctrl-f",
            command=f"ctrl-f:execute[nnn \"{library}/\"{{1}}]",
            description="Open in nnn",
            category="Navigation"
        ),
        FzfBinding(
            key="ctrl-y",
            command="Ctrl-y:execute[echo {+1} | wl-copy ]+abort",
            description="Copy the citation key",
            category="Export"
        ),
        FzfBinding(
            key="alt-n",
            command="alt-n:next-history",
            description="Next history",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-p",
            command="alt-p:previous-history",
            description="Previous history",
            category="Navigation"
        ),
        FzfBinding(
            key="/",
            command="/:toggle-preview",
            description="Toggle preview",
            category="Navigation"
        ),
        FzfBinding(
            key="ctrl-t",
            command="Ctrl-t:execute[timew start phd reading {1}]+abort",
            description="Track reading of bibliographic entry",
            category="Tracking"
        ),
        FzfBinding(
            key="ctrl-z",
            command=f"Ctrl-z:execute[hyprctl dispatch exec evince \" {library}/{{1}}/{{1}}.**\" ; echo {{1}} >> \"{bibhist}\" ]+abort",
            description="Open pdf in evince and abort",
            category="Document"
        ),
    ])
    
    # Add optional bindings based on available scripts
    if llm_path:
        manager.add_bindings([
            FzfBinding(
                key="alt-g",
                command=f"alt-g:execute[{llm_path} -t humanist-reading-critique -a {library}/{{1}}/{{1}}.pdf > {library}/{{1}}/{{1}}.pdf_analysis.md ]",
                description="Send to Gemini for humanist reading critique",
                category="Analysis"
            ),
            FzfBinding(
                key="alt-t",
                command=f"alt-t:execute[{llm_path} -t translate_to_english -a {library}/{{1}}/{{1}}.pdf > {library}/{{1}}/{{1}}.pdf_translation.md ]",
                description="Translate to English",
                category="Analysis"
            ),
        ])
    
    if zk_script and notes_dir_for_zk:
        manager.add_bindings([
            FzfBinding(
                key="ctrl-a",
                command=f"Ctrl-a:execute[{zk_script} -W \"{notes_dir_for_zk}\" list --format '{{{{path}}}} | {{{{title}}}} |{{{{tags}}}}' `rg {{1}} \"{notes_dir_for_zk}\" -l || echo 'error'` | ~/mybin/rgnotesearch ]",
                description="List ZK notes referencing this entry",
                category="Notes"
            ),
        ])
    
    # Add bindings dependent on specific tools
    if os.path.exists(os.path.expanduser("~/mybin/addToReadingList")):
        manager.add_bindings([
            FzfBinding(
                key="ctrl-r",
                command="ctrl-r:execute[~/mybin/addToReadingList {1} ]",
                description="Add to reading list",
                category="Tracking"
            ),
        ])
    
    # Add help binding
    manager.add_help_binding("bibview")
    
    return manager


def print_hotkeys(fzf_manager: FzfManager) -> None:
    """
    Print the key bindings and their actions.
    
    Args:
        fzf_manager: The FzfManager instance with all bindings
    """
    # Simple output formatting for bibview-specific help
    print("\033[1;36m=== BIBVIEW KEYBOARD SHORTCUTS ===\033[0m")
    
    # Group bindings by category for better organization
    categories = {}
    for binding in fzf_manager.bindings:
        if binding.category not in categories:
            categories[binding.category] = []
        categories[binding.category].append(binding)
    
    # Print bindings grouped by category
    for category, bindings in sorted(categories.items()):
        # Skip empty categories
        if not bindings:
            continue
            
        print(f"\n\033[1;33m{category}:\033[0m")
        for binding in sorted(bindings, key=lambda b: b.key):
            # Print key in cyan and action in italic
            print(f"\033[36m{binding.key:<10}\033[0m : \033[3m{binding.desc}\033[0m")
    
    print("\n\033[1;36mPress q to exit this help screen\033[0m")


def format_bibliography_data(data: str, args: argparse.Namespace) -> str:
    """
    Process the bibliographic data and format it for display.
    
    Args:
        data: Raw data from gojq
        args: Command line arguments
        
    Returns:
        Formatted table string
    """
    lines = data.strip().splitlines()
    formatted_lines = []
    
    # Add a header line
    header = "Citekey|Year| |Title|Authors/Editors|Abstract"
    formatted_lines.append(header)

    # Define color codes
    CYAN = "\033[36m"
    ITALIC = "\033[3m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    for line in lines:
        fields = line.split("\t")
        if len(fields) < 6:
            fields += [""] * (6 - len(fields))
        year, citekey, authors, title, typ, abstract = fields[:6]

        # Determine an icon based on the publication type
        if typ == "chapter":
            icon = "📖"
        elif typ == "book":
            icon = "📕"
        elif typ == "article-journal":
            icon = "📝"
        elif typ == "article-newspaper":
            icon = "🗞"
        elif typ == "thesis":
            icon = "🎓"
        else:
            icon = "❓"

        # Build table columns
        col1 = f"{CYAN}{ITALIC}{citekey}{RESET}"
        col2 = year
        col3 = icon
        col4 = f"{ITALIC}{title[:90]}{RESET}"
        col5 = f"{BLUE}{authors}{RESET}"
        col6 = abstract
        formatted_line = "|".join([col1, col2, col3, col4, col5, col6])
        formatted_lines.append(formatted_line)
    
    table_str = "\n".join(formatted_lines)
    
    # Use the 'column' command to format the table for a neat display
    try:
        col_result = subprocess.run(
            ["column", "-s", "|", "-t", "-N", "Citekey,Year, ,Title,Authors/Editors,Abstract"],
            input=table_str, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        formatted_table = col_result.stdout
    except Exception as e:
        logger.error(f"Error formatting table with column: {e}")
        formatted_table = table_str
        
    if args.debug:
        logger.debug("Formatted table:")
        logger.debug(formatted_table)
        
    return formatted_table


def main() -> None:
    """Main entry point for the bibview module."""
    parser = argparse.ArgumentParser(description="Bibliography viewer")
    parser.add_argument(
        "--sort", 
        choices=["year", "dateModified"], 
        default="dateModified",
        help="Sort bibliography by 'year' or 'dateModified'"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--list-hotkeys", action="store_true", 
                      help="Print a list of available fzf hotkeys and their functions, then exit")
    args = parser.parse_args()

    # Set the default editor
    os.environ["EDITOR"] = "nvim"

    # Try to load configuration from the standard location
    config = load_config()
    
    # Get bibview specific configuration
    bibview_config = get_config_value(config, "bibview", {})
    
    # Debug the loaded configuration
    logger.error(f"Loaded config: {config}")
    logger.error(f"Bibview config section: {bibview_config}")
    
    # If bibview_config is empty, try to load it directly from file
    if not bibview_config:
        logger.warning("Bibview section missing in config, loading directly from file")
        try:
            import yaml
            with open(os.path.expanduser("~/.config/zk_scripts/config.yaml"), 'r') as f:
                direct_config = yaml.safe_load(f)
                if direct_config and "bibview" in direct_config:
                    bibview_config = direct_config["bibview"]
                    logger.error(f"Direct loaded bibview config: {bibview_config}")
        except Exception as e:
            logger.error(f"Error loading config directly: {e}")
    
    # Set defaults for required values
    bib_json = get_config_value(bibview_config, "bibliography_json", "~/Dropbox/notes/biblib/bibliography.json")
    bibhist = get_config_value(bibview_config, "bibhist", "~/.cache/bibview.history")
    library = get_config_value(bibview_config, "library", "~/Dropbox/notes/biblib/")
    notes_dir_for_zk = get_config_value(bibview_config, "notes_dir_for_zk", "~/Dropbox/notes")
    
    # Optional values with empty defaults
    bibview_open_doc_script = get_config_value(bibview_config, "bibview_open_doc_script", "")
    llm_path = get_config_value(bibview_config, "llm_path", "")
    bat_theme = get_config_value(bibview_config, "bat_theme", "DEFAULT")
    zk_script = get_config_value(bibview_config, "zk_script", "")
    link_zathura_tmp_script = get_config_value(bibview_config, "link_zathura_tmp_script", "")
    obsidian_socket = get_config_value(bibview_config, "obsidian_socket", "")

    # Properly resolve all paths using resolve_path
    bib_json = resolve_path(bib_json)
    bibhist = resolve_path(bibhist)
    library = resolve_path(library)
    notes_dir_for_zk = resolve_path(notes_dir_for_zk)
    bibview_open_doc_script = resolve_path(bibview_open_doc_script)
    llm_path = resolve_path(llm_path)
    zk_script = resolve_path(zk_script)
    link_zathura_tmp_script = resolve_path(link_zathura_tmp_script)
    
    # Print loaded configuration values for debugging
    logger.error(f"DEBUG - Configuration values:")
    logger.error(f"bib_json: '{bib_json}'")
    logger.error(f"bibhist: '{bibhist}'")
    logger.error(f"library: '{library}'")
    logger.error(f"notes_dir_for_zk: '{notes_dir_for_zk}'")
    logger.error(f"bibview_open_doc_script: '{bibview_open_doc_script}'")
    logger.error(f"zk_script: '{zk_script}'")
    logger.error(f"link_zathura_tmp_script: '{link_zathura_tmp_script}'")
    
    # Check minimum required config values
    if not all([bib_json, bibhist, library, notes_dir_for_zk]):
        logger.error("Missing required configuration values. Please check your config file.")
        sys.exit(1)
        
    # Check if scripts exist and log warnings for missing scripts
    if bibview_open_doc_script and not os.path.exists(bibview_open_doc_script):
        logger.warning(f"Script not found: {bibview_open_doc_script}")
        logger.warning("Some document opening features will be disabled")
    
    if zk_script and not os.path.exists(zk_script):
        logger.warning(f"Script not found: {zk_script}")
        logger.warning("Note listing features will be disabled")
        # Don't fail, just disable the feature
        zk_script = ""
    
    if link_zathura_tmp_script and not os.path.exists(link_zathura_tmp_script):
        logger.warning(f"Script not found: {link_zathura_tmp_script}")
        logger.warning("Zathura features will be disabled")
        # Don't fail, just disable the feature
        link_zathura_tmp_script = ""
        
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Bibliography JSON file: {bib_json}")
        logger.debug(f"Bibliography history file: {bibhist}")
        logger.debug(f"Library path: {library}")
        logger.debug(f"Notes directory: {notes_dir_for_zk}")
    
    # Build the FzfManager with all bindings
    fzf_manager = build_bibview_fzf_manager(
        bibhist=bibhist,
        library=library,
        llm_path=llm_path,
        zk_script=zk_script,
        notes_dir_for_zk=notes_dir_for_zk,
        obsidian_socket=obsidian_socket,
        link_zathura_tmp_script=link_zathura_tmp_script,
        bat_theme=bat_theme
    )
    
    # Add bibview_open_doc_script bindings if available
    if bibview_open_doc_script:
        fzf_manager.add_bindings([
            FzfBinding(
                key="alt-b",
                command=f"alt-b:execute[{bibview_open_doc_script} {{1}} obsidian ; timew start phd reading {{1}} ; echo {{}} >> \"{bibhist}\"]+abort",
                description="Open PDF in Obsidian, track reading, and abort",
                category="Document"
            ),
            FzfBinding(
                key="ctrl-space",
                command=f"Ctrl-space:execute[ {bibview_open_doc_script} {{1}} evince ; timew start phd reading {{1}} ; echo {{1}} >> \"{bibhist}\" ]",
                description="Open PDF in evince and track reading",
                category="Document"
            ),
            FzfBinding(
                key="Enter",
                command=f"Enter:execute[{bibview_open_doc_script} {{1}} obsidian ; echo {{}} >> \"{bibhist}\" ]+abort",
                description="Open PDF in Obsidian and abort",
                category="Document"
            ),
        ])
    else:
        # Fallback binding for Enter if no open script is available
        fzf_manager.add_binding(
            FzfBinding(
                key="Enter",
                command=f"Enter:execute[echo {{1}} >> \"{bibhist}\" ]+abort",
                description="Save entry to history and abort",
                category="Navigation"
            )
        )
    
    # Add Obsidian socket binding if available
    if obsidian_socket:
        fzf_manager.add_binding(
            FzfBinding(
                key="ctrl-e",
                command=f"Ctrl-e:execute[nvim --server \"{obsidian_socket}\" --remote {library}/../@{{1}}.md]",
                description="Edit note in nvim via Obsidian socket",
                category="Editing"
            )
        )
    
    # Add Zathura script binding if available
    if link_zathura_tmp_script:
        fzf_manager.add_binding(
            FzfBinding(
                key="alt-z",
                command=f"alt-z:execute[{link_zathura_tmp_script} {{1}}]",
                description="Open PDF in Zathura",
                category="Document"
            )
        )
    
    # If --list-hotkeys was specified, print the hotkey help and exit
    if args.list_hotkeys:
        print_hotkeys(fzf_manager)
        sys.exit(0)
        
    # Build the gojq command based on sort order
    if args.sort == "year":
        jq_filter = (
            '. | sort_by(.year) | reverse[] | [ (.issued?."date-parts"?[0][0])?, '
            '.id?, ([.author[]? // .editor[]? | .given? + " " + .family?] | join(", ")), '
            '.title?, .type?, .abstract? ] | @tsv'
        )
    else:
        jq_filter = (
            '. | sort_by(.dateModified) | reverse[] | [ (.issued?."date-parts"?[0][0])?, '
            '.id?, ([.author[]? // .editor[]? | .given? + " " + .family?] | join(", ")), '
            '.title?, .type?, .abstract? ] | @tsv'
        )
    
    gojq_cmd = ["gojq", "-r", jq_filter, bib_json]
    
    if args.debug:
        logger.debug(f"Running gojq command: {' '.join(gojq_cmd)}")
        
    try:
        jq_result = subprocess.run(gojq_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        data = jq_result.stdout
    except Exception as e:
        logger.error(f"Error running gojq: {e}")
        sys.exit(1)
        
    # Format the bibliography data
    formatted_table = format_bibliography_data(data, args)
    
    # Prepare additional fzf arguments specific to bibview
    additional_args = [
        "--multi",
        "--tiebreak", "begin,index",
        "--header-lines", "1",
        "--preview-window", "right:wrap:38%,<80(up)",
        "--history", bibhist,
        "--info", "inline",
        "--preview", f"ls --color=always -c -l -h {library}/{{1}}/; echo '\\nCited in:' ; " + 
                   f"rg {{1}} \"{notes_dir_for_zk}\" -l --type markdown || echo 'error' ; " + 
                   f"bat --theme=\"{bat_theme}\" ~/Dropbox/notes/@{{1}}.md 2> /dev/null "
    ]
    
    # Get complete fzf arguments
    fzf_args = fzf_manager.get_fzf_args(additional_args)
    
    if args.debug:
        logger.debug(f"fzf arguments: {' '.join(fzf_args)}")
        
    try:
        # Run fzf with the formatted table as input
        fzf_result = fzf_manager.run_fzf(formatted_table, additional_args)
        if args.debug:
            logger.debug(f"fzf returned code {fzf_result.returncode}")
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)
        
    logger.debug("Bibliography viewer completed successfully")


if __name__ == "__main__":
    main()