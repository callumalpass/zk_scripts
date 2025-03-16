"""
Bibliography viewer module.

This module provides an interactive interface for viewing bibliography entries:
- FZF-based interface for browsing bibliography data
- Colored and formatted display of bibliographic entries
- Integration with external tools (zathura, evince, etc.)
- Custom keybindings for different actions
"""

import os
import sys
import logging
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.constants import DEFAULT_NOTES_DIR
from zk_core.fzf_manager import FzfManager, FzfBinding
from zk_core.fzf_utils import FzfHelper, colorize
from zk_core.commands import CommandExecutor
from zk_core.utils import get_path_for_script, get_socket_path

logger = logging.getLogger(__name__)


def format_bibliography_data(data: str, debug: bool = False) -> str:
    """
    Process the bibliographic data and format it for display.
    
    Args:
        data: Raw data from gojq
        debug: Whether to output debug information
        
    Returns:
        Formatted table string
    """
    lines = data.strip().splitlines()
    formatted_lines = []
    
    # Add a header line
    header = "Citekey|Year| |Title|Authors/Editors|Abstract"
    formatted_lines.append(header)

    for line in lines:
        fields = line.split("\t")
        if len(fields) < 6:
            fields += [""] * (6 - len(fields))
        year, citekey, authors, title, typ, abstract = fields[:6]

        # Determine an icon based on the publication type
        icon = get_pub_type_icon(typ)

        # Build table columns
        col1 = colorize(f"{citekey}", "cyan")
        col2 = year
        col3 = icon
        col4 = colorize(f"{title[:90]}", "italic")
        col5 = colorize(f"{authors}", "blue")
        col6 = abstract
        formatted_line = "|".join([col1, col2, col3, col4, col5, col6])
        formatted_lines.append(formatted_line)
    
    table_str = "\n".join(formatted_lines)
    
    # Use the 'column' command to format the table for a neat display
    try:
        rc, stdout, stderr = CommandExecutor.run(
            ["column", "-s", "|", "-t", "-N", "Citekey,Year, ,Title,Authors/Editors,Abstract"],
            input_data=table_str
        )
        if rc == 0:
            formatted_table = stdout
        else:
            logger.warning(f"Error formatting table with column: {stderr}")
            formatted_table = table_str
    except Exception as e:
        logger.warning(f"Exception formatting table with column: {e}")
        formatted_table = table_str
        
    if debug:
        logger.debug("Formatted table:")
        logger.debug(formatted_table)
        
    return formatted_table


def get_pub_type_icon(pub_type: str) -> str:
    """
    Get an icon for a publication type.
    
    Args:
        pub_type: Publication type string
        
    Returns:
        Icon string representation
    """
    # Define icons for different publication types
    icons = {
        "chapter": "📖",
        "book": "📕",
        "article-journal": "📝",
        "article-newspaper": "🗞",
        "thesis": "🎓", 
        "paper-conference": "🎤",
        "report": "📊",
        "webpage": "🌐",
        "post": "📮"
    }
    return icons.get(pub_type, "❓")


def build_bibview_fzf_manager(
    bibhist: str, 
    library: str, 
    llm_path: str = "", 
    zk_script: str = "", 
    notes_dir_for_zk: str = "",
    obsidian_socket: str = "", 
    link_zathura_tmp_script: str = "",
    bat_theme: str = "DEFAULT",
    bibview_open_doc_script: str = ""
) -> FzfManager:
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
        bibview_open_doc_script: Path to script for opening documents
        
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
    
    # Add bibview_open_doc_script bindings if available
    if bibview_open_doc_script:
        manager.add_bindings([
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
        manager.add_binding(
            FzfBinding(
                key="Enter",
                command=f"Enter:execute[echo {{1}} >> \"{bibhist}\" ]+abort",
                description="Save entry to history and abort",
                category="Navigation"
            )
        )
    
    # Add Obsidian socket binding if available
    if obsidian_socket:
        manager.add_binding(
            FzfBinding(
                key="ctrl-e",
                command=f"Ctrl-e:execute[nvim --server \"{obsidian_socket}\" --remote {library}/../@{{1}}.md]",
                description="Edit note in nvim via Obsidian socket",
                category="Editing"
            )
        )
    
    # Add Zathura script binding if available
    if link_zathura_tmp_script:
        manager.add_binding(
            FzfBinding(
                key="alt-z",
                command=f"alt-z:execute[{link_zathura_tmp_script} {{1}}]",
                description="Open PDF in Zathura",
                category="Document"
            )
        )
    
    # Add help binding
    manager.add_help_binding("bibview")
    
    return manager


def get_bibview_config(config: Dict[str, Any], args: Optional[Any] = None) -> Dict[str, str]:
    """
    Get bibliography viewer configuration with proper defaults.
    
    Args:
        config: Configuration dictionary
        args: Optional command line arguments
        
    Returns:
        Dictionary of bibliography viewer configuration
    """
    # Get bibview specific configuration
    bibview_config = get_config_value(config, "bibview", {})
    
    # Log loaded configuration
    logger.debug(f"Loaded config: {config}")
    logger.debug(f"Bibview config section: {bibview_config}")
    
    # If bibview_config is empty, try to load it directly from file
    if not bibview_config:
        logger.warning("Bibview section missing in config, loading directly from file")
        try:
            import yaml
            config_path = os.path.expanduser("~/.config/zk_scripts/config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    direct_config = yaml.safe_load(f)
                    if direct_config and "bibview" in direct_config:
                        bibview_config = direct_config["bibview"]
                        logger.debug(f"Direct loaded bibview config: {bibview_config}")
            else:
                logger.warning(f"Config file not found at {config_path}")
        except Exception as e:
            logger.warning(f"Error loading config directly: {e}")
    
    # Set defaults for required values
    bib_json = get_config_value(bibview_config, "bibliography_json", "~/Dropbox/notes/biblib/bibliography.json")
    bibhist = get_config_value(bibview_config, "bibhist", "~/.cache/bibview.history")
    library = get_config_value(bibview_config, "library", "~/Dropbox/notes/biblib/")
    notes_dir_for_zk = get_config_value(bibview_config, "notes_dir_for_zk", DEFAULT_NOTES_DIR)
    
    # Optional values with empty defaults
    bibview_open_doc_script = get_config_value(bibview_config, "bibview_open_doc_script", "")
    llm_path = get_config_value(bibview_config, "llm_path", "")
    bat_theme = get_config_value(bibview_config, "bat_theme", "DEFAULT")
    zk_script = get_config_value(bibview_config, "zk_script", "")
    link_zathura_tmp_script = get_config_value(bibview_config, "link_zathura_tmp_script", "")
    obsidian_socket = get_config_value(bibview_config, "obsidian_socket", "")
    
    # Properly resolve all paths
    bib_json = resolve_path(bib_json)
    bibhist = resolve_path(bibhist)
    library = resolve_path(library)
    notes_dir_for_zk = resolve_path(notes_dir_for_zk)
    
    # Resolve optional script paths
    if bibview_open_doc_script:
        bibview_open_doc_script = resolve_path(bibview_open_doc_script)
    if llm_path:
        llm_path = resolve_path(llm_path)
    if zk_script:
        zk_script = resolve_path(zk_script)
    if link_zathura_tmp_script:
        link_zathura_tmp_script = resolve_path(link_zathura_tmp_script)
        
    return {
        "bib_json": bib_json,
        "bibhist": bibhist,
        "library": library,
        "notes_dir_for_zk": notes_dir_for_zk,
        "bibview_open_doc_script": bibview_open_doc_script,
        "llm_path": llm_path,
        "bat_theme": bat_theme,
        "zk_script": zk_script,
        "link_zathura_tmp_script": link_zathura_tmp_script,
        "obsidian_socket": obsidian_socket
    }


def run_viewer(config: Optional[Dict[str, Any]] = None, args: Optional[argparse.Namespace] = None) -> int:
    """
    Main function to run bibliography viewer.
    
    Args:
        config: Optional configuration dictionary (loaded if not provided)
        args: Optional command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Create default args if none provided
    if args is None:
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
    
    # Set logging level based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration if not provided
    if config is None:
        config = load_config()
    
    # Get bibview configuration
    bibview_config = get_bibview_config(config, args)
    bib_json = bibview_config["bib_json"]
    bibhist = bibview_config["bibhist"]
    library = bibview_config["library"]
    notes_dir_for_zk = bibview_config["notes_dir_for_zk"]
    bibview_open_doc_script = bibview_config["bibview_open_doc_script"]
    llm_path = bibview_config["llm_path"]
    bat_theme = bibview_config["bat_theme"]
    zk_script = bibview_config["zk_script"]
    link_zathura_tmp_script = bibview_config["link_zathura_tmp_script"]
    obsidian_socket = bibview_config["obsidian_socket"]
    
    # Print loaded configuration values for debugging
    if args.debug:
        logger.debug("Configuration values:")
        logger.debug(f"bib_json: '{bib_json}'")
        logger.debug(f"bibhist: '{bibhist}'")
        logger.debug(f"library: '{library}'")
        logger.debug(f"notes_dir_for_zk: '{notes_dir_for_zk}'")
        logger.debug(f"bibview_open_doc_script: '{bibview_open_doc_script}'")
        logger.debug(f"zk_script: '{zk_script}'")
        logger.debug(f"link_zathura_tmp_script: '{link_zathura_tmp_script}'")
    
    # Check minimum required config values
    if not all([bib_json, bibhist, library, notes_dir_for_zk]):
        logger.error("Missing required configuration values. Please check your config file.")
        return 1
        
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
    
    # Build the FzfManager with all bindings
    fzf_manager = build_bibview_fzf_manager(
        bibhist=bibhist,
        library=library,
        llm_path=llm_path,
        zk_script=zk_script,
        notes_dir_for_zk=notes_dir_for_zk,
        obsidian_socket=obsidian_socket,
        link_zathura_tmp_script=link_zathura_tmp_script,
        bat_theme=bat_theme,
        bibview_open_doc_script=bibview_open_doc_script
    )
    
    # If --list-hotkeys was specified, print the hotkey help and exit
    if args.list_hotkeys:
        FzfHelper.print_hotkeys(fzf_manager, "BIBVIEW KEYBOARD SHORTCUTS")
        return 0
        
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
        # Run gojq command
        rc, data, err = CommandExecutor.run(gojq_cmd)
        if rc != 0:
            logger.error(f"Error running gojq: {err}")
            return 1
    except Exception as e:
        logger.error(f"Exception running gojq: {e}")
        return 1
        
    # Format the bibliography data
    formatted_table = format_bibliography_data(data, args.debug)
    
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
                   f"bat --theme=\"{bat_theme}\" {notes_dir_for_zk}/@{{1}}.md 2> /dev/null "
    ]
    
    # Get complete fzf arguments
    fzf_args = fzf_manager.get_fzf_args(additional_args)
    
    if args.debug:
        logger.debug(f"fzf arguments: {' '.join(fzf_args)}")
        
    try:
        # Run fzf with the formatted table as input using FzfHelper
        fzf_result = FzfHelper.run_fzf(fzf_manager, formatted_table, additional_args)
        if args.debug:
            logger.debug(f"fzf returned code {fzf_result.returncode}")
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        return 1
        
    logger.debug("Bibliography viewer completed successfully")
    return 0


def main() -> None:
    """Entry point for command-line use."""
    exit_code = run_viewer()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()