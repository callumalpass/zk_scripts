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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def show_key_bindings() -> None:
    """
    Print the key bindings and their actions.
    """
    bindings = [
        ("ctrl-space", "Open pdf in evince"),
        ("ctrl-z", "Open pdf in evince and abort"),
        ("ctrl-e", "Generate a zettel for bib entry, if none exists"),
        ("ctrl-v", "Open pdf in qpdfview in a right split"),
        ("ctrl-y", "Copy the citation key"),
        ("ctrl-f", "Open in nnn"),
        ("alt-n", "Next history"),
        ("alt-p", "Previous history"),
        ("alt-y", "Copy path of the pdf"),
        ("/", "Toggle preview"),
        ("ctrl-t", "Track reading of bibliographic entry"),
        ("alt-g", "Send to Gemini for humanist reading critique"),
        ("alt-t", "Translate to English"),
    ]
    for key, action in sorted(bindings):
        # Print key in cyan and action in italic.
        print(f"\033[36m{key:<10}\033[0m : \033[3m{action}\033[0m")


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
    # If _TASKFZF_SHOW is set to "keys", print hotkeys and exit
    if os.environ.get("_TASKFZF_SHOW") == "keys":
        show_key_bindings()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Bibliography viewer")
    parser.add_argument(
        "--sort", 
        choices=["year", "dateModified"], 
        default="dateModified",
        help="Sort bibliography by 'year' or 'dateModified'"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Set the default editor
    os.environ["EDITOR"] = "nvim"

    # Load configuration
    config = load_config()
    
    # Get bibview specific configuration
    bibview_config = get_config_value(config, "bibview", {})
    
    bib_json = get_config_value(bibview_config, "bibliography_json", "")
    bibhist = get_config_value(bibview_config, "bibhist", "~/.bibhist")
    bibview_open_doc_script = get_config_value(bibview_config, "bibview_open_doc_script", "")
    llm_path = get_config_value(bibview_config, "llm_path", "")
    library = get_config_value(bibview_config, "library", "")
    notes_dir_for_zk = get_config_value(bibview_config, "notes_dir_for_zk", "")
    bat_theme = get_config_value(bibview_config, "bat_theme", "DEFAULT")
    zk_script = get_config_value(bibview_config, "zk_script", "")
    link_zathura_tmp_script = get_config_value(bibview_config, "link_zathura_tmp_script", "")
    obsidian_socket = get_config_value(bibview_config, "obsidian_socket", "")

    # Resolve paths
    bib_json = resolve_path(bib_json)
    bibhist = resolve_path(bibhist)
    library = resolve_path(library)
    notes_dir_for_zk = resolve_path(notes_dir_for_zk)
    
    # Check required config values
    if not all([bib_json, bibhist, bibview_open_doc_script, library, notes_dir_for_zk, zk_script]):
        logger.error("Missing required configuration values. Please check your config file.")
        sys.exit(1)
        
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Bibliography JSON file: {bib_json}")
        logger.debug(f"Bibliography history file: {bibhist}")
        logger.debug(f"Library path: {library}")
        logger.debug(f"Notes directory: {notes_dir_for_zk}")
        
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
    
    # Build fzf arguments with various custom keybindings
    fzf_args = [
        "fzf",
        "--ansi",
        "--multi",
        "--tiebreak", "begin,index",
        "--header-lines", "1",
        "--preview-window", "right:wrap:38%,<80(up)",
        "--history", bibhist,
        "--bind", "alt-n:next-history",
        "--bind", "alt-p:previous-history",
        "--bind", f"alt-b:execute[{bibview_open_doc_script} {{1}} obsidian ; timew start phd reading {{1}} ; echo {{}} >> \"{bibhist}\"]+abort",
        "--info", "inline",
        "--bind", "/:toggle-preview",
        "--bind", f"alt-g:execute[{llm_path} -t humanist-reading-critique -a {library}/{{1}}/{{1}}.pdf  > {library}/{{1}}/{{1}}.pdf_analysis.md  ]",
        "--bind", f"alt-t:execute[{llm_path} -t translate_to_english -a {library}/{{1}}/{{1}}.pdf  > {library}/{{1}}/{{1}}.pdf_translation.md ]",
        "--bind", f"Ctrl-a:execute[{zk_script} -W \"{notes_dir_for_zk}\" list --format '{{{{path}}}} | {{{{title}}}} |{{{{tags}}}}' `rg {{1}}  \"{notes_dir_for_zk}\" -l || echo 'error'` | ~/mybin/rgnotesearch ]",
        "--bind", f"ctrl-f:execute[nnn \"{library}/\"{{1}}]",
        "--bind", "ctrl-r:execute[~/mybin/addToReadingList {1} ]",
        "--bind", f"Ctrl-e:execute[nvim --server \"{obsidian_socket}\" --remote  {library}/../@{{1}}.md]",
        "--bind", f"alt-z:execute[{link_zathura_tmp_script} {{1}}]",
        "--bind", "Ctrl-y:execute[echo {+1} | wl-copy ]+abort",
        "--bind", "Ctrl-t:execute[timew start phd reading {1}]+abort",
        "--bind", f"Ctrl-space:execute[ {bibview_open_doc_script} {{1}} evince ; timew start phd reading {{1}} ; echo {{1}} >> \"{bibhist}\" ]",
        "--bind", f"Ctrl-z:execute[hyprctl dispatch exec  evince \" {library}/{{1}}/{{1}}.**\" ; echo {{1}} >> \"{bibhist}\" ]+abort",
        "--bind", "?:execute(env _TASKFZF_SHOW=keys \"$0\" | fzf --ansi --header=Help --header-first --info=hidden --bind \"?:abort\")",
        "--bind", f"Enter:execute[{bibview_open_doc_script} {{1}} obsidian ; echo {{}} >> \"{bibhist}\" ]+abort",
        "--preview", f"ls --color=always -c -l -h {library}/{{1}}/; echo '\\nCited in:'  ;  rg {{1}}  \"{notes_dir_for_zk}\" -l --type markdown || echo 'error' ;  bat --theme=\"{bat_theme}\" ~/Dropbox/notes/@{{1}}.md 2> /dev/null "
    ]
    
    if args.debug:
        logger.debug(f"fzf arguments: {' '.join(fzf_args)}")
        
    try:
        subprocess.run(fzf_args, input=formatted_table, text=True)
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)
        
    logger.debug("Bibliography viewer completed successfully")


if __name__ == "__main__":
    main()