"""
A Python module for fuzzy searching Zettelkasten notes using fzf.
"""

import os
import sys
import subprocess
import tempfile
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml

from zk_core.config import load_config, get_config_value, resolve_path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_fzf_bindings(cfg: Dict[str, Any], templink: str, notes_dir: str, index_file: str,
                      notes_diary_subdir: str, bat_theme: str, 
                      script_path: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Return a list of fzf binding strings and also a separate list of tuples
    mapping hotkeys to a human-friendly description.
    """
    # A list of dictionaries; each dict has:
    #   'key': hotkey (for our help listing)
    #   'fzf_cmd': the fzf binding string (to provide to fzf)
    #   'desc': a human-readable description of the command
    bindings = [
        {
            "key": "Enter",
            "fzf_cmd": f"Enter:execute[nvim --server /tmp/obsidian.sock --remote {notes_dir}/{{+1}}.md]+abort",
            "desc": "Open the selected note in nvim (via the obsidian socket)."
        },
        {
            "key": "alt-a",
            "fzf_cmd": f"alt-a:execute[zk-index]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            "desc": "Run the index script and refresh the notes list."
        },
        {
            "key": "alt-c",
            "fzf_cmd": f"alt-c:execute[zk-index]+reload:(zk-query list -i {index_file} --color always -s dateCreated --fields filename --fields title --fields tags --fields dateCreated)",
            "desc": "Sort by creation date"
        },
        {
            "key": "alt-w",
            "fzf_cmd": f"alt-w:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --mode notes -i {notes_dir}/index.json --stdin --format-string '- [[{{filename}}|{{title}}]]' --separator='' >> workingMem.md]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            "desc": "Append selected note as a bullet to workingMem.md and refresh."
        },
        {
            "key": "alt-o",
            "fzf_cmd": f"alt-o:execute[zk-index]+reload:(zk-query list --mode orphans -i {index_file} --color always)",
            "desc": "Run index script and show orphan notes."
        },
        {
            "key": "alt-O",
            "fzf_cmd": f"alt-O:execute[zk-index]+reload:(zk-query list --mode untagged-orphans -i {index_file} --color always)",
            "desc": "Run index script and show untagged orphan notes."
        },
        {
            "key": "alt-e",
            "fzf_cmd": f"alt-e:reload:(zk-query list --mode notes -i {index_file} --color always --exclude-tag literature_note --exclude-tag person --exclude-tag task --exclude-tag diary --exclude-tag journal)",
            "desc": "Refresh list excluding specific tags."
        },
        {
            "key": "alt-y",
            "fzf_cmd": f"alt-y:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --mode notes -i {notes_dir}/index.json --stdin --format-string '[[{{filename}}|{{title}}]]' --separator='' > {templink}]+abort",
            "desc": "Save selected note into a temporary file and exit fzf."
        },
        {
            "key": "ctrl-e",
            "fzf_cmd": f"ctrl-e:execute[nvim {{+1}}.md ; zk-index]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            "desc": "Edit note in nvim then reindex and refresh."
        },
        {
            "key": "ctrl-alt-r",
            "fzf_cmd": f"ctrl-alt-r:execute[rm {{+1}}.md]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            "desc": "Delete the selected note and refresh."
        },
        {
            "key": "ctrl-alt-d",
            "fzf_cmd": f"ctrl-alt-d:execute[nvim {notes_dir}/{notes_diary_subdir}/$(date '+%Y-%m-%d' -d tomorrow).md;]",
            "desc": "Open tomorrow's diary note in nvim."
        },
        {
            "key": "alt-9",
            "fzf_cmd": f"alt-9:reload(zk-query list --mode unique-tags -i {index_file}  --color always)+clear-query",
            "desc": "Reload the list to show unique tags."
        },
        {
            "key": "alt-1",
            "fzf_cmd": f"alt-1:reload(rg -l {{1}} | sed 's/\\.md$//g' | zk-query list --mode notes -i {index_file} --color always --stdin)+clear-query",
            "desc": "Search files via ripgrep and show matching notes."
        },
        {
            "key": "alt-2",
            "fzf_cmd": f"alt-2:reload( zk-query search-embeddings -i {index_file} {{1}} --k 50)+clear-query",
            "desc": "Search notes using embeddings (k=50)."
        },
        {
            "key": "alt-3",
            "fzf_cmd": f"alt-3:reload( zk-query search-embeddings -i {index_file} --k 50 --query {{q}})+clear-query",
            "desc": "Search notes using a freeform query with embeddings."
        },
        {
            "key": "alt-8",
            "fzf_cmd": f"alt-8:reload(zk-query list --mode notes -i {index_file} --filter-tag {{1}}  --color always)+clear-query",
            "desc": "Reload the list filtering for a given tag."
        },
        {
            "key": "?",
            "fzf_cmd": f"?:reload(zk-query info -i {index_file} )",
            "desc": "Display additional info for the selected note."
        },
        {
            "key": "alt-?",
            "fzf_cmd": "alt-?:toggle-preview",
            "desc": "Toggle fzf preview window on/off."
        },
        {
            "key": "alt-j",
            "fzf_cmd": f"alt-j:reload(zk-query list --mode notes -i {index_file} --filter-tag diary --color always)",
            "desc": "Reload list to show notes tagged 'diary'."
        },
        {
            "key": "alt-g",
            "fzf_cmd": f"alt-g:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --stdin -i {index_file} --format-string '- [[{{filename}}|{{title}}]]' >> {{1}}.md]+clear-selection",
            "desc": "Append formatted selected note to file."
        },
        {
            "key": "alt-b",
            "fzf_cmd": f"alt-b:reload(zk-query list --mode notes -i {index_file} --filter-tag literature_note --color always)",
            "desc": "Reload list showing notes with tag 'literature_note'."
        },
        {
            "key": "ctrl-s",
            "fzf_cmd": "ctrl-s:reload(rg -U '^' --sortr modified --field-match-separator='::' --color=always --type md -n | sed 's/\\.md//' )",
            "desc": "Reload list sorted by modification time via ripgrep."
        },
        # Help hotkey
        {
            "key": "alt-h",
            "fzf_cmd": f"alt-h:execute[python3 {script_path} --list-hotkeys | fzf]",
            "desc": "Show this hotkeys help (prints the list of hotkeys)."
        }
    ]

    fzf_bindings = []
    hotkeys_info = []
    for binding in bindings:
        fzf_bindings += ["--bind", binding["fzf_cmd"]]
        hotkeys_info.append((binding["key"], binding["desc"]))
    return fzf_bindings, hotkeys_info

def print_hotkeys(hotkeys_info: List[Tuple[str, str]]) -> None:
    """Print a formatted list of hotkeys and their descriptions."""
    print("Available fzf hotkeys and their functions:")
    for key, desc in hotkeys_info:
        print(f"  {key:<12} : {desc}")
    print()

def main() -> None:
    """Main entry point for the fzf interface."""
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="A fuzzy search interface for Zettelkasten notes."
    )
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip running the index script before launching fzf.")
    parser.add_argument("--skip-notelist", action="store_true",
                        help="Skip launching the background process that updates the notelist.")
    parser.add_argument("--skip-tmux", action="store_true",
                        help="Skip sending the link output to tmux.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be executed, but do not execute them.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output.")
    parser.add_argument("--list-hotkeys", action="store_true",
                        help="Print a list of available fzf hotkeys and their functions, then exit.")
    parser.add_argument("--config-file", help="Specify config file path")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    
    # Get necessary configuration values from new config structure
    notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    notes_dir = resolve_path(notes_dir)
    
    index_file = get_config_value(config, "zk_index.index_file", "index.json")
    # If index_file doesn't include a path, join it with notes_dir
    if not os.path.dirname(index_file):
        index_file = os.path.join(notes_dir, index_file)
    index_file = resolve_path(index_file)
    
    # Use entry point script names directly
    py_zk = "zk-query"
    zk_index_script = "zk-index"
    
    notes_diary_subdir = get_config_value(config, "fzf_interface.diary_subdir", "")
    
    bat_theme = get_config_value(config, "fzf_interface.bat_theme", "default")

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("NOTES_DIR = %s", notes_dir)
        logger.debug("INDEX_FILE = %s", index_file)
        logger.debug("PY_ZK = %s", py_zk)
        logger.debug("ZK_INDEX_SCRIPT = %s", zk_index_script)

    try:
        os.chdir(notes_dir)
    except Exception as e:
        logger.error(f"Unable to change directory to {notes_dir}: {e}")
        sys.exit(1)

    # Create a temporary file for use by one of the keybindings.
    temp = tempfile.NamedTemporaryFile(delete=False)
    templink = temp.name
    temp.close()

    # Determine our script's full path (to use in the Alt-h binding).
    script_path = os.path.abspath(__file__)

    # Build the fzf bindings (and also the hotkeys information).
    fzf_bindings, hotkeys_info = build_fzf_bindings(
        {}, templink, notes_dir, index_file,
        notes_diary_subdir, bat_theme, script_path
    )
    
    if args.list_hotkeys:
        print_hotkeys(hotkeys_info)
        sys.exit(0)

    # Launch a background process that writes a notelist (unless skipped).
    home = Path.home()
    notelist_path = home / "Documents" / "notelist.md"
    if not args.skip_notelist:
        try:
            if args.dry_run:
                print(f"[DRY-RUN] Would launch: {py_zk} list --mode notes -i {index_file} --format-string '[[{{filename}}|{{title}}]]'")
            else:
                with open(notelist_path, "w") as nl:
                    # Use the entry point name directly
                    subprocess.Popen(["zk-query", "list", "--mode", "notes", "-i", index_file,
                                     "--format-string", "[[{filename}|{title}]]"],
                                     stdout=nl)
                if args.debug:
                    logger.debug("Launched background notelist update.")
        except Exception as e:
            logger.error(f"Error launching notelist update: {e}")
    else:
        if args.debug:
            logger.debug("Skipping notelist update (--skip-notelist).")

    # Run the index script (unless skipped)
    if not args.skip_index:
        if args.dry_run:
            print(f"[DRY-RUN] Would run index script: {zk_index_script}")
        else:
            try:
                # Use the entry point script name directly
                subprocess.run(["zk-index"], check=False)
                if args.debug:
                    logger.debug("Ran index script in foreground.")
            except Exception as e:
                logger.error(f"Error running zk-index script: {e}")
    else:
        if args.debug:
            logger.debug("Skipping index script (--skip-index).")

    # Build the py_zk list command.
    py_zk_list_cmd = ["zk-query", "list", "--mode", "notes", "-i", index_file, "--color", "always"]
    if args.debug:
        logger.debug("py_zk_list_cmd = %s", " ".join(py_zk_list_cmd))

    # Build the fzf command.
    fzf_args = [
        "fzf",
        "--tiebreak=chunk,begin",
        "--delimiter=::",
        "--scheme=default",
        "--info=right",
        "--ellipsis=",
        "--preview-label=",
        "--multi",
        "--wrap",
    ]
    fzf_args.extend(fzf_bindings)
    
    # Add preview command.
    preview_cmd = f"echo \"Backlinks:\"; zk-query list -i {index_file} --filter-outgoing-link {{1}} --color always; bat --theme=\"{bat_theme}\" --color=always --decorations=never {{1}}.md -H {{2}} 2> /dev/null || bat {{1}}.md"
    fzf_args.extend(["--preview", preview_cmd,
                    "--preview-window", "wrap:50%:<50(up)",
                    "--color", "16",
                    "--ansi"])
                    
    if args.debug:
        logger.debug("fzf args: %s", " ".join(fzf_args))

    if args.dry_run:
        print(f"[DRY-RUN] Would run: {' '.join(py_zk_list_cmd)} | {' '.join(fzf_args)}")
        sys.exit(0)
        
    # Run fzf with the py_zk list output piped into it.
    try:
        p1 = subprocess.Popen(py_zk_list_cmd, stdout=subprocess.PIPE)
        fzf_result = subprocess.run(fzf_args, stdin=p1.stdout)
        p1.stdout.close()
        if args.debug:
            logger.debug(f"fzf returned code {fzf_result.returncode}")
    except Exception as e:
        logger.error(f"Error running fzf: {e}")
        sys.exit(1)

    # Process the temporary link file (if it exists).
    if os.path.exists(templink):
        try:
            with open(templink, "r") as tf:
                link = tf.read().strip()
            if link:
                if args.skip_tmux:
                    if args.debug:
                        logger.debug(f"Skipping sending tmux; link: {link}")
                else:
                    subprocess.run(["tmux", "send-keys", "-l", link])
                    if args.debug:
                        logger.debug(f"Sent link to tmux: {link}")
        except Exception as e:
            logger.error(f"Error processing temporary link file: {e}")
    else:
        sys.exit(0)

    # Clean up temporary file.
    try:
        os.remove(templink)
    except Exception:
        pass

if __name__ == "__main__":
    main()