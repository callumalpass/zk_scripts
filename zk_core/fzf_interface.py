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

from zk_core.constants import DEFAULT_NOTES_DIR, DEFAULT_NVIM_SOCKET
from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.fzf_manager import FzfManager, FzfBinding

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def build_fzf_manager(templink: str, notes_dir: str, index_file: str,
                     notes_diary_subdir: str, bat_theme: str, socket_path: str) -> FzfManager:
    """
    Create and configure a FzfManager with all the bindings for the ZK interface.
    
    Args:
        templink: Path to the temporary link file
        notes_dir: Path to the notes directory
        index_file: Path to the index file
        notes_diary_subdir: Subdirectory for diary notes
        bat_theme: Theme for bat
        
    Returns:
        Configured FzfManager instance
    """
    # Create a new FzfManager instance
    manager = FzfManager()
    
    # Navigation & Viewing bindings
    manager.add_bindings([
        FzfBinding(
            key="Enter",
            command=f"Enter:execute[nvim --server {socket_path} --remote {notes_dir}/{{+1}}.md]+abort",
            description="Open the selected note in nvim (via the configured socket).",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-?",
            command="alt-?:toggle-preview",
            description="Toggle fzf preview window on/off.",
            category="Navigation"
        ),
        FzfBinding(
            key="?",
            command=f"?:reload(zk-query info -i {index_file} )",
            description="Display additional info for the selected note.",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-1",
            command=f"alt-1:reload(rg -l {{1}} | sed 's/\\.md$//g' | zk-query list --mode notes -i {index_file} --color always --stdin)+clear-query",
            description="Search files via ripgrep and show matching notes.",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-2", 
            command=f"alt-2:reload( zk-query search-embeddings -i {index_file} {{1}} --k 50)+clear-query",
            description="Search notes using embeddings (k=50).",
            category="Navigation"
        ),
        FzfBinding(
            key="alt-3",
            command=f"alt-3:reload( zk-query search-embeddings -i {index_file} --k 50 --query {{q}})+clear-query",
            description="Search notes using a freeform query with embeddings.",
            category="Navigation"
        ),
    ])
    
    # Filtering & Sorting bindings
    manager.add_bindings([
        FzfBinding(
            key="alt-8",
            command=f"alt-8:reload(zk-query list --mode notes -i {index_file} --filter-tag {{1}}  --color always)+clear-query",
            description="Reload the list filtering for a given tag.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-9",
            command=f"alt-9:reload(zk-query list --mode unique-tags -i {index_file}  --color always)+clear-query",
            description="Reload the list to show unique tags.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-e",
            command=f"alt-e:reload:(zk-query list --mode notes -i {index_file} --color always --exclude-tag literature_note --exclude-tag person --exclude-tag task --exclude-tag diary --exclude-tag journal)",
            description="Refresh list excluding specific tags.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-b",
            command=f"alt-b:reload(zk-query list --mode notes -i {index_file} --filter-tag literature_note --color always)",
            description="Reload list showing notes with tag 'literature_note'.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-j",
            command=f"alt-j:reload(zk-query list --mode notes -i {index_file} --filter-tag diary --color always)",
            description="Reload list to show notes tagged 'diary'.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-o",
            command=f"alt-o:execute[zk-index run]+reload:(zk-query list --mode orphans -i {index_file} --color always)",
            description="Run index script and show orphan notes.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-O",
            command=f"alt-O:execute[zk-index run]+reload:(zk-query list --mode untagged-orphans -i {index_file} --color always)",
            description="Run index script and show untagged orphan notes.",
            category="Filtering"
        ),
        FzfBinding(
            key="ctrl-s",
            command="ctrl-s:reload(rg -U '^' --sortr modified --field-match-separator='::' --color=always --type md -n | sed 's/\\.md//' )",
            description="Reload list sorted by modification time via ripgrep.",
            category="Filtering"
        ),
        FzfBinding(
            key="alt-c",
            command=f"alt-c:execute[zk-index run]+reload:(zk-query list -i {index_file} --color always -s dateCreated --fields filename --fields title --fields tags --fields dateCreated)",
            description="Sort by creation date",
            category="Filtering"
        ),
    ])
    
    # Editing & Management bindings
    manager.add_bindings([
        FzfBinding(
            key="ctrl-e",
            command=f"ctrl-e:execute[nvim {{+1}}.md ; zk-index run]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            description="Edit note in nvim then reindex and refresh.",
            category="Editing"
        ),
        FzfBinding(
            key="ctrl-alt-r",
            command=f"ctrl-alt-r:execute[rm {{+1}}.md]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            description="Delete the selected note and refresh.",
            category="Editing"
        ),
        FzfBinding(
            key="ctrl-alt-d",
            command=f"ctrl-alt-d:execute[nvim {notes_dir}/{notes_diary_subdir}/$(date '+%Y-%m-%d' -d tomorrow).md;]",
            description="Open tomorrow's diary note in nvim.",
            category="Editing"
        ),
        FzfBinding(
            key="alt-a",
            command=f"alt-a:execute[zk-index run]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            description="Run the index script and refresh the notes list.",
            category="Editing"
        ),
        FzfBinding(
            key="alt-w",
            command=f"alt-w:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --mode notes -i {notes_dir}/index.json --stdin --format-string '- [[{{filename}}|{{title}}]]' --separator='' >> workingMem.md]+reload:(zk-query list --mode notes -i {index_file} --color always)",
            description="Append selected note as a bullet to workingMem.md and refresh.",
            category="Editing"
        ),
        FzfBinding(
            key="alt-y",
            command=f"alt-y:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --mode notes -i {notes_dir}/index.json --stdin --format-string '[[{{filename}}|{{title}}]]' --separator='' > {templink}]+abort",
            description="Save selected note into a temporary file and exit fzf.",
            category="Editing"
        ),
        FzfBinding(
            key="alt-g",
            command=f"alt-g:execute[echo {{+1}} | sed 's/ /\\n/g' | zk-query list --stdin -i {index_file} --format-string '- [[{{filename}}|{{title}}]]' >> {{1}}.md]+clear-selection",
            description="Append formatted selected note to file.",
            category="Editing"
        ),
    ])
    
    # Add help binding
    manager.add_help_binding("zk-fzf")
    
    return manager


def print_hotkeys(fzf_manager: FzfManager) -> None:
    """
    Print a formatted list of hotkeys and their descriptions.
    
    Args:
        fzf_manager: The FzfManager instance with all bindings
    """
    # Define the custom categories and key assignments for printing
    custom_categories = {
        "Navigation & Viewing": ["Enter", "alt-?", "?", "alt-1", "alt-2", "alt-3", "alt-h"],
        "Filtering & Sorting": ["alt-8", "alt-9", "alt-e", "alt-b", "alt-j", "alt-o", "alt-O", "ctrl-s", "alt-c"],
        "Editing & Management": ["ctrl-e", "ctrl-alt-r", "ctrl-alt-d", "alt-a", "alt-w", "alt-y", "alt-g"],
    }
    
    # Override the header with ZK-specific one
    print("\033[1;36m=== ZK-FZF KEYBOARD SHORTCUTS ===\033[0m")
    
    # Use the custom categories for printing
    key_to_binding = {b.key: b for b in fzf_manager.bindings}
    
    for category_name, keys in custom_categories.items():
        print(f"\n\033[1;33m{category_name}:\033[0m")
        for key in keys:
            if key in key_to_binding:
                binding = key_to_binding[key]
                print(f"  \033[1;32m{binding.key:<12}\033[0m : {binding.desc}")
    
    # Print any remaining keys in "Other Commands" category
    all_categorized_keys = []
    for keys in custom_categories.values():
        all_categorized_keys.extend(keys)
    
    other_keys = [k for k in key_to_binding.keys() if k not in all_categorized_keys]
    if other_keys:
        print("\n\033[1;33mOther Commands:\033[0m")
        for key in other_keys:
            binding = key_to_binding[key]
            print(f"  \033[1;32m{binding.key:<12}\033[0m : {binding.desc}")
    
    print("\n\033[1;36mPress q to exit this help screen\033[0m")

def main() -> None:
    """Main entry point for the fzf interface."""
    try:
        import typer
        from typing import Optional as OptionalType
        from enum import Enum
        from pathlib import Path
        
        app = typer.Typer(help="A fuzzy search interface for Zettelkasten notes")
        
        @app.command()
        def search(
            skip_index: bool = typer.Option(
                False,
                "--skip-index",
                help="Skip running the index script before launching fzf"
            ),
            skip_notelist: bool = typer.Option(
                False,
                "--skip-notelist",
                help="Skip launching the background process that updates the notelist"
            ),
            skip_tmux: bool = typer.Option(
                False,
                "--skip-tmux",
                help="Skip sending the link output to tmux"
            ),
            dry_run: bool = typer.Option(
                False,
                "--dry-run",
                help="Print the commands that would be executed, but do not execute them"
            ),
            debug: bool = typer.Option(
                False,
                "--debug", "-d",
                help="Enable debug output"
            ),
            list_hotkeys: bool = typer.Option(
                False,
                "--list-hotkeys",
                help="Print a list of available fzf hotkeys and their functions, then exit"
            ),
            config_file: OptionalType[str] = typer.Option(
                None,
                "--config-file", "-c",
                help="Specify config file path"
            ),
            socket_path: OptionalType[str] = typer.Option(
                None,
                "--socket-path", "-s",
                help="Specify custom Neovim socket path"
            ),
        ) -> None:
            """
            Search and interact with your Zettelkasten notes using a fuzzy finder interface.
            
            This interactive tool provides a powerful way to browse, search, and manage your notes
            with keyboard shortcuts for common operations.
            """
            # Create a class to mimic the argparse namespace for backward compatibility
            class Args:
                pass
                
            args = Args()
            args.skip_index = skip_index
            args.skip_notelist = skip_notelist
            args.skip_tmux = skip_tmux
            args.dry_run = dry_run 
            args.debug = debug
            args.list_hotkeys = list_hotkeys
            args.config_file = config_file
            args.socket_path = socket_path
            
            # Call the run function
            run_fzf_interface(args)
        
        # Run the app if typer is available
        app()
        
    except ImportError:
        # Fall back to argparse if typer is not available
        import argparse
        
        print("WARNING: Typer package not found, falling back to basic argparse implementation.")
        print("For a better CLI experience, install typer: pip install typer")
        
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
        parser.add_argument("--socket-path", help="Specify custom Neovim socket path")
        args = parser.parse_args()
        
        # Call the run function
        run_fzf_interface(args)


def run_fzf_interface(args) -> None:
    """Run the FZF interface with the given arguments."""

    # Load configuration
    config_file = getattr(args, 'config_file', None)
    config = load_config(config_file)
    
    # Get necessary configuration values using utility functions
    notes_dir = get_config_value(config, "notes_dir", DEFAULT_NOTES_DIR)
    notes_dir = resolve_path(notes_dir)
    
    from zk_core.utils import get_index_file_path
    index_file = get_index_file_path(config, notes_dir, args)
    
    # Use entry point script names directly
    py_zk = "zk-query"
    zk_index_script = "zk-index"
    
    notes_diary_subdir = get_config_value(config, "fzf_interface.diary_subdir", "")
    
    bat_theme = get_config_value(config, "fzf_interface.bat_theme", "default")
    
    # Get socket path using utility function for consistent handling
    from zk_core.utils import get_socket_path
    socket_path = get_socket_path(config, args)

    if hasattr(args, 'debug') and args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("NOTES_DIR = %s", notes_dir)
        logger.debug("INDEX_FILE = %s", index_file)
        logger.debug("PY_ZK = %s", py_zk)
        logger.debug("ZK_INDEX_SCRIPT = %s", zk_index_script)
        logger.debug("SOCKET_PATH = %s", socket_path)

    try:
        os.chdir(notes_dir)
    except Exception as e:
        logger.error(f"Unable to change directory to {notes_dir}: {e}")
        sys.exit(1)

    # Create a temporary file for use by one of the keybindings.
    temp = tempfile.NamedTemporaryFile(delete=False)
    templink = temp.name
    temp.close()

    # Build the fzf manager with all bindings
    fzf_manager = build_fzf_manager(
        templink, notes_dir, index_file,
        notes_diary_subdir, bat_theme, socket_path
    )
    
    if hasattr(args, 'list_hotkeys') and args.list_hotkeys:
        print_hotkeys(fzf_manager)
        sys.exit(0)

    # Launch a background process that writes a notelist (unless skipped).
    home = Path.home()
    notelist_path = home / "Documents" / "notelist.md"
    if not hasattr(args, 'skip_notelist') or not args.skip_notelist:
        try:
            if hasattr(args, 'dry_run') and args.dry_run:
                print(f"[DRY-RUN] Would launch: {py_zk} list --mode notes -i {index_file} --format-string '[[{{filename}}|{{title}}]]'")
            else:
                with open(notelist_path, "w") as nl:
                    # Use the entry point name directly
                    subprocess.Popen(["zk-query", "list", "--mode", "notes", "-i", index_file,
                                     "--format-string", "[[{filename}|{title}]]"],
                                     stdout=nl)
                if hasattr(args, 'debug') and args.debug:
                    logger.debug("Launched background notelist update.")
        except Exception as e:
            logger.error(f"Error launching notelist update: {e}")
    else:
        if hasattr(args, 'debug') and args.debug:
            logger.debug("Skipping notelist update (--skip-notelist).")

    # Run the index script (unless skipped)
    if not hasattr(args, 'skip_index') or not args.skip_index:
        if hasattr(args, 'dry_run') and args.dry_run:
            print(f"[DRY-RUN] Would run index script: {zk_index_script} run")
        else:
            try:
                # Use the entry point script name directly
                subprocess.run(["zk-index", "run"], check=False)
                if hasattr(args, 'debug') and args.debug:
                    logger.debug("Ran index script in foreground.")
            except Exception as e:
                logger.error(f"Error running zk-index script: {e}")
    else:
        if hasattr(args, 'debug') and args.debug:
            logger.debug("Skipping index script (--skip-index).")

    # Build the py_zk list command.
    py_zk_list_cmd = ["zk-query", "list", "--mode", "notes", "-i", index_file, "--color", "always"]
    if hasattr(args, 'debug') and args.debug:
        logger.debug("py_zk_list_cmd = %s", " ".join(py_zk_list_cmd))

    # Build the fzf command with additional arguments
    additional_args = [
        "--tiebreak=chunk,begin",
        "--delimiter=::",
        "--scheme=default",
        "--info=right",
        "--ellipsis=",
        "--preview-label=",
        "--multi",
        "--wrap",
    ]
    
    # Add preview command.
    preview_cmd = f"echo \"Backlinks:\"; zk-query list -i {index_file} --filter-outgoing-link {{1}} --color always; bat --theme=\"{bat_theme}\" --color=always --decorations=never {{1}}.md -H {{2}} 2> /dev/null || bat {{1}}.md"
    additional_args.extend([
        "--preview", preview_cmd,
        "--preview-window", "wrap:50%:<50(up)",
        "--ansi"
    ])
    
    # Get complete fzf arguments
    fzf_args = fzf_manager.get_fzf_args(additional_args)
                    
    if hasattr(args, 'debug') and args.debug:
        logger.debug("fzf args: %s", " ".join(fzf_args))

    if hasattr(args, 'dry_run') and args.dry_run:
        print(f"[DRY-RUN] Would run: {' '.join(py_zk_list_cmd)} | {' '.join(fzf_args)}")
        sys.exit(0)
        
    # Run fzf with the py_zk list output piped into it.
    try:
        p1 = subprocess.Popen(py_zk_list_cmd, stdout=subprocess.PIPE)
        fzf_result = subprocess.run(fzf_args, stdin=p1.stdout)
        p1.stdout.close()
        if hasattr(args, 'debug') and args.debug:
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
                if hasattr(args, 'skip_tmux') and args.skip_tmux:
                    if hasattr(args, 'debug') and args.debug:
                        logger.debug(f"Skipping sending tmux; link: {link}")
                else:
                    subprocess.run(["tmux", "send-keys", "-l", link])
                    if hasattr(args, 'debug') and args.debug:
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
