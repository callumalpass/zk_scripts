"""
Working memory note management module.

This module provides functionality for:
  - Creating a new note by capturing content
  - Adding YAML frontmatter (title, date, tags)
  - Appending a wikilink to the working memory file
  - Integration with Neovim for direct editing
"""

import argparse
import os
import sys
import subprocess
import tempfile
import random
import string
import datetime
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import yaml

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.utils import extract_frontmatter_and_body, generate_filename
from zk_core.query import app as query_app

# ANSI Colors for UI
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
NC = "\033[0m"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Default values
DEFAULT_LLM_INSTRUCT = (
    "Print five oneline potential headers for this zettel.\n"
    "The titles should be appropriate to aid in retrieval of the note at a later point in time.\n"
    "They should, therefore, prioritize searchability over colour, with some effort made to incorporate key terms, names, etc.\n"
    "Avoid referring to other files.\n"
    "Use sentence case.\n"
    "Prefer 'atomic' titles (i.e. the title should be a statement), and don't be afraid of somewhat lengthy titles.\n"
    "Do not markup your response in any way."
)

def eprint(msg: str) -> None:
    """Print to stderr."""
    print(msg, file=sys.stderr)

def check_directories(notes_dir: str, working_mem_file: str) -> None:
    """Ensure necessary directories and files exist."""
    if not os.path.isdir(notes_dir):
        eprint(f"{RED}Error:{NC} Notes directory '{notes_dir}' does not exist or is not a directory.")
        sys.exit(1)
    if not os.path.isfile(working_mem_file):
        print(f"{YELLOW}Working memory file '{working_mem_file}' does not exist. Creating it.{NC}")
        try:
            with open(working_mem_file, "w", encoding="utf-8"):
                pass
        except Exception as e:
            eprint(f"{RED}Error:{NC} Failed to create working memory file '{working_mem_file}'. {e}")
            sys.exit(1)

def get_temp_file(notes_dir: str) -> tempfile.NamedTemporaryFile:
    """Create a temporary file for editing."""
    tmp_dir = os.path.join(notes_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    temp = tempfile.NamedTemporaryFile(
        mode="w+", suffix=".md", prefix="note_", 
        dir=tmp_dir, delete=False, encoding="utf-8"
    )
    return temp

def call_editor(editor: str, filepath: str) -> None:
    """Launch the editor to edit a file."""
    try:
        subprocess.run([editor, filepath], check=True)
    except subprocess.CalledProcessError:
        eprint(f"{RED}Error:{NC} Editor '{CYAN}{editor}{RED}' exited with an error.")
        sys.exit(1)

def quick_capture(temp_file_path: str, working_mem_file: str) -> None:
    """Append content directly to working memory file."""
    print(f"{CYAN}Quick capture mode:{NC} Appending content directly to working memory file...")
    try:
        with open(temp_file_path, "r", encoding="utf-8") as tf, open(working_mem_file, "a", encoding="utf-8") as wm:
            wm.write("\n")
            wm.write(tf.read())
    except Exception as e:
        eprint(f"{RED}Error:{NC} Failed to append content to working memory file '{working_mem_file}'. {e}")
        sys.exit(1)
    print(f"{GREEN}Content appended to '{CYAN}{working_mem_file}{GREEN}'.{NC}")
    print(f"{GREEN}Done (quick capture).{NC}")
    sys.exit(0)

def run_fzf(input_text: str, header: str = "", prompt: str = "", 
            multi: bool = False, preview_cmd: Optional[str] = None, 
            extra_args: Optional[List[str]] = None) -> str:
    """Run fzf with the given parameters."""
    args = ["fzf", "--ansi"]
    if header:
        args.extend(["--header", header])
    if prompt:
        args.extend(["--prompt", prompt])
    if multi:
        args.append("--multi")
    if preview_cmd:
        args.extend(["--preview", preview_cmd])
    if extra_args:
        args.extend(extra_args)
    try:
        proc = subprocess.run(args, input=input_text, text=True, capture_output=True)
    except Exception as e:
        eprint(f"{RED}Error running fzf:{NC} {e}")
        return ""
    return proc.stdout.strip()

def generate_random_suffix(n: int = 3) -> str:
    """Generate a random string of lowercase letters.
    
    Note: This function is kept for backward compatibility.
    New code should use utils.generate_random_string() instead.
    """
    from zk_core.utils import generate_random_string
    return generate_random_string(n)

def get_title_from_llm_content(note_content: str, llm_cmd: str, llm_instruct: str) -> Optional[str]:
    """Generate title suggestions using an LLM."""
    try:
        llm_args = llm_cmd.split() + ["-s", llm_instruct]
        proc = subprocess.run(llm_args, input=note_content, text=True, capture_output=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            print(f"{YELLOW}Warning:{NC} LLM command failed to generate a title.")
            return None
        return proc.stdout.strip()
    except Exception as e:
        eprint(f"{RED}Error calling LLM command:{NC} {e}")
        return None

def choose_title(note_content: str, llm_cmd: str, llm_instruct: str, editor: str) -> Tuple[str, bool, bool]:
    """Choose a title for the note, returning the title and flags for scratch/journal mode."""
    scratch_mode = False
    journal_mode = False
    title = ""
    
    while True:
        menu = (
            f"{CYAN}1: Generate title with LLM and edit{NC}\n"
            f"{CYAN}2: Enter title manually{NC}\n"
            f"{CYAN}3: Use 'Scratch' title (quick note){NC}\n"
            f"{CYAN}4: Use 'Journal' title (simple date){NC}\n"
            f"{CYAN}q: Quit without saving{NC}\n"
        )
        
        # Use bat if available for previewing the note content
        preview_cmd = None
        temp_prev = None
        if subprocess.run(["which", "bat"], capture_output=True).returncode == 0:
            # Create a temp file for preview
            temp_prev = tempfile.NamedTemporaryFile(
                mode="w+", suffix=".md", prefix="preview_", delete=False, encoding="utf-8"
            )
            temp_prev.write(note_content)
            temp_prev.flush()
            preview_cmd = f"bat {temp_prev.name}"
            
        choice = run_fzf(
            menu, 
            header="Choose title option:", 
            prompt="Title option: ", 
            extra_args=["--height=8", "--bind", "one:accept"], 
            preview_cmd=preview_cmd
        )
        
        # Clean up preview temp file if created
        if temp_prev:
            try:
                os.unlink(temp_prev.name)
            except Exception:
                pass
                
        choice = choice.split(":")[0].strip()
        
        if choice == "1" or choice == "":
            # Generate with LLM
            print("Generating title with LLM...")
            llm_result = get_title_from_llm_content(note_content, llm_cmd, llm_instruct)
            if not llm_result:
                continue
                
            # Write LLM suggestions to temp file for editing
            title_temp = get_temp_file(os.path.dirname(tempfile.gettempdir()))
            title_temp_path = title_temp.name
            title_temp.write(llm_result)
            title_temp.close()
            
            print(f"Opening editor to edit LLM titles... {CYAN}{editor}{NC}")
            call_editor(editor, title_temp_path)
            
            try:
                with open(title_temp_path, "r", encoding="utf-8") as f:
                    edited_titles = f.readlines()
                title_choice = ""
                for line in edited_titles:
                    line = line.strip()
                    if line:
                        title_choice = line
                        break
                if not title_choice:
                    print(f"{YELLOW}Warning:{NC} No title chosen from editor. Please try again.")
                    os.unlink(title_temp_path)
                    continue
                print(f"Selected title (edited): '{GREEN}{title_choice}{NC}'")
                title = title_choice
                os.unlink(title_temp_path)
                break
            except Exception as e:
                eprint(f"{RED}Error processing edited titles:{NC} {e}")
                os.unlink(title_temp_path)
                continue
                
        elif choice == "2":
            # Manual title entry
            title = input("Enter your title: ").strip()
            break
            
        elif choice == "3":
            # Scratch note
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"Scratch {now}"
            scratch_mode = True
            break
            
        elif choice == "4":
            # Journal note
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            title = now
            journal_mode = True
            break
            
        elif choice.lower() == "q":
            print("Quitting without saving notes.")
            sys.exit(0)
            
        else:
            print(f"{RED}Invalid choice.{NC} Please enter 1, 2, 3, 4 or q.")
            
    return title, scratch_mode, journal_mode

def select_tags(index_file: str) -> List[str]:
    """Select tags for the note using fzf."""
    tags = []
    try:
        # Use the query module directly
        from typer.testing import CliRunner
        runner = CliRunner()
        query_args = [
            "list",
            "--mode", "unique-tags",
            "-i", index_file
        ]
        result = runner.invoke(query_app, query_args)
        valid_tags = result.stdout.strip() if result.exit_code == 0 else ""
    except Exception as e:
        print(f"{YELLOW}Warning:{NC} Could not fetch valid tags. {e}")
        valid_tags = ""
        
    if valid_tags:
        selection = run_fzf(
            valid_tags, 
            header="Select tags (TAB/SPACE for multiple, ENTER when done):",
            prompt="System Tags: ", 
            multi=True, 
            extra_args=["--height=28"]
        )
        if selection:
            system_tags = [tag.strip() for tag in selection.splitlines() if tag.strip()]
            tags.extend(system_tags)
    else:
        print(f"{YELLOW}Warning:{NC} No valid tags")
        
    return tags

def write_note_file(filepath: str, title: str, note_content: str) -> None:
    """Write the note to a file with YAML frontmatter."""
    now = datetime.datetime.now()
    datetime_iso = now.strftime("%Y-%m-%dT%H:%M:%S")
    filepath = Path(filepath)
    old_yaml = {}
    body = note_content

    # If file exists, try to preserve existing frontmatter
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Extract existing frontmatter
            meta, extracted_body = extract_frontmatter_and_body(content)
            old_yaml = meta
            body = extracted_body
        except Exception as e:
            eprint(f"{RED}Error:{NC} Failed to read existing note file '{filepath}'. {e}")

    # Determine dateCreated: preserve if present; otherwise use file creation time
    if "dateCreated" in old_yaml:
        date_created_iso = old_yaml["dateCreated"]
    else:
        try:
            # Try to get creation time (stat birthtime if available, otherwise ctime)
            stat = filepath.stat() if filepath.exists() else None
            if hasattr(stat, 'st_birthtime'):
                creation_timestamp = stat.st_birthtime
            else:
                creation_timestamp = os.path.getctime(str(filepath)) if filepath.exists() else now.timestamp()
            date_created_iso = datetime.datetime.fromtimestamp(creation_timestamp).strftime("%Y-%m-%dT%H:%M:%S")
        except Exception as e:
            logger.warning(f"Error getting file creation time: {e}. Using current time.")
            date_created_iso = datetime_iso

    # Merge metadata
    merged_yaml = old_yaml.copy()  # preserve any preexisting fields
    merged_yaml.update({
        "title": title,
        "dateCreated": date_created_iso,
        "dateModified": datetime_iso,
        "tags": old_yaml.get("tags", []),  # tags will be updated later
        "zettelid": filepath.stem,
    })
    
    updated_yaml_str = yaml.dump(merged_yaml, indent=2, sort_keys=False)
    frontmatter = f"---\n{updated_yaml_str}---\n\n# {title}\n\n"
    
    # Write the new file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.write(body)
    except Exception as e:
        eprint(f"{RED}Error:{NC} Failed to write note file '{filepath}'. {e}")
        sys.exit(1)

def add_tags_to_note(filepath: str, tags: List[str]) -> None:
    """Add tags to a note's YAML frontmatter."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract frontmatter and body
        meta, body = extract_frontmatter_and_body(content)
        
        # Update tags
        current_tags = meta.get('tags', [])
        if not isinstance(current_tags, list):
            current_tags = []
        
        updated_tags = list(set(current_tags + tags))
        meta['tags'] = updated_tags
        
        # Write updated frontmatter and content
        updated_yaml_str = yaml.dump(meta, indent=2, sort_keys=False)
        new_content = f"---\n{updated_yaml_str}---\n\n{body}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
    except Exception as e:
        eprint(f"{RED}Error:{NC} Failed to update tags in note file '{filepath}'. {e}")

def append_working_mem_link(working_mem_file: str, zettelid_base: str, title: str) -> None:
    """Append a wikilink to the working memory file."""
    link_text = f"- [[{zettelid_base}|{title}]]\n"
    try:
        with open(working_mem_file, "a", encoding="utf-8") as f:
            f.write("\n" + link_text)
    except Exception as e:
        eprint(f"{RED}Error:{NC} Failed to append link to working memory file '{working_mem_file}'. {e}")
        sys.exit(1)

def clear_screen() -> None:
    """Clear the terminal screen."""
    print("\033[H\033[J", end="")

def get_nvim_buffer_content(socket_path: str) -> Tuple[str, str, Any]:
    """Get the content and filename of the current Neovim buffer."""
    try:
        import pynvim
    except ImportError:
        eprint(f"{RED}Error:{NC} pynvim is required in --nvim mode. Please install it (pip install pynvim).")
        sys.exit(1)
        
    try:
        nvim = pynvim.attach('socket', path=socket_path)
        buf = nvim.current.buffer
        content = "\n".join(buf[:])
        file_name = buf.name
        return content, file_name, nvim
    except Exception as e:
        eprint(f"{RED}Error connecting to Neovim:{NC} {e}")
        sys.exit(1)

def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Create a new note and update working memory.")
    parser.add_argument("-s", action="store_true", help="Scratch mode")
    parser.add_argument("-q", action="store_true", help="Quick capture mode")
    parser.add_argument("--nvim", action="store_true", help="Use current Neovim buffer (no temporary file)")
    parser.add_argument("--config-file", help="Path to configuration file")
    args = parser.parse_args()

    scratch_mode_flag = args.s
    quick_capture_mode = args.q
    use_nvim = args.nvim
    journal_mode = False

    # Load configuration
    config = load_config(args.config_file)
    
    # Get configuration values
    notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    notes_dir = resolve_path(notes_dir)
    
    working_mem_file = get_config_value(config, "working_mem.file", os.path.join(notes_dir, "workingMem.md"))
    working_mem_file = resolve_path(working_mem_file)
    
    editor = os.getenv("EDITOR_CMD", "nvim")
    
    llm_path = get_config_value(config, "bibview.llm_path", os.path.join(os.path.expanduser("~/mybin"), "swllm"))
    llm_path = resolve_path(llm_path)
    
    llm_cmd = os.getenv("LLM_CMD", f"{llm_path} -m gemini-2.0-flash")
    llm_instruct = os.getenv("LLM_INSTRUCT", DEFAULT_LLM_INSTRUCT)
    
    # Get index file path
    index_file = get_config_value(config, "zk_fzf.index_file", os.path.join(notes_dir, "index.json"))
    index_file = resolve_path(index_file)
    
    # Get filename configuration
    filename_format = get_config_value(config, "filename.format", None)
    filename_extension = get_config_value(config, "filename.extension", None)
    
    # Make sure directories exist
    check_directories(notes_dir, working_mem_file)

    if use_nvim:
        # Use the current Neovim buffer
        socket_path = os.getenv("NVIM_SOCKET", "/tmp/obsidian.sock")
        note_content, file_name, nvim_instance = get_nvim_buffer_content(socket_path)
        if not note_content.strip():
            print(f"{YELLOW}No content in the current Neovim buffer. Nothing to save.{NC}")
            sys.exit(0)
        if not file_name:
            eprint(f"{RED}Error:{NC} Current buffer is not associated with a file.")
            sys.exit(1)
        note_path = Path(file_name)
        print(f"{GREEN}Using Neovim buffer from file: {CYAN}{file_name}{NC}")
    else:
        # Normal mode: open a temporary file in the editor
        temp = get_temp_file(notes_dir)
        temp_file_path = temp.name
        temp.close()
        print(f"{GREEN}Opening editor ({CYAN}{editor}{GREEN})...{NC}")
        call_editor(editor, temp_file_path)
        try:
            with open(temp_file_path, "r", encoding="utf-8") as tf:
                note_content = tf.read()
        except Exception as e:
            eprint(f"{RED}Error reading temporary file:{NC} {e}")
            os.unlink(temp_file_path)
            sys.exit(1)
        if not note_content.strip():
            print(f"{YELLOW}No notes written. Temporary file was empty. Nothing saved.{NC}")
            os.unlink(temp_file_path)
            sys.exit(0)

    if quick_capture_mode and (not use_nvim):
        quick_capture(temp_file_path, working_mem_file)

    # Display the captured note for preview
    clear_screen()
    print("\n" + note_content + "\n" + "-" * 40 + "\n")

    title, chosen_scratch, chosen_journal = ("", False, False)
    if not scratch_mode_flag:
        title, chosen_scratch, chosen_journal = choose_title(note_content, llm_cmd, llm_instruct, editor)
    if scratch_mode_flag or chosen_scratch:
        title = f"Scratch {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        scratch_mode_flag = True
    if chosen_journal:
        journal_mode = True

    clear_screen()
    print(f"Using title: '{CYAN}{title}{NC}'")
    selected_tags = select_tags(index_file)

    if use_nvim:
        # In nvim mode we use the file already open in nvim
        note_path = Path(file_name)
    else:
        # Create a new note file with configured filename format
        generated_filename = generate_filename(filename_format, filename_extension)
        note_path = Path(notes_dir) / generated_filename

    # Write the note file with YAML frontmatter
    write_note_file(str(note_path), title, note_content)
    print(f"{GREEN}Note created/updated successfully: '{CYAN}{note_path}{GREEN}'.{NC}")

    if not use_nvim:
        os.unlink(temp_file_path)

    # Add tags
    if selected_tags:
        add_tags_to_note(str(note_path), selected_tags)
    if scratch_mode_flag:
        add_tags_to_note(str(note_path), ["scratch"])
    if journal_mode:
        add_tags_to_note(str(note_path), ["journal"])

    # Append a wikilink to the working memory file
    append_working_mem_link(working_mem_file, note_path.stem, title)
    print(f"{GREEN}Link appended to working memory file.{NC}")
    print(f"{GREEN}Done.{NC}")

    # In nvim mode, trigger a buffer reload
    if use_nvim:
        try:
            nvim_instance.command("edit!")
        except Exception as e:
            eprint(f"{YELLOW}Warning:{NC} Could not force a reload in Neovim: {e}")

if __name__ == "__main__":
    main()