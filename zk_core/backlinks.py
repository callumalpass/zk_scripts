"""
Backlinks and Note Viewer module.

This module provides a terminal UI for viewing:
- Backlinks: notes that reference the current note
- Similar notes: computed using embeddings
- Outgoing links: notes that the current note links to

Features:
- Connects to Neovim over a socket
- Toggleable views for different types of links
- Content preview with scrolling
- Send wikilinks to the working memory file
"""

import os
import sys
import time
import curses
import logging
import json
import textwrap
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union

import numpy as np
import pynvim

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.models import Note
from zk_core.utils import load_json_file
from zk_core.constants import DEFAULT_NVIM_SOCKET

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Preview:
    """Content preview with scrolling."""
    
    def __init__(self, filepath: str) -> None:
        """Initialize with filepath to preview."""
        self.filepath = filepath
        self.file_lines = []
        self.total_lines = 0
        self.view_start = 0  # current top line in the preview window
        try:
            with open(self.filepath, "r", encoding="utf8") as f:
                self.file_lines = [line.rstrip() for line in f.readlines()]
            self.total_lines = len(self.file_lines)
        except Exception as e:
            logger.exception(f"Failed to load preview for {self.filepath}: {e}")
            self.file_lines = [f"Error loading preview: {e}"]
            self.total_lines = 1

    def get_visible_lines(self, height: int, width: int) -> Tuple[List[str], float]:
        """Get lines to display in the preview window."""
        output = []
        # Determine scroll percentage for UI feedback.
        scroll_pct = (self.view_start / max(1, self.total_lines - height)) * 100 if self.total_lines > height else 0
        
        # Show up indicator if scrolled down
        if self.view_start > 0:
            output.append("^^^")
            
        # Get visible lines
        for idx in range(self.view_start, min(self.view_start + height, self.total_lines)):
            line = self.file_lines[idx]
            wrapped = textwrap.wrap(line, width=width-2) if line else [""]
            for wline in wrapped:
                output.append(" " + wline)
                
        # Show down indicator if more content below
        if self.view_start + height < self.total_lines:
            output.append("vvv")
            
        return output, scroll_pct

    def scroll_down(self, amount: int, height: int) -> None:
        """Scroll the preview down by the given amount."""
        max_start = max(0, self.total_lines - height)
        self.view_start = min(self.view_start + amount, max_start)

    def scroll_up(self, amount: int) -> None:
        """Scroll the preview up by the given amount."""
        self.view_start = max(0, self.view_start - amount)

    def jump_to_top(self) -> None:
        """Jump to the top of the preview."""
        self.view_start = 0

    def jump_to_bottom(self, height: int) -> None:
        """Jump to the bottom of the preview."""
        self.view_start = max(0, self.total_lines - height)


class UIState:
    """Maintains the state of the UI."""
    
    def __init__(self) -> None:
        """Initialize UI state."""
        self.selected_idx = 0
        self.list_offset = 0
        self.current_file = ""
        self.all_notes: List[Note] = []
        self.current_note: Optional[Note] = None
        self.list_mode = "backlinks"  # "backlinks", "similar", "outgoing"
        self.displayed_list: List[str] = []
        self.status_message = ""
        self.status_message_time = 0
        self.preview_cache: Dict[str, Preview] = {}
        self.show_help = False

    def set_status(self, msg: str) -> None:
        """Set status message with timestamp."""
        self.status_message = msg
        self.status_message_time = time.time()

    def reset_selection(self) -> None:
        """Reset selection to top of list."""
        self.selected_idx = 0
        self.list_offset = 0


def get_note_body(note: Note) -> str:
    """Get the body text of a note."""
    if hasattr(note, 'body') and note.body:
        return note.body
    filepath = os.path.join(os.getenv("NOTES_DIR", ""), note.filename + ".md")
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf8") as f:
                return f.read()
        except Exception as e:
            logger.exception(f"Failed to read file {filepath} for body: {e}")
    return ""


def format_note_item(note: Note) -> str:
    """Format a note for display in the list."""
    return f"{note.filename}: {note.title or '(No Title)'}"


def get_relative_note_name(full_path: str, notes_dir: str) -> str:
    """Get the relative note name from a full path."""
    abs_notes = os.path.abspath(notes_dir)
    abs_file = os.path.abspath(full_path)
    try:
        rel = os.path.relpath(abs_file, abs_notes)
        if rel.lower().endswith(".md"):
            rel = rel[:-3]
        return rel
    except Exception as e:
        logger.exception(f"Error computing relative note name: {e}")
        return full_path


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    return vec / (np.linalg.norm(vec) + 1e-10)


def compute_similar_notes(current_note: Note, all_notes: List[Note], 
                          embeddings_path: Path, k: int) -> List[str]:
    """Compute similar notes based on embeddings."""
    try:
        with embeddings_path.open("r", encoding="utf-8") as f:
            embeddings_map = json.load(f)
    except Exception as e:
        logger.exception(f"Error loading embeddings from {embeddings_path}: {e}")
        return [f"Error loading embeddings: {e}"]

    query_id = current_note.filename
    if query_id in embeddings_map:
        query_embedding = embeddings_map[query_id]
    else:
        return ["No embedding found for current note."]

    available_ids = []
    embeddings_list = []
    for note in all_notes:
        nid = note.filename
        if nid in embeddings_map and nid != query_id:
            available_ids.append(nid)
            embeddings_list.append(embeddings_map[nid])
    
    if not embeddings_list:
        return ["No other notes with embeddings available."]

    # Compute similarities
    query_vector = normalize_vector(np.array(query_embedding, dtype="float32"))
    sim_list = []
    for embed in embeddings_list:
        vec = normalize_vector(np.array(embed, dtype="float32"))
        sim = float(np.dot(query_vector, vec))
        sim_list.append(sim)

    # Sort by similarity and format results
    sim_indices = sorted(range(len(sim_list)), key=lambda i: sim_list[i], reverse=True)
    results = []
    for idx in sim_indices:
        if len(results) >= k:
            break
        note = next((n for n in all_notes if n.filename == available_ids[idx]), None)
        if note:
            results.append(format_note_item(note) + f" (score: {sim_list[idx]:.4f})")
    
    if not results:
        return ["No similar notes found."]
    
    return results


def send_wikilink(ui: UIState, working_mem_file: str) -> None:
    """Send a wikilink for the currently selected note to working memory file."""
    if not ui.displayed_list or ui.selected_idx >= len(ui.displayed_list):
        ui.set_status("No note selected to send wikilink.")
        return
        
    entry = ui.displayed_list[ui.selected_idx]
    note_key = entry.split(":", 1)[0].strip()
    
    # Get note title
    note_title = ""
    for note in ui.all_notes:
        if note.filename == note_key:
            note_title = note.title or note_key
            break
    
    wikilink = f"- [[{note_key}|{note_title}]]\n"
    
    try:
        with open(working_mem_file, "a", encoding="utf8") as f:
            f.write(wikilink)
        ui.set_status(f"Wikilink for {note_key} added to workingMem.md")
    except Exception as e:
        logger.exception(f"Error writing wikilink to {working_mem_file}: {e}")
        ui.set_status(f"Error writing wikilink: {e}")


def draw_help_overlay(stdscr, height: int, width: int) -> None:
    """Draw a help overlay with key bindings."""
    # Create a full-screen overlay with a reverse-video attribute
    overlay_win = curses.newwin(height, width, 0, 0)
    overlay_win.bkgd(' ', curses.color_pair(3))
    overlay_text = [
        "Help - Available Commands:",
        "",
        "↑ / k         : Move selection up",
        "↓ / j         : Move selection down",
        "PgUp / PgDn   : Scroll preview",
        "h             : Jump preview to top",
        "Enter         : Open selected note in Neovim",
        "l             : Toggle list mode (backlinks, similar, outgoing)",
        "g             : Send wikilink for selected note to workingMem.md",
        "?             : Toggle this help overlay",
        "q             : Quit",
        "",
        "Press any key to dismiss help..."
    ]
    
    # Calculate vertical starting point
    start_y = (height - len(overlay_text)) // 2
    for idx, line in enumerate(overlay_text):
        x = 2  # Indent from left
        try:
            overlay_win.addstr(start_y+idx, x, line, curses.A_BOLD)
        except curses.error:
            pass
    overlay_win.refresh()
    overlay_win.getch()  # Wait for any key press


def draw_ui(stdscr, ui: UIState, notes_dir: str) -> None:
    """Draw the user interface."""
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    min_height = 15
    min_width = 40
    
    if height < min_height or width < min_width:
        msg = f"Terminal too small! (min: {min_width}x{min_height}, current: {width}x{height})"
        stdscr.addnstr(0, 0, msg, width, curses.A_BOLD)
        stdscr.refresh()
        return

    # --- Header ---
    header_height = 3
    try:
        header_win = stdscr.subwin(header_height, width, 0, 0)
    except curses.error:
        header_win = stdscr
    header_win.erase()
    header_win.border()
    header_text = f"File: {os.path.basename(ui.current_file) if ui.current_file else 'None'} | Mode: {ui.list_mode.upper()}"
    header_hint = " | Press '?' for help"
    header_win.addnstr(1, 2, header_text + header_hint, width-4, curses.color_pair(1) | curses.A_BOLD)

    # --- Main Area: List & Preview ---
    main_area_y = header_height
    status_bar_height = 3
    main_area_height = height - main_area_y - status_bar_height

    if width >= 100:
        # Vertical split: notes list on the left, preview on the right.
        list_panel_width = int(width * 0.3)
        preview_panel_width = width - list_panel_width
        
        # List Panel:
        try:
            list_win = stdscr.subwin(main_area_height, list_panel_width, main_area_y, 0)
        except curses.error:
            list_win = stdscr
        list_win.erase()
        list_win.border()
        list_win.addnstr(0, 2, " Notes List ", list_panel_width-4, curses.A_BOLD | curses.color_pair(1))
        visible_lines = main_area_height - 2
        for i in range(visible_lines):
            idx = ui.list_offset + i
            if idx >= len(ui.displayed_list):
                break
            line = ui.displayed_list[idx]
            attr = curses.A_NORMAL
            if idx == ui.selected_idx:
                attr = curses.color_pair(2) | curses.A_BOLD
            try:
                list_win.addnstr(i+1, 2, line, list_panel_width-4, attr)
            except curses.error:
                pass

        # Preview Panel:
        try:
            preview_win = stdscr.subwin(main_area_height, preview_panel_width, main_area_y, list_panel_width)
        except curses.error:
            preview_win = stdscr
        preview_win.erase()
        preview_win.border()
        preview_win.addnstr(0, 2, " Preview ", preview_panel_width-4, curses.A_BOLD | curses.color_pair(1))
        scroll_pct = None
        
        if ui.displayed_list and ui.selected_idx < len(ui.displayed_list):
            entry = ui.displayed_list[ui.selected_idx]
            note_key = entry.split(":", 1)[0].strip()
            filepath = os.path.join(notes_dir, note_key + ".md")
            
            if note_key in ui.preview_cache:
                preview = ui.preview_cache[note_key]
            else:
                preview = Preview(filepath)
                ui.preview_cache[note_key] = preview
                
            vis_lines, scroll_pct = preview.get_visible_lines(main_area_height-2, preview_panel_width-4)
            for i, line in enumerate(vis_lines[:main_area_height-2]):
                try:
                    preview_win.addnstr(i+1, 2, line, preview_panel_width-4)
                except curses.error:
                    pass
                    
        # Display scroll percentage in the lower right corner
        if scroll_pct is not None:
            perc_text = f"{scroll_pct:3.0f}%"
            try:
                preview_win.addnstr(main_area_height-1, preview_panel_width-len(perc_text)-2, 
                                   perc_text, len(perc_text), curses.A_DIM)
            except curses.error:
                pass
    else:
        # Stacked layout for smaller terminals
        half_area = (main_area_height - 2) // 2
        
        # List window (top)
        try:
            list_win = stdscr.subwin(half_area + 2, width, main_area_y, 0)
        except curses.error:
            list_win = stdscr
        list_win.erase()
        list_win.border()
        list_win.addnstr(0, 2, " Notes List ", width-4, curses.A_BOLD | curses.color_pair(1))
        visible_lines = half_area - 1
        for i in range(visible_lines):
            idx = ui.list_offset + i
            if idx >= len(ui.displayed_list):
                break
            line = ui.displayed_list[idx]
            attr = curses.A_NORMAL
            if idx == ui.selected_idx:
                attr = curses.color_pair(2) | curses.A_BOLD
            try:
                list_win.addnstr(i+1, 2, line, width-4, attr)
            except curses.error:
                pass

        # Preview window (bottom)
        try:
            preview_win = stdscr.subwin(half_area + 2, width, main_area_y + half_area + 2, 0)
        except curses.error:
            preview_win = stdscr
        preview_win.erase()
        preview_win.border()
        preview_win.addnstr(0, 2, " Preview ", width-4, curses.A_BOLD | curses.color_pair(1))
        scroll_pct = None
        
        if ui.displayed_list and ui.selected_idx < len(ui.displayed_list):
            entry = ui.displayed_list[ui.selected_idx]
            note_key = entry.split(":", 1)[0].strip()
            filepath = os.path.join(notes_dir, note_key + ".md")
            
            if note_key in ui.preview_cache:
                preview = ui.preview_cache[note_key]
            else:
                preview = Preview(filepath)
                ui.preview_cache[note_key] = preview
                
            vis_lines, scroll_pct = preview.get_visible_lines(half_area - 1, width-4)
            for i, line in enumerate(vis_lines[:half_area - 1]):
                try:
                    preview_win.addnstr(i+1, 2, line, width-4)
                except curses.error:
                    pass
                    
        if scroll_pct is not None:
            perc_text = f"{scroll_pct:3.0f}%"
            try:
                preview_win.addnstr(half_area, width-len(perc_text)-2, perc_text, len(perc_text), curses.A_DIM)
            except curses.error:
                pass

    # --- Status Bar ---
    try:
        status_win = stdscr.subwin(status_bar_height, width, height - status_bar_height, 0)
    except curses.error:
        status_win = stdscr
    status_win.erase()
    status_win.border()
    
    help_text = ("Commands: ↑/k, ↓/j: Move  |  PgUp/PgDn: Scroll Preview  |  l: Toggle Mode  | "
                "Enter: Open  |  g: Wikilink  |  h: Top  |  q: Quit")
                
    # Use status message if within timeout, otherwise show default help
    status_timeout = 3  # seconds
    disp_status = ui.status_message if (time.time() - ui.status_message_time < status_timeout) else help_text
    
    status_win.addnstr(1, 2, disp_status, width-4, curses.A_BOLD | curses.color_pair(3))
    stdscr.refresh()


def update_displayed_list(ui: UIState, embeddings_path: Path, notes_dir: str) -> None:
    """Update the displayed list based on the current mode."""
    if ui.current_note is None:
        ui.displayed_list = ["No note found in index."]
        return
        
    if ui.list_mode == "backlinks":
        if ui.current_note.backlinks:
            disp = []
            for bk in ui.current_note.backlinks:
                note_obj = next((n for n in ui.all_notes if n.filename == bk), None)
                if note_obj:
                    disp.append(format_note_item(note_obj))
                else:
                    disp.append(bk + ": (Not found)")
            ui.displayed_list = disp
        else:
            ui.displayed_list = ["No backlinks found."]
            
    elif ui.list_mode == "outgoing":
        if ui.current_note.outgoing_links:
            disp = []
            for ot in ui.current_note.outgoing_links:
                note_obj = next((n for n in ui.all_notes if n.filename == ot), None)
                if note_obj:
                    disp.append(format_note_item(note_obj))
                else:
                    disp.append(ot + ": (Not found)")
            ui.displayed_list = disp
        else:
            ui.displayed_list = ["No outgoing links."]
            
    elif ui.list_mode == "similar":
        k = 30  # Number of similar notes to show
        similar = compute_similar_notes(ui.current_note, ui.all_notes, embeddings_path, k)
        ui.displayed_list = similar
        
    # Clear preview cache when the list changes
    ui.preview_cache = {}


def get_current_file(nvim) -> str:
    """Get the current file path from Neovim."""
    try:
        full = nvim.eval("expand('%:p')")
        return full
    except Exception as e:
        logger.exception(f"Error getting current file: {e}")
        return ""


def update_ui_state(nvim, ui: UIState, index_path: Path, embeddings_path: Path, notes_dir: str) -> None:
    """Update UI state based on the current Neovim file."""
    current_file = get_current_file(nvim)
    
    # If the file has changed, update the note data
    if current_file != ui.current_file:
        ui.current_file = current_file
        try:
            # Load all notes from the index
            with open(index_path, 'r') as f:
                notes_data = json.load(f)
            ui.all_notes = [Note.from_dict(item) for item in notes_data]
        except Exception as e:
            ui.set_status(f"Error loading index: {e}")
            ui.all_notes = []
            
        # Find the current note in the index
        note = None
        relative_path = get_relative_note_name(current_file, notes_dir)
        
        for n in ui.all_notes:
            fname = n.filename
            if fname.lower().endswith(".md"):
                fname = fname[:-3]
            if fname == relative_path:
                note = n
                break
                
        ui.current_note = note
        ui.reset_selection()
        update_displayed_list(ui, embeddings_path, notes_dir)


def handle_input(stdscr, nvim, ui: UIState, index_path: Path, embeddings_path: Path, 
                working_mem_file: str, notes_dir: str) -> bool:
    """Handle user input, return True to continue or False to exit."""
    key = stdscr.getch()
    
    # If no key was pressed (timeout)
    if key == -1:
        return True

    # Toggle help overlay
    if key == ord("?"):
        ui.show_help = True
        draw_help_overlay(stdscr, *stdscr.getmaxyx())
        ui.show_help = False
        return True

    # Toggle list mode
    if key in (ord("l"), ord("L")):
        if ui.list_mode == "backlinks":
            ui.list_mode = "similar"
            ui.set_status("Switched to SIMILAR")
        elif ui.list_mode == "similar":
            ui.list_mode = "outgoing"
            ui.set_status("Switched to OUTGOING")
        else:
            ui.list_mode = "backlinks"
            ui.set_status("Switched to BACKLINKS")
        ui.reset_selection()
        update_displayed_list(ui, embeddings_path, notes_dir)
        return True

    # Send wikilink to working memory
    if key in (ord("g"), ord("G")):
        send_wikilink(ui, working_mem_file)
        return True

    # Jump preview to top with "h" key
    if key == ord("h"):
        curr_entry = ui.displayed_list[ui.selected_idx] if ui.displayed_list and ui.selected_idx < len(ui.displayed_list) else None
        if curr_entry:
            note_key = curr_entry.split(":", 1)[0].strip()
            if note_key in ui.preview_cache:
                ui.preview_cache[note_key].jump_to_top()
                ui.set_status("Preview set to top.")
        return True

    # Up/down navigation
    if key in (curses.KEY_UP, ord("k")):
        if ui.selected_idx > 0:
            ui.selected_idx -= 1
            if ui.selected_idx < ui.list_offset:
                ui.list_offset = ui.selected_idx
    elif key in (curses.KEY_DOWN, ord("j")):
        if ui.selected_idx < len(ui.displayed_list) - 1:
            ui.selected_idx += 1
            # Adjust list offset based on layout
            if width := stdscr.getmaxyx()[1]:
                max_visible = ((stdscr.getmaxyx()[0]-6)//2 - 1) if width < 100 else (stdscr.getmaxyx()[0]-6-2)
                if ui.selected_idx >= ui.list_offset + max_visible:
                    ui.list_offset += 1
                    
    # Preview scrolling
    elif key == curses.KEY_NPAGE:  # Page Down
        curr_entry = ui.displayed_list[ui.selected_idx] if ui.displayed_list and ui.selected_idx < len(ui.displayed_list) else None
        if curr_entry:
            note_key = curr_entry.split(":", 1)[0].strip()
            if note_key in ui.preview_cache:
                preview_height = stdscr.getmaxyx()[0] // 2 - 10
                ui.preview_cache[note_key].scroll_down(3, preview_height)  # Scroll 3 lines
                
    elif key == curses.KEY_PPAGE:  # Page Up
        curr_entry = ui.displayed_list[ui.selected_idx] if ui.displayed_list and ui.selected_idx < len(ui.displayed_list) else None
        if curr_entry:
            note_key = curr_entry.split(":", 1)[0].strip()
            if note_key in ui.preview_cache:
                ui.preview_cache[note_key].scroll_up(3)  # Scroll 3 lines
                
    # Open note in Neovim
    elif key in (10, 13, curses.KEY_ENTER):  # Enter key
        if ui.selected_idx < len(ui.displayed_list):
            entry = ui.displayed_list[ui.selected_idx]
            note_key = entry.split(":", 1)[0].strip()
            filepath = os.path.join(notes_dir, note_key + ".md")
            try:
                nvim.command(f"edit {filepath}")
                ui.set_status(f"Opened {note_key}")
            except Exception as e:
                ui.set_status(f"Error opening {note_key}: {e}")
                
    # Quit
    elif key in (ord("q"), ord("Q")):
        return False
        
    return True


def interactive_ui(stdscr, nvim, index_path: str, embeddings_path: str, working_mem_file: str, notes_dir: str):
    """Run the interactive user interface."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking input
    stdscr.timeout(100)  # Check for input every 100ms

    # Initialize color pairs
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)      # Headers & labels
    curses.init_pair(2, curses.COLOR_GREEN, -1)     # Selected items
    curses.init_pair(3, curses.COLOR_MAGENTA, -1)   # Status / help overlay

    # Initialize UI state
    ui = UIState()
    
    # Main loop
    running = True
    while running:
        # Handle terminal resize
        new_height, new_width = stdscr.getmaxyx()
        if curses.is_term_resized(new_height, new_width):
            curses.resizeterm(new_height, new_width)
            
        # Update state and draw UI
        update_ui_state(nvim, ui, Path(index_path), Path(embeddings_path), notes_dir)
        draw_ui(stdscr, ui, notes_dir)
        
        # Handle input
        running = handle_input(stdscr, nvim, ui, Path(index_path), Path(embeddings_path), 
                             working_mem_file, notes_dir)
                             
        # Clear status message after timeout
        status_timeout = 3  # seconds
        if ui.status_message and (time.time() - ui.status_message_time > status_timeout):
            ui.status_message = ""
            
        time.sleep(0.01)  # Prevent CPU hogging


def main() -> None:
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Interactive terminal UI for viewing note backlinks, similar notes, and outgoing links. "
                    "Connects to Neovim over a socket to open and interact with notes."
    )
    parser.add_argument("--config-file", help="Specify custom config file path")
    parser.add_argument("--socket-path", help="Specify custom Neovim socket path")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Get configuration values
    notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    notes_dir = resolve_path(notes_dir)
    
    index_file = get_config_value(config, "zk_index.index_file", os.path.join(notes_dir, "index.json"))
    index_file = resolve_path(index_file)
    
    embeddings_file = os.path.join(os.path.dirname(index_file), "embeddings.json")
    
    working_mem_file = get_config_value(config, "working_mem.file", os.path.join(notes_dir, "workingMem.md"))
    working_mem_file = resolve_path(working_mem_file)
    
    # Log file for debugging
    log_file = os.path.join(os.path.expanduser("~/.cache"), "poll_backlinks.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    
    # Check files exist
    if not os.path.isdir(notes_dir):
        sys.exit(f"Error: Notes directory does not exist: {notes_dir}")
    if not os.path.exists(index_file):
        sys.exit(f"Error: Index file does not exist: {index_file}")
    
    # Get socket path from (in order of precedence):
    # 1. Command line argument
    # 2. Global configuration
    # 3. Section-specific configuration (backward compatibility)
    # 4. Environment variable
    # 5. Default value
    socket_path = args.socket_path if args.socket_path else get_config_value(
        config, "socket_path", get_config_value(
            config, "backlinks.socket_path", os.getenv("NVIM_SOCKET", DEFAULT_NVIM_SOCKET)
        )
    )
    
    if args.debug:
        logger.debug(f"Using notes directory: {notes_dir}")
        logger.debug(f"Using index file: {index_file}")
        logger.debug(f"Using socket path: {socket_path}")
    
    # Connect to Neovim
    try:
        nvim = pynvim.attach("socket", path=socket_path)
    except Exception as e:
        sys.exit(f"Failed to attach to Neovim socket: {e}")
    
    # Run the UI
    curses.wrapper(interactive_ui, nvim, index_file, embeddings_file, working_mem_file, notes_dir)


if __name__ == "__main__":
    main()