#!/usr/bin/env python3
"""
workout_log.by

An workout logging tool.
In addition to recording workout sessions and exercises,
you can now:

  • Log new workouts with a live preview that updates as you add each exercise
  • Create, load, edit, and delete workout templates (marked with the tag 'workout_template')
  • View workout history with basic ASCII statistics and recent session summaries
  • Create your own exercises
  • Enjoy immediate feedback and tips across the user interface

Usage:
    $ export NOTES_DIR=/path/to/notes
    $ ./workout_log.py
    OR
    $ ./workout_log.py --notes-dir /path/to/notes

New Features and Improvements:
  • Polished UI using curses with multiple “panels” for a better workflow.
  • A new “View Workout History” option showing summary stats and miniature graphs.
  • Enhanced key–hints displayed on the screen.
  • Improved modularity and error handling.

The NOTES_DIR folder must have your note files (both exercise and workout sessions)
and an index.json. The workouts and templates are stored as markdown files with YAML frontmatter.
"""

import curses
import curses.textpad
import curses.panel
import datetime
import json
import logging
import os
import random
import string
import sys
from pathlib import Path
import re
import yaml
import argparse

# --- Constants & Configurations ---
FRONTMATTER_DELIM = "---"
DATE_KEY = "date"
DATETIME_KEY = "datetime"
MODIFIED_KEY = "dateModified"
PLANNED_KEY = "planned_exercise"
TEMPLATE_TAG = "workout_template"  # Identifies workout templates within notes

# Global log file path will be set later.
LOG_FILE = None

# --- Utility Functions ---
def generate_zkid() -> str:
    """Generate a Zettel ID in the format YYMMDD + 3 random lowercase letters."""
    today = datetime.datetime.now().strftime("%y%m%d")
    rand = "".join(random.choices(string.ascii_lowercase, k=3))
    return today + rand

def current_iso_dt(with_seconds: bool = True) -> str:
    fmt = "%Y-%m-%dT%H:%M:%S" if with_seconds else "%Y-%m-%dT%H:%M"
    return datetime.datetime.now().strftime(fmt)

def draw_box(win, title=""):
    """Helper to draw a box around a window and add an optional title."""
    win.box()
    if title:
        try:
            win.addstr(0, 2, f" {title} ", curses.A_BOLD)
        except Exception:
            pass

# === Data Management ===
class DataManager:
    """
    Provides file I/O routines that create/update exercise notes,
    workout sessions, template notes, and read the workout history.
    """
    def __init__(self, notes_dir: Path, index_file: Path, log_file: Path):
        self.notes_dir = notes_dir
        self.index_file = index_file
        self.log_file = log_file
        self._index_cache = None
        self._index_mtime = None
        self._workout_history_cache = None

    # --- YAML & File Helpers ---
    def read_file_with_frontmatter(self, filepath: Path) -> (dict, str):
        """Read a file with YAML frontmatter, returning (frontmatter, body)"""
        try:
            txt = filepath.read_text(encoding="utf-8")
        except Exception as e:
            logging.error("Failed to read file %s: %s", filepath, e)
            return {}, ""
        if not txt.startswith(FRONTMATTER_DELIM):
            return {}, txt
        parts = txt.split(FRONTMATTER_DELIM, 2)
        if len(parts) < 3:
            return {}, txt
        try:
            fm = yaml.safe_load(parts[1])
            if fm is None:
                fm = {}
        except Exception as err:
            logging.error("Error parsing frontmatter in %s: %s", filepath, err)
            fm = {}
        body = parts[2].lstrip("\n")
        return fm, body

    def write_file_with_frontmatter(self, filepath: Path, frontmatter: dict, body: str = ""):
        """Write a markdown file with YAML frontmatter."""
        try:
            fm_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logging.error("YAML dump error for %s: %s", filepath, e)
            fm_text = ""
        content = f"{FRONTMATTER_DELIM}\n{fm_text}{FRONTMATTER_DELIM}\n\n{body}"
        try:
            filepath.write_text(content, encoding="utf-8")
        except Exception as e:
            logging.error("Failed to write file %s: %s", filepath, e)

    def update_date_modified(self, filepath: Path):
        fm, body = self.read_file_with_frontmatter(filepath)
        fm[MODIFIED_KEY] = current_iso_dt()
        self.write_file_with_frontmatter(filepath, fm, body)

    def read_planned_status(self, filepath: Path) -> bool:
        fm, _ = self.read_file_with_frontmatter(filepath)
        val = fm.get(PLANNED_KEY, False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return False

    # --- Exercise File Operations ---
    def write_exercise_file(self, title: str, equipment: str) -> str:
        """Write a new exercise note with title and equipment."""
        zkid = generate_zkid()
        filename = f"{zkid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        eq_items = [item.strip() for item in equipment.split(",") if item.strip()]
        equipment_list = eq_items if eq_items else ["none"]
        frontmatter = {
            "title": title,
            "tags": ["exercise"],
            DATE_KEY: today_str,
            DATETIME_KEY: datetime_str,
            PLANNED_KEY: False,
            "exercise_equipment": equipment_list,
        }
        self.write_file_with_frontmatter(filepath, frontmatter)
        return filename

    def toggle_planned_status(self, exercise_filename: str) -> bool:
        """Toggle the planned state of an exercise note."""
        if not exercise_filename.endswith(".md"):
            exercise_filename += ".md"
        filepath = self.notes_dir / exercise_filename
        fm, body = self.read_file_with_frontmatter(filepath)
        cur_val = fm.get(PLANNED_KEY, False)
        new_val = (not cur_val) if type(cur_val) is bool else cur_val.lower() != "true"
        fm[PLANNED_KEY] = new_val
        fm[MODIFIED_KEY] = current_iso_dt()
        self.write_file_with_frontmatter(filepath, fm, body)
        self._index_cache = None  # Invalidate index cache
        return self.read_planned_status(filepath)

    def load_index_exercises(self) -> list:
        """
        Load exercise index from index.json and add planned and equipment data.
        Returns a list of dictionary objects.
        """
        try:
            stat = self.index_file.stat()
        except Exception as err:
            logging.error("Failed to stat index file %s: %s", self.index_file, err)
            return []
        if self._index_cache is None or stat.st_mtime != self._index_mtime:
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8"))
                exercises = []
                for note in data:
                    tags = note.get("tags", [])
                    if isinstance(tags, list) and "exercise" in tags:
                        filename = note.get("filename")
                        filepath = self.notes_dir / (filename + ".md")
                        planned = self.read_planned_status(filepath) if filepath.exists() else False
                        exercises.append({
                            "filename": filename,
                            "title": note.get("title", filename),
                            "planned": planned,
                            "equipment": note.get("exercise_equipment", ["none"]),
                        })
                self._index_cache = exercises
                self._index_mtime = stat.st_mtime
            except Exception as err:
                logging.error("Error reading index file %s: %s", self.index_file, err)
                return []
        return self._index_cache

    # --- Workout Note Operations & History ---
    def write_workout_note(self, session_exercises: list) -> str:
        """Create a workout session note that stores the session's exercises."""
        zkid = generate_zkid()
        filename = f"{zkid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        datetime_str = current_iso_dt(with_seconds=False)
        frontmatter = {
            "zettelid": zkid,
            "title": f"Workout session {today_str}",
            "tags": ["workout"],
            DATE_KEY: today_str,
            DATETIME_KEY: datetime_str,
            "exercises": session_exercises,
        }
        self.write_file_with_frontmatter(filepath, frontmatter)
        self._workout_history_cache = None
        return filename

    def parse_workout_frontmatter(self, text: str) -> dict:
        """Extract frontmatter from a workout note."""
        parts = text.split(FRONTMATTER_DELIM, 2)
        if len(parts) < 3:
            return {}
        try:
            fm = yaml.safe_load(parts[1])
        except Exception as err:
            logging.error("Error parsing workout YAML: %s", err)
            return {}
        return fm if fm else {}

    def build_workout_history_cache(self):
        """
        Scan NOTES_DIR for workout notes and build a history cache keyed by exercise id.
        Each entry contains date, set count, average reps and average weight.
        """
        self._workout_history_cache = {}
        for md_file in self.notes_dir.glob("*.md"):
            try:
                txt = md_file.read_text(encoding="utf-8")
            except Exception as err:
                logging.error("Error reading file %s: %s", md_file, err)
                continue
            fm = self.parse_workout_frontmatter(txt)
            if not fm:
                continue
            tags = fm.get("tags") or []
            if "workout" not in tags:
                continue
            f_date = fm.get("date", "unknown")
            for ex in fm.get("exercises", []):
                sets = ex.get("sets", [])
                num = len(sets)
                total_reps = 0
                total_weight = 0.0
                valid = 0
                for s in sets:
                    try:
                        reps = int(s.get("reps", 0))
                        weight = float(s.get("weight", 0))
                        total_reps += reps
                        total_weight += weight
                        valid += 1
                    except Exception as err:
                        logging.error("Conversion error in %s: %s", md_file, err)
                avg_reps = total_reps/valid if valid > 0 else 0
                avg_weight = total_weight/valid if valid > 0 else 0
                entry = {"date": f_date, "sets": num, "avg_reps": avg_reps, "avg_weight": avg_weight}
                self._workout_history_cache.setdefault(ex.get("filename"), []).append(entry) # Use filename as key
        if self._workout_history_cache:
            for ex_id in self._workout_history_cache:
                self._workout_history_cache[ex_id].sort(key=lambda x: x.get("date", ""), reverse=True)

    def get_workout_history_for_exercise(self, ex_id: str) -> list:
        """Return the history for a given exercise id."""
        if self._workout_history_cache is None:
            self.build_workout_history_cache()
        return self._workout_history_cache.get(ex_id, [])

    def list_workout_sessions(self) -> list:
        """
        Return a list of all workout sessions (files with tag 'workout').
        Sorted descending by date.
        Each entry is a tuple (filename, date, title).
        """
        sessions = []
        for md_file in self.notes_dir.glob("*.md"):
            fm, _ = self.read_file_with_frontmatter(md_file)
            if not fm:
                continue
            tags = fm.get("tags") or []
            if "workout" in tags:
                sessions.append({
                    "filename": md_file.name,
                    "date": fm.get("date","unknown"),
                    "title": fm.get("title", md_file.stem),
                    "exercises": fm.get("exercises") or []
                })
        sessions.sort(key=lambda s: s.get("date", ""), reverse=True)
        return sessions

    # --- Workout Template Operations ---
    def write_workout_template(self, name: str, description: str, exercises: list) -> str:
        """Create a workout template note with the tag 'workout_template'."""
        zkid = generate_zkid()
        filename = f"{zkid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        datetime_str = current_iso_dt(with_seconds=False)
        frontmatter = {
            "title": name,
            "description": description,
            DATE_KEY: today_str,
            DATETIME_KEY: datetime_str,
            MODIFIED_KEY: datetime_str,
            "exercises": [{"exercise_filename": ex["filename"], "order": idx + 1} for idx, ex in enumerate(exercises)],
            "tags": [TEMPLATE_TAG],
        }
        body = f"# {name}\n\n{description}"
        self.write_file_with_frontmatter(filepath, frontmatter, body)
        return filename

    def load_workout_templates(self) -> list:
        """
        Scan NOTES_DIR for workout templates (notes that have TEMPLATE_TAG)
        and return them.
        """
        templates = []
        for md_file in self.notes_dir.glob("*.md"):
            fm, body = self.read_file_with_frontmatter(md_file)
            if not fm:
                continue
            tags = fm.get("tags") or []
            if TEMPLATE_TAG not in tags:
                continue
            templates.append({
                "filename": md_file.name,
                "title": fm.get("title", md_file.stem),
                "description": fm.get("description", ""),
                "exercises": fm.get("exercises") or []
            })
        return templates

    def update_workout_template(self, tmpl_filename: str, name: str, description: str, exercises: list):
        """Update an existing workout template note."""
        filepath = self.notes_dir / tmpl_filename
        if not filepath.exists():
            logging.error("Template %s not found", tmpl_filename)
            return
        datetime_str = current_iso_dt(with_seconds=False)
        fm, _ = self.read_file_with_frontmatter(filepath)
        fm["title"] = name
        fm["description"] = description
        fm[MODIFIED_KEY] = datetime_str
        fm["exercises"] = [{"exercise_filename": ex["filename"], "order": idx + 1} for idx, ex in enumerate(exercises)]
        body = f"# {name}\n\n{description}"
        self.write_file_with_frontmatter(filepath, fm, body)

    def delete_workout_template(self, tmpl_filename: str):
        """Delete the selected workout template note."""
        filepath = self.notes_dir / tmpl_filename
        try:
            filepath.unlink()
        except Exception as err:
            logging.error("Error deleting template %s: %s", tmpl_filename, err)

# === UI Management: Using curses ===
class UIManager:
    """
    Handles a polished curses–interface. Includes menus for recording sessions,
    selecting exercises, viewing workout history, and managing workout templates.
    """
    def __init__(self, stdscr, data_manager: DataManager):
        self.stdscr = stdscr
        self.dm = data_manager
        curses.curs_set(1)
        self.stdscr.nodelay(False)
        self.stdscr.keypad(True)
        self.init_colors()

    def init_colors(self):
        """Initialize color pairs for header, selected items and footer/status."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)     # Selected
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Footer/Status
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Normal text

    # --- Screen Helpers ---
    def draw_header(self, title: str):
        self.stdscr.attron(curses.color_pair(1))
        try:
            self.stdscr.addstr(0, 0, (" " * (curses.COLS - 1))[:curses.COLS-1])
            self.stdscr.addstr(0, 2, title)
        except Exception:
            pass
        self.stdscr.attroff(curses.color_pair(1))
        self.stdscr.refresh()

    def show_footer(self, message: str, color_pair: int = 3):
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.attron(curses.color_pair(color_pair))
        self.stdscr.addstr(max_y - 1, 0, message[:max_x-1])
        self.stdscr.clrtoeol()
        self.stdscr.attroff(curses.color_pair(color_pair))
        self.stdscr.refresh()

    def pause_message(self, message: str = "Press any key to continue..."):
        self.show_footer(message, color_pair=3)
        self.stdscr.getch()

    def clear_screen(self):
        self.stdscr.clear()
        self.stdscr.bkgd(' ', curses.color_pair(4))
        self.stdscr.refresh()

    def prompt_input(self, prompt_str: str, y: int, x: int, default="") -> str:
        """Prompt the user for input at a given location."""
        self.stdscr.addstr(y, x, prompt_str)
        self.stdscr.clrtoeol()
        curses.echo()
        try:
            inp = self.stdscr.getstr(y, x + len(prompt_str)).decode("utf-8").strip()
        except Exception as err:
            logging.error("Input error: %s", err)
            inp = ""
        curses.noecho()
        return inp if inp else default

    # --- Workout Session Recording ---
    def record_exercise_session(self, exercise: dict) -> dict:
        """
        Record a set-by-set entry for a given exercise.
        Returns a dictionary with the exercise file id and the recorded sets.
        """
        self.clear_screen()
        header = f"Recording Workout for: {exercise.get('title', '')}"
        if exercise.get("planned"):
            header += " [PLANNED]"
        self.draw_header(header)
        set_number = 1
        sets_recorded = []
        last_weight = ""
        max_y, max_x = self.stdscr.getmaxyx()

        # Create a window for the summary of sets.
        sum_h = 7; sum_w = max_x - 6; sum_y = 2; sum_x = 3
        summary_win = curses.newwin(sum_h, sum_w, sum_y, sum_x)
        draw_box(summary_win, " Recorded Sets ")
        summary_win.refresh()

        input_y = sum_y + sum_h + 1
        while True:
            # Provide instructions for the set.
            self.stdscr.addstr(input_y, 2, f"Set #{set_number}: (Leave 'Reps' empty to finish)")
            reps = self.prompt_input(" Reps: ", input_y + 1, 4)
            if not reps:
                break
            prompt_w = f" Weight [{last_weight}]: " if last_weight else " Weight: "
            weight = self.prompt_input(prompt_w, input_y + 2, 4)
            if not weight and last_weight:
                weight = last_weight
            else:
                last_weight = weight
            sets_recorded.append({"reps": reps, "weight": weight})
            # Update summary window
            summary_win.erase()
            draw_box(summary_win, " Recorded Sets ")
            for idx, s in enumerate(sets_recorded, start=1):
                if idx < sum_h - 1:
                    summary_win.addstr(idx, 2, f"Set {idx}: {s['reps']} reps @ {s['weight']} wt")
            summary_win.refresh()
            set_number += 1
            input_y += 4
            if input_y > max_y - 6:
                self.pause_message("Press any key to continue recording sets...")
                self.clear_screen()
                self.draw_header(header)
                input_y = sum_y + sum_h + 1
                summary_win.mvwin(sum_y, sum_x)
                summary_win.refresh()
        self.pause_message("Exercise sets recorded. Press any key to continue...")
        return {"id": exercise["filename"], "title": exercise.get("title"), "sets": sets_recorded} # Keep title for preview

    def draw_session_preview(self, session_exercises: list):
        """Display an updated session preview panel showing exercises recorded so far."""
        max_y, max_x = self.stdscr.getmaxyx()
        preview_h = max(10, max_y // 2)  # Increased height for preview
        preview_win = curses.newwin(preview_h, max_x - 2, max_y - preview_h - 2, 1)
        draw_box(preview_win, " Session Preview ")
        line = 1
        if not session_exercises:
            preview_win.addstr(line, 2, " (No exercises recorded yet)")
        else:
            for ex in session_exercises:
                if "id" in ex and "sets" in ex:
                    preview_win.addstr(line, 2, f"• {ex['title']}: {len(ex['sets'])} set(s)") # Use title here
                    for set_data in ex['sets']:
                        line += 1
                        preview_win.addstr(line, 4, f"  - {set_data.get('reps', '?')} reps @ {set_data.get('weight', '?')} wt")
                else:
                    preview_win.addstr(line, 2, f"• {ex.get('title','')}")
                line += 1
                if line >= preview_h - 2: # Give a bit more padding
                    preview_win.addstr(preview_h - 2, 2, "(...) more exercises below (...)")
                    break
        preview_win.refresh()

    def record_workout_session(self, prepopulated_exercises: list = None):
        """
        Record a complete workout session.
        If prepopulated_exercises is provided (e.g. from a template),
        those exercises are recorded first.
        """
        self.clear_screen()
        self.draw_header("Start Workout Session")
        self.pause_message("Press any key to begin your workout...")
        session_exercises = []
        # Record exercises coming from a template
        if prepopulated_exercises:
            for ex in prepopulated_exercises:
                rec = self.record_exercise_session(ex)
                if rec["sets"]:
                    session_exercises.append(rec)
            add_more = self.prompt_input("Add another exercise? (y/N): ", curses.LINES-4, 2)
            if add_more.lower() == "y":
                while True:
                    ex = self.select_exercise(session_exercises)
                    if ex is None:
                        break
                    rec = self.record_exercise_session(ex)
                    if rec["sets"]:
                        session_exercises.append(rec)
                    self.clear_screen()
                    self.draw_header("Recording Workout Session")
                    self.draw_session_preview(session_exercises)
                    add_more = self.prompt_input("Add another exercise? (y/N): ", curses.LINES-4, 2)
                    if add_more.lower() != "y":
                        break
        else:
            while True:
                ex = self.select_exercise(session_exercises)
                if ex is None:
                    break
                rec = self.record_exercise_session(ex)
                if rec["sets"]:
                    session_exercises.append(rec)
                self.clear_screen()
                self.draw_header("Recording Workout Session")
                self.draw_session_preview(session_exercises)
                add_more = self.prompt_input("Add another exercise? (y/N): ", curses.LINES-4, 2)
                if add_more.lower() != "y":
                    break
        # Save if any sessions were recorded
        if not session_exercises:
            self.show_footer("No exercises recorded for this session.", color_pair=3)
            self.pause_message()
            return
        self.dm.write_workout_note(session_exercises)
        self.show_footer("Workout session saved.", color_pair=3)
        self.pause_message("Session complete. Press any key to return to main menu...")

    # --- Exercise Selection ---
    def select_exercise(self, session_exercises: list) -> dict:
        """
        Allow the user to select an exercise from the index.
        Also allow for toggling planned status.
        """
        cursor = 0
        while True:
            exercises = self.dm.load_index_exercises()
            max_y, max_x = self.stdscr.getmaxyx()
            list_h = max_y - 10
            self.clear_screen()
            self.draw_header("Select an Exercise")
            header_str = f"{'Title':<{max_x-35}} {'Status':<10} Equipment"
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
            self.stdscr.addstr(2, 2, header_str[:max_x-4])
            self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))
            for idx, ex in enumerate(exercises[:list_h-4]):
                status = "Planned" if ex.get("planned") else ""
                equip = ", ".join(ex.get("equipment") or [])
                line = f"{ex.get('title',''):<{max_x-35}} {status:<10} {equip}"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3 + idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3 + idx, 2, line[:max_x-4])
            key_hint = "↑/↓: Navigate | Enter: Select | P: Toggle Planned | Q: Cancel"
            self.show_footer(key_hint, color_pair=3)
            self.stdscr.refresh()
            # Secondary panel: preview for the selected exercise and session summary
            preview_h = 8
            preview_win = curses.newwin(preview_h, max_x-2, max_y - preview_h - 2, 1)
            draw_box(preview_win, " Exercise & Session Preview ")
            if exercises:
                sel = exercises[cursor]
                preview_win.addstr(1, 2, f"Title: {sel.get('title','')}")
                equip = ", ".join(sel.get("equipment" or []))
                preview_win.addstr(2, 2, f"Equipment: {equip}")
                status = "Planned" if sel.get("planned") else "Not Planned"
                preview_win.addstr(3, 2, f"Status: {status}")
                preview_win.addstr(4, 2, "Recent History:")
                hist = self.dm.get_workout_history_for_exercise(sel["filename"]) # Still use filename here for data access
                line = 5
                if hist:
                    for h in hist[:preview_h - line - 1]:
                        preview_win.addstr(line, 2, f"{h['date']}: {h['sets']} set(s), "
                                                     f"avgR: {h['avg_reps']:.1f}, wt: {h['avg_weight']:.1f}")
                        line += 1
                else:
                    preview_win.addstr(5, 2, "No previous sessions.")
            preview_win.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(exercises)-1:
                    cursor += 1
            elif k in (10, 13):
                return exercises[cursor]
            elif k in (ord('p'), ord('P')):
                if exercises:
                    ex = exercises[cursor]
                    new_state = self.dm.toggle_planned_status(ex["filename"])
                    ex["planned"] = new_state
                    state_txt = "Planned" if new_state else "Not Planned"
                    self.show_footer(f"'{ex.get('title','')}' toggled to {state_txt}", color_pair=3)
                    self.pause_message("Press any key...")
            elif k in (ord('q'), ord('Q')):
                return None

    # --- Workout Template Management ---
    def create_workout_template(self):
        """Allow user to create a workout template from selected exercises."""
        self.clear_screen()
        self.draw_header("Create Workout Template")
        tmpl_name = self.prompt_input("Enter template name: ", 4, 4)
        if not tmpl_name:
            self.show_footer("Template name missing. Cancelled.", color_pair=3)
            self.pause_message()
            return
        tmpl_desc = self.prompt_input("Enter template description (optional): ", 5, 4)
        exercises = []
        while True:
            add = self.prompt_input("Add an exercise? (y/N): ", 7, 4)
            if add.lower() != "y":
                break
            ex = self.select_exercise(exercises)
            if ex:
                exercises.append(ex)
            else:
                break
        if not exercises:
            self.show_footer("No exercises selected. Template creation cancelled.", color_pair=3)
            self.pause_message()
            return
        self.dm.write_workout_template(tmpl_name, tmpl_desc, exercises)
        self.show_footer(f"Template '{tmpl_name}' created.", color_pair=3)
        self.pause_message("Press any key to return to main menu...")

    def list_workout_templates(self) -> list:
        """Display a list of available workout templates for selection."""
        templates = self.dm.load_workout_templates()
        if not templates:
            self.show_footer("No workout templates available.", color_pair=3)
            self.pause_message()
            return None
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Select Workout Template")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, tmpl in enumerate(templates):
                y_pos = 3 + idx
                disp = f"{tmpl['title']}: {tmpl['description']}"
                if y_pos < max_y - 2:
                    if idx == cursor:
                        self.stdscr.attron(curses.color_pair(2))
                        self.stdscr.addstr(y_pos, 2, disp[:max_x-4])
                        self.stdscr.attroff(curses.color_pair(2))
                    else:
                        self.stdscr.addstr(y_pos, 2, disp[:max_x-4])
            self.show_footer("↑/↓: Navigate | Enter: Select | Q: Back", color_pair=3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(templates)-1:
                    cursor += 1
            elif k in (10, 13):
                return templates[cursor]
            elif k in (ord('q'), ord('Q')):
                return None

    def start_workout_from_template(self):
        tmpl = self.list_workout_templates()
        if not tmpl:
            return
        index_exercises = self.dm.load_index_exercises()
        prepopulated = []
        for item in tmpl["exercises"]:
            for ex in index_exercises:
                if ex["filename"] == item.get("exercise_filename"):
                    prepopulated.append(ex)
                    break
        self.show_footer(f"Starting workout from template '{tmpl['title']}'", color_pair=3)
        self.pause_message("Press any key to continue...")
        self.record_workout_session(prepopulated)

    def edit_workout_template(self):
        """Edit an existing workout template."""
        tmpl = self.list_workout_templates()
        if not tmpl:
            return
        self.clear_screen()
        self.draw_header("Edit Workout Template")
        new_name = self.prompt_input(f"Enter new name [{tmpl['title']}]: ", 4, 4, tmpl['title'])
        new_desc = self.prompt_input(f"Enter new description [{tmpl['description']}]: ", 5, 4, tmpl['description'])
        rebuild = self.prompt_input("Rebuild exercise list? (y/N): ", 7, 4)
        exercises = []
        if rebuild.lower() == 'y':
            while True:
                add = self.prompt_input("Add an exercise? (y/N): ", 9, 4)
                if add.lower() != "y":
                    break
                ex = self.select_exercise(exercises)
                if ex:
                    exercises.append(ex)
                else:
                    break
            if not exercises:
                self.show_footer("No exercises selected. Edit cancelled.", color_pair=3)
                self.pause_message()
                return
        else:
            index_exercises = self.dm.load_index_exercises()
            for item in tmpl["exercises"]:
                for ex in index_exercises:
                    if ex["filename"] == item.get("exercise_filename"):
                        exercises.append(ex)
                        break
        self.dm.update_workout_template(tmpl['filename'], new_name, new_desc, exercises)
        self.show_footer(f"Template '{new_name}' updated.", color_pair=3)
        self.pause_message("Press any key to return to main menu...")

    def delete_workout_template(self):
        """Select and delete a workout template."""
        tmpl = self.list_workout_templates()
        if not tmpl:
            return
        self.clear_screen()
        self.draw_header("Delete Workout Template")
        confirm = self.prompt_input(f"Delete template '{tmpl['title']}'? (y/N): ", 4, 4)
        if confirm.lower() == "y":
            self.dm.delete_workout_template(tmpl["filename"])
            self.show_footer(f"Template '{tmpl['title']}' deleted.", color_pair=3)
        else:
            self.show_footer("Deletion cancelled.", color_pair=3)
        self.pause_message("Press any key to return to main menu...")

    # --- New Exercise Creation ---
    def create_new_exercise(self):
        self.clear_screen()
        self.draw_header("Create New Exercise")
        title = self.prompt_input("Enter exercise title: ", 4, 4)
        equipment = self.prompt_input("Enter comma separated equipment: ", 5, 4)
        if title:
            self.dm.write_exercise_file(title, equipment)
            self.show_footer(f"New exercise '{title}' created.", color_pair=3)
            self.pause_message("Exercise created. Press any key to continue...")
        else:
            self.show_footer("No title provided. Cancelled.", color_pair=3)
            self.pause_message("Press any key to return to main menu...")

    # --- Workout History Viewer ---
    def view_workout_history(self):
        """Display a list of workout sessions with basic graphs for recent exercises."""
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No past workout sessions found.", color_pair=3)
            self.pause_message()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Workout History")
            max_y, max_x = self.stdscr.getmaxyx()
            list_h = max_y - 8
            for idx, session in enumerate(sessions[:list_h]):
                date = session.get("date", "unknown")
                title = session.get("title", "")
                line = f"{date} - {title}"
                y = 3+idx
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 2, line[:max_x-4])
            key_hint = "↑/↓: Navigate | Enter: View Session | Q: Back"
            self.show_footer(key_hint, color_pair=3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(sessions)-1:
                    cursor += 1
            elif k in (10, 13):
                self.display_session_details(sessions[cursor])
            elif k in (ord('q'), ord('Q')):
                break

    def display_session_details(self, session: dict):
        """Show detailed info of a chosen workout session with a simple ASCII chart."""
        self.clear_screen()
        self.draw_header(f"Session Details: {session['title']}")
        max_y, max_x = self.stdscr.getmaxyx()
        details_win = curses.newwin(max_y-6, max_x-4, 3, 2)
        draw_box(details_win, " Session Info ")
        details_win.addstr(1, 2, f"Date: {session.get('date','')}")
        details_win.addstr(2, 2, f"Title: {session.get('title','')}")
        details_win.addstr(4, 2, "Exercises:")
        line = 5
        for ex in session.get("exercises") or []:
            ex_title = ""
            exercises_index = self.dm.load_index_exercises() # Load exercise index to find title
            for indexed_ex in exercises_index:
                if indexed_ex["filename"] == ex.get("id"):
                    ex_title = indexed_ex["title"]
                    break
            rec_line = f"- {ex_title}: {len(ex.get('sets', []))} set(s)" # Use title from index
            details_win.addstr(line, 4, rec_line[:max_x-8])
            line += 1
        # Simple ASCII–bar for average reps for the first exercise (if exists)
        if session.get("exercises"):
            first = session["exercises"][0]
            try:
                total = sum(int(s.get("reps", 0)) for s in first.get("sets", []))
                count = len(first.get("sets", []))
                avg = total/count if count>0 else 0
                bar = "#" * int(avg)
                details_win.addstr(line+1, 2, f"Avg Reps (first exercise): {avg:.1f} {bar}")
            except Exception:
                pass
        details_win.refresh()
        self.pause_message("Press any key to return to history view...")

    # --- Main Menu ---
    def main_menu(self):
        menu_options = [
            "Record a Workout Session",
            "View Workout History",
            "Create a New Exercise",
            "Create Workout Template",
            "Start Workout from Template",
            "Edit Workout Template",
            "Delete Workout Template",
            "Quit"
        ]
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Workout Logger – Main Menu")
            for idx, opt in enumerate(menu_options):
                y = 3 + idx * 2
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 4, f"> {opt}")
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 4, f"  {opt}")
            hint = "↑/↓: Navigate | Enter: Select | Q: Quit"
            self.show_footer(hint, color_pair=3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif k in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(menu_options)-1, cursor+1)
            elif k in (10, 13):
                if cursor == 0:
                    self.record_workout_session()
                elif cursor == 1:
                    self.view_workout_history()
                elif cursor == 2:
                    self.create_new_exercise()
                elif cursor == 3:
                    self.create_workout_template()
                elif cursor == 4:
                    self.start_workout_from_template()
                elif cursor == 5:
                    self.edit_workout_template()
                elif cursor == 6:
                    self.delete_workout_template()
                elif cursor == 7:
                    break
            elif k in (ord('q'), ord('Q')):
                break

# === Main Entrypoint ===
def main(stdscr):
    parser = argparse.ArgumentParser(description="Workout Logging Script")
    parser.add_argument('--notes-dir', type=str,
                        help='Path to the notes directory (overrides NOTES_DIR env variable).')
    args = parser.parse_args()
    notes_dir_cli = args.notes_dir
    notes_dir_env = os.environ.get("NOTES_DIR")
    if notes_dir_cli:
        notes_dir_path = notes_dir_cli
    elif notes_dir_env:
        notes_dir_path = notes_dir_env
    else:
        print("Error: Provide NOTES_DIR env variable or --notes-dir flag.")
        sys.exit(1)
    NOTES_DIR = Path(notes_dir_path)
    INDEX_FILE = NOTES_DIR / "index.json"
    LOG_FILE_PATH = NOTES_DIR / "workout_log_error.log"
    global LOG_FILE
    LOG_FILE = LOG_FILE_PATH
    logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    dm = DataManager(NOTES_DIR, INDEX_FILE, LOG_FILE_PATH)
    ui = UIManager(stdscr, dm)
    ui.main_menu()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        if LOG_FILE:
            logging.exception("Critical error in Workout Logger")
            print(f"An error occurred. Check log file at {LOG_FILE}.")
        else:
            print("An error occurred. Check your log for details.")
        sys.exit(1)

