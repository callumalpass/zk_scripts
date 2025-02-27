#!/usr/bin/env python3
"""
workout_log.py

A re‐imagined workout logging script using a curses interface.
It records workout sessions as YAML–frontmatter “notes” tied to individual “exercise” notes,
and now supports workout templates (routines) to pre‐populate sessions.
Templates are identified by the tag 'workout_template' rather than by their location.

Usage:
    # Using environment variable
    $ export NOTES_DIR=/path/to/notes
    $ ./workout_log.py

    # Using command line flag (overrides environment variable if both are set)
    $ ./workout_log.py --notes-dir /path/to/notes

New Features (Workout Templates):
  • Create workout templates by selecting exercises from your exercise library,
    setting a name and (optionally) a description. Templates are tagged (workout_template).
  • Start a new workout session based on a selected template.
  • Edit or delete existing workout templates.

Environment Variable:
    NOTES_DIR  –  Directory where your note files and workout templates reside.
"""

import curses
import curses.textpad
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

# --- Constants & Configuration ---
FRONTMATTER_DELIM = "---"
DATE_KEY = "date"
DATETIME_KEY = "datetime"
MODIFIED_KEY = "dateModified"
PLANNED_KEY = "planned_exercise"
TEMPLATE_TAG = "workout_template"  # Tag used to identify workout templates

# Logging setup
LOG_FILE = None  # Global log file path (set later)

# --- Utility Functions ---
def generate_zkid() -> str:
    """Generate a zettel ID: YYMMDD + 3 random lowercase letters."""
    today = datetime.datetime.now().strftime("%y%m%d")
    rand = "".join(random.choices(string.ascii_lowercase, k=3))
    return today + rand

def current_iso_dt(with_seconds: bool = True) -> str:
    fmt = "%Y-%m-%dT%H:%M:%S" if with_seconds else "%Y-%m-%dT%H:%M"
    return datetime.datetime.now().strftime(fmt)

# === Data Management ===
class DataManager:
    """
    Handles file reading/writing, caching for exercises, workout notes,
    and workout templates (which are now identified by the tag 'workout_template').
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
        """Read a file and return (frontmatter_dict, body)."""
        try:
            content = filepath.read_text(encoding="utf-8")
        except Exception as e:
            logging.error("Cannot read file %s: %s", filepath, e)
            return {}, ""
        if not content.startswith(FRONTMATTER_DELIM):
            return {}, content
        parts = content.split(FRONTMATTER_DELIM, 2)
        if len(parts) < 3:
            return {}, content
        try:
            frontmatter = yaml.safe_load(parts[1])
            if frontmatter is None:
                frontmatter = {}
        except Exception as e:
            logging.error("Error parsing YAML frontmatter in %s: %s", filepath, e)
            frontmatter = {}
        body = parts[2].lstrip("\n")
        return frontmatter, body

    def write_file_with_frontmatter(self, filepath: Path, frontmatter: dict, body: str = ""):
        """Write a file with YAML frontmatter."""
        try:
            fm_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logging.error("YAML dump error for file %s: %s", filepath, e)
            fm_text = ""
        content = f"{FRONTMATTER_DELIM}\n{fm_text}{FRONTMATTER_DELIM}\n\n{body}"
        try:
            filepath.write_text(content, encoding="utf-8")
        except Exception as e:
            logging.error("Cannot write file %s: %s", filepath, e)

    def update_date_modified(self, filepath: Path):
        frontmatter, body = self.read_file_with_frontmatter(filepath)
        frontmatter[MODIFIED_KEY] = current_iso_dt()
        self.write_file_with_frontmatter(filepath, frontmatter, body)

    def read_planned_status(self, filepath: Path) -> bool:
        frontmatter, _ = self.read_file_with_frontmatter(filepath)
        val = frontmatter.get(PLANNED_KEY, False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return False

    # --- Exercise File Operations ---
    def write_exercise_file(self, title: str, equipment: str) -> str:
        """
        Write a new exercise file (with a generated zettel ID) that includes title,
        equipment list, and planned status. Returns the generated filename.
        """
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
        """
        Toggle the planned status for an exercise file.
        Returns the new planned status.
        """
        if not exercise_filename.endswith(".md"):
            exercise_filename += ".md"
        filepath = self.notes_dir / exercise_filename
        frontmatter, body = self.read_file_with_frontmatter(filepath)
        current_val = frontmatter.get(PLANNED_KEY, False)
        if isinstance(current_val, bool):
            new_val = not current_val
        elif isinstance(current_val, str):
            new_val = current_val.lower() != "true"
        else:
            new_val = True
        frontmatter[PLANNED_KEY] = new_val
        frontmatter[MODIFIED_KEY] = current_iso_dt()
        self.write_file_with_frontmatter(filepath, frontmatter, body)
        self._index_cache = None
        return self.read_planned_status(filepath)

    def load_index_exercises(self) -> list:
        """
        Read the index JSON file and filter for exercise notes.
        Returns a list with keys: filename, title, planned, equipment.
        """
        try:
            stat = self.index_file.stat()
        except Exception as e:
            logging.error("Unable to stat index file %s: %s", self.index_file, e)
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
            except Exception as e:
                logging.error("Error reading index file %s: %s", self.index_file, e)
                return []
        return self._index_cache

    # --- Workout Note Operations & History ---
    def write_workout_note(self, session_exercises: list) -> str:
        """
        Write a new workout note that records the exercises performed in the session.
        Returns the generated filename.
        """
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
        self.write_file_with_frontmatter(filepath, frontmatter, body="")
        self._workout_history_cache = None
        return filename

    def parse_workout_frontmatter(self, content: str) -> dict:
        """Return workout front matter from the content string."""
        parts = content.split(FRONTMATTER_DELIM, 2)
        if len(parts) < 3:
            return {}
        try:
            fm = yaml.safe_load(parts[1])
        except Exception as e:
            logging.error("Error parsing workout YAML: %s", e)
            return {}
        return fm if fm else {}

    def build_workout_history_cache(self):
        """
        Build a cache for workout history keyed by exercise id.
        """
        self._workout_history_cache = {}
        for md_file in self.notes_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logging.error("Error reading file %s: %s", md_file, e)
                continue
            fm = self.parse_workout_frontmatter(content)
            if not fm:
                continue
            tags = fm.get("tags") or []
            if "workout" not in tags:
                continue
            f_date = fm.get("date", "unknown")
            for ex in fm.get("exercises", []):
                sets = ex.get("sets", [])
                num_sets = len(sets)
                total_reps = 0
                total_weight = 0.0
                valid_sets = 0
                for s in sets:
                    try:
                        reps = int(s.get("reps", "0"))
                        weight = float(s.get("weight", "0"))
                        total_reps += reps
                        total_weight += weight
                        valid_sets += 1
                    except Exception as err:
                        logging.error("Error converting set values in file %s: %s", md_file, err)
                avg_reps = total_reps / valid_sets if valid_sets > 0 else 0
                avg_weight = total_weight / valid_sets if valid_sets > 0 else 0
                entry = {"date": f_date, "sets": num_sets, "avg_reps": avg_reps, "avg_weight": avg_weight}
                self._workout_history_cache.setdefault(ex.get("id"), []).append(entry)
        if self._workout_history_cache:
            for ex_id, entries in self._workout_history_cache.items():
                entries.sort(key=lambda x: x.get("date", ""), reverse=True)

    def get_workout_history_for_exercise(self, ex_id: str) -> list:
        """Return workout history for a given exercise id."""
        if self._workout_history_cache is None:
            self.build_workout_history_cache()
        return self._workout_history_cache.get(ex_id, [])

    # --- Workout Template Operations (Tag-based) ---
    def write_workout_template(self, name: str, description: str, exercises: list) -> str:
        """
        Write a new workout template file as a regular note in NOTES_DIR,
        tagged as 'workout_template'.
        """
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
            "exercises": [{"exercise_filename": ex["filename"], "order": idx+1} for idx, ex in enumerate(exercises)],
            "tags": [TEMPLATE_TAG],
        }
        self.write_file_with_frontmatter(filepath, frontmatter, body=f"# {name}\n\n{description}")
        return filename

    def load_workout_templates(self) -> list:
        """
        Load all workout templates by scanning NOTES_DIR for notes tagged as 'workout_template'.
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
                "exercises": fm.get("exercises", []),
            })
        return templates

    def update_workout_template(self, tmpl_filename: str, name: str, description: str, exercises: list):
        """Update an existing workout template file (tag-based)."""
        filepath = self.notes_dir / tmpl_filename
        if not filepath.exists():
            logging.error("Template file %s does not exist for updating.", tmpl_filename)
            return
        datetime_str = current_iso_dt(with_seconds=False)
        fm, _ = self.read_file_with_frontmatter(filepath)
        fm["title"] = name
        fm["description"] = description
        fm[MODIFIED_KEY] = datetime_str
        fm["exercises"] = [{"exercise_filename": ex["filename"], "order": idx+1} for idx, ex in enumerate(exercises)]
        self.write_file_with_frontmatter(filepath, fm, body=f"# {name}\n\n{description}")

    def delete_workout_template(self, tmpl_filename: str):
        """Delete a workout template file (tag-based)."""
        filepath = self.notes_dir / tmpl_filename
        try:
            filepath.unlink()
        except Exception as e:
            logging.error("Error deleting template file %s: %s", tmpl_filename, e)

# === UI Management ===
class UIManager:
    """
    Handles the curses–based user interface.
    """
    def __init__(self, stdscr, data_manager: DataManager):
        self.stdscr = stdscr
        self.dm = data_manager
        curses.curs_set(1)
        self.stdscr.nodelay(False)
        self.stdscr.keypad(True)
        self.init_colors()

    def init_colors(self):
        """Initialize three color pairs: header, selected, footer/status."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)    # Selected
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Footer/status

    # --- Screen Helpers ---
    def draw_header(self, title: str):
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(0, 0, " " * (curses.COLS - 1))
        self.stdscr.addstr(0, 2, title)
        self.stdscr.attroff(curses.color_pair(1))
        self.stdscr.refresh()

    def show_footer(self, message: str, color_pair: int = 3):
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.attron(curses.color_pair(color_pair))
        self.stdscr.addstr(max_y - 1, 0, message[:max_x - 1])
        self.stdscr.clrtoeol()
        self.stdscr.attroff(curses.color_pair(color_pair))
        self.stdscr.refresh()

    def pause_message(self, message: str = "Press any key to continue..."):
        self.show_footer(message, color_pair=3)
        self.stdscr.getch()

    def clear_screen_area(self):
        """Clear the area between header and footer."""
        max_y = curses.LINES
        self.stdscr.clear()
        self.stdscr.bkgd(' ', curses.color_pair(0))
        self.stdscr.refresh()
        for line in range(1, max_y - 2):
            self.stdscr.move(line, 0)
            self.stdscr.clrtoeol()

    def prompt_input(self, prompt_str: str, y: int, x: int) -> str:
        self.stdscr.addstr(y, x, prompt_str)
        self.stdscr.clrtoeol()
        curses.echo()
        try:
            inp = self.stdscr.getstr(y, x + len(prompt_str)).decode("utf-8").strip()
        except Exception as e:
            logging.error("Error reading input: %s", e)
            inp = ""
        curses.noecho()
        return inp

    # --- Workout Session UI ---
    def record_exercise_session(self, exercise: dict) -> dict:
        """
        Record a workout session for an exercise.
        Returns a dictionary containing the exercise id and the recorded sets.
        """
        self.clear_screen_area()
        header = f"Workout for: {exercise.get('title','')}"
        if exercise.get("planned"):
            header += " [PLANNED]"
        self.draw_header(header)
        set_number = 1
        sets_recorded = []
        last_weight = ""
        max_y, max_x = self.stdscr.getmaxyx()

        # Setup summary window for recorded sets.
        summary_height = 7
        summary_width = max_x - 6
        summary_y = 2
        summary_x = 3
        summary_win = self.stdscr.subwin(summary_height, summary_width, summary_y, summary_x)
        summary_win.box()
        summary_win.attron(curses.A_BOLD)
        summary_win.addstr(0, 1, " Recorded Sets ")
        summary_win.attroff(curses.A_BOLD)
        summary_win.refresh()

        input_start_row = summary_y + summary_height + 1
        while True:
            self.stdscr.addstr(input_start_row, 2, f"Set #{set_number}: (Leave 'Reps' empty to finish)")
            self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
            reps = self.prompt_input("  Reps: ", input_start_row + 1, 4)
            self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
            if not reps:
                break
            prompt = f"  Weight [{last_weight}]: " if last_weight else "  Weight: "
            self.stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
            weight = self.prompt_input(prompt, input_start_row + 2, 4)
            self.stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
            if not weight and last_weight:
                weight = last_weight
            else:
                last_weight = weight
            current_set = {"reps": reps, "weight": weight}
            sets_recorded.append(current_set)

            # Update summary window.
            summary_win.clear()
            summary_win.box()
            summary_win.attron(curses.A_BOLD)
            summary_win.addstr(0, 1, " Recorded Sets ")
            summary_win.attroff(curses.A_BOLD)
            for idx, s in enumerate(sets_recorded, start=1):
                if idx < summary_height - 1:
                    summary_win.addstr(idx, 1, f"Set {idx}: {s['reps']} reps @ {s['weight']} wt")
            summary_win.refresh()

            self.stdscr.addstr(input_start_row+4, 4, f"Recorded: {reps} reps @ {weight}")
            self.stdscr.refresh()
            set_number += 1
            input_start_row += 6
            if input_start_row > curses.LINES - 8:
                self.pause_message("Press any key to continue recording sets...")
                self.clear_screen_area()
                self.draw_header(header)
                input_start_row = summary_y + summary_height + 1
                summary_win.mvwin(summary_y, summary_x)
                summary_win.refresh()
        self.pause_message("Exercise sets recorded. Press any key to continue...")
        # Return a dictionary with "id" for later session lookup.
        return {"id": exercise["filename"], "sets": sets_recorded}

    def draw_session_preview(self, session_exercises: list):
        max_y, max_x = self.stdscr.getmaxyx()
        preview_height = max_y // 2
        session_win = self.stdscr.subwin(preview_height, max_x, max_y - preview_height - 1, 0)
        session_win.box()
        session_win.attron(curses.A_BOLD)
        session_win.addstr(0, 2, " Session Preview ")
        session_win.attroff(curses.A_BOLD)
        line = 1
        # Check each exercise – if it contains session details ("id" and "sets"), show them;
        # otherwise, simply display the title.
        if not session_exercises:
            session_win.addstr(line, 2, " (No exercises recorded yet)")
        else:
            for ex in session_exercises:
                if "id" in ex and "sets" in ex:
                    display_line = f"• {ex['id']}: {len(ex['sets'])} set(s)"
                else:
                    display_line = f"• {ex.get('title','')}"
                session_win.addstr(line, 2, display_line)
                line += 1
        session_win.refresh()

    def record_workout_session(self, prepopulated_exercises: list = None):
        """
        Record a workout session.
        If prepopulated_exercises (list of exercise dicts) is provided, use them first.
        """
        self.clear_screen_area()
        self.draw_header("Start Workout Session")
        self.pause_message("Press any key to begin your workout...")
        session_exercises = []
        if prepopulated_exercises:
            for ex in prepopulated_exercises:
                rec = self.record_exercise_session(ex)
                if rec["sets"]:
                    session_exercises.append(rec)
            ans = self.prompt_input("Add another exercise? (y/N): ", curses.LINES - 4, 2)
            if ans.lower() == "y":
                while True:
                    ex = self.select_exercise(session_exercises)
                    if ex is None:
                        break
                    rec = self.record_exercise_session(ex)
                    if rec["sets"]:
                        session_exercises.append(rec)
                    self.clear_screen_area()
                    self.draw_header("Recording Workout Session")
                    self.draw_session_preview(session_exercises)
                    ans = self.prompt_input("Add another exercise? (y/N): ", curses.LINES - 4, 2)
                    if ans.lower() != "y":
                        break
        else:
            while True:
                ex = self.select_exercise(session_exercises)
                if ex is None:
                    break
                rec = self.record_exercise_session(ex)
                if rec["sets"]:
                    session_exercises.append(rec)
                self.clear_screen_area()
                self.draw_header("Recording Workout Session")
                self.draw_session_preview(session_exercises)
                ans = self.prompt_input("Add another exercise? (y/N): ", curses.LINES - 4, 2)
                if ans.lower() != "y":
                    break

        if not session_exercises:
            self.show_footer("No exercises recorded for this session.", color_pair=3)
            self.pause_message()
            return
        new_filename = self.dm.write_workout_note(session_exercises)
        self.show_footer("Workout session saved.", color_pair=3)
        self.pause_message("Session complete. Press any key to return to main menu...")

    # --- Exercise Selection UI ---
    def select_exercise(self, session_exercises: list) -> dict:
        """
        Present a list of exercises (from index) for selection.
        The bottom preview shows the already selected exercises.
        """
        cursor_pos = 0
        while True:
            exercises = self.dm.load_index_exercises()
            max_y, max_x = self.stdscr.getmaxyx()
            preview_height = max_y // 3
            list_height = max_y - preview_height - 3
            self.clear_screen_area()
            self.draw_header("Select an Exercise")

            col_width_title = max_x - 35
            col_width_status = 10
            header_line = f"{'Title':<{col_width_title}} {'Status':<{col_width_status}} Equipment"
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
            self.stdscr.addstr(1, 2, header_line[:max_x-4])
            self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))

            for idx, ex in enumerate(exercises[:list_height-2]):
                status_label = "Planned" if ex.get("planned") else ""
                eq = ex.get("equipment", [])
                if not isinstance(eq, (list, tuple)):
                    eq = [str(eq)]
                equipment = ", ".join(eq)
                line = f"{ex.get('title',''):<{col_width_title}} {status_label:<{col_width_status}} {equipment}"
                display_line = line[:max_x-4]
                if idx == cursor_pos:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(2 + idx, 2, display_line)
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(2 + idx, 2, display_line)

            key_hints = "↑/↓/j/k: Navigate | Enter: Select | P: Toggle Planned | Q: Back to Menu"
            self.show_footer(key_hints, color_pair=3)

            preview_win = self.stdscr.subwin(preview_height, max_x, max_y - preview_height - 1, 0)
            preview_win.box()
            split_y = preview_height // 2
            top_win = preview_win.derwin(split_y - 1, max_x - 2, 1, 1)
            top_win.clear()
            top_win.box()
            top_win.attron(curses.A_BOLD)
            top_win.addstr(0, 2, " Exercise Preview ")
            top_win.attroff(curses.A_BOLD)
            if exercises:
                selected_ex = exercises[cursor_pos]
                pline = 1
                top_win.addstr(pline, 2, f"Title: {selected_ex.get('title','')}")
                pline += 1
                eq = selected_ex.get("equipment", [])
                if not isinstance(eq, (list, tuple)):
                    eq = [str(eq)]
                equipment = ", ".join(eq)
                top_win.addstr(pline, 2, f"Equipment: {equipment}")
                pline += 1
                status = "Planned" if selected_ex.get("planned") else "Not planned"
                top_win.addstr(pline, 2, f"Status: {status}")
                pline += 1
                top_win.addstr(pline, 2, "Recent History:")
                pline += 1
                history = self.dm.get_workout_history_for_exercise(selected_ex["filename"])
                if history:
                    for h in history[:(split_y - pline - 1)]:
                        top_win.addstr(pline, 2, f"{h['date']}: {h['sets']} set(s), avg reps: {h['avg_reps']:.1f}, wt: {h['avg_weight']:.1f}")
                        pline += 1
                        if pline >= split_y - 2:
                            break
                else:
                    top_win.addstr(pline, 2, "No previous sessions.")
            top_win.refresh()

            bottom_win = preview_win.derwin(preview_height - split_y - 2, max_x - 2, split_y + 1, 1)
            bottom_win.clear()
            bottom_win.box()
            bottom_win.attron(curses.A_BOLD)
            bottom_win.addstr(0, 2, " Session Preview ")
            bottom_win.attroff(curses.A_BOLD)
            rline = 1
            if not session_exercises:
                bottom_win.addstr(rline, 2, " (No exercises recorded yet)")
            else:
                for ex in session_exercises:
                    if "id" in ex and "sets" in ex:
                        display_line = f"• {ex['id']}: {len(ex['sets'])} set(s)"
                    else:
                        display_line = f"• {ex.get('title','')}"
                    bottom_win.addstr(rline, 2, display_line[:max_x-4])
                    rline += 1
            bottom_win.refresh()
            preview_win.refresh()
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor_pos > 0:
                    cursor_pos -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if exercises and cursor_pos < len(exercises)-1:
                    cursor_pos += 1
            elif k in (10, 13):
                return exercises[cursor_pos]
            elif k in (ord('p'), ord('P')):
                if exercises:
                    ex = exercises[cursor_pos]
                    new_state = self.dm.toggle_planned_status(ex["filename"])
                    ex["planned"] = new_state
                    state_str = "Planned" if new_state else "Not planned"
                    self.show_footer(f"Exercise '{ex.get('title','')}' toggled to {state_str}.", color_pair=3)
                    self.pause_message("Press any key to continue...")
            elif k in (ord('q'), ord('Q')):
                return None

    # --- Workout Template UI (Tag-based) ---
    def create_workout_template(self):
        """
        Build a workout template by choosing exercises (using the index) and
        entering a template name and optional description.
        """
        self.clear_screen_area()
        self.draw_header("Create Workout Template")
        tmpl_name = self.prompt_input("Enter template name: ", 4, 4)
        if not tmpl_name:
            self.show_footer("No template name provided. Cancelled.", color_pair=3)
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
        tmpl_file = self.dm.write_workout_template(tmpl_name, tmpl_desc, exercises)
        self.show_footer(f"Template '{tmpl_name}' created.", color_pair=3)
        self.pause_message("Press any key to return to main menu...")

    def list_workout_templates(self) -> list:
        """
        Display a list of available workout templates.
        Returns the selected template dictionary or None if cancelled.
        """
        templates = self.dm.load_workout_templates()
        if not templates:
            self.show_footer("No workout templates available.", color_pair=3)
            self.pause_message()
            return None
        cursor_pos = 0
        while True:
            self.clear_screen_area()
            self.draw_header("Select Workout Template")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, tmpl in enumerate(templates):
                y_pos = 3 + idx
                display_text = f"{tmpl['title']}: {tmpl['description']}"
                if y_pos < max_y - 2:
                    if idx == cursor_pos:
                        self.stdscr.attron(curses.color_pair(2))
                        self.stdscr.addstr(y_pos, 2, display_text[:max_x-4])
                        self.stdscr.attroff(curses.color_pair(2))
                    else:
                        self.stdscr.addstr(y_pos, 2, display_text[:max_x-4])
            self.show_footer("↑/↓: Navigate | Enter: Select | Q: Cancel", color_pair=3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor_pos > 0:
                    cursor_pos -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor_pos < len(templates) - 1:
                    cursor_pos += 1
            elif k in (10, 13):
                return templates[cursor_pos]
            elif k in (ord('q'), ord('Q')):
                return None

    def start_workout_from_template(self):
        """
        Let the user choose a workout template and start a session
        pre-populated with the template's exercises.
        """
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
        self.show_footer(f"Starting workout from template '{tmpl['title']}'.", color_pair=3)
        self.pause_message("Press any key to continue...")
        self.record_workout_session(prepopulated)

    def edit_workout_template(self):
        """
        Allow the user to select and edit an existing workout template.
        """
        tmpl = self.list_workout_templates()
        if not tmpl:
            return
        self.clear_screen_area()
        self.draw_header("Edit Workout Template")
        new_name = self.prompt_input(f"Enter new name [{tmpl['title']}]: ", 4, 4)
        if not new_name:
            new_name = tmpl["title"]
        new_desc = self.prompt_input(f"Enter new description [{tmpl['description']}]: ", 5, 4)
        if not new_desc:
            new_desc = tmpl["description"]
        self.stdscr.addstr(7, 4, "Rebuild exercise list? (y/N): ")
        k = self.stdscr.getch()
        exercises = []
        if chr(k).lower() == 'y':
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
        """
        Let the user select a workout template and confirm its deletion.
        """
        tmpl = self.list_workout_templates()
        if not tmpl:
            return
        self.clear_screen_area()
        self.draw_header("Delete Workout Template")
        confirm = self.prompt_input(f"Delete template '{tmpl['title']}'? (y/N): ", 4, 4)
        if confirm.lower() == "y":
            self.dm.delete_workout_template(tmpl["filename"])
            self.show_footer(f"Template '{tmpl['title']}' deleted.", color_pair=3)
        else:
            self.show_footer("Deletion cancelled.", color_pair=3)
        self.pause_message("Press any key to return to main menu...")

    # --- Main Menu ---
    def main_menu(self):
        menu_options = [
            "Record a Workout Session",
            "Create a New Exercise",
            "Create Workout Template",
            "Start Workout from Template",
            "Edit Workout Template",
            "Delete Workout Template",
            "Quit"
        ]
        menu_cursor_pos = 0
        while True:
            self.clear_screen_area()
            max_y, _ = self.stdscr.getmaxyx()
            self.draw_header("Workout Logger – Main Menu")
            for idx, opt in enumerate(menu_options):
                y_pos = 3 + idx * 2
                if idx == menu_cursor_pos:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y_pos, 4, f"> {opt}")
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y_pos, 4, f"  {opt}")
            self.show_footer("↑/↓/j/k: Navigate | Enter: Select | Q: Quit", color_pair=3)
            self.stdscr.refresh()
            key = self.stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                menu_cursor_pos = max(0, menu_cursor_pos - 1)
            elif key in (curses.KEY_DOWN, ord('j')):
                menu_cursor_pos = min(len(menu_options) - 1, menu_cursor_pos + 1)
            elif key in (10, 13):
                if menu_cursor_pos == 0:
                    self.record_workout_session()
                elif menu_cursor_pos == 1:
                    self.create_new_exercise()
                elif menu_cursor_pos == 2:
                    self.create_workout_template()
                elif menu_cursor_pos == 3:
                    self.start_workout_from_template()
                elif menu_cursor_pos == 4:
                    self.edit_workout_template()
                elif menu_cursor_pos == 5:
                    self.delete_workout_template()
                elif menu_cursor_pos == 6:
                    break
            elif key in (ord('q'), ord('Q')):
                break

    # --- New Exercise UI ---
    def create_new_exercise(self):
        self.clear_screen_area()
        self.draw_header("Create New Exercise")
        title = self.prompt_input("Enter exercise title: ", 4, 4)
        equipment = self.prompt_input("Enter comma separated equipment: ", 5, 4)
        if title:
            filename = self.dm.write_exercise_file(title, equipment)
            self.show_footer(f"New exercise '{title}' created.", color_pair=3)
            self.pause_message("Exercise created successfully. Press any key to continue...")
            return filename
        else:
            self.show_footer("No title provided. Exercise creation cancelled.", color_pair=3)
            self.pause_message("Press any key to return to main menu...")
            return ""

# === Main Entrypoint ===
def main(stdscr):
    parser = argparse.ArgumentParser(description="Workout Logging Script")
    parser.add_argument('--notes-dir', type=str,
                        help='Specify the notes directory path. Overrides NOTES_DIR environment variable.')
    args = parser.parse_args()

    notes_dir_cli = args.notes_dir
    notes_dir_env = os.environ.get("NOTES_DIR")

    if notes_dir_cli:
        NOTES_DIR_ENV = notes_dir_cli
    elif notes_dir_env:
        NOTES_DIR_ENV = notes_dir_env
    else:
        print("Error: NOTES_DIR environment variable or --notes-dir flag must be set.")
        sys.exit(1)

    NOTES_DIR = Path(NOTES_DIR_ENV)
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
            logging.exception("An error occurred while running the Workout Logger.")
            print(f"An error occurred. Check the log file {LOG_FILE} for details.")
        else:
            print("An error occurred. Check the log for details.")
        sys.exit(1)


