#!/usr/bin/env python3
"""
User-Friendly & Feature-Rich Workout Logger

This tool helps you log your workouts easily. You can record sessions,
manage your exercise library, create and use workout templates, view
your workout history, and export data to CSV.

Usage:
    $ export NOTES_DIR=/path/to/notes
    $ ./workout_log.py [--verbose]
    OR
    $ ./workout_log.py --notes-dir /path/to/notes [--export-history] [--verbose]

Command Line Options:
    --notes-dir       Specify the path to your notes directory.
    --export-history  Export all workout sessions to CSV and exit.
    --verbose         Enable detailed logging.
"""

import argparse
import curses
import csv
import datetime
import json
import logging
import os
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from logging.handlers import RotatingFileHandler

# --- Constants ---
FRONTMATTER_DELIMITER = "---"
DATE_KEY = "date"
DATETIME_KEY = "dateCreated"
MODIFIED_KEY = "dateModified"
PLANNED_KEY = "planned_exercise"
TEMPLATE_TAG = "workout_template"

# Set up module logger
logger = logging.getLogger(__name__)

# --- Utility Functions ---
def generate_unique_id() -> str:
    """Generate a unique ID using the current date and three random letters."""
    today = datetime.datetime.now().strftime("%y%m%d")
    rand = "".join(random.choices(string.ascii_lowercase, k=3))
    uid = today + rand
    logger.debug("Generated unique ID: %s", uid)
    return uid

def get_current_iso(with_seconds: bool = True) -> str:
    """Return the current date/time in ISO format."""
    fmt = "%Y-%m-%dT%H:%M:%S" if with_seconds else "%Y-%m-%dT%H:%M"
    iso_time = datetime.datetime.now().strftime(fmt)
    logger.debug("Current ISO time: %s", iso_time)
    return iso_time

def draw_box(win: Any, title: str = "") -> None:
    """Draw a border around a curses window and, if provided, add a title."""
    win.box()
    if title:
        try:
            win.addstr(0, 2, f" {title} ", curses.A_BOLD)
        except Exception as err:
            logger.exception("Error drawing box title: %s", err)

# --- Data Management Class ---
class DataManager:
    """
    Handles file operations: reading/writing markdown notes (with YAML frontmatter),
    managing exercises, workout sessions, and workout templates.
    """

    def __init__(self, notes_dir: Path, index_file: Path, log_file: Path):
        self.notes_dir = notes_dir
        self.index_file = index_file
        self.log_file = log_file
        self._index_cache: Optional[List[Dict[str, Any]]] = None
        self._index_mtime: Optional[float] = None
        self._workout_history_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
        logger.debug("DataManager initialized with notes dir: %s", self.notes_dir)

    def read_file(self, filepath: Path) -> Tuple[Dict[str, Any], str]:
        """Read a markdown file with YAML frontmatter."""
        try:
            text = filepath.read_text(encoding="utf-8")
            logger.debug("Read file: %s", filepath)
        except Exception as e:
            logger.error("Could not read file %s: %s", filepath, e)
            return {}, ""
        if not text.startswith(FRONTMATTER_DELIMITER):
            return {}, text
        parts = text.split(FRONTMATTER_DELIMITER, 2)
        if len(parts) < 3:
            return {}, text
        try:
            fm = yaml.safe_load(parts[1]) or {}
            logger.debug("Parsed YAML frontmatter from %s", filepath)
        except Exception as err:
            logger.error("Error parsing frontmatter in %s: %s", filepath, err)
            fm = {}
        body = parts[2].lstrip("\n")
        return fm, body

    def write_file(self, filepath: Path, frontmatter: Dict[str, Any], body: str = "") -> None:
        """Write a markdown file with YAML frontmatter."""
        try:
            fm_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error("Error dumping YAML for %s: %s", filepath, e)
            fm_text = ""
        content = f"{FRONTMATTER_DELIMITER}\n{fm_text}{FRONTMATTER_DELIMITER}\n\n{body}"
        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info("Saved file: %s", filepath)
        except Exception as e:
            logger.error("Error writing file %s: %s", filepath, e)

    def update_modified_date(self, filepath: Path) -> None:
        """Update the modified date in a file's frontmatter."""
        fm, body = self.read_file(filepath)
        fm[MODIFIED_KEY] = get_current_iso()
        self.write_file(filepath, fm, body)
        logger.debug("Updated modified date for: %s", filepath)

    def is_planned(self, filepath: Path) -> bool:
        """Return the planned status from a note's frontmatter."""
        fm, _ = self.read_file(filepath)
        val = fm.get(PLANNED_KEY, False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() == "true"
        return False

    def create_exercise(self, title: str, equipment: str) -> str:
        """
        Create a new exercise note with a title and equipment.
        Returns the created filename.
        """
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        equipment_list = [item.strip() for item in equipment.split(",") if item.strip()] or ["none"]
        frontmatter = {
            "title": title,
            "tags": ["exercise"],
            DATE_KEY: today_str,
            DATETIME_KEY: iso_time,
            PLANNED_KEY: False,
            "exercise_equipment": equipment_list,
        }
        self.write_file(filepath, frontmatter)
        self._index_cache = None  # Invalidate cache
        logger.info("Created exercise: %s", filename)
        return filename

    def toggle_exercise_planned(self, filename: str) -> bool:
        """
        Toggle the planned status of an exercise note.
        Returns the new status.
        """
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = self.notes_dir / filename
        fm, body = self.read_file(filepath)
        current = fm.get(PLANNED_KEY, False)
        new_status = (not current) if isinstance(current, bool) else (current.lower() != "true")
        fm[PLANNED_KEY] = new_status
        fm[MODIFIED_KEY] = get_current_iso()
        self.write_file(filepath, fm, body)
        self._index_cache = None  # Invalidate cache
        logger.info("Toggled planned status for %s to %s", filename, new_status)
        return self.is_planned(filepath)

    def load_exercise_index(self) -> List[Dict[str, Any]]:
        """
        Load the exercise index from index.json.
        Each exercise is enriched with planned status and equipment info.
        """
        try:
            stat = self.index_file.stat()
        except Exception as err:
            logger.error("Error accessing index file %s: %s", self.index_file, err)
            return []

        if self._index_cache is None or stat.st_mtime != self._index_mtime:
            try:
                data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
                exercises = []
                for note in data:
                    tags = note.get("tags") or []
                    if "exercise" in tags:
                        filename = note.get("filename")
                        filepath = self.notes_dir / (filename + ".md")
                        planned = self.is_planned(filepath) if filepath.exists() else False
                        exercises.append({
                            "filename": filename,
                            "title": note.get("title", filename),
                            "planned": planned,
                            "equipment": note.get("exercise_equipment", ["none"]),
                        })
                self._index_cache = exercises
                self._index_mtime = stat.st_mtime
            except Exception as err:
                logger.error("Error loading index from %s: %s", self.index_file, err)
                return []
        return self._index_cache

    def save_workout_session(self, exercises: List[Dict[str, Any]]) -> str:
        """
        Create and save a workout session note.
        Returns the filename of the saved session.
        """
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        frontmatter = {
            "zettelid": uid,
            "title": f"Workout Session on {today_str}",
            "tags": ["workout"],
            DATE_KEY: today_str,
            DATETIME_KEY: iso_time,
            "exercises": exercises,
        }
        self.write_file(filepath, frontmatter)
        self._workout_history_cache = None
        logger.info("Saved workout session: %s", filename)
        return filename

    def extract_frontmatter(self, text: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from a workout note text."""
        parts = text.split(FRONTMATTER_DELIMITER, 2)
        if len(parts) < 3:
            return {}
        try:
            return yaml.safe_load(parts[1]) or {}
        except Exception as err:
            logger.error("Error parsing workout YAML: %s", err)
            return {}

    def build_history_cache(self) -> None:
        """Build a cache of workout sessions by scanning all notes."""
        self._workout_history_cache = {}
        for md_file in self.notes_dir.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as err:
                logger.error("Error reading file %s: %s", md_file, err)
                continue
            fm = self.extract_frontmatter(text)
            if not fm or "workout" not in (fm.get("tags") or []):
                continue
            session_date = fm.get("date", "unknown")
            for ex in fm.get("exercises", []):
                sets = ex.get("sets", [])
                num_sets = len(sets)
                total_reps, total_weight, valid = 0, 0.0, 0
                for s in sets:
                    try:
                        reps = int(s.get("reps", 0))
                        weight = float(s.get("weight", 0))
                        total_reps += reps
                        total_weight += weight
                        valid += 1
                    except Exception as err:
                        logger.error("Conversion error in %s: %s", md_file, err)
                avg_reps = total_reps / valid if valid else 0
                avg_weight = total_weight / valid if valid else 0
                entry = {
                    "date": session_date,
                    "sets": num_sets,
                    "avg_reps": avg_reps,
                    "avg_weight": avg_weight,
                }
                ex_id = ex.get("filename")
                if ex_id:
                    self._workout_history_cache.setdefault(ex_id, []).append(entry)
        if self._workout_history_cache:
            for ex_id in self._workout_history_cache:
                self._workout_history_cache[ex_id].sort(key=lambda x: x.get("date", ""), reverse=True)
            logger.debug("Workout history cache built with %d exercises", len(self._workout_history_cache))

    def get_history_for_exercise(self, ex_id: str) -> List[Dict[str, Any]]:
        """Return the workout history for a given exercise."""
        if self._workout_history_cache is None:
            self.build_history_cache()
        return self._workout_history_cache.get(ex_id, [])

    def list_workout_sessions(self) -> List[Dict[str, Any]]:
        """
        List all workout sessions (notes tagged with 'workout'),
        sorted by date (newest first).
        """
        sessions = []
        for md_file in self.notes_dir.glob("*.md"):
            fm, _ = self.read_file(md_file)
            if not fm or "workout" not in (fm.get("tags") or []):
                continue
            sessions.append({
                "filename": md_file.name,
                "date": fm.get("date", "unknown"),
                "title": fm.get("title", md_file.stem),
                "exercises": fm.get("exercises", []),
            })
        sessions.sort(key=lambda s: s.get("date", ""), reverse=True)
        logger.debug("Found %d workout sessions", len(sessions))
        return sessions

    def create_workout_template(self, name: str, description: str, exercises: List[Dict[str, Any]]) -> str:
        """
        Create a workout template using the provided name, description, and exercise list.
        Returns the filename of the new template.
        """
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        frontmatter = {
            "title": name,
            "description": description,
            DATE_KEY: today_str,
            DATETIME_KEY: iso_time,
            MODIFIED_KEY: iso_time,
            "exercises": [
                {"exercise_filename": ex["filename"], "order": idx + 1}
                for idx, ex in enumerate(exercises)
            ],
            "tags": [TEMPLATE_TAG],
        }
        body = f"# {name}\n\n{description}"
        self.write_file(filepath, frontmatter, body)
        logger.info("Created workout template: %s", filename)
        return filename

    def load_templates(self) -> List[Dict[str, Any]]:
        """
        Scan for workout template notes (tagged with TEMPLATE_TAG)
        and return a list of templates.
        """
        templates = []
        for md_file in self.notes_dir.glob("*.md"):
            fm, body = self.read_file(md_file)
            if not fm or TEMPLATE_TAG not in (fm.get("tags") or []):
                continue
            templates.append({
                "filename": md_file.name,
                "title": fm.get("title", md_file.stem),
                "description": fm.get("description", ""),
                "exercises": fm.get("exercises", []),
            })
        logger.debug("Loaded %d workout templates", len(templates))
        return templates

    def update_template(self, tmpl_filename: str, name: str, description: str, exercises: List[Dict[str, Any]]) -> None:
        """Update an existing workout template."""
        filepath = self.notes_dir / tmpl_filename
        if not filepath.exists():
            logger.error("Template %s not found", tmpl_filename)
            return
        iso_time = get_current_iso(with_seconds=False)
        fm, _ = self.read_file(filepath)
        fm["title"] = name
        fm["description"] = description
        fm[MODIFIED_KEY] = iso_time
        fm["exercises"] = [
            {"exercise_filename": ex["filename"], "order": idx + 1}
            for idx, ex in enumerate(exercises)
        ]
        body = f"# {name}\n\n{description}"
        self.write_file(filepath, fm, body)
        logger.info("Updated template: %s", tmpl_filename)

    def delete_template(self, tmpl_filename: str) -> None:
        """Delete a workout template note."""
        filepath = self.notes_dir / tmpl_filename
        try:
            filepath.unlink()
            logger.info("Deleted template: %s", tmpl_filename)
        except Exception as err:
            logger.error("Error deleting template %s: %s", tmpl_filename, err)

    def export_history(self, export_path: Path) -> None:
        """Export all workout sessions to a CSV file."""
        sessions = self.list_workout_sessions()
        if not sessions:
            logger.info("No workout sessions found for export.")
            return
        try:
            with export_path.open("w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["Session Date", "Session Title", "Exercise Title", "Sets", "Avg Reps", "Avg Weight"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for session in sessions:
                    for ex in session.get("exercises") or []:
                        exercise_title = ""
                        for ex_item in self.load_exercise_index():
                            if ex_item["filename"] == ex.get("id"):
                                exercise_title = ex_item["title"]
                                break
                        writer.writerow({
                            "Session Date": session.get("date", ""),
                            "Session Title": session.get("title", ""),
                            "Exercise Title": exercise_title,
                            "Sets": len(ex.get("sets", [])),
                            "Avg Reps": (sum(int(s.get("reps", 0)) for s in ex.get("sets", [])) / len(ex.get("sets", []))) if ex.get("sets") else 0,
                            "Avg Weight": (sum(float(s.get("weight", 0)) for s in ex.get("sets", [])) / len(ex.get("sets", []))) if ex.get("sets") else 0,
                        })
            logger.info("Workout history exported to %s", export_path)
        except Exception as err:
            logger.exception("Failed to export workout history: %s", err)

# --- UI Manager (curses based) ---
class UIManager:
    """
    Manages the terminal UI using curses.
    Provides menus, data entry, integrated preview, and feedback.
    """

    def __init__(self, stdscr: Any, data_manager: DataManager):
        self.stdscr = stdscr
        self.dm = data_manager
        curses.curs_set(1)
        self.stdscr.nodelay(False)
        self.stdscr.keypad(True)
        self.setup_colors()
        logger.debug("UI Manager initialized.")

    def setup_colors(self) -> None:
        """Set up color pairs for the UI."""
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)     # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)     # Highlight
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Footer/Status
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Normal text

    def draw_header(self, title: str) -> None:
        """Display a header across the top of the screen."""
        self.stdscr.attron(curses.color_pair(1))
        try:
            self.stdscr.addstr(0, 0, " " * (curses.COLS - 1))
            self.stdscr.addstr(0, 2, title)
        except Exception:
            pass
        self.stdscr.attroff(curses.color_pair(1))
        self.stdscr.refresh()

    def show_footer(self, message: str, color_pair: int = 3) -> None:
        """Show a message in the footer."""
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.attron(curses.color_pair(color_pair))
        self.stdscr.addstr(max_y - 1, 0, message[:max_x - 1])
        self.stdscr.clrtoeol()
        self.stdscr.attroff(curses.color_pair(color_pair))
        self.stdscr.refresh()

    def pause(self, prompt: str = "Press any key to continue...") -> None:
        """Display a pause message and wait for a key press."""
        self.show_footer(prompt, color_pair=3)
        self.stdscr.getch()

    def clear_screen(self) -> None:
        """Clear the screen."""
        self.stdscr.clear()
        self.stdscr.bkgd(" ", curses.color_pair(4))
        self.stdscr.refresh()

    def prompt_input(self, prompt: str, y: int, x: int, default: str = "") -> str:
        """
        Prompt for input at a given coordinate.
        Returns the entered text or default.
        """
        self.stdscr.addstr(y, x, prompt)
        self.stdscr.clrtoeol()
        curses.echo()
        try:
            inp = self.stdscr.getstr(y, x + len(prompt)).decode("utf-8").strip()
        except Exception as err:
            logger.error("Input error: %s", err)
            inp = ""
        curses.noecho()
        return inp if inp else default

    def record_exercise(self, exercise: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record sets for a given exercise.
        Returns a dictionary with the exercise id, title, and set details.
        """
        self.clear_screen()
        header = f"Recording Workout for: {exercise.get('title', '')}"
        if exercise.get("planned"):
            header += " [PLANNED]"
        self.draw_header(header)
        logger.info("Recording exercise: %s", exercise.get("title", ""))
        set_number = 1
        recorded_sets: List[Dict[str, Any]] = []
        last_weight = ""
        max_y, max_x = self.stdscr.getmaxyx()

        # Create a window for set summary
        sum_h = 7
        sum_w = max_x - 6
        sum_y = 2
        sum_x = 3
        summary_win = curses.newwin(sum_h, sum_w, sum_y, sum_x)
        draw_box(summary_win, " Recorded Sets ")
        summary_win.refresh()

        input_y = sum_y + sum_h + 1
        while True:
            self.stdscr.addstr(input_y, 2, f"Set #{set_number} (Leave 'Reps' blank to finish)")
            reps = self.prompt_input(" Reps: ", input_y + 1, 4)
            if not reps:
                break
            weight_prompt = f" Weight [{last_weight}]: " if last_weight else " Weight: "
            weight = self.prompt_input(weight_prompt, input_y + 2, 4)
            if not weight and last_weight:
                weight = last_weight
            else:
                last_weight = weight
            recorded_sets.append({"reps": reps, "weight": weight})
            summary_win.erase()
            draw_box(summary_win, " Recorded Sets ")
            for idx, s in enumerate(recorded_sets, start=1):
                if idx < sum_h - 1:
                    summary_win.addstr(idx, 2, f"Set {idx}: {s['reps']} reps @ {s['weight']} weight")
            summary_win.refresh()
            set_number += 1
            input_y += 4
            if input_y > max_y - 6:
                self.pause("Press any key to continue recording sets...")
                self.clear_screen()
                self.draw_header(header)
                input_y = sum_y + sum_h + 1
                summary_win.mvwin(sum_y, sum_x)
                summary_win.refresh()
        self.pause("Exercise sets recorded. Press any key to continue...")
        logger.debug("Recorded %d sets for %s", len(recorded_sets), exercise.get("title", ""))
        return {"id": exercise["filename"], "title": exercise.get("title"), "sets": recorded_sets}

    def record_session(self, prepopulated: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Record a full workout session.
        Uses the same exercise–selection logic whether from a template or full list.
        """
        self.clear_screen()
        self.draw_header("Start Workout Session")
        self.pause("Get ready! Press any key to begin your workout...")
        session_exercises: List[Dict[str, Any]] = []

        while True:
            ex = self.choose_exercise(session_exercises, exercises=prepopulated)
            if ex is None:
                break
           

