#!/usr/bin/env python3
"""
Ultimate Feature-Rich & User-Friendly Workout Logger

This tool lets you log workouts, manage exercises (create, edit, delete, search),
manage workout templates, view & filter history, view aggregated statistics,
delete sessions, backup data, and export data in CSV and JSON.
It auto-creates directories/files if needed and provides an advanced curses-based UI.

It uses an external index (index.json) produced by zk_index.py for reading information.
New workouts, exercises, and templates are written as markdown files.
The index file is never written to by this script.

Usage:
    $ export NOTES_DIR=/path/to/notes
    $ ./workout_log.py [--verbose] [--export-history] [--export-json]
    OR
    $ ./workout_log.py --notes-dir /path/to/notes [--export-history] [--verbose] [--export-json]
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
import shutil
import yaml
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field

# --- Data Classes ---
@dataclass
class Exercise:
    filename: str
    title: str
    planned: bool = False
    equipment: List[str] = field(default_factory=lambda: ["none"])

@dataclass
class WorkoutSet:
    reps: int
    weight: float

@dataclass
class WorkoutExercise:
    id: str
    title: str  # Now we save the title so it can be displayed in session summaries.
    sets: List[WorkoutSet]

@dataclass
class WorkoutSession:
    filename: str
    date: str
    title: str
    exercises: List[WorkoutExercise]

@dataclass
class WorkoutTemplate:
    filename: str
    title: str
    description: str
    exercises: List[str]  # list of exercise filenames

# --- Utility Functions ---
def generate_unique_id() -> str:
    """Generate a unique ID using today's date and three random letters."""
    today = datetime.datetime.now().strftime("%y%m%d")
    rand = "".join(random.choices(string.ascii_lowercase, k=3))
    uid = today + rand
    logging.getLogger(__name__).debug("Generated unique ID: %s", uid)
    return uid

def get_current_iso(with_seconds: bool = True) -> str:
    """Return the current ISO-formatted date/time."""
    fmt = "%Y-%m-%dT%H:%M:%S" if with_seconds else "%Y-%m-%dT%H:%M"
    iso_time = datetime.datetime.now().strftime(fmt)
    logging.getLogger(__name__).debug("Current ISO time: %s", iso_time)
    return iso_time

def draw_box(win: Any, title: str = "") -> None:
    """Draw a border around a curses window and optionally add a title."""
    win.box()
    if title:
        try:
            win.addstr(0, 2, f" {title} ", curses.A_BOLD)
        except Exception as err:
            logging.getLogger(__name__).exception("Error drawing box title: %s", err)

# --- Modal & Flash Helpers ---
def modal_confirm(stdscr: Any, message: str) -> bool:
    """Display a centered modal asking the user to confirm (Y/N)."""
    max_y, max_x = stdscr.getmaxyx()
    modal_h, modal_w = 7, min(60, max_x - 4)
    begin_y = (max_y - modal_h) // 2
    begin_x = (max_x - modal_w) // 2
    win = curses.newwin(modal_h, modal_w, begin_y, begin_x)
    draw_box(win, " Confirmation ")
    win.addstr(2, 2, message[:modal_w-4])
    win.addstr(4, 2, "Press Y to confirm or N to cancel.")
    win.refresh()
    while True:
        ch = stdscr.getch()
        if ch in (ord('y'), ord('Y')):
            return True
        elif ch in (ord('n'), ord('N'), ord('q'), ord('Q'), 27): # Added ESC to cancel
            return False

def flash_message(stdscr: Any, message: str, duration_ms: int = 1500) -> None:
    """Display a centered flash message for a brief duration."""
    max_y, max_x = stdscr.getmaxyx()
    modal_h, modal_w = 5, min(50, max_x - 4)
    begin_y = (max_y - modal_h) // 2
    begin_x = (max_x - modal_w) // 2
    win = curses.newwin(modal_h, modal_w, begin_y, begin_x)
    draw_box(win, " Info ")
    win.addstr(2, 2, message[:modal_w-4])
    win.refresh()
    curses.napms(duration_ms)
    win.erase()
    stdscr.refresh()

# --- Data Manager ---
class DataManager:
    """
    Handles file operations for exercises, sessions, and templates.
    Reads data from an external index (index.json) for fast access.
    New notes are written to markdown files as before.
    """
    def __init__(self, notes_dir: Path, index_file: Path, log_file: Path, backup_interval_days: int = 1):
        self.notes_dir = notes_dir
        self.index_file = index_file  # This is produced externally (do not write to it)
        self.log_file = log_file
        self._exercise_cache: Optional[List[Exercise]] = None
        self._session_cache: Optional[List[WorkoutSession]] = None
        self._template_cache: Optional[List[WorkoutTemplate]] = None
        self.logger = logging.getLogger(__name__)
        self._ensure_files_exist()
        self.logger.debug("DataManager initialized with notes directory: %s", self.notes_dir)
        self.backup_interval_days = backup_interval_days
        self._last_backup_check = '2100-01-01 00:00:00'  # Initialize to a date in the distant future (change this if we want to implement backing up)


    def _ensure_files_exist(self) -> None:
        """Ensure that the notes directory exists.
        Do not create or modify the index file—it is managed externally."""
        if not self.notes_dir.exists():
            self.notes_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Created notes directory: %s", self.notes_dir)

    def read_file(self, filepath: Path) -> (Dict[str, Any], str):
        """Read a markdown file with YAML frontmatter and return its frontmatter and body."""
        try:
            text = filepath.read_text(encoding="utf-8")
            self.logger.debug("Read file: %s", filepath)
        except Exception as e:
            self.logger.error("Could not read file %s: %s", filepath, e)
            return {}, ""
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        try:
            fm = yaml.safe_load(parts[1]) or {}
            self.logger.debug("Parsed YAML frontmatter from %s", filepath)
        except Exception as err:
            self.logger.error("Error parsing YAML from %s: %s", filepath, err)
            fm = {}
        body = parts[2].lstrip("\n")
        return fm, body

    def write_file(self, filepath: Path, frontmatter: Dict[str, Any], body: str = "") -> None:
        """Write a markdown file with YAML frontmatter."""
        try:
            fm_text = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        except Exception as e:
            self.logger.error("Error dumping YAML for %s: %s", filepath, e)
            fm_text = ""
        content = f"---\n{fm_text}---\n\n{body}"
        try:
            filepath.write_text(content, encoding="utf-8")
            self.logger.info("Saved file: %s", filepath)
        except Exception as e:
            self.logger.error("Error writing file %s: %s", filepath, e)

    def update_modified_date(self, filepath: Path) -> None:
        """Update the modified date in a file's frontmatter."""
        fm, body = self.read_file(filepath)
        fm["dateModified"] = get_current_iso()
        self.write_file(filepath, fm, body)
        self.logger.debug("Updated modified date for: %s", filepath)

    # --- Writing Operations ---
    def create_exercise(self, title: str, equipment: str) -> str:
        """Create a new exercise file.
        Note: This writes a new markdown file; the external index is not updated here."""
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        equipment_list = [item.strip() for item in equipment.split(",") if item.strip()] or ["none"]
        frontmatter = {
            "title": title,
            "tags": ["exercise"],
            "date": today_str,
            "dateCreated": iso_time,
            "dateModified": iso_time,
            "planned_exercise": False,
            "exercise_equipment": equipment_list,
        }
        self.write_file(filepath, frontmatter)
        self._exercise_cache = None  # Invalidate cache
        self.logger.info("Created new exercise: %s", filename)
        return filename

    def append_to_workout_session(self, session_filename: str, new_exercises: List[Dict[str, Any]]) -> None:
        if not session_filename.endswith(".md"):
            session_filename += ".md"
        filepath = self.notes_dir / session_filename
        fm, body = self.read_file(filepath)
        if "exercises" not in fm or not isinstance(fm["exercises"], list):
            fm["exercises"] = []
        all_exercises = fm["exercises"] + new_exercises
        fm["exercises"] = self.merge_exercise_entries(all_exercises)
        fm["dateModified"] = get_current_iso(with_seconds=False)
        # Optionally, ignore the old body if you don't need it:
        self.write_file(filepath, fm, "")
        self._session_cache = None
        self.logger.info("Merged %d exercise entry(ies) into session %s", len(new_exercises), session_filename)


    def toggle_exercise_planned(self, filename: str) -> bool:
        """Toggle the planned status of an exercise."""
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = self.notes_dir / filename
        fm, body = self.read_file(filepath)
        current = fm.get("planned_exercise", False)
        new_status = (not current) if isinstance(current, bool) else (current.lower() != "true")
        fm["planned_exercise"] = new_status
        fm["dateModified"] = get_current_iso()
        self.write_file(filepath, fm, body)
        self._exercise_cache = None
        self.logger.info("Toggled planned status for %s to %s", filename, new_status)
        return new_status

    def edit_exercise(self, filename: str, new_title: str, new_equipment: str) -> None:
        """Edit an existing exercise note.
        Note: Does not update the external index."""
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = self.notes_dir / filename
        fm, body = self.read_file(filepath)
        fm["title"] = new_title
        new_equipment_list = [item.strip() for item in new_equipment.split(",") if item.strip()] or ["none"]
        fm["exercise_equipment"] = new_equipment_list
        fm["dateModified"] = get_current_iso()
        self.write_file(filepath, fm, body)
        self._exercise_cache = None
        self.logger.info("Edited exercise: %s", filename)

    def delete_exercise(self, filename: str) -> None:
        """Delete an exercise note.
        Note: Does not update the external index."""
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = self.notes_dir / filename
        try:
            filepath.unlink()
            self.logger.info("Deleted exercise file: %s", filename)
        except Exception as err:
            self.logger.error("Error deleting exercise %s: %s", filename, err)
        self._exercise_cache = None

    def merge_exercise_entries(self, exercises: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate exercise entries (by 'id') so that all sets for the same exercise are combined."""
        merged = {}
        for ex in exercises:
            ex_id = ex.get("id")
            if ex_id in merged:
                merged[ex_id]["sets"].extend(ex.get("sets", []))
            else:
                # Create a new copy to avoid modifying the original list.
                merged[ex_id] = {
                    "id": ex.get("id"),
                    "title": ex.get("title"),
                    "sets": ex.get("sets", []).copy(),
                }
        return list(merged.values())


    def save_workout_session(self, exercises: List[Dict[str, Any]]) -> str:
        """Save a workout session note with recorded exercises after merging duplicate entries."""
        merged_exercises = self.merge_exercise_entries(exercises)
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        frontmatter = {
            "zettelid": uid,
            "title": f"Workout Session on {today_str}",
            "tags": ["workout"],
            "date": today_str,
            "dateCreated": iso_time,
            "dateModified": iso_time,
            "exercises": merged_exercises,
        }
        self.write_file(filepath, frontmatter)
        self._session_cache = None
        self.logger.info("Saved workout session: %s", filename)
        return filename


    def delete_workout_session(self, filename: str) -> None:
        """Delete a workout session note."""
        if not filename.endswith(".md"):
            filename += ".md"
        filepath = self.notes_dir / filename
        try:
            filepath.unlink()
            self.logger.info("Deleted workout session file: %s", filename)
        except Exception as err:
            self.logger.error("Error deleting workout session %s: %s", filename, err)
        self._session_cache = None

    def create_workout_template(self, name: str, description: str, exercises: List[Exercise]) -> str:
        """Create a new workout template note and return its filename."""
        uid = generate_unique_id()
        filename = f"{uid}.md"
        filepath = self.notes_dir / filename
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        iso_time = get_current_iso(with_seconds=False)
        frontmatter = {
            "title": name,
            "description": description,
            "date": today_str,
            "dateCreated": iso_time,
            "dateModified": iso_time,
            "exercises": [{"exercise_filename": ex.filename, "order": idx+1} for idx, ex in enumerate(exercises)],
            "tags": ["workout_template"],
        }
        body = f"# {name}\n\n{description}"
        self.write_file(filepath, frontmatter, body)
        self._template_cache = None
        self.logger.info("Created workout template: %s", filename)
        return filename

    def update_template(self, tmpl_filename: str, name: str, description: str, exercises: List[Exercise]) -> None:
        """Update an existing workout template.
        Note: Does not update the external index."""
        filepath = self.notes_dir / tmpl_filename
        if not filepath.exists():
            self.logger.error("Template %s not found", tmpl_filename)
            return
        iso_time = get_current_iso(with_seconds=False)
        fm, _ = self.read_file(filepath)
        fm["title"] = name
        fm["description"] = description
        fm["dateModified"] = iso_time
        fm["exercises"] = [{"exercise_filename": ex.filename, "order": idx+1} for idx, ex in enumerate(exercises)]
        body = f"# {name}\n\n{description}"
        self.write_file(filepath, fm, body)
        self._template_cache = None
        self.logger.info("Updated template: %s", tmpl_filename)

    def delete_template(self, tmpl_filename: str) -> None:
        """Delete a workout template note."""
        filepath = self.notes_dir / tmpl_filename
        try:
            filepath.unlink()
            self._template_cache = None
            self.logger.info("Deleted template: %s", tmpl_filename)
        except Exception as err:
            self.logger.error("Error deleting template %s: %s", tmpl_filename, err)

    def export_history_csv(self, export_path: Path) -> None:
        """Export all workout sessions to a CSV file."""
        sessions = self.list_workout_sessions()
        if not sessions:
            self.logger.info("No workout sessions found for export.")
            return
        try:
            with export_path.open("w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["Session Date", "Session Title", "Exercise Title", "Sets", "Avg Reps", "Avg Weight"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                exercises = self.load_exercises()
                for session in sessions:
                    for ex in session.exercises:
                        exercise_title = next((e.title for e in exercises if e.filename == ex.id), ex.title)
                        num_sets = len(ex.sets)
                        avg_reps = sum(s.reps for s in ex.sets) / num_sets if num_sets else 0
                        avg_weight = sum(s.weight for s in ex.sets) / num_sets if num_sets else 0
                        writer.writerow({
                            "Session Date": session.date,
                            "Session Title": session.title,
                            "Exercise Title": exercise_title,
                            "Sets": num_sets,
                            "Avg Reps": f"{avg_reps:.1f}",
                            "Avg Weight": f"{avg_weight:.1f}",
                        })
            self.logger.info("Workout history exported to CSV at %s", export_path)
        except Exception as err:
            self.logger.exception("Failed to export workout history CSV: %s", err)

    def export_history_json(self, export_path: Path) -> None:
        """Export all workout sessions to a JSON file."""
        sessions = self.list_workout_sessions()
        if not sessions:
            self.logger.info("No workout sessions found for export.")
            return
        try:
            data = []
            exercises = self.load_exercises()
            for session in sessions:
                session_data = {
                    "date": session.date,
                    "title": session.title,
                    "exercises": []
                }
                for ex in session.exercises:
                    exercise_title = next((e.title for e in exercises if e.filename == ex.id), ex.title)
                    session_data["exercises"].append({
                        "exercise_title": exercise_title,
                        "sets": [{"reps": s.reps, "weight": s.weight} for s in ex.sets]
                    })
                data.append(session_data)
            export_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.logger.info("Workout history exported to JSON at %s", export_path)
        except Exception as err:
            self.logger.exception("Failed to export workout history JSON: %s", err)

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate statistics from all workout sessions."""
        stats = {
            "total_sessions": 0,
            "total_exercises": 0,
            "total_sets": 0,
            "total_reps": 0,
            "total_weight": 0.0,
        }
        sessions = self.list_workout_sessions()
        stats["total_sessions"] = len(sessions)
        for session in sessions:
            for ex in session.exercises:
                stats["total_exercises"] += 1
                num_sets = len(ex.sets)
                stats["total_sets"] += num_sets
                stats["total_reps"] += sum(s.reps for s in ex.sets)
                stats["total_weight"] += sum(s.weight for s in ex.sets)
        if stats["total_sets"]:
            stats["avg_reps"] = stats["total_reps"] / stats["total_sets"]
            stats["avg_weight"] = stats["total_weight"] / stats["total_sets"]
        else:
            stats["avg_reps"] = 0
            stats["avg_weight"] = 0
        return stats

    def _should_perform_backup(self) -> bool:
        """Check if a backup should be performed based on the backup interval."""
        today = datetime.date.today()
        if (today - self._last_backup_check).days >= self.backup_interval_days:
            self._last_backup_check = today
            return True
        return False


    def backup_data(self, automatic: bool = False) -> Optional[Path]:
        """Create a ZIP archive backup of all markdown files.

        Args:
            automatic (bool): Whether this is an automatic or manual backup.

        Returns:
            Optional[Path]: The path to the backup file, or None if no backup was created.
        """
        if automatic and not self._should_perform_backup():
            return None

        backup_dir = self.notes_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.zip"

        try:
            with zipfile.ZipFile(backup_file, "w") as zf:
                for md_file in self.notes_dir.glob("*.md"):
                    zf.write(md_file, arcname=md_file.name)
            self.logger.info("Backup created at %s", backup_file)
            return backup_file
        except Exception as err:
            self.logger.exception("Failed to create backup: %s", err)
            return None


    def load_exercises(self) -> List[Exercise]:
        """Load and return all exercises from the external index (index.json)."""
        if self._exercise_cache is not None:
            return self._exercise_cache
        try:
            if not self.index_file.exists():
                self.logger.warning("Index file %s does not exist. Returning empty list.", self.index_file)
                data = []
            else:
                data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        exercises = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []
            if "exercise" in tags:
                filename = note.get("filename")
                title = meta.get("title", filename)
                planned = meta.get("planned_exercise", False)
                equipment = meta.get("exercise_equipment", ["none"])
                exercises.append(Exercise(
                    filename=filename,
                    title=title,
                    planned=planned,
                    equipment=equipment
                ))
        self._exercise_cache = exercises
        self.logger.debug("Loaded %d exercises from index", len(exercises))
        return exercises

    def search_exercises(self, keyword: str) -> List[Exercise]:
        """Return a list of exercises whose title or equipment matches the given keyword."""
        keyword_lower = keyword.lower()
        return [ex for ex in self.load_exercises()
                if keyword_lower in ex.title.lower()
                or any(keyword_lower in eq.lower() for eq in ex.equipment)]

    def list_workout_sessions(self) -> List[WorkoutSession]:
        """Return a list of all workout sessions from the external index."""
        if self._session_cache is not None:
            return self._session_cache
        try:
            if not self.index_file.exists():
                self.logger.warning("Index file %s does not exist. Returning empty list.", self.index_file)
                data = []
            else:
                data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        sessions = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []
            if "workout" in tags:
                filename = note.get("filename")
                date_val = meta.get("date", "unknown")
                title = meta.get("title", filename)
                exercises_data = meta.get("exercises", [])
                session_exercises = []
                for ex in exercises_data:
                    sets = ex.get("sets", [])
                    workout_sets = []
                    for s in sets:
                        try:
                            reps = int(s.get("reps", 0))
                            weight_str = s.get("weight", "0")
# If weight is empty, treat it as 0.0
                            weight = float(weight_str) if weight_str.strip() != "" else 0.0
                            workout_sets.append(WorkoutSet(reps=reps, weight=weight))
                        except Exception as e:
                            self.logger.error("Error processing set: %s", e)
# Optionally append a default set instead of skipping
                            workout_sets.append(WorkoutSet(reps=0, weight=0.0))

                    # Preserve the title from the saved session data.
                    session_exercises.append(WorkoutExercise(id=ex.get("id"), title=ex.get("title", ""), sets=workout_sets))
                sessions.append(WorkoutSession(filename=filename, date=date_val, title=title, exercises=session_exercises))
        sessions.sort(key=lambda s: s.date, reverse=True)
        self._session_cache = sessions
        self.logger.debug("Found %d workout sessions in index", len(sessions))
        return sessions

    def load_templates(self) -> List[WorkoutTemplate]:
        """Load and return all workout templates from the external index."""
        if self._template_cache is not None:
            return self._template_cache
        try:
            if not self.index_file.exists():
                self.logger.warning("Index file %s does not exist. Returning empty list.", self.index_file)
                data = []
            else:
                data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        templates = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []
            if "workout_template" in tags:
                filename = note.get("filename")
                title = meta.get("title", filename)
                description = meta.get("description", "")
                exercises_field = meta.get("exercises", [])
                exercise_filenames = [item.get("exercise_filename") for item in exercises_field if item.get("exercise_filename")]
                templates.append(WorkoutTemplate(filename=filename, title=title, description=description, exercises=exercise_filenames))
        self._template_cache = templates
        self.logger.debug("Loaded %d workout templates from index", len(templates))
        return templates

# --- UI Manager ---
class UIManager:
    """
    Manages the terminal user interface using curses.
    Offers an enhanced menu with options to record sessions,
    manage exercises/templates, search/filter data, view statistics,
    delete sessions, backup data, and display About information.
    """
    def __init__(self, stdscr: Any, data_manager: DataManager):
        self.stdscr = stdscr
        self.dm = data_manager
        curses.curs_set(1)
        self.stdscr.nodelay(False)
        self.stdscr.keypad(True)
        self.setup_colors()
        self.dm.logger.debug("UI Manager initialized.")
        self.history_index = None
        self.menu_history = []  # Stack to store menu states (options, cursor)

    def setup_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)     # Highlight
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Footer/Status
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Normal text
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_RED)      # Modal / Alerts

    def draw_header(self, title: str) -> None:
        self.stdscr.attron(curses.color_pair(1))
        try:
            self.stdscr.addstr(0, 0, " " * (curses.COLS - 1))
            self.stdscr.addstr(0, 2, title)
        except Exception:
            pass
        self.stdscr.attroff(curses.color_pair(1))
        self.stdscr.refresh()

    def show_footer(self, message: str, color_pair: int = 3) -> None:
        max_y, max_x = self.stdscr.getmaxyx()
        self.stdscr.attron(curses.color_pair(color_pair))
        self.stdscr.addstr(max_y - 1, 0, message[:max_x-1])
        self.stdscr.clrtoeol()
        self.stdscr.attroff(curses.color_pair(color_pair))
        self.stdscr.refresh()

    def pause(self, prompt: str = "Press any key to continue...") -> None:
        self.show_footer(prompt)
        self.stdscr.getch()

    def clear_screen(self) -> None:
        self.stdscr.clear()
        self.stdscr.bkgd(" ", curses.color_pair(4))
        self.stdscr.refresh()

    def show_help(self, context: str = "main") -> None:
        """Display help, optionally contextual."""
        self.clear_screen()

        if context == "main":
            self.draw_header("Help & Instructions - Main Menu")
            help_text = [
                "Navigation:",
                "  ↑/↓ or k/j : Move selection",
                "  Enter      : Choose option",
                "  Esc        : Go Back",
                "  Q          : Quit",
                "",
                "Press any key to return..."
            ]
        elif context == "exercise_select":
            self.draw_header("Help - Exercise Selection")
            help_text = [
                "  ↑/↓ or k/j : Move selection",
                "  Enter      : Select exercise",
                "  P          : Toggle 'Planned' status",
                "  S          : Search",
                "  D          : Session Summary",
                "  H          : Exercise History",
                "  Esc        : Cancel",
                "",
                "Press any key to return..."
            ]
        elif context == "edit_exercise":
            self.draw_header("Help - Edit Exercise")
            help_text = [
                "  ↑/↓ or k/j : Move selection",
                "  Enter      : Edit Exercise",
                "  T          : Toggle 'Planned' status",
                "  D          : Delete Exercise",
                "  Esc        : Back to Menu",
                "",
                "Press any key to return..."
            ]
        elif context == "history_view":
            self.draw_header("Help - Workout History")
            help_text = [
                "  ↑/↓ or k/j : Move selection",
                "  Enter      : View Session Details",
                "  F          : Filter Sessions",
                "  Esc        : Back to Menu",
                "",
                "Press any key to return..."
            ]
        else: # Default main help
            self.draw_header("Help & Instructions")
            help_text = [
                "Navigation:",
                "  ↑/↓ or k/j : Move selection",
                "  Enter      : Choose option",
                "  Esc        : Go back / Quit",
                "  Q          : Quit",
                "",
                "Common Keys:",
                "  P          : Toggle Planned status (Exercises)",
                "  S          : Search exercises",
                "  F1 or H    : Show this help screen",
                "",
                "Features:",
                "  Create, edit, delete exercises & templates",
                "  Record workouts, view/filter history, export CSV/JSON",
                "  View statistics, delete sessions, backup data",
                "",
                "Press any key to return to the main menu..."
            ]

        max_y, max_x = self.stdscr.getmaxyx()
        for idx, line in enumerate(help_text, start=3):
            if idx < max_y - 2:
                self.stdscr.addstr(idx, 2, line[:max_x-4])
        self.stdscr.refresh()
        self.stdscr.getch()

    def _validate_numeric_input(self, prompt: str, y: int, x: int, allow_decimal: bool = False, default: str = "") -> str:
        """Helper function to get validated numeric input."""
        while True:
            user_input = self.prompt_input(prompt, y, x, default=default)
            if not user_input and default:  # Allow default value
                return default
            if not user_input: # Empty input
                return ""
            try:
                if allow_decimal:
                    float(user_input)
                else:
                    int(user_input)
                return user_input
            except ValueError:
                flash_message(self.stdscr, "Invalid input. Please enter a number.", duration_ms=1000)
                self.stdscr.move(y, x) # Restore cursor
                self.stdscr.clrtoeol()


    # --- Updated "Add to Most Recent Workout" function ---
    def add_to_recent_workout(self, session_filename: str = None) -> None:
        """Allow the user to add one or more exercises to the most recent workout session."""
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No recent workout found.", 3)
            self.pause("Press any key to return to main menu...")
            return

        if session_filename is None:
          recent_session = sessions[0]
        else:
          for session in sessions:
            if session.filename == session_filename:
              recent_session = session
              break

        # Build the current session summary from the session data.
        session_exercises = [
            {
                "id": we.id,
                "title": we.title,
                "sets": [{"reps": s.reps, "weight": s.weight} for s in we.sets]
            }
            for we in recent_session.exercises
        ]

        # Allow the user to add multiple exercises in a loop.
        while True:
            self.clear_screen()
            self.draw_header(f"Resume Workout: {recent_session.title} ({recent_session.date})") # Show Date
            self.show_footer("Select an exercise to add (or press Q to finish)", 3)
            ex = self.choose_exercise(session_exercises)
            if not ex:
                break  # User cancelled selection.
            result = self.record_exercise(ex)
            if not result:
                self.show_footer("No exercise recorded.", 3)
                self.pause("Press any key to return to main menu...")
                return
            # Append new exercise both to file and local summary for preview.
            session_exercises.append(result)
            self.dm.append_to_workout_session(recent_session.filename, [result])
            flash_message(self.stdscr, f"Exercise added to '{recent_session.title}'")
            # Ask if user wants to add another exercise.
            cont = self.prompt_input("Add another exercise? (y/N): ", curses.LINES - 3, 2)
            if cont.lower() == "n":
                break

    def prompt_input(self, prompt: str, y: int, x: int, default: str = "") -> str:
        self.stdscr.addstr(y, x, prompt)
        self.stdscr.clrtoeol()
        curses.echo()
        try:
            inp = self.stdscr.getstr(y, x + len(prompt)).decode("utf-8").strip()
        except Exception as err:
            self.dm.logger.error("Input error: %s", err)
            inp = ""
        curses.noecho()
        return inp if inp else default

    def record_exercise(self, exercise: Exercise) -> Optional[Dict[str, Any]]:
        self.clear_screen()
        header = f"Recording Workout for: {exercise.title}"
        if exercise.planned:
            header += " [PLANNED]"
        self.draw_header(header)
        self.dm.logger.info("Recording exercise: %s", exercise.title)
        set_number = 1
        recorded_sets = []
        last_weight = ""
        max_y, max_x = self.stdscr.getmaxyx()
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
                    summary_win.addstr(idx, 2, f"Set {idx}: {s['reps']} reps @ {s['weight']}")
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
        if not recorded_sets:
            self.show_footer("No sets recorded for this exercise.", 3)
            self.pause()
            return None
        # flash_message(self.stdscr, "Exercise sets recorded.")
        self.dm.logger.debug("Recorded %d sets for %s", len(recorded_sets), exercise.title)
        return {"id": exercise.filename, "title": exercise.title, "sets": recorded_sets}


    def record_session(self, prepopulated: Optional[List[Exercise]] = None) -> None:
        self.clear_screen()
        self.draw_header("Start Workout Session")
        # self.pause("Get ready! Press any key to begin your workout...")
        session_exercises = []
        while True:
            ex = self.choose_exercise(session_exercises, exercises=prepopulated)
            if ex is None:
                break
            result = self.record_exercise(ex)
            if result and result.get("sets"):
                session_exercises.append(result)
            self.clear_screen()
            self.draw_header("Recording Workout Session")
            add_more = self.prompt_input("Add another exercise? (y/N): ", curses.LINES - 4, 2)
            if add_more.lower() == "n":
                break
        if not session_exercises:
            self.show_footer("No exercises recorded for this session.", 3)
            self.pause()
            return
        self.dm.save_workout_session(session_exercises)
        flash_message(self.stdscr, "Workout session saved.")
        self.pause("Session complete. Press any key to return to main menu...")
        self.dm.logger.info("Session recorded with %d exercises", len(session_exercises))

    def build_history_index(self) -> dict:
        """Build and return a history index mapping exercise id to a list of tuples (session date, sets)."""
        all_sessions = self.dm.list_workout_sessions()
        history_index = {}
        for session in all_sessions:
            for ex in session.exercises:
                history_index.setdefault(ex.id, []).append((session.date, ex.sets))
        for ex_id in history_index:
            history_index[ex_id].sort(key=lambda item: item[0], reverse=True)
        return history_index

    def show_exercise_history_popup(self, exercise: Exercise) -> None:
        if not self.history_index:
            self.history_index = self.build_history_index()
        details = self.history_index.get(exercise.filename, [])
        max_y, max_x = self.stdscr.getmaxyx()
        popup_h = min(20, max_y - 4)
        popup_w = min(60, max_x - 4)
        begin_y = (max_y - popup_h) // 2
        begin_x = (max_x - popup_w) // 2
        win = curses.newwin(popup_h, popup_w, begin_y, begin_x)
        draw_box(win, f" History for '{exercise.title}' ")
        if not details:
            win.addstr(2, 2, "No session history available.")
        else:
            row = 2
            for (sess_date, sets) in details:
                sets_summary = ", ".join([f"{s.reps}r@{s.weight}" for s in sets])
                line = f"{sess_date}: {sets_summary}"
                win.addnstr(row, 2, line, popup_w - 4)
                row += 1
                if row >= popup_h - 2:
                    break
        win.addstr(popup_h - 2, 2, "Press any key to close...")
        win.refresh()
        win.getch()
        self.clear_screen()
        self.draw_header("Select an Exercise (or / to search)")

    def show_session_summary_popup(self, session_exercises: List[Dict[str, Any]]) -> None:
        max_y, max_x = self.stdscr.getmaxyx()
        popup_h = min(20, max_y - 4)
        popup_w = min(70, max_x - 4)
        begin_y = (max_y - popup_h) // 2
        begin_x = (max_x - popup_w) // 2

        # Build the content lines
        lines = []
        if not session_exercises:
            lines.append("No exercises recorded yet.")
        else:
            for ex in session_exercises:
                title = ex.get("title", "Unknown")
                sets = ex.get("sets", [])
                lines.append(f"{title} - {len(sets)} set(s)")
                for idx, s in enumerate(sets, start=1):
                    detail = f"  Set {idx}: {s.get('reps', '')} reps @ {s.get('weight', '')}"
                    lines.append(detail)
                lines.append("")  # Blank line for spacing

        # Determine pad height and create pad (subtracting border width)
        pad_height = len(lines)
        pad_width = popup_w - 2
        pad = curses.newpad(pad_height, pad_width)
        for i, line in enumerate(lines):
            try:
                pad.addstr(i, 0, line)
            except curses.error:
                pass

        # Create the popup window for the border and title
        win = curses.newwin(popup_h, popup_w, begin_y, begin_x)
        draw_box(win, " Session Summary (scrollable) ")
        win.refresh()

        # Initial pad display starting row
        pad_top = 0
        # Instructions shown in footer
        self.show_footer("Use UP/DOWN arrows (or k/j) to scroll; any other key to close", 3)
        while True:
            try:
                pad.refresh(pad_top, 0, begin_y + 1, begin_x + 1, begin_y + popup_h - 2, begin_x + popup_w - 2)
            except curses.error:
                pass
            ch = self.stdscr.getch()
            if ch in (curses.KEY_UP, ord('k')):
                if pad_top > 0:
                    pad_top -= 1
            elif ch in (curses.KEY_DOWN, ord('j')):
                if pad_top < pad_height - (popup_h - 2):
                    pad_top += 1
            else:
                break

        self.clear_screen()
        self.draw_header("Select an Exercise (or / to search)")


    def choose_exercise(self, session_exercises: List[Dict[str, Any]], exercises: Optional[List[Exercise]] = None) -> Optional[Exercise]:
        if exercises is None:
            exercises = self.dm.load_exercises()
        if not self.history_index:
            self.history_index = self.build_history_index()
        # Sort exercises by the total number of recorded sets (descending order)
        exercises.sort(key=lambda ex: sum(len(sets) for (_, sets) in self.history_index.get(ex.filename, [])), reverse=True)

        cursor = 0
        offset = 0  # This is still useful for initial positioning
        max_y, max_x = self.stdscr.getmaxyx()
        list_h = max_y - 20  # Height of the list area

        # Calculate pad dimensions.  +1 is important for scrolling to the last item.
        pad_height = max(list_h, len(exercises) + 1)
        pad_width = max_x - 2
        pad = curses.newpad(pad_height, pad_width)

        while True:
            # Clear the pad, not the whole screen.  This avoids flicker.
            pad.clear()
            self.clear_screen()  # Clear *only* the main screen once.
            self.draw_header("Select an Exercise (or / to search)")

            header_text = f"{'Title':<{max_x-35}} {'Status':<10} Equipment"
            pad.attron(curses.A_BOLD | curses.color_pair(3))
            pad.addstr(0, 2, header_text[:max_x-4])  # Add header to the pad
            pad.attroff(curses.A_BOLD | curses.color_pair(3))

            # Draw the exercise list onto the pad.
            for idx, ex in enumerate(exercises):
                status = "Planned" if ex.planned else ""
                equip = ", ".join(ex.equipment) if isinstance(ex.equipment, list) else str(ex.equipment or "")
                line = f"{ex.title:<{max_x-35}} {status:<10} {equip}"
                row = 1 + idx  # Start below the header on the pad.

                if idx == cursor:
                    pad.attron(curses.color_pair(2))
                    pad.addstr(row, 2, line[:max_x-4])
                    pad.attroff(curses.color_pair(2))
                else:
                    pad.addstr(row, 2, line[:max_x-4])


            # --- PREVIEW WINDOW (Draw this *outside* the pad loop) ---
            preview_h = 18
            preview_win = curses.newwin(preview_h, max_x-2, max_y - preview_h - 2, 1)
            draw_box(preview_win, " Preview ")

            if exercises: # Handle empty exercise list
                selected = exercises[cursor]
                preview_win.addstr(1, 2, f"Title: {selected.title}")
                equip = ", ".join(selected.equipment) if isinstance(selected.equipment, list) else str(selected.equipment or "")
                preview_win.addstr(2, 2, f"Equipment: {equip}")
                status = "Planned" if selected.planned else "Not Planned"
                preview_win.addstr(3, 2, f"Status: {status}")

                # Compute all-time statistics (as before)
                history = self.history_index.get(selected.filename, [])
                total_sets = 0
                highest_reps = 0
                highest_weight = 0.0
                for sess_date, sets in history:
                    total_sets += len(sets)
                    for s in sets:
                        if hasattr(s, "get"):  # Handle both dict and dataclass
                            reps = int(s.get("reps", 0)) if s.get("reps") else 0
                            weight = float(s.get("weight", 0)) if s.get("weight") else 0.0
                        else:
                            reps = s.reps
                            weight = s.weight
                        highest_reps = max(highest_reps, reps)
                        highest_weight = max(highest_weight, weight)

                preview_win.addstr(4, 2, f"All Time Sets: {total_sets}")
                preview_win.addstr(5, 2, f"Highest Reps: {highest_reps}")
                preview_win.addstr(6, 2, f"Highest Weight: {highest_weight}")
                preview_win.addstr(7, 2, "Recent History:")

                recent_history_max_lines = preview_h - 10
                line_idx = 8
                if history:
                    for date, sets in history[:3]:  # Limit to 3 recent sessions
                        if (line_idx - 8) < recent_history_max_lines:
                            set_str_items = []
                            for s in sets:
                                if hasattr(s, "get"):
                                    reps = s.get("reps", "")
                                    weight = s.get("weight", "")
                                else:
                                    reps = s.reps
                                    weight = s.weight
                                set_str_items.append(f"{reps}r@{weight}")
                            set_str = ", ".join(set_str_items)
                            preview_win.addstr(line_idx, 2, f"{date}: {set_str}"[:max_x-4])
                            line_idx += 1
                        else:
                            break
                else:
                    preview_win.addstr(line_idx, 2, "No past sessions.")

                divider_line = preview_h - 8
                preview_win.addstr(divider_line, 2, "-" * (max_x - 6))
                summary_title_line = divider_line + 1
                preview_win.addstr(summary_title_line, 2, "Current Session Summary:")
                summary_line = summary_title_line + 1

                if session_exercises:
                    for recorded_ex in session_exercises:
                        if summary_line < preview_h - 1:
                            if hasattr(recorded_ex, "get"):
                                rec_title = recorded_ex.get("title", "")
                                rec_sets = recorded_ex.get("sets", [])
                            else:
                                rec_title = recorded_ex.title
                                rec_sets = []  # No sets if not a dictionary
                            preview_win.addstr(summary_line, 4, f"• {rec_title} - {len(rec_sets)} set(s)")
                            summary_line += 1
                else:
                    preview_win.addstr(summary_line, 4, "(none)")

            preview_win.refresh()  # Refresh the preview window

            # --- KEY HANDLING & DISPLAY ---

            key_hint = ("↑/↓: Move | Enter: Select | P: Toggle Planned | " +
                        "S: Search | D: Session Summary | H: History | F1: Help | Esc: Back")
            self.show_footer(key_hint, 3)

            # Refresh *only* the visible portion of the pad.
            # Calculate the top-left corner for pad.refresh.
            pad_top = max(0, cursor - list_h // 2)  # Center cursor if possible
            pad_top = min(pad_top, pad_height - list_h) # Don't scroll past the end

            try:
                # The arguments to refresh are:
                #   pad_top, pad_left, screen_top, screen_left, screen_bottom, screen_right
                pad.refresh(pad_top, 0, 2, 1, list_h - 1, max_x - 2)
            except curses.error:
                pass  # Ignore errors during refresh (can happen at boundaries)

            self.stdscr.refresh() # Refresh main screen after pad


            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif k in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(exercises) - 1, cursor + 1)
            elif k in (10, 13):  # Enter
                if exercises: # Check for empty list
                    self.dm.logger.debug("Exercise selected: %s", exercises[cursor].title)
                    return exercises[cursor]
            elif k in (ord('p'), ord('P')):
                if exercises:
                    ex = exercises[cursor]
                    new_state = self.dm.toggle_exercise_planned(ex.filename)
                    ex.planned = new_state  # Update local cache
                    state_text = "Planned" if new_state else "Not Planned"
                    flash_message(self.stdscr, f"'{ex.title}' toggled to {state_text}")
            elif k in (ord('s'), ord('S')):
                keyword = self.prompt_input("Search keyword: ", 2, 2)
                if keyword:
                    exercises = self.dm.search_exercises(keyword)
                    # Re-sort the filtered list
                    exercises.sort(key=lambda ex: sum(len(sets) for (_, sets) in self.history_index.get(ex.filename, [])), reverse=True)
                    cursor = 0
                    offset = 0  # Reset offset after search
                    # Recalculate pad height after search results change.
                    pad_height = max(list_h, len(exercises) + 1)
                    pad = curses.newpad(pad_height, pad_width)

            elif k in (ord('d'), ord('D')):
                self.show_session_summary_popup(session_exercises)
            elif k in (ord('h'), ord('H')):
                if exercises:
                    self.show_exercise_history_popup(exercises[cursor])
            elif k in (curses.KEY_F1, ord('h')):
                self.show_help("exercise_select")
            elif k in (27, ):  # ESC Key
                return None


    def list_templates(self) -> Optional[WorkoutTemplate]:
        templates = self.dm.load_templates()
        if not templates:
            self.show_footer("No workout templates available.", 3)
            self.pause()
            return None

        cursor = 0
        max_y, max_x = self.stdscr.getmaxyx()
        list_h = max_y - 4  # Height of the list area (adjust as needed)

        # Calculate pad dimensions
        pad_height = max(list_h, len(templates) + 1)
        pad_width = max_x - 2
        pad = curses.newpad(pad_height, pad_width)

        while True:
            pad.clear()
            self.clear_screen()
            self.draw_header("Select Workout Template")

            # Draw the template list onto the pad
            for idx, tmpl in enumerate(templates):
                y = idx  # Start at the top of the pad
                disp = f"{tmpl.title}: {tmpl.description}"
                if idx == cursor:
                    pad.attron(curses.color_pair(2))
                    pad.addstr(y, 2, disp[:max_x-4])
                    pad.attroff(curses.color_pair(2))
                else:
                    pad.addstr(y, 2, disp[:max_x-4])

            self.show_footer("↑/↓: Move | Enter: Select | Esc: Back", 3)

            # Calculate the top-left corner for pad.refresh
            pad_top = max(0, cursor - list_h // 2)
            pad_top = min(pad_top, pad_height - list_h)

            try:
                pad.refresh(pad_top, 0, 2, 1, 2 + list_h, max_x - 2)
            except curses.error:
                pass

            self.stdscr.refresh() # Refresh main screen

            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif k in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(templates) - 1, cursor + 1)
            elif k in (10, 13):
                if templates:
                    self.dm.logger.debug("Template selected: %s", templates[cursor].title)
                    return templates[cursor]
            elif k in (27, ):  # ESC Key
                return None

    def start_session_from_template(self) -> None:
        tmpl = self.list_templates()
        if not tmpl:
            return
        exercises = self.dm.load_exercises()
        tmpl_exercises = [ex for ex in exercises if ex.filename in tmpl.exercises]
        if not tmpl_exercises:
            self.show_footer("No valid exercises found in template.", 3)
            if modal_confirm(self.stdscr, "Edit the template now?"):
                self.edit_template(tmpl)  # Pass the template object
            return
        flash_message(self.stdscr, f"Starting session from template '{tmpl.title}'")
        # self.pause("Press any key to continue...")
        self.dm.logger.info("Starting session from template: %s", tmpl.title)
        self.record_session(prepopulated=tmpl_exercises)


    def create_template(self) -> None:
        self.clear_screen()
        self.draw_header("Create Workout Template")
        tmpl_name = self.prompt_input("Template name: ", 4, 4)
        if not tmpl_name:
            self.show_footer("Template name missing. Cancelled.", 3)
            self.pause()
            return
        tmpl_desc = self.prompt_input("Template description (optional): ", 5, 4)
        exercises = []
        while True:
            add = self.prompt_input("Add an exercise? (y/N): ", 7, 4)
            if add.lower() != "y":
                break
            ex = self.choose_exercise([])
            if ex:
                exercises.append(ex)
            else:
                break
        if not exercises:
            self.show_footer("No exercises selected. Template creation cancelled.", 3)
            self.pause()
            return
        self.dm.create_workout_template(tmpl_name, tmpl_desc, exercises)
        flash_message(self.stdscr, f"Template '{tmpl_name}' created.")
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Created workout template: %s", tmpl_name)

    def edit_template(self, tmpl_to_edit: Optional[WorkoutTemplate] = None) -> None:
        """Edit an existing template, optionally pre-selected."""
        if tmpl_to_edit:
            tmpl = tmpl_to_edit
        else:
            tmpl = self.list_templates()
            if not tmpl:
                return

        self.clear_screen()
        self.draw_header("Edit Workout Template")
        new_name = self.prompt_input(f"New name [{tmpl.title}]: ", 4, 4, tmpl.title)
        new_desc = self.prompt_input(f"New description [{tmpl.description}]: ", 5, 4, tmpl.description)

        # Convert stored exercise filenames into exercise objects
        current_exercises = []
        all_exercises = self.dm.load_exercises()
        for ex_id in tmpl.exercises:
            for ex in all_exercises:
                if ex.filename == ex_id:
                    current_exercises.append(ex)
                    break

        # Start with the current exercise list and allow editing
        edited_exercises = current_exercises[:]  # Make a copy for editing

        while True:
            self.clear_screen()
            self.draw_header("Edit Template Exercise List")
            # Display the current list of exercises in the template
            self.stdscr.addstr(3, 4, "Current exercises in template:")
            for idx, ex in enumerate(edited_exercises, start=1):
                self.stdscr.addstr(3 + idx, 6, f"{idx}. {ex.title}")
            # Show interactive options
            self.show_footer("A: Add, R: Remove, M: Move, D: Done, Esc: Back", 3)
            self.stdscr.refresh()
            ch = self.stdscr.getch()

            if ch in (ord('a'), ord('A')):
                # Add an exercise
                ex_to_add = self.choose_exercise(edited_exercises)
                if ex_to_add and ex_to_add not in edited_exercises:
                    edited_exercises.append(ex_to_add)
                    flash_message(self.stdscr, f"Added '{ex_to_add.title}'")

            elif ch in (ord('r'), ord('R')):
                # Remove an exercise by number
                num_str = self.prompt_input("Enter number to remove: ", curses.LINES - 3, 2)
                try:
                    num = int(num_str)
                    if 1 <= num <= len(edited_exercises):
                        removed = edited_exercises.pop(num - 1)
                        flash_message(self.stdscr, f"Removed '{removed.title}'")
                    else:
                        flash_message(self.stdscr, "Invalid number.")
                except ValueError:  # Catch non-numeric input
                    flash_message(self.stdscr, "Invalid input.")

            elif ch in (ord('m'), ord('M')):
                # Move (reorder) an exercise
                num_str = self.prompt_input("Enter number to move: ", curses.LINES - 3, 2)
                try:
                    num = int(num_str)
                    if 1 <= num <= len(edited_exercises):
                        ex_to_move = edited_exercises.pop(num - 1)
                        new_pos_str = self.prompt_input(f"Enter new position (1-{len(edited_exercises) + 1}): ", curses.LINES - 3, 2)
                        try:
                            new_pos = int(new_pos_str)
                            if 1 <= new_pos <= len(edited_exercises) + 1:
                                edited_exercises.insert(new_pos - 1, ex_to_move)
                                flash_message(self.stdscr, f"Moved '{ex_to_move.title}' to position {new_pos}")
                            else:
                                flash_message(self.stdscr, "Invalid position.")
                                edited_exercises.insert(num -1, ex_to_move) # Put it back
                        except ValueError:
                            flash_message(self.stdscr, "Invalid position.")
                            edited_exercises.insert(num -1, ex_to_move) # Put it back
                    else:
                        flash_message(self.stdscr, "Invalid number.")
                except ValueError:
                    flash_message(self.stdscr, "Invalid input.")

            elif ch in (ord('d'), ord('D')):
                break

            elif ch in (27, ): # ESC Key
                return

        self.dm.update_template(tmpl.filename, new_name, new_desc, edited_exercises)
        flash_message(self.stdscr, f"Template '{new_name}' updated.")
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Edited template: %s", new_name)


    def delete_template_ui(self) -> None:
        tmpl = self.list_templates()
        if not tmpl:
            return
        self.clear_screen()
        self.draw_header("Delete Workout Template")
        if modal_confirm(self.stdscr, f"Delete template '{tmpl.title}'?"):
            self.dm.delete_template(tmpl.filename)
            flash_message(self.stdscr, f"Template '{tmpl.title}' deleted.")
            self.dm.logger.info("Deleted template: %s", tmpl.title)
        else:
            flash_message(self.stdscr, "Deletion cancelled.")
        self.pause("Press any key to return to main menu...")

    def add_new_exercise(self) -> None:
        self.clear_screen()
        self.draw_header("Create New Exercise")
        title = self.prompt_input("Exercise title: ", 4, 4)

        # Multi-select equipment (similar to exercise selection)
        equipment_options = ["none", "barbell", "dumbbell", "kettlebell", "machine", "bodyweight", "bands", "other"]
        selected_equipment = self._multi_select_list(equipment_options, "Select Equipment")

        if title:
            self.dm.create_exercise(title, ", ".join(selected_equipment))
            flash_message(self.stdscr, f"New exercise '{title}' created.")
            self.pause("Press any key to continue...")
            self.dm.logger.info("New exercise created: %s", title)
        else:
            flash_message(self.stdscr, "No title provided. Cancelled.")
            self.pause("Press any key to return to main menu...")

    def _multi_select_list(self, options: List[str], title: str) -> List[str]:
        """Helper function for a multi-select list using curses."""
        cursor = 0
        selected = []  # List of selected indices
        while True:
            self.clear_screen()
            self.draw_header(title)
            for idx, opt in enumerate(options):
                y = 3 + idx
                prefix = "[x] " if idx in selected else "[ ] "
                line = prefix + opt
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 2, line)
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 2, line)

            self.show_footer("↑/↓: Move | Space: Toggle | Enter: Confirm | Esc: Cancel", 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')) and cursor > 0:
                cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and cursor < len(options) - 1:
                cursor += 1
            elif k == ord(' '):  # Space bar
                if cursor in selected:
                    selected.remove(cursor)
                else:
                    selected.append(cursor)
            elif k in (10, 13):  # Enter
                return [options[i] for i in selected]
            elif k in (27,):  # ESC
                return []


    def edit_exercise_ui(self) -> None:
        exercises = self.dm.load_exercises()
        if not exercises:
            self.show_footer("No exercises available to edit.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Edit Exercise (Inline)")
            max_y, max_x = self.stdscr.getmaxyx()

            #--- Display in a table-like format ---
            headers = ["Title", "Equipment", "Planned"]
            col_widths = [max_x - 40, 25, 7]  # Adjust as needed
            header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
            self.stdscr.addstr(2, 2, header_line[:max_x - 4])
            self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))


            for idx, ex in enumerate(exercises):
                y = 3 + idx
                # Display with columns
                title_str = f"{ex.title:<{col_widths[0]}}"
                equipment_str = f"{', '.join(ex.equipment):<{col_widths[1]}}"
                planned_str = f"{'Yes' if ex.planned else 'No':<{col_widths[2]}}"
                line = f"{title_str}  {equipment_str}  {planned_str}"

                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 2, line[:max_x - 4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 2, line[:max_x - 4])

            hint = "↑/↓: Move | Enter: Edit | T: Toggle Planned | D: Delete | Esc: Back" # Add 'T' for Toggle, 'D' for Delete
            self.show_footer(hint, 3)
            self.stdscr.refresh()

            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')) and cursor > 0:
                cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and cursor < len(exercises) - 1:
                cursor += 1
            elif k in (10, 13):  # Enter
                ex = exercises[cursor]
                # Inline editing using a helper function.
                self.inline_edit_exercise(ex)
            elif k in (ord('t'), ord('T')): # Toggle Planned
                ex = exercises[cursor]
                new_state = self.dm.toggle_exercise_planned(ex.filename)
                ex.planned = new_state # Update local cache
                flash_message(self.stdscr, f"'{ex.title}' planned status toggled.")
            elif k in (ord('d'), ord('D')):
                ex = exercises[cursor]
                if modal_confirm(self.stdscr, f"Delete exercise '{ex.title}'?"):
                    self.dm.delete_exercise(ex.filename)
                    exercises.pop(cursor) # Remove from local list.
                    flash_message(self.stdscr, "Exercise deleted.")
                    if cursor >= len(exercises) and cursor > 0:
                        cursor -= 1 # Adjust cursor if we deleted the last item
            elif k in (27,): # Esc
                break

    def inline_edit_exercise(self, exercise: Exercise) -> None:
        """Helper function for inline editing of an exercise."""
        max_y, max_x = self.stdscr.getmaxyx()
        edit_win = curses.newwin(5, max_x - 4, 2, 2) # Window for input
        draw_box(edit_win, "Edit Exercise")
        edit_win.refresh()

        new_title = self.prompt_input(f"Title [{exercise.title}]: ", 3, 4, exercise.title, edit_win)

        # Use the multi-select for equipment editing
        new_equipment_list = self._multi_select_list(
            ["none", "barbell", "dumbbell", "kettlebell", "machine", "bodyweight", "bands", "other"],
            "Select Equipment",
            initial_selection=[i for i, item in enumerate(["none", "barbell", "dumbbell", "kettlebell", "machine", "bodyweight", "bands", "other"]) if item in exercise.equipment]
        )
        new_equipment = ", ".join(new_equipment_list)


        self.dm.edit_exercise(exercise.filename, new_title, new_equipment)
        exercise.title = new_title  # Update the *local* Exercise object.
        exercise.equipment = [item.strip() for item in new_equipment.split(",") if item.strip()] or ["none"]
        flash_message(self.stdscr, "Exercise updated.")


    def delete_exercise_ui(self) -> None:
        exercises = self.dm.load_exercises()
        if not exercises:
            self.show_footer("No exercises available to delete.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Delete an Exercise")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, ex in enumerate(exercises):
                line = f"{ex.title}"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Delete | Esc: Back", 3) # Changed Q to Esc
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')) and cursor > 0:
                cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and cursor < len(exercises) - 1:
                cursor += 1
            elif k in (10, 13):
                ex = exercises[cursor]
                if modal_confirm(self.stdscr, f"Delete exercise '{ex.title}'?"):
                    self.dm.delete_exercise(ex.filename)
                    flash_message(self.stdscr, "Exercise deleted.")
                    exercises.pop(cursor) # Remove from local list
                    if cursor >= len(exercises) and cursor > 0:
                        cursor -= 1 # Adjust cursor if we deleted the last item
                    # No break, stay in delete screen to allow deleting multiple.
                else:
                    flash_message(self.stdscr, "Deletion cancelled.")
                    self.pause("Press any key...")
            elif k in (27, ): # ESC Key
                break

    def view_history(self) -> None:
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No past sessions found.", 3)
            self.pause()
            return

        filtered_sessions = sessions
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Workout History (F: Filter)")
            max_y, max_x = self.stdscr.getmaxyx()
            list_h = max_y - 8  # Height available for the list

            # --- Table-like display (with Sets) ---
            headers = ["Date", "Title", "Exercises", "Sets"] # Added "Sets"
            col_widths = [12, max_x - 38, 10, 8]  # Adjusted widths
            header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
            self.stdscr.addstr(2, 2, header_line[:max_x-4])
            self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))

            for idx, session in enumerate(filtered_sessions[:list_h]):
                y = 3 + idx
                date_str = f"{session.date:<{col_widths[0]}}"
                title_str = f"{session.title:<{col_widths[1]}}"
                exercises_count_str = f"{len(session.exercises):<{col_widths[2]}}"

                # Calculate total sets for the session
                total_sets = sum(len(ex.sets) for ex in session.exercises)
                sets_str = f"{total_sets:<{col_widths[3]}}"

                line = f"{date_str}  {title_str}  {exercises_count_str}  {sets_str}"

                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 2, line[:max_x - 4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 2, line[:max_x - 4])

            hint = "↑/↓: Move | Enter: Details | R: Resume | F: Filter | F1: Help | Esc: Back"
            self.show_footer(hint, 3)
            self.stdscr.refresh()

            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')) and cursor > 0:
                cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and cursor < len(filtered_sessions) - 1:
                cursor += 1
            elif k in (10, 13):
                self.show_session_details(filtered_sessions[cursor])
            elif k in (ord('r'), ord('R')):
                self.add_to_recent_workout(filtered_sessions[cursor].filename)
                break
            elif k in (ord('f'), ord('F')):
                keyword = self.prompt_input("Filter by exercise: ", 2, 2)
                if keyword:
                    all_ex = self.dm.load_exercises()
                    filtered_sessions = [s for s in sessions if any(keyword.lower() in next((e.title for e in all_ex if e.filename == ex.id), ex.id).lower() for ex in s.exercises)]
                    cursor = 0
                else:
                    filtered_sessions = sessions
            elif k in (curses.KEY_F1, ord('h')):
                self.show_help("history_view")
            elif k in (27,):
                break

    def show_session_details(self, session: WorkoutSession) -> None:
        max_y, max_x = self.stdscr.getmaxyx()
        # Define popup dimensions (adjust as needed)
        popup_h = min(20, max_y - 4)
        popup_w = min(70, max_x - 4)
        begin_y = (max_y - popup_h) // 2
        begin_x = (max_x - popup_w) // 2

        # Build session details content as a list of strings
        lines = []
        lines.append(f"Date: {session.date}")
        lines.append(f"Title: {session.title}")
        lines.append("")
        lines.append("Exercises:")
        lines.append("")
        all_exercises = self.dm.load_exercises()
        for ex in session.exercises:
            ex_title = next((e.title for e in all_exercises if e.filename == ex.id), ex.title or ex.id)
            lines.append(f"- {ex_title} ({len(ex.sets)} set(s))")
            for idx, s in enumerate(ex.sets, start=1):
                lines.append(f"  Set {idx}: Reps: {s.reps}, Weight: {s.weight}")
            total_reps = sum(s.reps for s in ex.sets)
            total_weight = sum(s.weight for s in ex.sets)
            avg_weight = total_weight / len(ex.sets) if ex.sets else 0
            lines.append(f"  Totals: {total_reps} reps, Total Weight: {total_weight:.1f}, Avg Weight: {avg_weight:.1f}")
            lines.append("")  # Blank line for spacing

        # Create a pad with enough height for all lines and a width a bit smaller than the popup.
        pad_height = len(lines)
        pad_width = popup_w - 2
        pad = curses.newpad(pad_height, pad_width)
        for i, line in enumerate(lines):
            try:
                pad.addnstr(i, 0, line, pad_width - 1)
            except curses.error:
                pass

        # Create the popup window (for border and title)
        win = curses.newwin(popup_h, popup_w, begin_y, begin_x)
        draw_box(win, " Session Details (scrollable) ")
        win.refresh()

        pad_top = 0
        # Display instructions in the footer
        self.show_footer("Use UP/DOWN arrows (or k/j) to scroll; any other key to close", 3)
        while True:
            try:
                pad.refresh(pad_top, 0, begin_y + 1, begin_x + 1, begin_y + popup_h - 2, begin_x + popup_w - 2)
            except curses.error:
                pass
            ch = self.stdscr.getch()
            if ch in (curses.KEY_UP, ord('k')):
                if pad_top > 0:
                    pad_top -= 1
            elif ch in (curses.KEY_DOWN, ord('j')):
                if pad_top < pad_height - (popup_h - 2):
                    pad_top += 1
            else:
                break

        self.clear_screen()
        self.draw_header("Workout History (F: Filter)") # Go back to history list


    def delete_session_ui(self) -> None:
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No sessions available to delete.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Delete a Session")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, session in enumerate(sessions):
                line = f"{session.date} - {session.title}"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Delete | Esc: Back", 3) # Changed Q to Esc
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')) and cursor > 0:
                cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')) and cursor < len(sessions) - 1:
                cursor += 1
            elif k in (10, 13):
                sess = sessions[cursor]
                if modal_confirm(self.stdscr, f"Delete session '{sess.title}'?"):
                    self.dm.delete_workout_session(sess.filename)
                    flash_message(self.stdscr, "Session deleted.")
                    sessions.pop(cursor) # Remove from local list
                    if cursor >= len(sessions) and cursor > 0:
                        cursor -= 1 # Adjust cursor if we deleted the last item
                    # No break, stay in delete screen to allow deleting multiple.
                else:
                    flash_message(self.stdscr, "Deletion cancelled.")
                    self.pause("Press any key...")
            elif k in (27, ): # ESC Key
                break

    def view_statistics(self) -> None:
        stats = self.dm.get_statistics()
        self.clear_screen()
        self.draw_header("Workout Statistics")
        max_y, max_x = self.stdscr.getmaxyx()
        stat_lines = [
            f"Total Sessions: {stats['total_sessions']}",
            f"Total Exercises: {stats['total_exercises']}",
            f"Total Sets: {stats['total_sets']}",
            f"Total Reps: {stats['total_reps']}",
            f"Total Weight: {stats['total_weight']:.1f}",
            f"Average Reps per Set: {stats['avg_reps']:.1f}",
            f"Average Weight per Set: {stats['avg_weight']:.1f}",
        ]
        for idx, line in enumerate(stat_lines, start=3):
            if idx < max_y - 2:
                self.stdscr.addstr(idx, 2, line[:max_x-4])
        self.stdscr.refresh()
        self.pause("Press any key to return to main menu...")

    def backup_data_ui(self) -> None:
        backup_file = self.dm.backup_data()
        if backup_file:
            self.clear_screen()
            self.draw_header("Backup Data")
            self.stdscr.addstr(3, 2, f"Backup created at: {backup_file}")
        else:
            flash_message(self.stdscr, "Backup failed.", 3)
        self.stdscr.refresh()
        self.pause("Press any key to return to main menu...")

    def export_history_ui(self) -> None:
        self.clear_screen()
        self.draw_header("Export Workout History")
        choice = self.prompt_input("Export as (1) CSV or (2) JSON? [1]: ", 4, 4, "1")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Include time in filename.
        default_filename = f"workout_history_{timestamp}." + ("csv" if choice=="1" else "json")
        export_path_str = self.prompt_input("Export file path: ", 5, 4, default_filename)
        export_path = Path(export_path_str).expanduser()
        if choice == "2":
            self.dm.export_history_json(export_path)
        else:
            self.dm.export_history_csv(export_path)
        flash_message(self.stdscr, f"History exported to {export_path}")
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Workout history exported to: %s", export_path)
    def _save_menu_state(self, menu_options: List[Any], cursor: int) -> None:
        """Saves the current menu state (options and cursor position)."""
        self.menu_history.append((menu_options, cursor))

    def _restore_menu_state(self) -> Tuple[List[Any], int]:
        """Restores the previous menu state."""
        return self.menu_history.pop() if self.menu_history else ([], 0)


    def navigate_menu(self, menu_options: List[Any], title: str, breadcrumb: str = "") -> None:
        """Handles navigation through hierarchical menus, with breadcrumbs and shortcut keys."""

        # Add shortcut keys to top-level menu options
        if not breadcrumb:  # Only for the main menu
            numbered_menu_options = []
            for i, option in enumerate(menu_options, start=1):
                if isinstance(option, tuple):
                    numbered_menu_options.append((f"{i}. {option[0]}", option[1]))
                else: # Submenu
                    numbered_menu_options.append((f"{i}. {option[0]}...", option[1]))
            menu_options = numbered_menu_options

        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header(f"{title} – {breadcrumb}" if breadcrumb else title)  # Breadcrumb

            for idx, option in enumerate(menu_options):
                y = 3 + idx
                if isinstance(option, tuple):  # Regular menu item
                    label = option[0]
                else:  # Submenu
                    label = option[0] + "..."
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 4, f"> {label}")
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 4, f"  {label}")

            hint = "↑/↓: Move | Enter: Select | Esc: Back | Q: Quit | F1: Help"
            self.show_footer(hint, 3)
            self.stdscr.refresh()

            k = self.stdscr.getch()

            # Handle shortcut keys (only on the main menu)
            if not breadcrumb and ord('1') <= k <= ord(str(len(menu_options))):
                selected_option = menu_options[k - ord('1')]
                if isinstance(selected_option, tuple):
                    if selected_option[1] is None:  # Quit
                        return
                    elif isinstance(selected_option[1], list): # Submenu
                        self._save_menu_state(menu_options, cursor)
                        new_breadcrumb = selected_option[0].split(". ", 1)[1]  # Remove number
                        self.navigate_menu(selected_option[1], title, new_breadcrumb)
                        if not self.menu_history: # Returned to top level, re-apply numbers
                            return self.main_menu()
                        menu_options, cursor = self._restore_menu_state()

                    elif callable(selected_option[1]):
                        self._save_menu_state(menu_options, cursor)  # Save before action
                        selected_option[1]()
                        menu_options, cursor = self._restore_menu_state() # Restore
                continue


            if k in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif k in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(menu_options) - 1, cursor + 1)
            elif k in (10, 13):  # Enter key
                selected_option = menu_options[cursor]
                if isinstance(selected_option, tuple):
                    if selected_option[1] is None: # Quit
                        return
                    elif isinstance(selected_option[1], list): # Submenu
                        self._save_menu_state(menu_options, cursor)  # Save before entering submenu
                        new_breadcrumb = f"{breadcrumb} > {selected_option[0].split('. ',1)[1]}" if breadcrumb else selected_option[0].split(". ", 1)[1]  # Remove leading number
                        self.navigate_menu(selected_option[1], title, new_breadcrumb)

                        if not self.menu_history: # We've returned to the top-level. Re-apply shortcut keys.
                           return self.main_menu()
                        menu_options, cursor = self._restore_menu_state() # Restore state after returning from submenu

                    elif callable(selected_option[1]):
                        self._save_menu_state(menu_options, cursor)  # Save before action
                        selected_option[1]()  # Execute
                        if selected_option[1] == self.add_to_recent_workout:
                            pass # Don't go back to the menu if adding to recent workout.
                        elif selected_option[1] == self.record_session:
                            pass  #Don't go back if recording a regular session
                        elif not self.menu_history: # If we are on the main menu, re-number:
                            return self.main_menu()
                        else:
                            menu_options, cursor = self._restore_menu_state()  # Restore state after action
                else: # Should not happen
                    self.dm.logger.error(f"Unexpected menu option format: {selected_option}")
            elif k in (27,):  # ESC key
                return # Go back one level.
            elif k in (ord('q'), ord('Q')):
                if modal_confirm(self.stdscr, "Are you sure you want to quit?"): # Confirmation
                    return
            elif k in (curses.KEY_F1, ord('h')):
                self.show_help()

    def main_menu(self) -> None:
        menu_options = [
            ("Record Workout", [
                ("Start New Session", self.record_session),
                ("Resume Recent Session", self.add_to_recent_workout),
                ("Start from Template", self.start_session_from_template),
            ]),
            ("Manage Exercises", [
                ("Create New", self.add_new_exercise),
                ("Edit Existing", self.edit_exercise_ui),
                ("Delete", self.delete_exercise_ui),
            ]),
            ("Manage Templates", [
                ("Create New", self.create_template),
                ("Edit Existing", self.edit_template),
                ("Delete", self.delete_template_ui),
            ]),
            ("View History & Stats", [
                ("View History", self.view_history),
                ("View Statistics", self.view_statistics),
                ("Export Data", self.export_history_ui),
            ]),
            ("Settings & Data", [
                ("Backup Data", self.backup_data_ui),
            ]),
            ("About & Help", [  # Keep these together
                ("About", self.show_help),
                ("Help", lambda: self.show_help("main")), # Context for main menu help
            ]),
            ("Quit", None),
        ]
        self.navigate_menu(menu_options, "Ultimate Workout Logger")


# --- Main Entrypoint ---
def main(stdscr: Any) -> None:
    parser = argparse.ArgumentParser(description="Ultimate Workout Logger")
    parser.add_argument("--notes-dir", type=str, help="Path to your notes directory.")
    parser.add_argument("--export-history", action="store_true", help="Export workout history and exit.")
    parser.add_argument("--export-json", action="store_true", help="Export workout history in JSON format and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
    parser.add_argument("--backup-interval", type=int, default=1, help="Automatic backup interval in days (default: 1).")
    args = parser.parse_args()

    notes_dir_cli = args.notes_dir
    notes_dir_env = os.environ.get("NOTES_DIR")
    if notes_dir_cli:
        notes_dir_path = notes_dir_cli
    elif notes_dir_env:
        notes_dir_path = notes_dir_env
    else:
        print("Error: Please set the NOTES_DIR environment variable or use the --notes-dir option.")
        sys.exit(1)
    NOTES_DIR = Path(notes_dir_path)
    INDEX_FILE = NOTES_DIR / "index.json"  # This file is produced by zk_index.py
    LOG_FILE_PATH = NOTES_DIR / "workout_log_error.log"

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    try:
        handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    except Exception as err:
        print("Error setting up log file handler:", err)
        sys.exit(1)
    if args.verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    logger.info("Ultimate Workout Logger started")
    logger.debug("Using notes directory: %s", NOTES_DIR)

    dm = DataManager(NOTES_DIR, INDEX_FILE, LOG_FILE_PATH, backup_interval_days=args.backup_interval)
    if args.export_history or args.export_json:
        export_file = NOTES_DIR / ("workout_history." + ("json" if args.export_json else "csv"))
        if args.export_json:
            dm.export_history_json(export_file)
        else:
            dm.export_history_csv(export_file)
        print(f"Workout history exported to {export_file}")
        logger.info("Exported workout history to %s and exiting", export_file)
        return

    ui = UIManager(stdscr, dm)
    ui.main_menu()
    logger.info("Ultimate Workout Logger exiting")

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        logging.getLogger(__name__).exception("Critical error in Ultimate Workout Logger")
        print("An error occurred. Please check the log file for details.")
        sys.exit(1)
