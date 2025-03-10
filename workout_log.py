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
from typing import Any, Dict, List, Optional
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
    title: str
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

# --- Data Manager ---
class DataManager:
    """
    Handles file operations for exercises, sessions, and templates.
    Reads data from an external index (index.json) for fast access.
    New notes are written to markdown files as before.
    """
    def __init__(self, notes_dir: Path, index_file: Path, log_file: Path):
        self.notes_dir = notes_dir
        self.index_file = index_file  # This is produced externally (do not write to it)
        self.log_file = log_file
        self._exercise_cache: Optional[List[Exercise]] = None
        self._session_cache: Optional[List[WorkoutSession]] = None
        self._template_cache: Optional[List[WorkoutTemplate]] = None
        self.logger = logging.getLogger(__name__)
        self._ensure_files_exist()
        self.logger.debug("DataManager initialized with notes directory: %s", self.notes_dir)

    def _ensure_files_exist(self) -> None:
        """Ensure that the notes directory exists.
        Do not create or modify the index file—it is managed externally."""
        if not self.notes_dir.exists():
            self.notes_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info("Created notes directory: %s", self.notes_dir)
        # Do not create the index file if missing.

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

    # --- Writing operations (do not update the external index) ---
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
        """Append new exercises to an existing workout session note."""
        filepath = self.notes_dir / session_filename
        fm, body = self.read_file(filepath)
        if "exercises" not in fm or not isinstance(fm["exercises"], list):
            fm["exercises"] = []
        fm["exercises"].extend(new_exercises)
        fm["dateModified"] = get_current_iso(with_seconds=False)
        self.write_file(filepath, fm, body)
        self._session_cache = None
        self.logger.info("Appended %d exercise(s) to session %s", len(new_exercises), session_filename)

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

    def save_workout_session(self, exercises: List[Dict[str, Any]]) -> str:
        """Save a workout session note with recorded exercises."""
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
            "exercises": exercises,
        }
        self.write_file(filepath, frontmatter)
        self._session_cache = None
        self.logger.info("Saved workout session: %s", filename)
        return filename

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
                        exercise_title = next((e.title for e in exercises if e.filename == ex.id), "")
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
                    exercise_title = next((e.title for e in exercises if e.filename == ex.id), "")
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

    def backup_data(self) -> Path:
        """Create a ZIP archive backup of all markdown files in the notes directory."""
        backup_dir = self.notes_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{timestamp}.zip"
        try:
            with zipfile.ZipFile(backup_file, "w") as zf:
                for md_file in self.notes_dir.glob("*.md"):
                    zf.write(md_file, arcname=md_file.name)
            self.logger.info("Backup created at %s", backup_file)
        except Exception as err:
            self.logger.exception("Failed to create backup: %s", err)
        return backup_file

    def load_exercises(self) -> List[Exercise]:
        """Load and return all exercises from the external index (index.json)."""
        if self._exercise_cache is not None:
            return self._exercise_cache
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        exercises = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []  # Ensure tags is always a list.
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

    def list_workout_sessions(self) -> List[WorkoutSession]:
        """Return a list of all workout sessions from the external index."""
        if self._session_cache is not None:
            return self._session_cache
        try:
            data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        sessions = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []  # Ensure tags is always iterable.
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
                            workout_sets.append(WorkoutSet(reps=int(s.get("reps", 0)), weight=float(s.get("weight", 0))))
                        except Exception:
                            continue
                    session_exercises.append(WorkoutExercise(id=ex.get("id"), title="", sets=workout_sets))
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
            data = json.loads(self.index_file.read_text(encoding="utf-8")) or []
        except Exception as err:
            self.logger.error("Error reading index file: %s", err)
            data = []
        templates = []
        for note in data:
            meta = note.get("extra", note)
            tags = meta.get("tags") or []  # Ensure tags is a list.
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

    def setup_colors(self) -> None:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)    # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)     # Highlight
        curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)   # Footer/Status
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)    # Normal text

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

    def add_to_recent_workout(self) -> None:
        """Allow the user to add an exercise to the most recent workout session."""
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No recent workout found.", 3)
            self.pause("Press any key to return to main menu...")
            return

        # Assume the first session is the most recent.
        recent_session = sessions[0]

        # Convert session exercises into a list of dicts for display.
        session_exercises = [
            {
                "id": we.id,
                "title": we.title,
                "sets": [{"reps": s.reps, "weight": s.weight} for s in we.sets]
            }
            for we in recent_session.exercises
        ]

        self.clear_screen()
        self.draw_header(f"Resume Workout: {recent_session.title}")
        self.show_footer("Select an exercise to add to your workout", 3)

        ex = self.choose_exercise(session_exercises)
        if not ex:
            self.show_footer("No exercise selected.", 3)
            self.pause("Press any key to return to main menu...")
            return

        result = self.record_exercise(ex)
        if not result:
            self.show_footer("No exercise recorded.", 3)
            self.pause("Press any key to return to main menu...")
            return

        self.dm.append_to_workout_session(recent_session.filename, [result])
        self.show_footer(f"Exercise added to session '{recent_session.title}'.", 3)
        self.pause("Press any key to return to main menu...")

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
        self.pause("Exercise sets recorded. Press any key to continue...")
        self.dm.logger.debug("Recorded %d sets for %s", len(recorded_sets), exercise.title)
        return {"id": exercise.filename, "title": exercise.title, "sets": recorded_sets}

    def record_session(self, prepopulated: Optional[List[Exercise]] = None) -> None:
        self.clear_screen()
        self.draw_header("Start Workout Session")
        self.pause("Get ready! Press any key to begin your workout...")
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
            if add_more.lower() != "y":
                break
        if not session_exercises:
            self.show_footer("No exercises recorded for this session.", 3)
            self.pause()
            return
        self.dm.save_workout_session(session_exercises)
        self.show_footer("Workout session saved.", 3)
        self.pause("Session complete. Press any key to return to the main menu...")
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
        if not hasattr(self, "history_index") or self.history_index is None:
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
        del win
        self.clear_screen()
        self.draw_header("Select an Exercise (or / to search)")

    def show_session_summary_popup(self, session_exercises: List[Dict[str, Any]]) -> None:
        max_y, max_x = self.stdscr.getmaxyx()
        popup_h = min(20, max_y - 4)
        popup_w = min(70, max_x - 4)
        begin_y = (max_y - popup_h) // 2
        begin_x = (max_x - popup_w) // 2
        win = curses.newwin(popup_h, popup_w, begin_y, begin_x)
        draw_box(win, " Session Summary ")
        if not session_exercises:
            win.addstr(2, 2, "No exercises recorded yet.")
        else:
            row = 2
            for ex in session_exercises:
                title = ex.get("title", "Unknown")
                sets = ex.get("sets", [])
                line = f"{title} - {len(sets)} set(s)"
                win.addnstr(row, 2, line, popup_w - 4)
                row += 1
                for idx, s in enumerate(sets, start=1):
                    detail = f"  Set {idx}: {s.get('reps', '')} reps @ {s.get('weight', '')}"
                    if row < popup_h - 2:
                        win.addnstr(row, 4, detail, popup_w - 6)
                        row += 1
                    else:
                        break
                if row < popup_h - 2:
                    row += 1
                else:
                    break
        win.addnstr(popup_h - 2, 2, "Press any key to close...", popup_w - 4)
        win.refresh()
        win.getch()
        self.clear_screen()
        self.draw_header("Select an Exercise (or / to search)")

    def choose_exercise(self, session_exercises: List[Dict[str, Any]], exercises: Optional[List[Exercise]] = None) -> Optional[Exercise]:
        if exercises is None:
            exercises = self.dm.load_exercises()
        if not hasattr(self, "history_index") or self.history_index is None:
            self.history_index = self.build_history_index()
        cursor = 0
        offset = 0
        while True:
            max_y, max_x = self.stdscr.getmaxyx()
            list_h = max_y - 20
            self.clear_screen()
            self.draw_header("Select an Exercise (or / to search)")
            header_text = f"{'Title':<{max_x-35}} {'Status':<10} Equipment"
            self.stdscr.attron(curses.A_BOLD | curses.color_pair(3))
            self.stdscr.addstr(2, 2, header_text[:max_x-4])
            self.stdscr.attroff(curses.A_BOLD | curses.color_pair(3))
            visible_exercises = exercises[offset:offset + (list_h - 4)]
            for idx, ex in enumerate(visible_exercises):
                actual_idx = offset + idx
                status = "Planned" if ex.planned else ""
                equip = ", ".join(ex.equipment) if isinstance(ex.equipment, list) else str(ex.equipment or "")
                line = f"{ex.title:<{max_x-35}} {status:<10} {equip}"
                row = 3 + idx
                if actual_idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(row, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(row, 2, line[:max_x-4])
            key_hint = ("↑/↓: Move | Enter: Select | P: Toggle Planned | " +
                        "S: Search | D: Session Summary | H: Exercise History | Q: Cancel")
            self.show_footer(key_hint, 3)
            self.stdscr.refresh()
            preview_h = 18
            preview_win = curses.newwin(preview_h, max_x-2, max_y - preview_h - 2, 1)
            draw_box(preview_win, " Preview ")
            selected = exercises[cursor]
            preview_win.addstr(1, 2, f"Title: {selected.title}")
            equip = ", ".join(selected.equipment) if isinstance(selected.equipment, list) else str(selected.equipment or "")
            preview_win.addstr(2, 2, f"Equipment: {equip}")
            status = "Planned" if selected.planned else "Not Planned"
            preview_win.addstr(3, 2, f"Status: {status}")
            preview_win.addstr(4, 2, "Recent History:")
            recent_history_max_lines = preview_h - 8
            line_idx = 5
            recent_sessions = self.history_index.get(selected.filename, [])
            if recent_sessions:
                for date, sets in recent_sessions[:3]:
                    if (line_idx - 5) < recent_history_max_lines:
                        set_str = ", ".join([f"{s.reps}r@{s.weight}" for s in sets])
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
                        preview_win.addstr(summary_line, 4,
                                            f"• {recorded_ex.get('title','')} - {len(recorded_ex.get('sets', []))} set(s)")
                        summary_line += 1
            else:
                preview_win.addstr(summary_line, 4, "(none)")
            preview_win.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
                    if cursor < offset:
                        offset = cursor
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(exercises) - 1:
                    cursor += 1
                    if cursor >= offset + (list_h - 4):
                        offset += 1
            elif k in (10, 13):
                self.dm.logger.debug("Exercise selected: %s", exercises[cursor].title)
                return exercises[cursor]
            elif k in (ord('p'), ord('P')):
                if exercises:
                    ex = exercises[cursor]
                    new_state = self.dm.toggle_exercise_planned(ex.filename)
                    ex.planned = new_state
                    state_text = "Planned" if new_state else "Not Planned"
                    self.show_footer(f"'{ex.title}' toggled to {state_text}", 3)
                    self.pause("Press any key...")
                    self.dm.logger.info("Toggled planned status for: %s", ex.title)
            elif k in (ord('s'), ord('S')):
                keyword = self.prompt_input("Search keyword: ", 2, 2)
                if keyword:
                    exercises = self.dm.search_exercises(keyword)
                    cursor = 0
                    offset = 0
            elif k in (ord('d'), ord('D')):
                self.show_session_summary_popup(session_exercises)
            elif k in (ord('h'), ord('H')):
                self.show_exercise_history_popup(exercises[cursor])
            elif k in (ord('q'), ord('Q')):
                return None

    def list_templates(self) -> Optional[WorkoutTemplate]:
        templates = self.dm.load_templates()
        if not templates:
            self.show_footer("No workout templates available.", 3)
            self.pause()
            return None
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Select Workout Template")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, tmpl in enumerate(templates):
                y = 3 + idx
                disp = f"{tmpl.title}: {tmpl.description}"
                if y < max_y - 2:
                    if idx == cursor:
                        self.stdscr.attron(curses.color_pair(2))
                        self.stdscr.addstr(y, 2, disp[:max_x-4])
                        self.stdscr.attroff(curses.color_pair(2))
                    else:
                        self.stdscr.addstr(y, 2, disp[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Select | Q: Back", 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(templates) - 1:
                    cursor += 1
            elif k in (10, 13):
                self.dm.logger.debug("Template selected: %s", templates[cursor].title)
                return templates[cursor]
            elif k in (ord('q'), ord('Q')):
                return None

    def start_session_from_template(self) -> None:
        tmpl = self.list_templates()
        if not tmpl:
            return
        exercises = self.dm.load_exercises()
        tmpl_exercises = [ex for ex in exercises if ex.filename in tmpl.exercises]
        if not tmpl_exercises:
            self.show_footer("No valid exercises found in template.", 3)
            self.pause()
            return
        self.show_footer(f"Starting workout from template '{tmpl.title}'", 3)
        self.pause("Press any key to continue...")
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
        self.show_footer(f"Template '{tmpl_name}' created.", 3)
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Created workout template: %s", tmpl_name)

    def edit_template(self) -> None:
        tmpl = self.list_templates()
        if not tmpl:
            return
        self.clear_screen()
        self.draw_header("Edit Workout Template")
        new_name = self.prompt_input(f"New name [{tmpl.title}]: ", 4, 4, tmpl.title)
        new_desc = self.prompt_input(f"New description [{tmpl.description}]: ", 5, 4, tmpl.description)
        rebuild = self.prompt_input("Rebuild exercise list? (y/N): ", 7, 4)
        exercises = []
        if rebuild.lower() == 'y':
            while True:
                add = self.prompt_input("Add an exercise? (y/N): ", 9, 4)
                if add.lower() != "y":
                    break
                ex = self.choose_exercise([])
                if ex:
                    exercises.append(ex)
                else:
                    break
            if not exercises:
                self.show_footer("No exercises selected. Edit cancelled.", 3)
                self.pause()
                return
        else:
            ex_list = self.dm.load_exercises()
            for ex_id in tmpl.exercises:
                for ex in ex_list:
                    if ex.filename == ex_id:
                        exercises.append(ex)
                        break
        self.dm.update_template(tmpl.filename, new_name, new_desc, exercises)
        self.show_footer(f"Template '{new_name}' updated.", 3)
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Edited template: %s", new_name)

    def delete_template_ui(self) -> None:
        tmpl = self.list_templates()
        if not tmpl:
            return
        self.clear_screen()
        self.draw_header("Delete Workout Template")
        confirm = self.prompt_input(f"Delete template '{tmpl.title}'? (y/N): ", 4, 4)
        if confirm.lower() == "y":
            self.dm.delete_template(tmpl.filename)
            self.show_footer(f"Template '{tmpl.title}' deleted.", 3)
            self.dm.logger.info("Deleted template: %s", tmpl.title)
        else:
            self.show_footer("Deletion cancelled.", 3)
        self.pause("Press any key to return to main menu...")

    def add_new_exercise(self) -> None:
        self.clear_screen()
        self.draw_header("Create New Exercise")
        title = self.prompt_input("Exercise title: ", 4, 4)
        equipment = self.prompt_input("Equipment (comma separated): ", 5, 4)
        if title:
            self.dm.create_exercise(title, equipment)
            self.show_footer(f"New exercise '{title}' created.", 3)
            self.pause("Exercise created. Press any key to continue...")
            self.dm.logger.info("New exercise created: %s", title)
        else:
            self.show_footer("No title provided. Cancelled.", 3)
            self.pause("Press any key to return to main menu...")

    def edit_exercise_ui(self) -> None:
        exercises = self.dm.load_exercises()
        if not exercises:
            self.show_footer("No exercises available to edit.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Select an Exercise to Edit")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, ex in enumerate(exercises):
                line = f"{ex.title} (Equipment: {', '.join(ex.equipment)})"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Edit | Q: Cancel", 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(exercises) - 1:
                    cursor += 1
            elif k in (10, 13):
                ex = exercises[cursor]
                new_title = self.prompt_input(f"New title [{ex.title}]: ", 2, 2, ex.title)
                new_equipment = self.prompt_input(f"New equipment (comma separated) [{', '.join(ex.equipment)}]: ", 3, 2, ", ".join(ex.equipment))
                self.dm.edit_exercise(ex.filename, new_title, new_equipment)
                self.show_footer("Exercise updated.", 3)
                self.pause("Press any key to return to main menu...")
                break
            elif k in (ord('q'), ord('Q')):
                break

    def delete_exercise_ui(self) -> None:
        exercises = self.dm.load_exercises()
        if not exercises:
            self.show_footer("No exercises available to delete.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Select an Exercise to Delete")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, ex in enumerate(exercises):
                line = f"{ex.title}"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Delete | Q: Cancel", 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(exercises) - 1:
                    cursor += 1
            elif k in (10, 13):
                ex = exercises[cursor]
                confirm = self.prompt_input(f"Delete exercise '{ex.title}'? (y/N): ", 2, 2)
                if confirm.lower() == "y":
                    self.dm.delete_exercise(ex.filename)
                    self.show_footer("Exercise deleted.", 3)
                    self.pause("Press any key to return to main menu...")
                    break
            elif k in (ord('q'), ord('Q')):
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
            list_h = max_y - 8
            for idx, session in enumerate(filtered_sessions[:list_h]):
                line = f"{session.date} - {session.title}"
                y = 3 + idx
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 2, line[:max_x-4])
            hint = "↑/↓: Move | Enter: View Details | F: Filter | Q: Back"
            self.show_footer(hint, 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(filtered_sessions) - 1:
                    cursor += 1
            elif k in (10, 13):
                self.show_session_details(filtered_sessions[cursor])
            elif k in (ord('f'), ord('F')):
                keyword = self.prompt_input("Filter sessions by exercise title: ", 2, 2)
                if keyword:
                    all_ex = self.dm.load_exercises()
                    filtered_sessions = [s for s in sessions if any(keyword.lower() in next((e.title for e in all_ex if e.filename == ex.id), "").lower() for ex in s.exercises)]
                    cursor = 0
                else:
                    filtered_sessions = sessions
            elif k in (ord('q'), ord('Q')):
                break

    def show_session_details(self, session: WorkoutSession) -> None:
        self.clear_screen()
        self.draw_header(f"Session Details: {session.title}")
        max_y, max_x = self.stdscr.getmaxyx()
        details_win = curses.newwin(max_y - 6, max_x - 4, 3, 2)
        draw_box(details_win, " Session Info ")
        details_win.addstr(1, 2, f"Date: {session.date}")
        details_win.addstr(2, 2, f"Title: {session.title}")
        details_win.addstr(4, 2, "Exercises:")
        line = 5
        all_exercises = self.dm.load_exercises()
        for ex in session.exercises:
            ex_title = next((e.title for e in all_exercises if e.filename == ex.id), ex.id)
            details_win.addstr(line, 4, f"- {ex_title} ({len(ex.sets)} set(s))")
            line += 1
            for idx, s in enumerate(ex.sets, start=1):
                details_win.addstr(line, 6, f"Set {idx}: Reps: {s.reps}, Weight: {s.weight}")
                line += 1
            total_reps = sum(s.reps for s in ex.sets)
            total_weight = sum(s.weight for s in ex.sets)
            avg_weight = total_weight / len(ex.sets) if ex.sets else 0
            details_win.addstr(line, 6, f"Totals: {total_reps} reps, Total Weight: {total_weight:.1f}, Avg Weight: {avg_weight:.1f}")
            line += 2
            if line > max_y - 3:
                details_win.addstr(max_y - 3, 2, "-- More details available, resize window for full view --")
                break
        details_win.refresh()
        self.pause("Press any key to return to history view...")

    def delete_session_ui(self) -> None:
        sessions = self.dm.list_workout_sessions()
        if not sessions:
            self.show_footer("No sessions available to delete.", 3)
            self.pause()
            return
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Select a Session to Delete")
            max_y, max_x = self.stdscr.getmaxyx()
            for idx, session in enumerate(sessions):
                line = f"{session.date} - {session.title}"
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(3+idx, 2, line[:max_x-4])
            self.show_footer("↑/↓: Move | Enter: Delete | Q: Cancel", 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                if cursor > 0:
                    cursor -= 1
            elif k in (curses.KEY_DOWN, ord('j')):
                if cursor < len(sessions) - 1:
                    cursor += 1
            elif k in (10, 13):
                sess = sessions[cursor]
                confirm = self.prompt_input(f"Delete session '{sess.title}'? (y/N): ", 2, 2)
                if confirm.lower() == "y":
                    self.dm.delete_workout_session(sess.filename)
                    self.show_footer("Session deleted.", 3)
                    self.pause("Press any key to return to main menu...")
                    break
            elif k in (ord('q'), ord('Q')):
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
        self.clear_screen()
        self.draw_header("Backup Data")
        self.stdscr.addstr(3, 2, f"Backup created at: {backup_file}")
        self.stdscr.refresh()
        self.pause("Press any key to return to main menu...")

    def show_about(self) -> None:
        self.clear_screen()
        self.draw_header("About Ultimate Workout Logger")
        about_text = [
            "Ultimate Workout Logger v2.0",
            "Developed for power users who want complete control",
            "over their fitness data.",
            "",
            "Features:",
            " - Record & manage workouts",
            " - Edit, delete, search exercises & sessions",
            " - View aggregated statistics",
            " - Export data in CSV/JSON",
            " - Automated backup",
            "",
            "Press any key to return to the main menu..."
        ]
        max_y, max_x = self.stdscr.getmaxyx()
        for idx, line in enumerate(about_text, start=3):
            if idx < max_y - 2:
                self.stdscr.addstr(idx, 2, line[:max_x-4])
        self.stdscr.refresh()
        self.stdscr.getch()

    def export_history_ui(self) -> None:
        self.clear_screen()
        self.draw_header("Export Workout History")
        choice = self.prompt_input("Export as (1) CSV or (2) JSON? [1]: ", 4, 4, "1")
        export_path_str = self.prompt_input("Export file path: ", 5, 4, "workout_history." + ("csv" if choice=="1" else "json"))
        export_path = Path(export_path_str).expanduser()
        if choice == "2":
            self.dm.export_history_json(export_path)
        else:
            self.dm.export_history_csv(export_path)
        self.show_footer(f"History exported to {export_path}", 3)
        self.pause("Press any key to return to main menu...")
        self.dm.logger.info("Workout history exported to: %s", export_path)

    def show_help(self) -> None:
        self.clear_screen()
        self.draw_header("Help & Instructions")
        help_text = [
            "Navigation:",
            "  ↑/↓ or k/j : Move selection",
            "  Enter      : Choose option",
            "  Q          : Go back / Quit",
            "",
            "In Exercise Selection:",
            "  P          : Toggle Planned status",
            "  S          : Search exercises",
            "",
            "Other Features:",
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

    def main_menu(self) -> None:
        menu_options = [
            ("Record a Workout Session", self.record_session),
            ("Add to Most Recent Workout", self.add_to_recent_workout),  # New option
            ("View Workout History", self.view_history),
            ("Delete a Workout Session", self.delete_session_ui),
            ("Create a New Exercise", self.add_new_exercise),
            ("Edit an Exercise", self.edit_exercise_ui),
            ("Delete an Exercise", self.delete_exercise_ui),
            ("Create Workout Template", self.create_template),
            ("Start Workout from Template", self.start_session_from_template),
            ("Edit Workout Template", self.edit_template),
            ("Delete Workout Template", self.delete_template_ui),
            ("Export Workout History", self.export_history_ui),
            ("View Statistics", self.view_statistics),
            ("Backup Data", self.backup_data_ui),
            ("About", self.show_about),
            ("Help", self.show_help),
            ("Quit", None),
        ]
        cursor = 0
        while True:
            self.clear_screen()
            self.draw_header("Ultimate Workout Logger – Main Menu")
            for idx, (option, _) in enumerate(menu_options):
                y = 3 + idx 
                if idx == cursor:
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(y, 4, f"> {option}")
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(y, 4, f"  {option}")
            hint = "↑/↓: Move | Enter: Select | Q: Quit"
            self.show_footer(hint, 3)
            self.stdscr.refresh()
            k = self.stdscr.getch()
            if k in (curses.KEY_UP, ord('k')):
                cursor = max(0, cursor - 1)
            elif k in (curses.KEY_DOWN, ord('j')):
                cursor = min(len(menu_options) - 1, cursor + 1)
            elif k in (10, 13):
                if menu_options[cursor][1] is None or menu_options[cursor][0] == "Quit":
                    self.dm.logger.info("Exiting workout logger.")
                    break
                menu_options[cursor][1]()
            elif k in (ord('q'), ord('Q')):
                self.dm.logger.info("User requested exit from main menu.")
                break

# --- Main Entrypoint ---
def main(stdscr: Any) -> None:
    parser = argparse.ArgumentParser(description="Ultimate Workout Logger")
    parser.add_argument("--notes-dir", type=str, help="Path to your notes directory.")
    parser.add_argument("--export-history", action="store_true", help="Export workout history and exit.")
    parser.add_argument("--export-json", action="store_true", help="Export workout history in JSON format and exit.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging.")
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

    dm = DataManager(NOTES_DIR, INDEX_FILE, LOG_FILE_PATH)
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

