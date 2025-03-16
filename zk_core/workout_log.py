"""
Workout logging module.

This module provides functionality for logging and tracking workout sessions,
managing exercises and templates, and viewing workout history and statistics.
"""

import os
import sys
import csv
import json
import yaml
import curses
import logging
import zipfile
import argparse
import datetime
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

from zk_core.config import load_config, get_config_value, resolve_path
from zk_core.utils import extract_frontmatter_and_body, json_ready
from zk_core.models import WorkoutSet, WorkoutExercise, WorkoutSession

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
WORKOUT_LOG_DIR = "workout_log"
EXERCISE_DIR = "exercises"
TEMPLATE_DIR = "templates"
SESSION_DIR = "sessions"
BACKUP_DIR = "backups"
DATA_FILE = "workout_data.json"


class WorkoutLog:
    """Main class for workout logging functionality."""
    
    def __init__(self, notes_dir: str, backup_interval: int = 1, verbose: bool = False):
        """
        Initialize the workout logging system.
        
        Args:
            notes_dir: Path to the notes directory
            backup_interval: Interval in days to create backups
            verbose: Enable verbose logging
        """
        self.notes_dir = Path(notes_dir)
        self.backup_interval = backup_interval
        self.verbose = verbose
        
        # Set up logging level
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        # Initialize directories
        self.workout_dir = self.notes_dir / WORKOUT_LOG_DIR
        self.exercise_dir = self.workout_dir / EXERCISE_DIR
        self.template_dir = self.workout_dir / TEMPLATE_DIR
        self.session_dir = self.workout_dir / SESSION_DIR
        self.backup_dir = self.workout_dir / BACKUP_DIR
        self.data_file = self.workout_dir / DATA_FILE
        
        # Create required directories
        self._create_directories()
        
        # Load data
        self.exercises = self._load_exercises()
        self.templates = self._load_templates()
        self.sessions = self._load_sessions()
        
        # Create backup if needed
        self._create_backup_if_needed()
    
    def _create_directories(self) -> None:
        """Create the necessary directories for the workout logging system."""
        for directory in [self.workout_dir, self.exercise_dir, self.template_dir, 
                        self.session_dir, self.backup_dir]:
            directory.mkdir(exist_ok=True, parents=True)
    
    def _load_exercises(self) -> Dict[str, Dict[str, Any]]:
        """Load all exercises from the exercise directory."""
        exercises = {}
        for file in self.exercise_dir.glob("*.md"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                meta, _ = extract_frontmatter_and_body(content)
                if meta:
                    exercise_id = file.stem
                    exercises[exercise_id] = meta
            except Exception as e:
                logger.error(f"Error loading exercise {file}: {e}")
        
        logger.debug(f"Loaded {len(exercises)} exercises")
        return exercises
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all templates from the template directory."""
        templates = {}
        for file in self.template_dir.glob("*.md"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                meta, _ = extract_frontmatter_and_body(content)
                if meta:
                    template_id = file.stem
                    templates[template_id] = meta
            except Exception as e:
                logger.error(f"Error loading template {file}: {e}")
        
        logger.debug(f"Loaded {len(templates)} templates")
        return templates
    
    def _load_sessions(self) -> Dict[str, WorkoutSession]:
        """Load all workout sessions from the session directory."""
        sessions = {}
        for file in self.session_dir.glob("*.md"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
                meta, _ = extract_frontmatter_and_body(content)
                if meta and 'date' in meta and 'exercises' in meta:
                    session_id = file.stem
                    
                    # Convert date string to date object if needed
                    if isinstance(meta['date'], str):
                        try:
                            meta['date'] = datetime.datetime.strptime(meta['date'], "%Y-%m-%d").date()
                        except ValueError:
                            logger.warning(f"Invalid date format in {file}")
                            continue
                    
                    # Create exercise objects
                    exercises = []
                    for ex in meta['exercises']:
                        sets = []
                        for s in ex.get('sets', []):
                            sets.append(WorkoutSet(
                                weight=float(s.get('weight', 0)),
                                reps=int(s.get('reps', 0)),
                                notes=s.get('notes')
                            ))
                        
                        exercises.append(WorkoutExercise(
                            name=ex.get('name', ""),
                            sets=sets,
                            template_id=ex.get('template_id'),
                            notes=ex.get('notes')
                        ))
                    
                    sessions[session_id] = WorkoutSession(
                        date=meta['date'],
                        exercises=exercises,
                        duration=meta.get('duration'),
                        notes=meta.get('notes')
                    )
            except Exception as e:
                logger.error(f"Error loading session {file}: {e}")
        
        logger.debug(f"Loaded {len(sessions)} workout sessions")
        return sessions
    
    def add_exercise(self, name: str, muscle_groups: List[str], equipment: Optional[str] = None, 
                     description: Optional[str] = None) -> str:
        """
        Add a new exercise.
        
        Args:
            name: Name of the exercise
            muscle_groups: List of muscle groups
            equipment: Optional equipment needed
            description: Optional description
            
        Returns:
            ID of the new exercise
        """
        exercise_id = name.lower().replace(" ", "_")
        
        # Add a number suffix if ID already exists
        if exercise_id in self.exercises:
            count = 1
            while f"{exercise_id}_{count}" in self.exercises:
                count += 1
            exercise_id = f"{exercise_id}_{count}"
        
        exercise = {
            "name": name,
            "muscle_groups": muscle_groups,
            "equipment": equipment,
            "description": description,
            "date_created": datetime.date.today().isoformat()
        }
        
        # Remove None values
        exercise = {k: v for k, v in exercise.items() if v is not None}
        
        # Save to file
        exercise_path = self.exercise_dir / f"{exercise_id}.md"
        with open(exercise_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            yaml.dump(exercise, f, default_flow_style=False)
            f.write("---\n\n")
            if description:
                f.write(description)
        
        # Add to in-memory dictionary
        self.exercises[exercise_id] = exercise
        
        logger.debug(f"Added exercise: {name} (ID: {exercise_id})")
        return exercise_id
    
    def add_template(self, name: str, exercises: List[Dict[str, Any]]) -> str:
        """
        Add a new workout template.
        
        Args:
            name: Name of the template
            exercises: List of exercise configurations
            
        Returns:
            ID of the new template
        """
        template_id = name.lower().replace(" ", "_")
        
        # Add a number suffix if ID already exists
        if template_id in self.templates:
            count = 1
            while f"{template_id}_{count}" in self.templates:
                count += 1
            template_id = f"{template_id}_{count}"
        
        template = {
            "name": name,
            "exercises": exercises,
            "date_created": datetime.date.today().isoformat()
        }
        
        # Save to file
        template_path = self.template_dir / f"{template_id}.md"
        with open(template_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            yaml.dump(template, f, default_flow_style=False)
            f.write("---\n\n")
            f.write(f"# {name} Workout\n\n")
            
            for ex in exercises:
                f.write(f"## {ex['name']}\n\n")
                if ex.get('sets'):
                    for i, s in enumerate(ex['sets'], 1):
                        f.write(f"- Set {i}: {s['reps']} reps")
                        if s.get('weight'):
                            f.write(f" @ {s['weight']} kg")
                        if s.get('notes'):
                            f.write(f" - {s['notes']}")
                        f.write("\n")
                f.write("\n")
        
        # Add to in-memory dictionary
        self.templates[template_id] = template
        
        logger.debug(f"Added template: {name} (ID: {template_id})")
        return template_id
    
    def add_session(self, session: WorkoutSession) -> str:
        """
        Add a new workout session.
        
        Args:
            session: WorkoutSession object
            
        Returns:
            ID of the new session
        """
        # Generate a unique ID based on date and time
        session_date = session.date.isoformat()
        session_id = f"workout_{session_date}"
        
        # Add timestamp if ID already exists
        if session_id in self.sessions:
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            session_id = f"{session_id}_{timestamp}"
        
        # Convert to dictionary for saving
        session_dict = {
            "date": session.date.isoformat(),
            "exercises": [],
            "duration": session.duration,
            "notes": session.notes
        }
        
        for ex in session.exercises:
            exercise_dict = {
                "name": ex.name,
                "template_id": ex.template_id,
                "sets": [],
                "notes": ex.notes
            }
            
            for s in ex.sets:
                set_dict = {
                    "weight": s.weight,
                    "reps": s.reps,
                    "notes": s.notes
                }
                exercise_dict["sets"].append(set_dict)
            
            session_dict["exercises"].append(exercise_dict)
        
        # Remove None values
        session_dict = json_ready(session_dict)
        session_dict = {k: v for k, v in session_dict.items() if v is not None}
        
        # Save to file
        session_path = self.session_dir / f"{session_id}.md"
        with open(session_path, "w", encoding="utf-8") as f:
            f.write("---\n")
            yaml.dump(session_dict, f, default_flow_style=False)
            f.write("---\n\n")
            f.write(f"# Workout Session: {session_date}\n\n")
            
            for ex in session.exercises:
                f.write(f"## {ex.name}\n\n")
                for i, s in enumerate(ex.sets, 1):
                    f.write(f"- Set {i}: {s.reps} reps @ {s.weight} kg")
                    if s.notes:
                        f.write(f" - {s.notes}")
                    f.write("\n")
                if ex.notes:
                    f.write(f"\nNotes: {ex.notes}\n")
                f.write("\n")
            
            if session.notes:
                f.write(f"## Notes\n\n{session.notes}\n")
        
        # Add to in-memory dictionary
        self.sessions[session_id] = session
        
        logger.debug(f"Added session: {session_id}")
        return session_id
    
    def get_workout_history(self, start_date: Optional[datetime.date] = None, 
                            end_date: Optional[datetime.date] = None,
                            exercise_filter: Optional[str] = None) -> List[WorkoutSession]:
        """
        Get workout history filtered by date range and/or exercise.
        
        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            exercise_filter: Optional exercise name filter
            
        Returns:
            List of WorkoutSession objects matching the criteria
        """
        filtered_sessions = []
        
        for session_id, session in self.sessions.items():
            # Apply date filters
            if start_date and session.date < start_date:
                continue
            if end_date and session.date > end_date:
                continue
            
            # Apply exercise filter
            if exercise_filter:
                exercise_match = False
                for ex in session.exercises:
                    if exercise_filter.lower() in ex.name.lower():
                        exercise_match = True
                        break
                if not exercise_match:
                    continue
            
            filtered_sessions.append(session)
        
        # Sort by date
        filtered_sessions.sort(key=lambda s: s.date, reverse=True)
        
        return filtered_sessions
    
    def export_history_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export workout history to CSV.
        
        Args:
            output_path: Optional path for the CSV file
            
        Returns:
            Path to the created CSV file
        """
        if output_path:
            csv_path = Path(output_path)
        else:
            csv_path = self.workout_dir / "workout_history.csv"
        
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Exercise", "Set", "Weight (kg)", "Reps", "Notes"])
            
            for session_id, session in sorted(self.sessions.items(), 
                                              key=lambda x: x[1].date):
                for ex in session.exercises:
                    for i, s in enumerate(ex.sets, 1):
                        writer.writerow([
                            session.date.isoformat(),
                            ex.name,
                            i,
                            s.weight,
                            s.reps,
                            s.notes or ""
                        ])
        
        logger.info(f"Exported workout history to CSV: {csv_path}")
        return str(csv_path)
    
    def export_history_json(self, output_path: Optional[str] = None) -> str:
        """
        Export workout history to JSON.
        
        Args:
            output_path: Optional path for the JSON file
            
        Returns:
            Path to the created JSON file
        """
        if output_path:
            json_path = Path(output_path)
        else:
            json_path = self.workout_dir / "workout_history.json"
        
        # Prepare data for JSON serialization
        data = []
        for session_id, session in self.sessions.items():
            session_dict = {
                "id": session_id,
                "date": session.date.isoformat(),
                "exercises": [],
                "duration": session.duration,
                "notes": session.notes
            }
            
            for ex in session.exercises:
                exercise_dict = {
                    "name": ex.name,
                    "template_id": ex.template_id,
                    "sets": [],
                    "notes": ex.notes
                }
                
                for s in ex.sets:
                    set_dict = {
                        "weight": s.weight,
                        "reps": s.reps,
                        "notes": s.notes
                    }
                    exercise_dict["sets"].append(set_dict)
                
                session_dict["exercises"].append(exercise_dict)
            
            data.append(session_dict)
        
        # Convert to JSON-serializable format
        data = json_ready(data)
        
        # Write to file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported workout history to JSON: {json_path}")
        return str(json_path)
    
    def _create_backup_if_needed(self) -> None:
        """Create a backup if the backup interval has passed since the last backup."""
        today = datetime.date.today()
        backup_files = list(self.backup_dir.glob("workout_backup_*.zip"))
        
        if not backup_files:
            self._create_backup()
            return
        
        # Find the most recent backup
        most_recent = max(backup_files, key=lambda p: p.stat().st_mtime)
        most_recent_time = datetime.datetime.fromtimestamp(most_recent.stat().st_mtime)
        most_recent_date = most_recent_time.date()
        
        days_since_backup = (today - most_recent_date).days
        
        if days_since_backup >= self.backup_interval:
            self._create_backup()
    
    def _create_backup(self) -> str:
        """
        Create a backup of all workout data.
        
        Returns:
            Path to the created backup file
        """
        today = datetime.date.today().isoformat()
        backup_path = self.backup_dir / f"workout_backup_{today}.zip"
        
        with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add exercise files
            for file in self.exercise_dir.glob("*.md"):
                zipf.write(file, f"{EXERCISE_DIR}/{file.name}")
            
            # Add template files
            for file in self.template_dir.glob("*.md"):
                zipf.write(file, f"{TEMPLATE_DIR}/{file.name}")
            
            # Add session files
            for file in self.session_dir.glob("*.md"):
                zipf.write(file, f"{SESSION_DIR}/{file.name}")
        
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)


def run_tui(workout_log: WorkoutLog) -> None:
    """Run the text-based user interface for the workout logger."""
    # This would be a complex TUI using curses
    # For simplicity, we'll implement a basic version here
    try:
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(True)
        
        # Handle actual TUI implementation here
        # This would involve menu rendering, navigation, input handling, etc.
        
        # Display a simple menu for now
        stdscr.clear()
        stdscr.addstr(0, 0, "Workout Logger TUI")
        stdscr.addstr(2, 0, "1. Add Exercise")
        stdscr.addstr(3, 0, "2. Add Template")
        stdscr.addstr(4, 0, "3. Log Workout")
        stdscr.addstr(5, 0, "4. View History")
        stdscr.addstr(6, 0, "5. Export Data")
        stdscr.addstr(7, 0, "6. Exit")
        stdscr.addstr(9, 0, "Press a number key to select an option...")
        stdscr.refresh()
        
        # Wait for user input
        stdscr.getch()
        
    finally:
        # Clean up curses
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()


def main() -> None:
    """Main entry point for the workout log module."""
    parser = argparse.ArgumentParser(description="Workout logging system")
    parser.add_argument("--notes-dir", help="Path to notes directory")
    parser.add_argument("--export-history", action="store_true", help="Export workout history to CSV and exit")
    parser.add_argument("--export-json", action="store_true", help="Export workout history to JSON and exit")
    parser.add_argument("--backup", action="store_true", help="Create a backup and exit")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--backup-interval", type=int, default=1, help="Automatic backup interval in days")
    args = parser.parse_args()

    # Load configuration
    config = load_config()
    
    # Get notes directory
    notes_dir = args.notes_dir
    if not notes_dir:
        notes_dir = os.environ.get("NOTES_DIR")
    
    if not notes_dir:
        notes_dir = get_config_value(config, "notes_dir", os.path.expanduser("~/notes"))
    
    notes_dir = resolve_path(notes_dir)
    
    # Create the workout log instance
    workout_log = WorkoutLog(
        notes_dir=notes_dir, 
        backup_interval=args.backup_interval,
        verbose=args.verbose
    )
    
    # Handle command-line actions
    if args.export_history:
        workout_log.export_history_csv()
        sys.exit(0)
    
    if args.export_json:
        workout_log.export_history_json()
        sys.exit(0)
    
    if args.backup:
        workout_log._create_backup()
        sys.exit(0)
    
    # Run the interactive TUI
    run_tui(workout_log)


if __name__ == "__main__":
    main()