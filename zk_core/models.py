"""Data models for ZK Core."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import datetime

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


@dataclass(frozen=True)
class Note:
    """Represents a single note in the Zettelkasten system."""
    filename: str
    title: str = ""
    tags: List[str] = field(default_factory=list)
    dateModified: str = ""
    dateCreated: str = ""
    aliases: List[str] = field(default_factory=list)
    givenName: str = ""
    familyName: str = ""
    outgoing_links: List[str] = field(default_factory=list)
    backlinks: List[str] = field(default_factory=list)
    word_count: int = 0
    file_size: int = 0
    body: str = ""
    references: List[str] = field(default_factory=list)
    _extra_fields: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Note':
        """Create a Note object from a dictionary."""
        standard_fields = {
            'filename', 'title', 'tags', 'dateModified', 'dateCreated',
            'aliases', 'givenName', 'familyName', 'outgoing_links', 'backlinks', 
            'word_count', 'file_size', 'body', 'references'
        }
        extra_fields = {k: v for k, v in data.items() if k not in standard_fields}
        
        return cls(
            filename=data.get('filename', ''),
            title=data.get('title', '') or "",
            tags=data.get('tags', []) if isinstance(data.get('tags', []), list) else [],
            dateModified=data.get('dateModified', ''),
            dateCreated=data.get('dateCreated', ''),
            aliases=data.get('aliases', []) if isinstance(data.get('aliases', []), list) else [],
            givenName=data.get('givenName', ''),
            familyName=data.get('familyName', '') or "",
            outgoing_links=data.get('outgoing_links', []) if isinstance(data.get('outgoing_links', []), list) else [],
            backlinks=data.get('backlinks', []) if isinstance(data.get('backlinks', []), list) else [],
            word_count=data.get('word_count', 0) if isinstance(data.get('word_count', 0), int) else 0,
            file_size=data.get('file_size', 0) if isinstance(data.get('file_size', 0), int) else 0,
            body=data.get('body', ''),
            references=data.get('references', []) if isinstance(data.get('references', []), list) else [],
            _extra_fields=extra_fields
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Note to dictionary, including extra fields."""
        data = asdict(self)
        extra_fields = data.pop('_extra_fields', {})
        data.update(extra_fields)
        return data

    def get_field(self, field_name: str) -> Any:
        """Get a field value by name, looking in standard and extra fields."""
        if hasattr(self, field_name):
            return getattr(self, field_name)
        return self._extra_fields.get(field_name, "")


@dataclass
class IndexInfo:
    """Information about the note index."""
    note_count: int = 0
    total_word_count: int = 0
    average_word_count: float = 0.0
    median_word_count: float = 0.0
    min_word_count: int = 0
    max_word_count: int = 0
    total_file_size_bytes: int = 0
    average_file_size_bytes: float = 0.0
    median_file_size_bytes: float = 0.0
    min_file_size_bytes: int = 0
    max_file_size_bytes: int = 0
    average_tags_per_note: float = 0.0
    median_tags_per_note: float = 0.0
    orphan_notes_count: int = 0
    untagged_orphan_notes_count: int = 0
    date_range: str = "N/A"
    dangling_links_count: int = 0
    unique_tag_count: int = 0
    most_common_tags: List[tuple] = field(default_factory=list)
    extra_frontmatter_keys: List[tuple] = field(default_factory=list)
    notes_by_day_of_week: Dict[str, int] = field(default_factory=dict)
    peak_creation_day: str = "N/A"
    notes_by_year: Dict[int, int] = field(default_factory=dict)
    
    # Network and connectivity metrics
    total_links: int = 0
    average_outgoing_links: float = 0.0
    median_outgoing_links: float = 0.0
    average_backlinks: float = 0.0
    median_backlinks: float = 0.0
    highly_connected_notes: List[tuple] = field(default_factory=list)
    
    # Reference and alias statistics
    total_references: int = 0
    average_references: float = 0.0
    total_aliases: int = 0
    average_aliases: float = 0.0
    
    # Monthly patterns
    notes_by_month: Dict[str, int] = field(default_factory=dict)
    peak_creation_month: str = "N/A"


@dataclass
class WorkoutSet:
    """A single set in a workout exercise."""
    weight: float
    reps: int
    notes: Optional[str] = None


@dataclass
class WorkoutExercise:
    """A single exercise in a workout."""
    name: str
    sets: List[WorkoutSet] = field(default_factory=list)
    template_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class WorkoutSession:
    """A complete workout session."""
    date: datetime.date
    exercises: List[WorkoutExercise] = field(default_factory=list)
    duration: Optional[int] = None  # Duration in minutes
    notes: Optional[str] = None


# Pydantic models for API and validation

class NoteModel(BaseModel):
    """Pydantic model for a note, used for validation and API."""
    filename: str
    title: str = Field(default="")
    tags: List[str] = Field(default_factory=list)
    dateModified: str = Field(default="")
    dateCreated: str = Field(default="")
    aliases: List[str] = Field(default_factory=list)
    outgoing_links: List[str] = Field(default_factory=list)
    backlinks: List[str] = Field(default_factory=list)
    word_count: int = Field(default=0)
    file_size: int = Field(default=0)
    body: str = Field(default="")
    references: List[str] = Field(default_factory=list)

    class Config:
        """Model configuration."""
        extra = "allow"  # Allow extra fields


class WorkoutSetModel(BaseModel):
    """Pydantic model for a workout set."""
    weight: float
    reps: int
    notes: Optional[str] = None


class WorkoutExerciseModel(BaseModel):
    """Pydantic model for a workout exercise."""
    name: str
    sets: List[WorkoutSetModel] = Field(default_factory=list)
    template_id: Optional[str] = None
    notes: Optional[str] = None


class WorkoutSessionModel(BaseModel):
    """Pydantic model for a workout session."""
    date: datetime.date
    exercises: List[WorkoutExerciseModel] = Field(default_factory=list)
    duration: Optional[int] = None
    notes: Optional[str] = None


class NoteDict(TypedDict, total=False):
    """TypedDict for note data, providing better type checking."""
    filename: str
    title: str
    tags: List[str]
    dateModified: str
    dateCreated: str
    aliases: List[str]
    givenName: str
    familyName: str
    outgoing_links: List[str]
    backlinks: List[str]
    word_count: int
    file_size: int
    body: str
    references: List[str]