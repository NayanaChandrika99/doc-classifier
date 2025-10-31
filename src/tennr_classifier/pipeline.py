"""Core data models and pipeline skeleton for the Tennr classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class PageData:
    """Metadata for a single PDF page before OCR."""

    index: int
    image_path: Path
    width: int
    height: int
    persisted: bool = True


@dataclass
class OCRWord:
    """A single OCR-recognized word and its bounding box."""

    text: str
    bbox: Sequence[int]  # (x, y, width, height)
    confidence: float


@dataclass
class OCRResult:
    """OCR output for a page."""

    page_index: int
    text: str
    words: List[OCRWord] = field(default_factory=list)
    average_confidence: Optional[float] = None


@dataclass
class PatientEntity:
    """Structured representation of identifiers associated with a patient."""

    full_name: Optional[str] = None
    medical_record_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    phone_number: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SplitResult:
    """Resulting artifact for a patient-specific document split."""

    patient: PatientEntity
    pdf_path: Path
    metadata_path: Path
    page_indices: List[int] = field(default_factory=list)


@dataclass
class IdentifierMatch:
    """Represents a matched identifier on a page."""

    kind: str
    value: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    word_indices: List[int] = field(default_factory=list)


@dataclass
class PageEntities:
    """Collected identifier matches for a page."""

    page_index: int
    identifiers: List[IdentifierMatch] = field(default_factory=list)


@dataclass
class LinkedIdentifier:
    """Identifier aggregated across pages for a patient."""

    kind: str
    value: str
    confidence: float
    sources: List[Tuple[int, List[int]]] = field(default_factory=list)  # (page_index, word_indices)


@dataclass
class LinkedPatient:
    """Represents a patient cluster resulting from entity linking."""

    patient_id: str
    identifiers: List[LinkedIdentifier] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    score: float = 0.0


@dataclass
class AssignmentReason:
    """Reason contributing to a page assignment."""

    kind: str
    value: str
    score: float


@dataclass
class PageAssignment:
    """Assignment result for a page."""

    page_index: int
    patient_id: Optional[str]
    confidence: float
    reasons: List[AssignmentReason] = field(default_factory=list)
    manual_review: bool = False


@dataclass
class DocumentAssignmentSummary:
    """Aggregated page assignments for a document."""

    assignments: List[PageAssignment] = field(default_factory=list)
    unassigned_pages: List[int] = field(default_factory=list)
    ambiguous_pages: List[int] = field(default_factory=list)


@dataclass
class SplitArtifact:
    """Represents a generated patient-specific document."""

    patient_id: str
    pdf_path: Path
    metadata_path: Path
    pages: List[int]
    average_confidence: float


@dataclass
class DocumentSplitResult:
    """Bundle of split artifacts and summary metadata."""

    artifacts: List[SplitArtifact]
    global_metadata_path: Path
    unassigned_pdf_path: Optional[Path] = None
    unassigned_pages: List[int] = field(default_factory=list)
    ambiguous_pages: List[int] = field(default_factory=list)
    total_pages: int = 0
    assigned_pages: int = 0
    stage_durations: Dict[str, float] = field(default_factory=dict)
