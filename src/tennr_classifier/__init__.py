"""Expose Tennr classifier core modules."""

from .pipeline import (
    OCRResult,
    OCRWord,
    PageData,
    PatientEntity,
    SplitResult,
    IdentifierMatch,
    PageEntities,
    LinkedIdentifier,
    LinkedPatient,
    AssignmentReason,
    PageAssignment,
    DocumentAssignmentSummary,
    SplitArtifact,
    DocumentSplitResult,
)
from .page_extractor import PageExtractor, PageExtractionError
from .ocr_processor import OCRProcessingError, OCRProcessor
from .entity_extractor import EntityExtractor
from .fuzzy_matcher import FuzzyMatcher
from .entity_linker import PatientEntityLinker
from .page_assigner import PageAssigner
from .document_splitter import DocumentSplitter
from .orchestrator import PipelineOrchestrator, PipelineError

__all__ = [
    "OCRResult",
    "OCRWord",
    "PageData",
    "PatientEntity",
    "SplitResult",
    "IdentifierMatch",
    "PageEntities",
    "LinkedIdentifier",
    "LinkedPatient",
    "AssignmentReason",
    "PageAssignment",
    "DocumentAssignmentSummary",
    "SplitArtifact",
    "DocumentSplitResult",
    "PageExtractor",
    "PageExtractionError",
    "OCRProcessor",
    "OCRProcessingError",
    "EntityExtractor",
    "FuzzyMatcher",
    "PatientEntityLinker",
    "PageAssigner",
    "DocumentSplitter",
    "PipelineOrchestrator",
    "PipelineError",
]
