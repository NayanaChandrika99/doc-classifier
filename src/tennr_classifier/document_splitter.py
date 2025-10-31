"""Create patient-specific PDFs and metadata from page assignments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from PyPDF2 import PdfReader, PdfWriter
import yaml

from config import Settings
from .logging_utils import get_logger
from .pipeline import (
    DocumentAssignmentSummary,
    DocumentSplitResult,
    PageAssignment,
    SplitArtifact,
)

logger = get_logger(__name__)


class DocumentSplitter:
    """Split a PDF into patient-specific outputs based on page assignments."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def split(
        self,
        pdf_path: Path,
        assignments: DocumentAssignmentSummary,
    ) -> DocumentSplitResult:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"Source PDF not found: {pdf_path}")

        output_dir = self.settings.split_output_path
        logger.info("Splitting %s pages into patient documents in %s", len(assignments.assignments), output_dir)

        reader = PdfReader(str(pdf_path))
        artifacts: List[SplitArtifact] = []
        patient_writers: dict[str, PdfWriter] = {}
        patient_assignments: dict[str, List[PageAssignment]] = {}

        unassigned_writer = PdfWriter() if self.settings.split_include_unassigned else None

        for assignment in assignments.assignments:
            page_index = assignment.page_index
            if page_index >= len(reader.pages):
                logger.warning("Page %s exceeds source PDF length; skipping", page_index)
                continue
            if assignment.patient_id:
                writer = patient_writers.setdefault(assignment.patient_id, PdfWriter())
                writer.add_page(reader.pages[page_index])
                patient_assignments.setdefault(assignment.patient_id, []).append(assignment)
            elif unassigned_writer is not None:
                unassigned_writer.add_page(reader.pages[page_index])

        for patient_id, writer in patient_writers.items():
            artifact = self._write_patient_artifact(
                patient_id,
                writer,
                patient_assignments.get(patient_id, []),
                output_dir,
                pdf_path,
            )
            artifacts.append(artifact)

        unassigned_pdf_path: Optional[Path] = None
        if unassigned_writer and assignments.unassigned_pages:
            unassigned_pdf_path = output_dir / "unassigned.pdf"
            with unassigned_pdf_path.open("wb") as handle:
                unassigned_writer.write(handle)
            logger.info("Unassigned pages written to %s", unassigned_pdf_path)

        global_metadata_path = self._write_global_metadata(
            output_dir / "split_metadata",
            pdf_path,
            artifacts,
            assignments,
            unassigned_pdf_path,
        )

        return DocumentSplitResult(
            artifacts=artifacts,
            global_metadata_path=global_metadata_path,
            unassigned_pdf_path=unassigned_pdf_path,
            unassigned_pages=assignments.unassigned_pages,
            ambiguous_pages=assignments.ambiguous_pages,
            total_pages=len(assignments.assignments),
            assigned_pages=len(assignments.assignments) - len(assignments.unassigned_pages),
        )

    def _write_patient_artifact(
        self,
        patient_id: str,
        writer: PdfWriter,
        assignments: List[PageAssignment],
        output_dir: Path,
        source_pdf: Path,
    ) -> SplitArtifact:
        pdf_output = output_dir / f"{patient_id}.pdf"
        with pdf_output.open("wb") as handle:
            writer.write(handle)

        metadata_format = (self.settings.split_metadata_format or "json").lower()
        if metadata_format not in {"json", "yaml"}:
            logger.warning("Unsupported metadata format '%s'; defaulting to JSON.", metadata_format)
            metadata_format = "json"

        extension = ".yaml" if metadata_format == "yaml" else ".json"
        metadata_output = output_dir / f"{patient_id}{extension}"
        pages = [assignment.page_index for assignment in assignments]
        avg_confidence = (
            sum(assignment.confidence for assignment in assignments) / len(assignments)
            if assignments
            else 0.0
        )

        metadata = {
            "patient_id": patient_id,
            "source_pdf": str(source_pdf),
            "output_pdf": str(pdf_output),
            "pages": pages,
            "average_confidence": avg_confidence,
            "assignments": [
                {
                    "page_index": assignment.page_index,
                    "confidence": assignment.confidence,
                    "manual_review": assignment.manual_review,
                    "reasons": [reason.__dict__ for reason in assignment.reasons],
                }
                for assignment in assignments
            ],
        }

        with metadata_output.open("w", encoding="utf-8") as handle:
            if metadata_format == "yaml":
                yaml.safe_dump(metadata, handle, sort_keys=False)
            else:
                json.dump(metadata, handle, indent=2)

        logger.info(
            "Wrote split for %s: %s pages → %s",
            patient_id,
            len(pages),
            pdf_output,
        )

        return SplitArtifact(
            patient_id=patient_id,
            pdf_path=pdf_output,
            metadata_path=metadata_output,
            pages=pages,
            average_confidence=avg_confidence,
        )

    def _write_global_metadata(
        self,
        output_stub: Path,
        source_pdf: Path,
        artifacts: Iterable[SplitArtifact],
        assignments: DocumentAssignmentSummary,
        unassigned_pdf_path: Optional[Path],
    ) -> Path:
        metadata_format = (self.settings.split_metadata_format or "json").lower()
        if metadata_format not in {"json", "yaml"}:
            metadata_format = "json"
        extension = ".yaml" if metadata_format == "yaml" else ".json"
        output_path = output_stub.with_suffix(extension)

        data = {
            "source_pdf": str(source_pdf),
            "total_pages": len(assignments.assignments),
            "assigned_pages": len(assignments.assignments) - len(assignments.unassigned_pages),
            "unassigned_pages": assignments.unassigned_pages,
            "ambiguous_pages": assignments.ambiguous_pages,
            "unassigned_pdf": str(unassigned_pdf_path) if unassigned_pdf_path else None,
            "artifacts": [
                {
                    "patient_id": artifact.patient_id,
                    "pdf_path": str(artifact.pdf_path),
                    "metadata_path": str(artifact.metadata_path),
                    "pages": artifact.pages,
                    "average_confidence": artifact.average_confidence,
                }
                for artifact in artifacts
            ],
        }

        with output_path.open("w", encoding="utf-8") as handle:
            if metadata_format == "yaml":
                yaml.safe_dump(data, handle, sort_keys=False)
            else:
                json.dump(data, handle, indent=2)

        logger.info("Global split metadata written to %s", output_path)
        return output_path
