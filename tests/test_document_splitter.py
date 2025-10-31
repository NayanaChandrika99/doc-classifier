"""Tests for document splitting outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

from config import Settings
from src.tennr_classifier.document_splitter import DocumentSplitter
from src.tennr_classifier.pipeline import DocumentAssignmentSummary, PageAssignment


def _create_pdf(path: Path, pages: int) -> None:
    pdf = canvas.Canvas(str(path), pagesize=letter)
    for index in range(pages):
        pdf.drawString(72, 720, f"Page {index}")
        pdf.showPage()
    pdf.save()


def _assignment(page_index: int, patient_id: str | None, confidence: float, manual_review: bool = False):
    return PageAssignment(
        page_index=page_index,
        patient_id=patient_id,
        confidence=confidence,
        reasons=[],
        manual_review=manual_review,
    )


def test_document_splitter_creates_patient_pdfs(tmp_path):
    source_pdf = tmp_path / "source.pdf"
    _create_pdf(source_pdf, 3)

    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "tmp",
        split_output_dir=str(tmp_path / "splits"),
    )
    settings.ensure_directories()

    assignments = DocumentAssignmentSummary(
        assignments=[
            _assignment(0, "patient_001", 0.9),
            _assignment(1, "patient_002", 0.8),
            _assignment(2, None, 0.0, manual_review=True),
        ],
        unassigned_pages=[2],
        ambiguous_pages=[],
    )

    splitter = DocumentSplitter(settings)
    result = splitter.split(source_pdf, assignments)

    patient_artifact = result.artifacts[0]
    assert patient_artifact.pdf_path.exists()
    assert patient_artifact.metadata_path.exists()

    if result.unassigned_pdf_path:
        assert result.unassigned_pdf_path.exists()

    global_meta = result.global_metadata_path
    assert global_meta.exists()

    with patient_artifact.metadata_path.open() as handle:
        data = json.load(handle)
        assert data["pages"] == [0]

    with global_meta.open() as handle:
        global_data = json.load(handle)
        assert global_data["unassigned_pages"] == [2]
        assert global_data["total_pages"] == 3
        assert global_data["assigned_pages"] == 2
        assert any(artifact["patient_id"] == "patient_001" for artifact in global_data["artifacts"])


def test_document_splitter_handles_no_assignments(tmp_path):
    source_pdf = tmp_path / "source.pdf"
    _create_pdf(source_pdf, 1)

    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "tmp",
        split_output_dir=str(tmp_path / "splits"),
        split_include_unassigned=False,
    )
    settings.ensure_directories()

    assignments = DocumentAssignmentSummary(assignments=[], unassigned_pages=[], ambiguous_pages=[])

    splitter = DocumentSplitter(settings)
    result = splitter.split(source_pdf, assignments)

    assert result.total_pages == 0
    assert result.assigned_pages == 0
    assert result.artifacts == []


def test_document_splitter_yaml_metadata(tmp_path):
    source_pdf = tmp_path / "source.pdf"
    _create_pdf(source_pdf, 1)

    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "tmp",
        split_output_dir=str(tmp_path / "splits"),
        split_metadata_format="yaml",
    )
    settings.ensure_directories()

    assignments = DocumentAssignmentSummary(
        assignments=[_assignment(0, "patient_001", 0.9)],
        unassigned_pages=[],
        ambiguous_pages=[],
    )

    splitter = DocumentSplitter(settings)
    result = splitter.split(source_pdf, assignments)

    artifact = result.artifacts[0]
    assert artifact.metadata_path.suffix == ".yaml"
    with artifact.metadata_path.open() as handle:
        content = handle.read()
    assert "patient_id: patient_001" in content

    assert result.global_metadata_path.suffix == ".yaml"
