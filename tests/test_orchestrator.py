"""Tests for the PipelineOrchestrator wiring."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from src.tennr_classifier.orchestrator import PipelineOrchestrator
from src.tennr_classifier.pipeline import (
    DocumentAssignmentSummary,
    DocumentSplitResult,
    PageAssignment,
    PageData,
    PageEntities,
    SplitArtifact,
)


def _make_orchestrator(tmp_path: Path, *, persisted: bool = True):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("dummy")

    image_path = tmp_path / "page.png"
    image_path.write_text("tmp")

    page_data = PageData(index=0, image_path=image_path, width=10, height=10, persisted=persisted)

    class StubPageExtractor:
        def extract_pages(self, _pdf_path, persist_images=None):
            return [page_data]

    class StubOCRProcessor:
        def process_pages(self, pages):
            return [object() for _ in pages]

    class StubEntityExtractor:
        def extract_page(self, _):
            return PageEntities(page_index=0, identifiers=[])

    class StubEntityLinker:
        def link(self, pages):
            return []

    class StubPageAssigner:
        def assign_pages(self, pages, patients):
            assignment = PageAssignment(
                page_index=0,
                patient_id="patient_001",
                confidence=0.9,
                reasons=[],
                manual_review=False,
            )
            return DocumentAssignmentSummary(assignments=[assignment], unassigned_pages=[], ambiguous_pages=[])

    class StubSplitter:
        def split(self, pdf_path: Path, assignments: DocumentAssignmentSummary) -> DocumentSplitResult:
            return DocumentSplitResult(
                artifacts=[
                    SplitArtifact(
                        patient_id="patient_001",
                        pdf_path=pdf_path,
                        metadata_path=pdf_path.with_suffix(".json"),
                        pages=[0],
                        average_confidence=0.9,
                    )
                ],
                global_metadata_path=pdf_path.with_suffix(".meta.json"),
                unassigned_pages=assignments.unassigned_pages,
                ambiguous_pages=assignments.ambiguous_pages,
                total_pages=len(assignments.assignments),
                assigned_pages=len(assignments.assignments),
                stage_durations={"document_splitting": 0.01},
            )

    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / "temp",
    )

    orchestrator = PipelineOrchestrator(
        settings,
        page_extractor=StubPageExtractor(),
        ocr_processor=StubOCRProcessor(),
        entity_extractor=StubEntityExtractor(),
        entity_linker=StubEntityLinker(),
        page_assigner=StubPageAssigner(),
        document_splitter=StubSplitter(),
    )
    return orchestrator, pdf_path, image_path


def test_orchestrator_process_pdf(tmp_path):
    orchestrator, pdf_path, _ = _make_orchestrator(tmp_path)
    result = orchestrator.process_pdf(pdf_path)

    assert result.artifacts[0].patient_id == "patient_001"
    assert result.global_metadata_path == pdf_path.with_suffix(".meta.json")
    assert result.total_pages == 1
    assert "page_extraction" in result.stage_durations


def test_orchestrator_process_bytes(tmp_path):
    orchestrator, _, _ = _make_orchestrator(tmp_path)
    result = orchestrator.process_bytes(b"dummy", "upload.pdf")
    assert result.artifacts[0].patient_id == "patient_001"


def test_orchestrator_cleans_temp_images(tmp_path):
    orchestrator, pdf_path, image_path = _make_orchestrator(tmp_path, persisted=False)
    orchestrator.process_pdf(pdf_path)
    assert not image_path.exists()
