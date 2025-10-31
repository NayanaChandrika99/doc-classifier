"""High-level orchestrator wiring the document splitting pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Optional

from config import Settings, load_settings
from .entity_extractor import EntityExtractor
from .entity_linker import PatientEntityLinker
from .logging_utils import get_logger
from .ocr_processor import OCRProcessor
from .page_assigner import PageAssigner
from .page_extractor import PageExtractor
from .pipeline import DocumentAssignmentSummary, DocumentSplitResult, PageEntities
from .document_splitter import DocumentSplitter

logger = get_logger(__name__)


class PipelineError(RuntimeError):
    """Top-level pipeline failure."""


class PipelineOrchestrator:
    """Run the full Tennr document splitting pipeline on PDFs."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        page_extractor: Optional[PageExtractor] = None,
        ocr_processor: Optional[OCRProcessor] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        entity_linker: Optional[PatientEntityLinker] = None,
        page_assigner: Optional[PageAssigner] = None,
        document_splitter: Optional[DocumentSplitter] = None,
    ):
        self.settings = settings or load_settings()
        self.settings.ensure_directories()

        self.page_extractor = page_extractor or PageExtractor(self.settings)
        self.ocr_processor = ocr_processor or OCRProcessor(self.settings)
        self.entity_extractor = entity_extractor or EntityExtractor(self.settings)
        self.entity_linker = entity_linker or PatientEntityLinker(self.settings)
        self.page_assigner = page_assigner or PageAssigner(self.settings)
        self.document_splitter = document_splitter or DocumentSplitter(self.settings)

    def process_pdf(self, pdf_path: Path) -> DocumentSplitResult:
        """Run the full pipeline on a PDF file path."""

        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise PipelineError(f"PDF not found: {pdf_path}")

        logger.info("Starting pipeline for %s", pdf_path)
        start = time.time()

        stage_timings: dict[str, float] = {}

        temp_image_paths: list[Path] = []

        try:
            stage_start = time.perf_counter()
            pages = self.page_extractor.extract_pages(pdf_path)
            stage_timings["page_extraction"] = time.perf_counter() - stage_start
            logger.info("Extracted %s pages", len(pages))

            temp_image_paths = [page.image_path for page in pages if not getattr(page, "persisted", True)]

            stage_start = time.perf_counter()
            ocr_results = self.ocr_processor.process_pages(pages)
            stage_timings["ocr"] = time.perf_counter() - stage_start
            logger.info("OCR complete for %s pages", len(ocr_results))

            stage_start = time.perf_counter()
            page_entities: list[PageEntities] = [
                self.entity_extractor.extract_page(result) for result in ocr_results
            ]
            stage_timings["entity_extraction"] = time.perf_counter() - stage_start
            logger.info("Entity extraction produced %s page entity sets", len(page_entities))

            if temp_image_paths:
                cleanup_start = time.perf_counter()
                for path in temp_image_paths:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.debug("Failed to remove temporary image %s", path)
                stage_timings["image_cleanup"] = time.perf_counter() - cleanup_start

            stage_start = time.perf_counter()
            patients = self.entity_linker.link(page_entities)
            stage_timings["entity_linking"] = time.perf_counter() - stage_start
            logger.info("Linked %s patient clusters", len(patients))

            stage_start = time.perf_counter()
            assignments: DocumentAssignmentSummary = self.page_assigner.assign_pages(
                page_entities, patients
            )
            stage_timings["page_assignment"] = time.perf_counter() - stage_start
            logger.info(
                "Assignments prepared: %s unassigned, %s ambiguous",
                len(assignments.unassigned_pages),
                len(assignments.ambiguous_pages),
            )

            stage_start = time.perf_counter()
            split_result = self.document_splitter.split(pdf_path, assignments)
            stage_timings["document_splitting"] = time.perf_counter() - stage_start

            total_time = time.time() - start
            logger.info(
                "Pipeline finished in %.2fs (artifacts=%s, assigned=%s/%s)",
                total_time,
                len(split_result.artifacts),
                split_result.assigned_pages,
                split_result.total_pages,
            )

            split_result.stage_durations = stage_timings
            split_result.total_pages = len(page_entities)
            split_result.assigned_pages = split_result.total_pages - len(assignments.unassigned_pages)
            
            return split_result
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Pipeline failed for %s", pdf_path)
            raise PipelineError(str(exc)) from exc

    def process_bytes(self, data: bytes, filename: str = "upload.pdf") -> DocumentSplitResult:
        """Process an uploaded PDF provided as bytes."""

        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix or ".pdf",
            dir=self.settings.temp_dir,
            delete=False,
        ) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        try:
            return self.process_pdf(temp_path)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:  # pragma: no cover - best effort cleanup
                logger.warning("Failed to delete temporary file %s", temp_path)
