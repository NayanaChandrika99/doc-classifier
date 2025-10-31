"""FastAPI endpoint tests for the document splitter API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import api
from src.tennr_classifier.pipeline import DocumentSplitResult, SplitArtifact
from src.tennr_classifier.orchestrator import PipelineError


class _StubOrchestrator:
    def __init__(self, result: DocumentSplitResult, *, raise_error: bool = False):
        self.result = result
        self.raise_error = raise_error

    def process_bytes(self, data: bytes, filename: str = "upload.pdf") -> DocumentSplitResult:
        if self.raise_error:
            raise PipelineError("pipeline failure")
        return self.result


def _result(tmp_path: Path) -> DocumentSplitResult:
    artifact = SplitArtifact(
        patient_id="patient_001",
        pdf_path=tmp_path / "patient_001.pdf",
        metadata_path=tmp_path / "patient_001.json",
        pages=[0, 1],
        average_confidence=0.95,
    )
    return DocumentSplitResult(
        artifacts=[artifact],
        global_metadata_path=tmp_path / "split_metadata.json",
        unassigned_pdf_path=None,
        unassigned_pages=[2],
        ambiguous_pages=[],
        total_pages=3,
        assigned_pages=2,
        stage_durations={"page_extraction": 0.1},
    )


@pytest.fixture(autouse=True)
def _reset_dependencies():
    api.get_orchestrator.cache_clear()
    api.get_settings.cache_clear()
    api.app.dependency_overrides.clear()
    yield
    api.app.dependency_overrides.clear()


def test_healthz_endpoint():
    client = TestClient(api.app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_split_endpoint_success(tmp_path):
    result = _result(tmp_path)
    api.app.dependency_overrides[api.get_orchestrator] = lambda: _StubOrchestrator(result)
    client = TestClient(api.app)

    response = client.post(
        "/split",
        files={"file": ("sample.pdf", b"fake", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["artifacts"][0]["patient_id"] == "patient_001"
    assert data["unassigned_pages"] == [2]
    assert data["total_pages"] == result.total_pages


def test_split_endpoint_rejects_non_pdf():
    client = TestClient(api.app)
    response = client.post(
        "/split",
        files={"file": ("image.png", b"data", "image/png")},
    )
    assert response.status_code == 400


def test_split_endpoint_handles_pipeline_error(tmp_path):
    result = _result(tmp_path)
    api.app.dependency_overrides[api.get_orchestrator] = lambda: _StubOrchestrator(result, raise_error=True)
    client = TestClient(api.app)

    response = client.post(
        "/split",
        files={"file": ("sample.pdf", b"fake", "application/pdf")},
    )

    assert response.status_code == 500
