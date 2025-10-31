"""CLI tests for the Tennr pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.__main__ import main
from src.tennr_classifier.pipeline import DocumentSplitResult, SplitArtifact


class _StubPipeline:
    def __init__(self, settings):
        self.settings = settings

    def process_pdf(self, pdf_path: Path) -> DocumentSplitResult:
        artifact = SplitArtifact(
            patient_id="patient_001",
            pdf_path=pdf_path,
            metadata_path=pdf_path.with_suffix(".json"),
            pages=[0],
            average_confidence=1.0,
        )
        return DocumentSplitResult(
            artifacts=[artifact],
            global_metadata_path=pdf_path.with_suffix(".meta.json"),
        )


@pytest.fixture(autouse=True)
def _patch_orchestrator(monkeypatch):
    monkeypatch.setattr("src.__main__.PipelineOrchestrator", _StubPipeline)


def test_cli_requires_input(tmp_path, capsys):
    exit_code = main([])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "No input PDF provided" in captured.out


def test_cli_runs_pipeline(tmp_path, capsys):
    pdf = tmp_path / "sample.pdf"
    pdf.write_text("dummy")

    exit_code = main(["--input", str(pdf), "--output", str(tmp_path / "splits")])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Global metadata" in captured.out


def test_cli_handles_pipeline_error(monkeypatch, tmp_path, capsys):
    from src import __main__

    class _FailingPipeline(_StubPipeline):
        def process_pdf(self, pdf_path: Path) -> DocumentSplitResult:
            raise __main__.PipelineError("boom")

    monkeypatch.setattr("src.__main__.PipelineOrchestrator", _FailingPipeline)
    pdf = tmp_path / "sample.pdf"
    pdf.write_text("dummy")

    exit_code = main(["--input", str(pdf)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "Pipeline failed" in captured.out
