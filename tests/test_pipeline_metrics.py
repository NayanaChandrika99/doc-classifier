"""Tests covering pipeline metrics and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings
from src.tennr_classifier.orchestrator import PipelineError, PipelineOrchestrator
def test_pipeline_error_on_missing_pdf(tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "out",
        temp_dir=tmp_path / "temp",
    )
    orchestrator = PipelineOrchestrator(settings)
    with pytest.raises(PipelineError):
        orchestrator.process_pdf(tmp_path / "missing.pdf")
