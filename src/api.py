"""FastAPI application exposing the Tennr document splitter pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from config import Settings, load_settings
from src.tennr_classifier.orchestrator import PipelineError, PipelineOrchestrator


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = load_settings()
    settings.ensure_directories()
    return settings


@lru_cache(maxsize=1)
def get_orchestrator() -> PipelineOrchestrator:
    return PipelineOrchestrator(get_settings())


app = FastAPI(title="Tennr Document Splitter", version="0.1.0")


@app.get("/healthz")
def health(settings: Settings = Depends(get_settings)) -> Dict[str, Any]:
    return {
        "status": "ok",
        "ocr_backend": settings.ocr_backend,
        "split_output_dir": str(settings.split_output_path),
    }


@app.post("/split")
async def split_document(
    file: UploadFile = File(...),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> JSONResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = orchestrator.process_bytes(data, file.filename)
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {
        "global_metadata_path": str(result.global_metadata_path),
        "unassigned_pdf_path": str(result.unassigned_pdf_path) if result.unassigned_pdf_path else None,
        "unassigned_pages": result.unassigned_pages,
        "ambiguous_pages": result.ambiguous_pages,
        "total_pages": result.total_pages,
        "assigned_pages": result.assigned_pages,
        "stage_durations": result.stage_durations,
        "artifacts": [
            {
                "patient_id": artifact.patient_id,
                "pdf_path": str(artifact.pdf_path),
                "metadata_path": str(artifact.metadata_path),
                "pages": artifact.pages,
                "average_confidence": artifact.average_confidence,
            }
            for artifact in result.artifacts
        ],
    }

    return JSONResponse(content=payload)
