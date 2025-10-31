# Tennr Classifier Scaffold

This repository hosts the groundwork for a multi-patient document splitter. The pipeline ingests a multi-patient PDF, renders pages, runs OCR, extracts patient identifiers using rule-based heuristics, links entities into patient profiles, classifies pages to patients, and emits per-patient PDFs plus metadata via both a CLI and FastAPI API.

## Environment Setup

1. Ensure macOS has Python 3.10+ and Homebrew installed.  
2. Install [`uv`](https://github.com/astral-sh/uv) if it is not already present:

   ```
   brew install uv
   ```

3. Install the PDF/OCR prerequisites:

   ```
   brew install poppler tesseract
   ```

4. Clone the repository and bootstrap the environment:

   ```
   ./scripts/bootstrap_env.sh
   ```

   This creates `.venv` via `uv venv` and installs dependencies with `uv pip install -r requirements.txt`.

5. Later phases will expand `requirements.txt` with NER, fuzzy matching, and API dependencies outlined in `plan2.md`.

## Running Tests

Run tests directly through `uv` (activation optional):

```
uv run pytest
```

The tests cover configuration, logging, page extraction, OCR plumbing, and identifier extraction. They rely on mocked OCR calls by default. To exercise real Tesseract output, set `TENNR_TEST_USE_TESSERACT=1` and ensure the binary is installed.

## Command-Line Interface

The repository exposes a temporary CLI entry point for smoke testing. You can invoke it without manual activation:

```
uv run python -m src.__main__ --show-settings
```

To process a PDF end-to-end:

```
uv run python -m src.__main__ --input /path/to/sample.pdf --output /desired/output
```

Use the inspection helper to exercise page extraction and OCR end-to-end:

```
uv run python scripts/inspect_pdf.py --input /path/to/sample.pdf --backend tesseract
```

Environment variables control behaviour:

- `TENNR_OCR_BACKEND` (`tesseract`, `olmocr`, or `module:function`) selects the OCR strategy.
- `TENNR_PDF_RENDER_DPI` adjusts render quality (default 200).
- `TENNR_PERSIST_IMAGES=false` disables long-term storage of rendered PNGs.
- `TENNR_OLMOCR_HANDLER` points to the callable to use when `olmocr` is selected.
- `TENNR_REGEX_NAME`, `TENNR_REGEX_MRN`, `TENNR_REGEX_DOB`, `TENNR_REGEX_PHONE` override identifier patterns.
- `TENNR_ENTITY_MIN_CONFIDENCE` adjusts the minimum score required to accept a match.
- `TENNR_NAME_MATCH_THRESHOLD`, `TENNR_MRN_MATCH_THRESHOLD`, `TENNR_DOB_MATCH_THRESHOLD`, `TENNR_PHONE_MATCH_THRESHOLD` tune fuzzy matching thresholds.
- `TENNR_LINKER_STRICT_MODE` toggles conservative clustering behaviour.
- `TENNR_ASSIGN_MIN_CONFIDENCE`, `TENNR_ASSIGN_*_WEIGHT`, `TENNR_ASSIGN_AMBIGUITY_MARGIN`, `TENNR_ASSIGN_ALLOW_UNASSIGNED` configure page assignment scoring and manual review behaviour.
- `TENNR_SPLIT_OUTPUT_DIR`, `TENNR_SPLIT_INCLUDE_UNASSIGNED`, `TENNR_SPLIT_METADATA_FORMAT`, `TENNR_SPLIT_CLEAN_TEMP` control document splitting outputs.

## API Usage

Run the FastAPI app:

```
uv run uvicorn src.api:app --reload
```

Health check:

```
curl http://127.0.0.1:8000/healthz
```

Split a PDF:

```
curl -F "file=@/path/to/sample.pdf" http://127.0.0.1:8000/split
```

The response lists artifact metadata paths, patient PDFs, and unassigned page information.

## Further Reading

- `docs/metrics.md` – detailed explanation of collected metrics.
- `docs/operations.md` – runbook for operating the pipeline and service.
- `docs/troubleshooting.md` – guidance for resolving common issues.

## Project Layout

```
config.py                   # Application-wide settings loader
requirements.txt            # Phase 1 dependency pinning (expanded later)
src/
  __main__.py               # CLI entry point
  __init__.py               # Package marker
  api.py                    # FastAPI application
  tennr_classifier/
    __init__.py             # Exported data models
    logging_utils.py        # Logging configuration helpers
    pipeline.py             # Core data classes for pipeline components
    page_extractor.py       # PDF to image rendering utilities
    ocr_processor.py        # OCR pipeline with pluggable backends
    entity_extractor.py     # Regex-based identifier extraction
    fuzzy_matcher.py        # RapidFuzz-based similarity scoring helpers
    entity_linker.py        # Patient clustering logic
    page_assigner.py        # Page-to-patient assignment logic
    document_splitter.py    # Patient-specific PDF/metadata generation
    orchestrator.py         # High-level pipeline wiring
tests/
  test_skeleton.py          # Sanity tests for config/logging scaffolding
  test_page_extractor.py    # PDF rendering coverage
  test_ocr_processor.py     # OCR module coverage
  test_entity_extractor.py  # Identifier extraction coverage
  test_fuzzy_matcher.py     # Fuzzy matching coverage
  test_entity_linker.py     # Entity linking coverage
  test_page_assigner.py     # Page assignment coverage
  test_document_splitter.py # Document splitting coverage
  test_orchestrator.py      # Orchestrator wiring coverage
  test_api.py               # API endpoint coverage
  test_cli.py               # CLI behaviour coverage
scripts/
  bootstrap_env.sh          # Environment setup helper
  inspect_pdf.py            # Manual inspection CLI for extraction + OCR

```
