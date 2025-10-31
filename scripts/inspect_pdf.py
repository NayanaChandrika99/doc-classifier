#!/usr/bin/env python
"""Inspect a PDF by extracting pages and running OCR."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings
from src.tennr_classifier.logging_utils import configure_logging, get_logger
from src.tennr_classifier.ocr_processor import OCRProcessingError, OCRProcessor
from src.tennr_classifier.page_extractor import PageExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect PDF OCR output.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the PDF file to inspect.")
    parser.add_argument(
        "--backend",
        choices=["tesseract", "olmocr"],
        default=None,
        help="Override the configured OCR backend.",
    )
    parser.add_argument(
        "--no-persist-images",
        action="store_true",
        help="Do not keep rendered page images after processing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if args.backend:
        settings.ocr_backend = args.backend
    if args.no_persist_images:
        settings.persist_page_images = False

    configure_logging(settings, force=True)
    logger = get_logger("inspect_pdf")

    extractor = PageExtractor(settings)
    pages = extractor.extract_pages(args.input, persist_images=settings.persist_page_images)
    logger.info("Rendered %s pages from %s", len(pages), args.input)

    processor = OCRProcessor(settings)
    try:
        results = processor.process_pages(pages)
    except OCRProcessingError as exc:
        logger.error("OCR failed: %s", exc)
        return 1

    confidences = [result.average_confidence for result in results if result.average_confidence is not None]
    avg_conf = statistics.mean(confidences) if confidences else None
    logger.info("Extracted text for %s pages.", len(results))
    if avg_conf is not None:
        logger.info("Average confidence: %.2f", avg_conf)
    else:
        logger.info("No confidence scores available from OCR backend.")

    for result in results[:3]:
        snippet = result.text.strip().splitlines()[0] if result.text.strip() else ""
        logger.info("Page %s snippet: %s", result.page_index, snippet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
