"""Command-line interface entry point for Tennr classifier scaffolding."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from config import load_settings
from src.tennr_classifier.logging_utils import configure_logging, get_logger
from src.tennr_classifier.orchestrator import PipelineOrchestrator, PipelineError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tennr-classifier",
        description="Scaffold CLI for the Tennr multi-patient document splitter.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override the configured log level (default comes from TENNR_LOG_LEVEL).",
    )
    parser.add_argument("--show-settings", action="store_true", help="Print runtime settings and exit.")
    parser.add_argument("-i", "--input", help="Path to the multi-patient PDF to process.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional override for the split output directory (defaults to settings).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings()

    if args.log_level:
        settings.log_level = args.log_level

    configure_logging(settings, force=True)
    logger = get_logger(__name__)
    logger.info("Tennr document splitter CLI ready.")

    if args.show_settings:
        logger.info("Active settings: %s", settings.model_dump())
        if not args.input:
            return 0

    if args.output:
        settings.split_output_dir = args.output
    settings.ensure_directories()

    if not args.input:
        logger.error("No input PDF provided. Use --input to specify a file.")
        return 1

    pdf_path = Path(args.input)
    orchestrator = PipelineOrchestrator(settings)

    try:
        result = orchestrator.process_pdf(pdf_path)
    except PipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 2

    logger.info("Global metadata: %s", result.global_metadata_path)
    logger.info(
        "Pages assigned: %s/%s", result.assigned_pages, result.total_pages
    )
    if result.stage_durations:
        for stage, duration in result.stage_durations.items():
            logger.info("Stage %s took %.2fs", stage, duration)
    for artifact in result.artifacts:
        logger.info(
            "Patient %s → %s pages=%s",
            artifact.patient_id,
            artifact.pdf_path,
            artifact.pages,
        )
    if result.unassigned_pdf_path:
        logger.info("Unassigned pages written to %s", result.unassigned_pdf_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
