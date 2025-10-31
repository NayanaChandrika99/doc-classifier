"""Sanity checks for the Tennr scaffolding."""

from __future__ import annotations

import logging
from config import Settings, load_settings
from src.tennr_classifier.logging_utils import configure_logging, get_logger


def test_load_settings_respects_environment(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    temp_dir = tmp_path / "tmp"

    monkeypatch.setenv("TENNR_DATA_DIR", str(data_dir))
    monkeypatch.setenv("TENNR_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("TENNR_TEMP_DIR", str(temp_dir))
    monkeypatch.setenv("TENNR_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("TENNR_DEBUG", "true")
    monkeypatch.setenv("TENNR_OCR_BACKEND", "olmocr")
    monkeypatch.setenv("TENNR_PDF_RENDER_DPI", "150")
    monkeypatch.setenv("TENNR_PERSIST_IMAGES", "false")
    monkeypatch.setenv("TENNR_OCR_WARN_THRESHOLD", "0.25")
    monkeypatch.setenv("TENNR_OLMOCR_HANDLER", "tests.fake_module:fake_handler")
    monkeypatch.setenv("TENNR_REGEX_NAME", "(?P<name>[A-Za-z]+)")
    monkeypatch.setenv("TENNR_REGEX_MRN", "MRN:(?P<mrn>\\d+)")
    monkeypatch.setenv("TENNR_REGEX_DOB", "DOB:(?P<dob>\\d{2}/\\d{2}/\\d{4})")
    monkeypatch.setenv("TENNR_REGEX_PHONE", "Phone:(?P<phone>\\d{3}-\\d{3}-\\d{4})")
    monkeypatch.setenv("TENNR_ENTITY_MIN_CONFIDENCE", "0.3")
    monkeypatch.setenv("TENNR_NAME_MATCH_THRESHOLD", "0.75")
    monkeypatch.setenv("TENNR_MRN_MATCH_THRESHOLD", "0.9")
    monkeypatch.setenv("TENNR_DOB_MATCH_THRESHOLD", "1.0")
    monkeypatch.setenv("TENNR_PHONE_MATCH_THRESHOLD", "0.8")
    monkeypatch.setenv("TENNR_LINKER_STRICT_MODE", "true")
    monkeypatch.setenv("TENNR_ASSIGN_MIN_CONFIDENCE", "0.55")
    monkeypatch.setenv("TENNR_ASSIGN_NAME_WEIGHT", "0.25")
    monkeypatch.setenv("TENNR_ASSIGN_MRN_WEIGHT", "0.45")
    monkeypatch.setenv("TENNR_ASSIGN_DOB_WEIGHT", "0.2")
    monkeypatch.setenv("TENNR_ASSIGN_PHONE_WEIGHT", "0.1")
    monkeypatch.setenv("TENNR_ASSIGN_AMBIGUITY_MARGIN", "0.07")
    monkeypatch.setenv("TENNR_ASSIGN_ALLOW_UNASSIGNED", "false")
    monkeypatch.setenv("TENNR_SPLIT_OUTPUT_DIR", str(tmp_path / "splits"))
    monkeypatch.setenv("TENNR_SPLIT_INCLUDE_UNASSIGNED", "false")
    monkeypatch.setenv("TENNR_SPLIT_METADATA_FORMAT", "json")
    monkeypatch.setenv("TENNR_SPLIT_CLEAN_TEMP", "false")

    load_settings.cache_clear()
    settings = load_settings()

    assert settings.data_dir == data_dir
    assert settings.output_dir == output_dir
    assert settings.temp_dir == temp_dir
    assert settings.log_level == "DEBUG"
    assert settings.debug is True
    assert settings.ocr_backend == "olmocr"
    assert settings.pdf_render_dpi == 150
    assert settings.persist_page_images is False
    assert settings.ocr_warn_threshold == 0.25
    assert settings.olmocr_handler == "tests.fake_module:fake_handler"
    assert settings.regex_name == "(?P<name>[A-Za-z]+)"
    assert settings.regex_mrn == "MRN:(?P<mrn>\\d+)"
    assert settings.regex_dob == "DOB:(?P<dob>\\d{2}/\\d{2}/\\d{4})"
    assert settings.regex_phone == "Phone:(?P<phone>\\d{3}-\\d{3}-\\d{4})"
    assert settings.entity_min_confidence == 0.3
    assert settings.name_match_threshold == 0.75
    assert settings.mrn_match_threshold == 0.9
    assert settings.dob_match_threshold == 1.0
    assert settings.phone_match_threshold == 0.8
    assert settings.linker_strict_mode is True
    assert settings.assign_min_confidence == 0.55
    assert settings.assign_name_weight == 0.25
    assert settings.assign_mrn_weight == 0.45
    assert settings.assign_dob_weight == 0.2
    assert settings.assign_phone_weight == 0.1
    assert settings.assign_ambiguity_margin == 0.07
    assert settings.assign_allow_unassigned is False
    assert settings.split_output_dir == str(tmp_path / "splits")
    assert settings.split_include_unassigned is False
    assert settings.split_metadata_format == "json"
    assert settings.split_clean_temp is False

    # Directories should be created automatically.
    for directory in (data_dir, output_dir, temp_dir, settings.page_image_dir):
        assert directory.exists()


def test_configure_logging_emit_debug(tmp_path, caplog):
    settings = Settings(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        temp_dir=tmp_path / "tmp",
        log_level="DEBUG",
        debug=True,
    )

    settings.ensure_directories()
    configure_logging(settings, force=True)

    logger = get_logger("tennr.test")
    logger.debug("debug message emitted")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert any(isinstance(handler, logging.StreamHandler) for handler in root_logger.handlers)
