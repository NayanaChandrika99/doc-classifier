"""Project-level configuration helpers for the Tennr classifier pipeline."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, Optional, Pattern

from pydantic import BaseModel, Field, PrivateAttr, ValidationError


class Settings(BaseModel):
    """Runtime settings loaded from environment variables or defaults."""

    data_dir: Path = Field(default_factory=lambda: Path("data"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))
    temp_dir: Path = Field(default_factory=lambda: Path("tmp"))
    log_level: str = "INFO"
    debug: bool = False
    ocr_backend: str = "tesseract"
    pdf_render_dpi: int = 200
    persist_page_images: bool = True
    ocr_warn_threshold: float = 0.5
    olmocr_handler: Optional[str] = None
    regex_name: str = r"(?P<name>(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+)"
    regex_mrn: str = r"(?:MRN|Medical\s*Record\s*Number)[:\s]*(?P<mrn>[A-Z0-9-]{5,})"
    regex_dob: str = r"(?:DOB|Date\s*of\s*Birth)[:\s]*(?P<dob>\d{2}[/-]\d{2}[/-]\d{2,4})"
    regex_phone: str = r"(?:Phone|Tel)[:\s]*(?P<phone>\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})"
    entity_name_keywords: tuple[str, ...] = ("patient", "name")
    entity_min_confidence: float = 0.5
    name_match_threshold: float = 0.85
    mrn_match_threshold: float = 0.99
    dob_match_threshold: float = 0.99
    phone_match_threshold: float = 0.95
    linker_strict_mode: bool = False
    assign_min_confidence: float = 0.6
    assign_name_weight: float = 0.3
    assign_mrn_weight: float = 0.4
    assign_dob_weight: float = 0.2
    assign_phone_weight: float = 0.1
    assign_ambiguity_margin: float = 0.05
    assign_allow_unassigned: bool = True
    split_output_dir: Optional[str] = None
    split_include_unassigned: bool = True
    split_metadata_format: str = "json"
    split_clean_temp: bool = True
    _compiled_patterns: Dict[str, Pattern[str]] = PrivateAttr(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings by reading environment variables with Tennr-specific prefixes."""
        defaults = cls()
        env_overrides: Dict[str, Any] = {
            "data_dir": Path(os.getenv("TENNR_DATA_DIR", str(defaults.data_dir))),
            "output_dir": Path(os.getenv("TENNR_OUTPUT_DIR", str(defaults.output_dir))),
            "temp_dir": Path(os.getenv("TENNR_TEMP_DIR", str(defaults.temp_dir))),
            "log_level": os.getenv("TENNR_LOG_LEVEL", defaults.log_level),
            "debug": _coerce_bool(os.getenv("TENNR_DEBUG", str(defaults.debug))),
            "ocr_backend": os.getenv("TENNR_OCR_BACKEND", defaults.ocr_backend),
            "pdf_render_dpi": int(os.getenv("TENNR_PDF_RENDER_DPI", defaults.pdf_render_dpi)),
            "persist_page_images": _coerce_bool(
                os.getenv("TENNR_PERSIST_IMAGES", str(defaults.persist_page_images))
            ),
            "ocr_warn_threshold": float(
                os.getenv("TENNR_OCR_WARN_THRESHOLD", defaults.ocr_warn_threshold)
            ),
            "olmocr_handler": os.getenv("TENNR_OLMOCR_HANDLER", defaults.olmocr_handler),
            "regex_name": os.getenv("TENNR_REGEX_NAME", defaults.regex_name),
            "regex_mrn": os.getenv("TENNR_REGEX_MRN", defaults.regex_mrn),
            "regex_dob": os.getenv("TENNR_REGEX_DOB", defaults.regex_dob),
            "regex_phone": os.getenv("TENNR_REGEX_PHONE", defaults.regex_phone),
            "entity_min_confidence": float(
                os.getenv("TENNR_ENTITY_MIN_CONFIDENCE", defaults.entity_min_confidence)
            ),
            "name_match_threshold": float(
                os.getenv("TENNR_NAME_MATCH_THRESHOLD", defaults.name_match_threshold)
            ),
            "mrn_match_threshold": float(
                os.getenv("TENNR_MRN_MATCH_THRESHOLD", defaults.mrn_match_threshold)
            ),
            "dob_match_threshold": float(
                os.getenv("TENNR_DOB_MATCH_THRESHOLD", defaults.dob_match_threshold)
            ),
            "phone_match_threshold": float(
                os.getenv("TENNR_PHONE_MATCH_THRESHOLD", defaults.phone_match_threshold)
            ),
            "linker_strict_mode": _coerce_bool(
                os.getenv("TENNR_LINKER_STRICT_MODE", str(defaults.linker_strict_mode))
            ),
            "assign_min_confidence": float(
                os.getenv("TENNR_ASSIGN_MIN_CONFIDENCE", defaults.assign_min_confidence)
            ),
            "assign_name_weight": float(
                os.getenv("TENNR_ASSIGN_NAME_WEIGHT", defaults.assign_name_weight)
            ),
            "assign_mrn_weight": float(
                os.getenv("TENNR_ASSIGN_MRN_WEIGHT", defaults.assign_mrn_weight)
            ),
            "assign_dob_weight": float(
                os.getenv("TENNR_ASSIGN_DOB_WEIGHT", defaults.assign_dob_weight)
            ),
            "assign_phone_weight": float(
                os.getenv("TENNR_ASSIGN_PHONE_WEIGHT", defaults.assign_phone_weight)
            ),
            "assign_ambiguity_margin": float(
                os.getenv("TENNR_ASSIGN_AMBIGUITY_MARGIN", defaults.assign_ambiguity_margin)
            ),
            "assign_allow_unassigned": _coerce_bool(
                os.getenv("TENNR_ASSIGN_ALLOW_UNASSIGNED", str(defaults.assign_allow_unassigned))
            ),
            "split_output_dir": os.getenv("TENNR_SPLIT_OUTPUT_DIR", defaults.split_output_dir),
            "split_include_unassigned": _coerce_bool(
                os.getenv("TENNR_SPLIT_INCLUDE_UNASSIGNED", str(defaults.split_include_unassigned))
            ),
            "split_metadata_format": os.getenv(
                "TENNR_SPLIT_METADATA_FORMAT", defaults.split_metadata_format
            ),
            "split_clean_temp": _coerce_bool(
                os.getenv("TENNR_SPLIT_CLEAN_TEMP", str(defaults.split_clean_temp))
            ),
        }
        return cls(**env_overrides)

    def ensure_directories(self) -> None:
        """Create directories that the pipeline expects to exist."""
        page_dir = self.page_image_dir
        for path in (self.data_dir, self.output_dir, self.temp_dir, page_dir):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def page_image_dir(self) -> Path:
        """Directory that stores intermediate rendered page images."""
        return self.temp_dir / "pages"

    @property
    def split_output_path(self) -> Path:
        base = Path(self.split_output_dir) if self.split_output_dir else self.output_dir / "splits"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def compiled_pattern(self, key: str) -> Pattern[str]:
        """Return a compiled regex pattern, caching results for reuse."""
        mapping = {
            "name": self.regex_name,
            "mrn": self.regex_mrn,
            "dob": self.regex_dob,
            "phone": self.regex_phone,
        }
        if key not in mapping:
            raise KeyError(f"Unknown regex key: {key}")
        if key not in self._compiled_patterns:
            self._compiled_patterns[key] = re.compile(mapping[key], flags=re.IGNORECASE)
        return self._compiled_patterns[key]


def _coerce_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return cached settings, raising a helpful error if validation fails."""
    try:
        settings = Settings.from_env()
        settings.ensure_directories()
        return settings
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
