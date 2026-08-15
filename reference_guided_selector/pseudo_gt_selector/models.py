from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SelectionStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    REJECT = "REJECT"
    NO_PERSON = "NO_PERSON"
    INVALID_REFERENCE_PERSON = "INVALID_REFERENCE_PERSON"
    AMBIGUOUS_PERSON = "AMBIGUOUS_PERSON"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ToneFeatures:
    person: np.ndarray
    face: np.ndarray | None
    background: np.ndarray | None
    person_background: np.ndarray | None
    color: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    score: float
    person_error: float
    person_background_error: float | None
    face_error: float | None
    background_error: float | None
    weights: dict[str, float]
    color_differences: dict[str, float | None]


@dataclass(frozen=True)
class RegionMasks:
    person: np.ndarray
    face: np.ndarray | None
    local_background: np.ndarray | None


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    root: Path
    source: Path
    reference: Path
    ladder: dict[str, Path]
    missing_levels: tuple[str, ...] = ()


@dataclass
class SelectionResult:
    sample_id: str
    status: SelectionStatus
    best_level: str | None = None
    best_coefficient: float | None = None
    best_score: float | None = None
    second_best_level: str | None = None
    second_best_score: float | None = None
    margin: float | None = None
    confidence: float | None = None
    selected_path: Path | None = None
    region_validity: dict[str, bool] = field(default_factory=dict)
    best_components: dict[str, float | None] = field(default_factory=dict)
    levels: dict[str, CandidateScore] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    missing_levels: list[str] = field(default_factory=list)
    source_masks: RegionMasks | None = None
    reference_masks: RegionMasks | None = None
    message: str | None = None

    def without_masks(self) -> "SelectionResult":
        return replace(self, source_masks=None, reference_masks=None)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sample_id": self.sample_id,
            "status": self.status.value,
            "best_level": self.best_level,
            "best_coefficient": self.best_coefficient,
            "best_score": self.best_score,
            "second_best_level": self.second_best_level,
            "second_best_score": self.second_best_score,
            "margin": self.margin,
            "confidence": self.confidence,
            "region_validity": self.region_validity,
            "best_components": self.best_components,
            "weights": self.weights,
            "missing_levels": self.missing_levels,
            "levels": {},
        }
        for level, item in self.levels.items():
            data["levels"][level] = {
                "score": item.score,
                "person_error": item.person_error,
                "person_background_error": item.person_background_error,
                "face_error": item.face_error,
                "background_error": item.background_error,
                "color_debug": item.color_differences,
            }
        if self.message:
            data["message"] = self.message
        return data
