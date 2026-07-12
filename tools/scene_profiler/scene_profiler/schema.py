from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class LabelDecision:
    label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ElementDecision:
    status: str
    present: Optional[bool]
    confidence: Optional[float]
    area_ratio: Optional[float]
    detection_source: str
    types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewDecision:
    required: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class SceneProfile:
    scene_id: str
    image_path: str
    environment: LabelDecision
    time: LabelDecision
    content_scores: Dict[str, float]
    content_labels: List[str]
    dynamic_range_proxy: LabelDecision
    apparent_illumination: LabelDecision
    input_brightness: LabelDecision
    exposure: LabelDecision
    elements: Dict[str, ElementDecision]
    photometric_features: Dict[str, Any]
    semantic_region_features: Dict[str, Any]
    model_metadata: Dict[str, Any]
    review: ReviewDecision
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawSceneRecord:
    scene_id: str
    image_path: str
    image_width: int
    image_height: int
    semantic_probabilities: Dict[str, Dict[str, float]]
    segmentation_ratios: Dict[str, float]
    segmentation_valid_ratio: Optional[float]
    segmentation_mean_confidence: Optional[float]
    photometric_features: Dict[str, Any]
    regional_photometry: Dict[str, Dict[str, float]]
    open_vocab: Dict[str, Any]
    model_metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
