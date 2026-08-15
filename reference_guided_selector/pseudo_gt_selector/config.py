from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


LEVEL_COEFFICIENTS = {
    "a_m100": -1.0,
    "a_m075": -0.75,
    "a_m050": -0.5,
    "a_m025": -0.25,
    "a_000": 0.0,
    "a_p025": 0.25,
    "a_p050": 0.5,
    "a_p075": 0.75,
    "a_p100": 1.0,
}


def _frozen(values: dict[str, float]) -> Mapping[str, float]:
    return MappingProxyType(values)


@dataclass(frozen=True)
class SelectorConfig:
    levels: Mapping[str, float] = field(default_factory=lambda: _frozen(dict(LEVEL_COEFFICIENTS)))
    weights_with_face: Mapping[str, float] = field(
        default_factory=lambda: _frozen(
            {"person": 0.40, "person_background": 0.30, "face": 0.20, "background": 0.10}
        )
    )
    weights_without_face: Mapping[str, float] = field(
        default_factory=lambda: _frozen({"person": 0.50, "person_background": 0.35, "background": 0.15})
    )
    background_inner_ratio: float = 0.02
    background_outer_ratio: float = 0.10
    minimum_background_pixels: int = 8
    ambiguity_tolerance: float = 0.02
    accept_threshold: float = 0.18
    review_threshold: float = 0.30
    confidence_tau: float = 0.15
    confidence_margin_scale: float = 0.05
    debug_overlay_limit: int = 100
    debug_overlay_seed: int = 0
