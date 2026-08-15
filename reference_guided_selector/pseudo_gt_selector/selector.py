import math
from pathlib import Path

import numpy as np
from PIL import Image

from .config import LEVEL_COEFFICIENTS, SelectorConfig
from .dataset import validate_sample
from .mask_provider import MaskProvider, select_primary_person
from .models import RegionMasks, SamplePaths, SelectionResult, SelectionStatus
from .region_builder import build_local_background
from .scorer import ToneScorer
from .tone_descriptor import ToneDescriptorExtractor


class PseudoGTSelector:
    def __init__(
        self,
        mask_provider: MaskProvider | None = None,
        config: SelectorConfig | None = None,
        extractor: ToneDescriptorExtractor | None = None,
        scorer: ToneScorer | None = None,
    ):
        self.config = config or SelectorConfig()
        self.mask_provider = mask_provider
        self.extractor = extractor or ToneDescriptorExtractor()
        self.scorer = scorer or ToneScorer(self.config)

    def _provider_masks(self, image_input) -> tuple[RegionMasks | None, bool]:
        if self.mask_provider is None:
            raise ValueError("source_masks/reference_masks or a mask provider are required")
        primary = select_primary_person(
            self.mask_provider.get_person_masks(image_input), self.config.ambiguity_tolerance
        )
        if primary.mask is None:
            return None, False
        if primary.ambiguous:
            return None, True
        face = self.mask_provider.get_face_mask(image_input, primary.mask)
        background = build_local_background(primary.mask, self.config)
        return RegionMasks(primary.mask, face, background), False

    @staticmethod
    def _invalid(sample_id: str, message: str, missing: list[str] | None = None) -> SelectionResult:
        return SelectionResult(
            sample_id=sample_id,
            status=SelectionStatus.INVALID,
            missing_levels=missing or [],
            message=message,
        )

    def select(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        ladder: dict[str, np.ndarray],
        *,
        sample_id: str = "sample",
        source_masks: RegionMasks | None = None,
        reference_masks: RegionMasks | None = None,
        ladder_paths: dict[str, Path] | None = None,
        source_mask_input=None,
        reference_mask_input=None,
    ) -> SelectionResult:
        missing = [level for level in LEVEL_COEFFICIENTS if level not in ladder]
        if missing:
            return self._invalid(sample_id, "required ladder levels are missing", missing)
        source_shape = np.asarray(source).shape
        mismatched = [level for level in LEVEL_COEFFICIENTS if np.asarray(ladder[level]).shape != source_shape]
        if mismatched:
            return self._invalid(sample_id, f"ladder dimensions differ from source: {', '.join(mismatched)}")

        source_ambiguous = False
        if source_masks is None:
            source_masks, source_ambiguous = self._provider_masks(
                source if source_mask_input is None else source_mask_input
            )
        if source_ambiguous:
            return SelectionResult(sample_id, SelectionStatus.AMBIGUOUS_PERSON)
        if source_masks is None or not np.any(source_masks.person):
            return SelectionResult(sample_id, SelectionStatus.NO_PERSON)

        reference_ambiguous = False
        if reference_masks is None:
            reference_masks, reference_ambiguous = self._provider_masks(
                reference if reference_mask_input is None else reference_mask_input
            )
        if reference_ambiguous:
            return SelectionResult(sample_id, SelectionStatus.AMBIGUOUS_PERSON)
        if reference_masks is None or not np.any(reference_masks.person):
            return SelectionResult(sample_id, SelectionStatus.INVALID_REFERENCE_PERSON)

        reference_features = self.extractor.extract(
            reference,
            reference_masks.person,
            reference_masks.face,
            reference_masks.local_background,
        )
        scores = {}
        for level in LEVEL_COEFFICIENTS:
            candidate_features = self.extractor.extract(
                ladder[level],
                source_masks.person,
                source_masks.face,
                source_masks.local_background,
            )
            scores[level] = self.scorer.compare(candidate_features, reference_features)

        ranked = sorted(LEVEL_COEFFICIENTS, key=lambda level: (scores[level].score, list(LEVEL_COEFFICIENTS).index(level)))
        best_level, second_level = ranked[:2]
        best = scores[best_level]
        second = scores[second_level]
        margin = float(second.score - best.score)
        if best.score <= self.config.accept_threshold:
            status = SelectionStatus.ACCEPT
        elif best.score <= self.config.review_threshold:
            status = SelectionStatus.REVIEW
        else:
            status = SelectionStatus.REJECT
        confidence = math.exp(-best.score / self.config.confidence_tau) * min(
            1.0, margin / self.config.confidence_margin_scale
        )
        region_validity = {
            "person": True,
            "face": (
                source_masks.face is not None
                and reference_masks.face is not None
                and bool(np.any(source_masks.face))
                and bool(np.any(reference_masks.face))
            ),
            "local_background": (
                source_masks.local_background is not None
                and reference_masks.local_background is not None
                and bool(np.any(source_masks.local_background))
                and bool(np.any(reference_masks.local_background))
            ),
        }
        return SelectionResult(
            sample_id=sample_id,
            status=status,
            best_level=best_level,
            best_coefficient=LEVEL_COEFFICIENTS[best_level],
            best_score=best.score,
            second_best_level=second_level,
            second_best_score=second.score,
            margin=margin,
            confidence=float(confidence),
            selected_path=None if ladder_paths is None else ladder_paths[best_level],
            region_validity=region_validity,
            best_components={
                "person_error": best.person_error,
                "person_background_error": best.person_background_error,
                "face_error": best.face_error,
                "background_error": best.background_error,
            },
            levels=scores,
            weights=best.weights,
            source_masks=source_masks,
            reference_masks=reference_masks,
        )

    def select_paths(self, sample: SamplePaths) -> SelectionResult:
        errors = validate_sample(sample)
        if errors:
            return self._invalid(sample.sample_id, "; ".join(errors), list(sample.missing_levels))
        source = np.asarray(Image.open(sample.source).convert("RGB"))
        reference = np.asarray(Image.open(sample.reference).convert("RGB"))
        ladder = {level: np.asarray(Image.open(path).convert("RGB")) for level, path in sample.ladder.items()}
        return self.select(
            source,
            reference,
            ladder,
            sample_id=sample.sample_id,
            ladder_paths=sample.ladder,
            source_mask_input=sample.source,
            reference_mask_input=sample.reference,
        )
