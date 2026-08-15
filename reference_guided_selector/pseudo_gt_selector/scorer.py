import numpy as np

from .config import SelectorConfig
from .models import CandidateScore, ToneFeatures


def _mean_absolute(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


class ToneScorer:
    def __init__(self, config: SelectorConfig | None = None):
        self.config = config or SelectorConfig()

    def compare(self, candidate: ToneFeatures, reference: ToneFeatures) -> CandidateScore:
        person_error = _mean_absolute(candidate.person, reference.person)
        face_valid = candidate.face is not None and reference.face is not None
        background_valid = candidate.background is not None and reference.background is not None
        relation_valid = candidate.person_background is not None and reference.person_background is not None

        face_error = _mean_absolute(candidate.face, reference.face) if face_valid else None
        background_error = _mean_absolute(candidate.background, reference.background) if background_valid else None
        relation_error = None
        if relation_valid:
            delta = np.abs(candidate.person_background - reference.person_background)
            relation_error = float(delta[0] + 0.25 * delta[1] + 0.25 * delta[2])

        base = dict(self.config.weights_with_face if face_valid else self.config.weights_without_face)
        components: dict[str, float | None] = {
            "person": person_error,
            "person_background": relation_error,
            "face": face_error,
            "background": background_error,
        }
        active = {name: weight for name, weight in base.items() if components[name] is not None}
        weight_sum = sum(active.values())
        weights = {name: weight / weight_sum for name, weight in active.items()}
        total = sum(weights[name] * float(components[name]) for name in weights)

        color_differences: dict[str, float | None] = {}
        for region in ("person", "face", "background"):
            cand_color = candidate.color.get(region)
            ref_color = reference.color.get(region)
            color_differences[f"{region}_color_diff"] = (
                float(np.linalg.norm(np.asarray(cand_color) - np.asarray(ref_color)))
                if cand_color is not None and ref_color is not None
                else None
            )
        return CandidateScore(
            score=float(total),
            person_error=person_error,
            person_background_error=relation_error,
            face_error=face_error,
            background_error=background_error,
            weights=weights,
            color_differences=color_differences,
        )
