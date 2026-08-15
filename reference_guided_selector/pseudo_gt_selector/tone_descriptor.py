import numpy as np

from .models import ToneFeatures
from .photometric import chromaticity_descriptor, rgb_to_log_luminance


class ToneDescriptorExtractor:
    quantiles = (10, 25, 50, 75, 90)

    def _region(self, log_luminance: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or not np.any(mask):
            return None
        return np.percentile(log_luminance[np.asarray(mask, dtype=bool)], self.quantiles)

    def extract(
        self,
        image: np.ndarray,
        person_mask: np.ndarray,
        face_mask: np.ndarray | None = None,
        background_mask: np.ndarray | None = None,
    ) -> ToneFeatures:
        log_luminance = rgb_to_log_luminance(image)
        person = self._region(log_luminance, person_mask)
        if person is None:
            raise ValueError("person mask must contain pixels")
        face = self._region(log_luminance, face_mask)
        background = self._region(log_luminance, background_mask)
        relation = None
        if background is not None:
            relation = np.asarray(
                [person[2] - background[2], person[4] - person[0], background[4] - background[0]],
                dtype=float,
            )
        color = {"person": chromaticity_descriptor(image, person_mask)}
        if face is not None:
            color["face"] = chromaticity_descriptor(image, face_mask)
        if background is not None:
            color["background"] = chromaticity_descriptor(image, background_mask)
        return ToneFeatures(person, face, background, relation, color)
