import numpy as np

from .config import SelectorConfig


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    source = np.asarray(mask, dtype=bool)
    h, w = source.shape
    padded = np.pad(source, radius)
    result = np.zeros_like(source)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result |= padded[dy : dy + h, dx : dx + w]
    return result


def build_local_background(person_mask: np.ndarray, config: SelectorConfig | None = None) -> np.ndarray | None:
    config = config or SelectorConfig()
    person = np.asarray(person_mask, dtype=bool)
    ys, xs = np.nonzero(person)
    if not len(xs):
        return None
    scale = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    inner = max(1, int(round(config.background_inner_ratio * scale)))
    outer = max(inner + 1, int(round(config.background_outer_ratio * scale)))
    ring = _dilate(person, outer) & ~_dilate(person, inner)
    if np.count_nonzero(ring) < config.minimum_background_pixels:
        return None
    return ring
