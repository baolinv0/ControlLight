from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SelectorConfig


@dataclass(frozen=True)
class PersonDetection:
    mask: np.ndarray
    face_confidence: float = 0.0


@dataclass(frozen=True)
class PrimaryPersonSelection:
    mask: np.ndarray | None
    ambiguous: bool = False
    index: int | None = None
    scores: tuple[float, ...] = ()


class MaskProvider(ABC):
    @abstractmethod
    def get_person_masks(self, image) -> list[PersonDetection]:
        raise NotImplementedError

    def get_primary_person_mask(self, image) -> PrimaryPersonSelection:
        return select_primary_person(self.get_person_masks(image))

    @abstractmethod
    def get_face_mask(self, image, person_mask: np.ndarray) -> np.ndarray | None:
        raise NotImplementedError


def _center_score(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    h, w = mask.shape
    distance = np.hypot(xs.mean() - (w - 1) / 2, ys.mean() - (h - 1) / 2)
    max_distance = np.hypot(max((w - 1) / 2, 1), max((h - 1) / 2, 1))
    return float(max(0.0, 1.0 - distance / max_distance))


def select_primary_person(
    detections: list[PersonDetection], ambiguity_tolerance: float = 0.02
) -> PrimaryPersonSelection:
    valid = [item for item in detections if np.any(item.mask)]
    if not valid:
        return PrimaryPersonSelection(None)
    areas = np.asarray([np.count_nonzero(item.mask) for item in valid], dtype=float)
    normalized = areas / areas.max()
    scores = tuple(
        float(0.50 * area + 0.30 * _center_score(item.mask) + 0.20 * item.face_confidence)
        for area, item in zip(normalized, valid)
    )
    order = np.argsort(np.asarray(scores), kind="stable")[::-1]
    best = int(order[0])
    ambiguous = len(order) > 1 and scores[best] - scores[int(order[1])] <= ambiguity_tolerance
    return PrimaryPersonSelection(valid[best].mask, ambiguous, best, scores)


class FileMaskProvider(MaskProvider):
    """Deterministic provider for precomputed `<image_stem>_person[_N].png` masks."""

    def __init__(self, mask_root: str | Path, config: SelectorConfig | None = None):
        self.mask_root = Path(mask_root)
        self.config = config or SelectorConfig()
        self._person_cache: dict[Path, list[PersonDetection]] = {}
        self._face_cache: dict[Path, np.ndarray | None] = {}
        self.person_mask_reads: Counter[Path] = Counter()

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L")) > 0

    def _person_files(self, image: Path) -> list[Path]:
        stem = image.stem
        exact = self.mask_root / f"{stem}_person.png"
        numbered = sorted(self.mask_root.glob(f"{stem}_person_[0-9]*.png"))
        return [exact] if exact.exists() else numbered

    def get_person_masks(self, image) -> list[PersonDetection]:
        path = Path(image)
        if path not in self._person_cache:
            self.person_mask_reads[path] += 1
            face_path = self.mask_root / f"{path.stem}_face.png"
            face = self._read_mask(face_path) if face_path.exists() else None
            detections = []
            for mask_path in self._person_files(path):
                person = self._read_mask(mask_path)
                confidence = 0.0
                if face is not None and np.any(face):
                    confidence = float(np.count_nonzero(face & person) / np.count_nonzero(face))
                detections.append(PersonDetection(person, confidence))
            self._person_cache[path] = detections
        return self._person_cache[path]

    def get_primary_person_mask(self, image) -> PrimaryPersonSelection:
        return select_primary_person(self.get_person_masks(image), self.config.ambiguity_tolerance)

    def get_face_mask(self, image, person_mask: np.ndarray) -> np.ndarray | None:
        path = Path(image)
        if path not in self._face_cache:
            face_path = self.mask_root / f"{path.stem}_face.png"
            self._face_cache[path] = self._read_mask(face_path) if face_path.exists() else None
        face = self._face_cache[path]
        if face is None:
            return None
        overlap = face & np.asarray(person_mask, dtype=bool)
        return overlap if np.any(overlap) else None
