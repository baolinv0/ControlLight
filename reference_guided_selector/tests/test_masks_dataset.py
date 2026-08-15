from pathlib import Path

import numpy as np
from PIL import Image

from pseudo_gt_selector.config import LEVEL_COEFFICIENTS, SelectorConfig
from pseudo_gt_selector.dataset import discover_samples, validate_sample
from pseudo_gt_selector.mask_provider import FileMaskProvider, PersonDetection, select_primary_person
from pseudo_gt_selector.region_builder import build_local_background


def _save(path: Path, value: int = 100, size=(12, 10)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (value, value, value)).save(path)


def test_primary_person_uses_area_center_and_face_confidence():
    large_edge = np.zeros((20, 20), bool)
    large_edge[2:18, :8] = True
    centered_with_face = np.zeros((20, 20), bool)
    centered_with_face[5:15, 6:14] = True
    result = select_primary_person(
        [PersonDetection(large_edge, 0.0), PersonDetection(centered_with_face, 1.0)],
        ambiguity_tolerance=0.01,
    )
    assert result.mask is centered_with_face
    assert not result.ambiguous


def test_primary_person_reports_near_equal_candidates_as_ambiguous():
    left = np.zeros((20, 20), bool)
    right = np.zeros((20, 20), bool)
    left[5:15, 2:8] = True
    right[5:15, 12:18] = True
    result = select_primary_person([PersonDetection(left, 0.5), PersonDetection(right, 0.5)])
    assert result.ambiguous


def test_local_background_is_a_ring_outside_person():
    person = np.zeros((40, 40), bool)
    person[10:30, 15:25] = True
    ring = build_local_background(person, SelectorConfig(minimum_background_pixels=1))
    assert ring is not None
    assert not np.any(ring & person)
    assert ring.sum() > 0
    assert not ring[0, 0]


def test_file_provider_is_deterministic_and_returns_source_masks_once(tmp_path):
    image = tmp_path / "source.png"
    _save(image)
    mask_dir = tmp_path / "provided_masks"
    mask_dir.mkdir()
    person = np.zeros((10, 12), np.uint8)
    person[2:8, 3:9] = 255
    Image.fromarray(person).save(mask_dir / "source_person.png")
    Image.fromarray(person).save(mask_dir / "source_face.png")
    provider = FileMaskProvider(mask_dir)

    first = provider.get_person_masks(image)
    second = provider.get_person_masks(image)

    assert len(first) == 1
    assert np.array_equal(first[0].mask, second[0].mask)
    primary = provider.get_primary_person_mask(image)
    assert np.array_equal(primary.mask, person.astype(bool))
    assert provider.person_mask_reads[image] == 1


def test_file_provider_assigns_generic_face_to_the_overlapping_person(tmp_path):
    image = tmp_path / "source.png"
    _save(image, size=(20, 20))
    mask_dir = tmp_path / "provided_masks"
    mask_dir.mkdir()
    edge_person = np.zeros((20, 20), np.uint8)
    edge_person[2:18, :8] = 255
    centered_person = np.zeros((20, 20), np.uint8)
    centered_person[5:15, 6:14] = 255
    face = np.zeros((20, 20), np.uint8)
    face[6:10, 8:12] = 255
    Image.fromarray(edge_person).save(mask_dir / "source_person_0.png")
    Image.fromarray(centered_person).save(mask_dir / "source_person_1.png")
    Image.fromarray(face).save(mask_dir / "source_face.png")

    provider = FileMaskProvider(mask_dir)
    detections = provider.get_person_masks(image)
    primary = provider.get_primary_person_mask(image)

    assert [item.face_confidence for item in detections] == [0.0, 1.0]
    assert np.array_equal(primary.mask, centered_person.astype(bool))


def test_file_provider_returns_none_when_face_does_not_overlap_primary_person(tmp_path):
    image = tmp_path / "source.png"
    _save(image, size=(20, 20))
    mask_dir = tmp_path / "provided_masks"
    mask_dir.mkdir()
    person = np.zeros((20, 20), np.uint8)
    person[8:16, 8:16] = 255
    unrelated_face = np.zeros((20, 20), np.uint8)
    unrelated_face[1:4, 1:4] = 255
    Image.fromarray(person).save(mask_dir / "source_person.png")
    Image.fromarray(unrelated_face).save(mask_dir / "source_face.png")
    provider = FileMaskProvider(mask_dir)

    primary = provider.get_primary_person_mask(image)

    assert provider.get_face_mask(image, primary.mask) is None


def test_discovers_complete_sample_layout(tmp_path):
    sample = tmp_path / "scene001"
    _save(sample / "source.png")
    _save(sample / "reference.png")
    for level in LEVEL_COEFFICIENTS:
        _save(sample / "ladder" / f"{level}.png")

    found = discover_samples(tmp_path)

    assert len(found) == 1
    assert found[0].sample_id == "scene001"
    assert not found[0].missing_levels
    assert validate_sample(found[0]) == []


def test_missing_ladder_level_is_reported(tmp_path):
    _save(tmp_path / "source.png")
    _save(tmp_path / "reference.png")
    for level in list(LEVEL_COEFFICIENTS)[:-1]:
        _save(tmp_path / "ladder" / f"{level}.png")

    sample = discover_samples(tmp_path)[0]

    assert sample.missing_levels == ("a_p100",)
    assert "missing ladder levels: a_p100" in validate_sample(sample)


def test_ladder_size_mismatch_is_invalid(tmp_path):
    _save(tmp_path / "source.png", size=(12, 10))
    _save(tmp_path / "reference.png", size=(9, 9))
    for level in LEVEL_COEFFICIENTS:
        _save(tmp_path / "ladder" / f"{level}.png", size=(12, 10))
    _save(tmp_path / "ladder" / "a_p050.png", size=(11, 10))

    errors = validate_sample(discover_samples(tmp_path)[0])

    assert errors == ["ladder a_p050 size (11, 10) differs from source (12, 10)"]
