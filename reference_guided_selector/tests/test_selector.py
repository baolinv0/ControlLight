from pathlib import Path

import numpy as np

from pseudo_gt_selector.config import LEVEL_COEFFICIENTS
from pseudo_gt_selector.mask_provider import MaskProvider, PersonDetection
from pseudo_gt_selector.models import RegionMasks, SelectionStatus
from pseudo_gt_selector.selector import PseudoGTSelector


def _image(person_value: int, background_value: int, size=20):
    image = np.full((size, size, 3), background_value, np.uint8)
    image[7:13, 7:13] = person_value
    return image


def _masks(size=20, face=False, background=True):
    person = np.zeros((size, size), bool)
    person[7:13, 7:13] = True
    face_mask = None
    if face:
        face_mask = np.zeros_like(person)
        face_mask[7:10, 8:12] = True
    bg = ~person if background else None
    return RegionMasks(person, face_mask, bg)


def _ladder(candidate_a, candidate_b):
    ladder = {level: candidate_a.copy() for level in LEVEL_COEFFICIENTS}
    ladder["a_m025"] = candidate_a
    ladder["a_p025"] = candidate_b
    return ladder


def _assert_person_dominance(ref_person, ref_bg, a_person, a_bg, b_person, b_bg):
    reference = _image(ref_person, ref_bg)
    candidate_a = _image(a_person, a_bg)
    candidate_b = _image(b_person, b_bg)
    source = _image(100, 100)
    result = PseudoGTSelector().select(
        source,
        reference,
        _ladder(candidate_a, candidate_b),
        source_masks=_masks(),
        reference_masks=_masks(),
    )
    global_a = abs(float(candidate_a.mean()) - float(reference.mean()))
    global_b = abs(float(candidate_b.mean()) - float(reference.mean()))
    assert global_a < global_b
    assert result.best_level == "a_p025"


def test_person_dominates_global_brightness_case_1():
    _assert_person_dominance(180, 80, 80, 85, 180, 20)


def test_person_dominates_global_brightness_case_2():
    _assert_person_dominance(200, 90, 90, 95, 200, 25)


def test_person_dominates_global_brightness_case_3():
    _assert_person_dominance(160, 70, 70, 75, 160, 15)


def test_person_dominates_global_brightness_case_4():
    _assert_person_dominance(220, 100, 100, 105, 220, 30)


def test_person_dominates_global_brightness_case_5():
    _assert_person_dominance(190, 60, 60, 65, 190, 10)


def test_person_background_relation_discriminates_equal_person_tone():
    source = _image(100, 100)
    reference = _image(180, 80)
    wrong_background = _image(180, 160)
    matching_relation = _image(180, 80)
    result = PseudoGTSelector().select(
        source,
        reference,
        _ladder(wrong_background, matching_relation),
        source_masks=_masks(),
        reference_masks=_masks(),
    )
    assert result.best_level == "a_p025"
    assert result.levels["a_p025"].person_background_error < result.levels["a_m025"].person_background_error


def test_face_fallback_selects_without_error_and_weights_sum_to_one():
    result = PseudoGTSelector().select(
        _image(100, 100),
        _image(150, 80),
        _ladder(_image(100, 80), _image(150, 80)),
        source_masks=_masks(face=False),
        reference_masks=_masks(face=False),
    )
    assert result.best_level == "a_p025"
    assert result.region_validity["face"] is False
    assert sum(result.weights.values()) == 1.0


def test_empty_face_and_background_masks_match_fallback_validity_and_json():
    empty = np.zeros((20, 20), bool)
    source_masks = RegionMasks(_masks().person, empty, empty)
    reference_masks = RegionMasks(_masks().person, empty, empty)

    result = PseudoGTSelector().select(
        _image(100, 100),
        _image(150, 80),
        _ladder(_image(100, 80), _image(150, 80)),
        source_masks=source_masks,
        reference_masks=reference_masks,
    )
    payload = result.to_dict()

    assert result.weights == {"person": 1.0}
    assert result.region_validity == {"person": True, "face": False, "local_background": False}
    assert result.best_components["face_error"] is None
    assert result.best_components["person_background_error"] is None
    assert result.best_components["background_error"] is None
    assert payload["region_validity"] == result.region_validity


def test_selection_is_deterministic_and_all_levels_have_complete_scores():
    selector = PseudoGTSelector()
    arguments = dict(
        source=_image(100, 100),
        reference=_image(150, 80),
        ladder=_ladder(_image(100, 80), _image(150, 80)),
        source_masks=_masks(),
        reference_masks=_masks(),
    )
    first = selector.select(**arguments)
    second = selector.select(**arguments)
    assert first.best_level == second.best_level
    assert first.best_score == second.best_score
    assert list(first.levels) == list(LEVEL_COEFFICIENTS)
    for score in first.levels.values():
        assert score.person_error is not None
        assert score.person_background_error is not None
        assert score.background_error is not None


class _Provider(MaskProvider):
    def __init__(self, source_detections, reference_detections):
        self.responses = [source_detections, reference_detections]
        self.calls = 0

    def get_person_masks(self, image):
        response = self.responses[self.calls]
        self.calls += 1
        return response

    def get_face_mask(self, image, person_mask):
        return None


def test_no_source_person_has_mandated_status():
    provider = _Provider([], [PersonDetection(_masks().person)])
    result = PseudoGTSelector(provider).select(_image(1, 1), _image(1, 1), _ladder(_image(2, 2), _image(3, 3)))
    assert result.status is SelectionStatus.NO_PERSON


def test_no_reference_person_has_mandated_status():
    provider = _Provider([PersonDetection(_masks().person)], [])
    result = PseudoGTSelector(provider).select(_image(1, 1), _image(1, 1), _ladder(_image(2, 2), _image(3, 3)))
    assert result.status is SelectionStatus.INVALID_REFERENCE_PERSON


def test_ambiguous_primary_person_has_mandated_status():
    first = np.zeros((20, 20), bool)
    second = np.zeros((20, 20), bool)
    first[5:15, 2:8] = True
    second[5:15, 12:18] = True
    provider = _Provider([PersonDetection(first), PersonDetection(second)], [PersonDetection(first)])
    result = PseudoGTSelector(provider).select(_image(1, 1), _image(1, 1), _ladder(_image(2, 2), _image(3, 3)))
    assert result.status is SelectionStatus.AMBIGUOUS_PERSON


def test_missing_level_is_invalid_and_reported():
    ladder = _ladder(_image(2, 2), _image(3, 3))
    del ladder["a_p100"]
    result = PseudoGTSelector().select(
        _image(1, 1), _image(1, 1), ladder, source_masks=_masks(), reference_masks=_masks()
    )
    assert result.status is SelectionStatus.INVALID
    assert result.missing_levels == ["a_p100"]


def test_unaligned_ladder_is_invalid():
    ladder = _ladder(_image(2, 2), _image(3, 3))
    ladder["a_p100"] = np.zeros((19, 20, 3), np.uint8)
    result = PseudoGTSelector().select(
        _image(1, 1), _image(1, 1), ladder, source_masks=_masks(), reference_masks=_masks()
    )
    assert result.status is SelectionStatus.INVALID


def test_selected_path_is_the_original_ladder_path():
    paths = {level: Path(f"/ladder/{level}.png") for level in LEVEL_COEFFICIENTS}
    result = PseudoGTSelector().select(
        _image(100, 100),
        _image(150, 80),
        _ladder(_image(100, 80), _image(150, 80)),
        source_masks=_masks(),
        reference_masks=_masks(),
        ladder_paths=paths,
    )
    assert result.selected_path is paths[result.best_level]
