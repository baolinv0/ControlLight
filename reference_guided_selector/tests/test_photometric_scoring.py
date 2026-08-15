import numpy as np
import pytest

from pseudo_gt_selector.config import LEVEL_COEFFICIENTS, SelectorConfig
from pseudo_gt_selector.photometric import chromaticity_descriptor, rgb_to_log_luminance
from pseudo_gt_selector.scorer import ToneScorer
from pseudo_gt_selector.tone_descriptor import ToneDescriptorExtractor


def test_levels_and_default_weights_match_the_spec():
    assert LEVEL_COEFFICIENTS == {
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
    cfg = SelectorConfig()
    assert cfg.weights_with_face == {"person": 0.4, "person_background": 0.3, "face": 0.2, "background": 0.1}
    assert cfg.weights_without_face == {"person": 0.5, "person_background": 0.35, "background": 0.15}


def test_srgb_is_linearized_before_log_luminance():
    black_white = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    log_y = rgb_to_log_luminance(black_white)
    assert log_y[0, 0] == pytest.approx(np.log2(1e-6))
    assert log_y[0, 1] == pytest.approx(np.log2(1.0 + 1e-6))

    middle_gray = np.full((1, 1, 3), 128, dtype=np.uint8)
    expected_linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
    assert rgb_to_log_luminance(middle_gray)[0, 0] == pytest.approx(np.log2(expected_linear + 1e-6))


def test_extractor_returns_quantiles_and_person_background_relation():
    levels = np.array([16, 32, 64, 128, 240], dtype=np.uint8)
    image = np.stack([np.tile(levels, (2, 1))] * 3, axis=-1)
    person = np.zeros((2, 5), bool)
    person[0] = True
    background = np.zeros((2, 5), bool)
    background[1] = True

    desc = ToneDescriptorExtractor().extract(image, person, background_mask=background)

    expected = np.percentile(rgb_to_log_luminance(image)[0], [10, 25, 50, 75, 90])
    assert desc.person == pytest.approx(expected)
    assert desc.background == pytest.approx(expected)
    assert desc.person_background == pytest.approx((0.0, expected[4] - expected[0], expected[4] - expected[0]))


def test_chromaticity_is_a_region_debug_descriptor():
    image = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    red = chromaticity_descriptor(image, np.array([[True, False]]))
    green = chromaticity_descriptor(image, np.array([[False, True]]))
    assert red == pytest.approx((1.0, 0.0))
    assert green == pytest.approx((0.0, 1.0))


def _features(person, relation=(0, 0, 0), face=None, background=None, color=None):
    from pseudo_gt_selector.models import ToneFeatures

    return ToneFeatures(
        person=np.asarray(person, dtype=float),
        face=None if face is None else np.asarray(face, dtype=float),
        background=None if background is None else np.asarray(background, dtype=float),
        person_background=None if relation is None else np.asarray(relation, dtype=float),
        color={} if color is None else color,
    )


def test_default_score_components_and_weights():
    ref = _features([0] * 5, face=[0] * 5, background=[0] * 5)
    cand = _features([1] * 5, relation=(2, 4, 8), face=[3] * 5, background=[4] * 5)

    score = ToneScorer().compare(cand, ref)

    assert score.person_error == pytest.approx(1.0)
    assert score.person_background_error == pytest.approx(5.0)
    assert score.face_error == pytest.approx(3.0)
    assert score.background_error == pytest.approx(4.0)
    assert score.weights == {"person": 0.4, "person_background": 0.3, "face": 0.2, "background": 0.1}
    assert score.score == pytest.approx(2.9)


def test_face_fallback_uses_no_face_weights_summing_to_one():
    ref = _features([0] * 5, background=[0] * 5)
    cand = _features([1] * 5, relation=(2, 4, 8), background=[4] * 5)
    score = ToneScorer().compare(cand, ref)
    assert score.face_error is None
    assert score.weights == {"person": 0.5, "person_background": 0.35, "background": 0.15}
    assert sum(score.weights.values()) == pytest.approx(1.0)


def test_background_fallback_renormalizes_person_and_face_only():
    ref = _features([0] * 5, relation=None, face=[0] * 5, background=None)
    cand = _features([1] * 5, relation=None, face=[2] * 5, background=None)
    score = ToneScorer().compare(cand, ref)
    assert score.person_background_error is None
    assert score.background_error is None
    assert score.weights == {"person": pytest.approx(2 / 3), "face": pytest.approx(1 / 3)}
    assert score.score == pytest.approx(4 / 3)


def test_color_difference_is_reported_but_does_not_change_score():
    ref = _features([0] * 5, face=[0] * 5, background=[0] * 5, color={"person": (0.5, 0.5)})
    cand_a = _features([1] * 5, face=[1] * 5, background=[1] * 5, color={"person": (0.5, 0.5)})
    cand_b = _features([1] * 5, face=[1] * 5, background=[1] * 5, color={"person": (1.0, 0.0)})
    scorer = ToneScorer()
    a = scorer.compare(cand_a, ref)
    b = scorer.compare(cand_b, ref)
    assert a.score == b.score
    assert a.color_differences["person_color_diff"] == pytest.approx(0.0)
    assert b.color_differences["person_color_diff"] > 0
