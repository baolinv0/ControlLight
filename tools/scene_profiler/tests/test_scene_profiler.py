from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from scene_profiler.config import load_config
from scene_profiler.photometry import (
    compute_global_photometry,
    estimate_apparent_illumination,
    srgb_to_linear,
)
from scene_profiler.rules import DatasetCalibrator, SceneRuleEngine
from scene_profiler.schema import RawSceneRecord


DEFAULT_CONFIG = TEST_ROOT / "config" / "default.yaml"


def make_raw_record(config, *, indoor=True) -> RawSceneRecord:
    semantic = {
        "environment": {"indoor": 0.85, "outdoor": 0.10, "transition": 0.05}
        if indoor else {"indoor": 0.05, "outdoor": 0.90, "transition": 0.05},
        "time": {"day": 0.70, "twilight": 0.10, "night": 0.20},
        "content": {"portrait": 0.75, "city": 0.65, "nature": 0.15},
        "illumination_semantic": {"warm": 0.65, "neutral": 0.20, "cool": 0.10, "mixed": 0.05},
    }
    ratios = {
        "sky": 0.0 if indoor else 0.25,
        "person": 0.12,
        "vegetation": 0.02,
        "water": 0.0,
        "road_ground": 0.0 if indoor else 0.20,
        "architecture": 0.25,
        "indoor_structure": 0.40 if indoor else 0.0,
        "window_glass": 0.05,
        "vehicle": 0.02,
        "furniture": 0.25 if indoor else 0.0,
        "light_display": 0.01,
    }
    photometric = {
        "mid_luminance": 0.20,
        "mid_brightness_ev": float(np.log2(0.20)),
        "mean_luminance": 0.22,
        "mean_brightness_ev": float(np.log2(0.22)),
        "dr_p95_p05_ev": 5.8,
        "dr_p99_p01_ev": 8.1,
        "display_dr_proxy_ev": 6.2,
        "semantic_region_brightness_span_ev": 2.0,
        "dual_tail_score": 0.02,
        "deep_shadow_ratio": 0.08,
        "clip_ratio": 0.01,
        "non_sky_shadow_ratio": 0.08,
        "non_sky_clip_ratio": 0.01,
        "non_sky_mid_brightness_ev": float(np.log2(0.20)),
        "person_mid_brightness_ev": float(np.log2(0.18)),
        "apparent_illumination": {
            "label": "warm",
            "confidence": 0.85,
            "physical_cct_estimated": False,
        },
    }
    regional = {
        "person": {"mid_luminance": 0.18, "mid_brightness_ev": float(np.log2(0.18)), "pixels": 1000},
        "sky": {"mid_luminance": None, "mid_brightness_ev": None, "pixels": 0},
    }
    return RawSceneRecord(
        scene_id="scene_001",
        image_path="scene_001.png",
        image_width=256,
        image_height=256,
        semantic_probabilities=semantic,
        segmentation_ratios=ratios,
        segmentation_valid_ratio=0.92,
        segmentation_mean_confidence=0.88,
        photometric_features=photometric,
        regional_photometry=regional,
        open_vocab={},
        model_metadata={},
    )


class PhotometryTests(unittest.TestCase):
    def test_bright_image_has_higher_mid_brightness(self):
        dark = srgb_to_linear(np.full((64, 64, 3), 0.15, dtype=np.float32))
        bright = srgb_to_linear(np.full((64, 64, 3), 0.70, dtype=np.float32))
        dark_features, _ = compute_global_photometry(dark, {}, 0.01, 0.99)
        bright_features, _ = compute_global_photometry(bright, {}, 0.01, 0.99)
        self.assertGreater(bright_features["mid_brightness_ev"], dark_features["mid_brightness_ev"])

    def test_clipping_ratio_detects_white_pixels(self):
        srgb = np.full((32, 32, 3), 0.3, dtype=np.float32)
        srgb[:16] = 1.0
        features, _ = compute_global_photometry(srgb_to_linear(srgb), {}, 0.01, 0.99)
        self.assertAlmostEqual(features["clip_ratio"], 0.5, places=2)

    def test_mixed_apparent_illumination(self):
        config = load_config(DEFAULT_CONFIG)["classification"]["illumination"]
        config = copy.deepcopy(config)
        config["max_neutral_saturation"] = 0.9
        config["mixed_min_separation_log2_br"] = 0.25
        image = np.zeros((128, 128, 3), dtype=np.float32)
        image[:, :64] = np.asarray([0.55, 0.35, 0.20], dtype=np.float32)
        image[:, 64:] = np.asarray([0.20, 0.35, 0.55], dtype=np.float32)
        result = estimate_apparent_illumination(image, None, config)
        self.assertEqual(result["label"], "mixed")
        self.assertFalse(result["physical_cct_estimated"])


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG)

    def test_pure_indoor_time_is_unknown(self):
        raw = make_raw_record(self.config, indoor=True)
        calibration = DatasetCalibrator(self.config).fit([raw] * 10)
        profile = SceneRuleEngine(self.config, calibration).build_profile(raw)
        self.assertEqual(profile.environment.label, "indoor")
        self.assertEqual(profile.time.label, "unknown")

    def test_content_is_multilabel(self):
        raw = make_raw_record(self.config, indoor=False)
        calibration = DatasetCalibrator(self.config).fit([raw] * 10)
        profile = SceneRuleEngine(self.config, calibration).build_profile(raw)
        self.assertIn("portrait", profile.content_labels)
        self.assertIn("city", profile.content_labels)

    def test_reflection_is_not_false_when_backend_disabled(self):
        raw = make_raw_record(self.config, indoor=False)
        calibration = DatasetCalibrator(self.config).fit([raw] * 10)
        profile = SceneRuleEngine(self.config, calibration).build_profile(raw)
        reflection = profile.elements["reflective_surface"]
        self.assertEqual(reflection.status, "not_evaluated")
        self.assertIsNone(reflection.present)

    def test_dynamic_range_is_explicitly_proxy(self):
        raw = make_raw_record(self.config, indoor=False)
        calibration = DatasetCalibrator(self.config).fit([raw] * 10)
        profile = SceneRuleEngine(self.config, calibration).build_profile(raw)
        self.assertTrue(profile.dynamic_range_proxy.evidence["not_physical_sensor_dynamic_range"])


if __name__ == "__main__":
    unittest.main()
