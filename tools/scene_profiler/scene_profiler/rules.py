from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .schema import ElementDecision, LabelDecision, RawSceneRecord, ReviewDecision, SceneProfile

EPS = 1e-8


def normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    values = {k: max(float(v), 0.0) for k, v in scores.items()}
    total = sum(values.values())
    if total <= EPS:
        return {k: 1.0 / max(len(values), 1) for k in values}
    return {k: v / total for k, v in values.items()}


def top(scores: Mapping[str, float]) -> tuple[str, float, float]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return "unknown", 0.0, 0.0
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], float(ranked[0][1]), float(ranked[0][1] - second)


def ratio(value: float, reference: float) -> float:
    return float(np.clip(value / max(reference, EPS), 0.0, 1.0))


@dataclass
class CalibrationState:
    brightness_thresholds: tuple[float, float]
    brightness_source: str
    dynamic_range_thresholds: tuple[float, float]
    dynamic_range_source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brightness_thresholds": list(self.brightness_thresholds),
            "brightness_source": self.brightness_source,
            "dynamic_range_thresholds": list(self.dynamic_range_thresholds),
            "dynamic_range_source": self.dynamic_range_source,
        }


class DatasetCalibrator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def _thresholds(self, records: Sequence[RawSceneRecord], section: str, feature: str,
                    fixed_keys: tuple[str, str]) -> tuple[tuple[float, float], str]:
        cfg = self.config["classification"][section]
        if cfg["calibration_mode"] != "dataset_quantile":
            return (float(cfg[fixed_keys[0]]), float(cfg[fixed_keys[1]])), "fixed_config"
        values = np.asarray([
            float(r.photometric_features[feature]) for r in records
            if r.photometric_features.get(feature) is not None
        ])
        if values.size < 10:
            raise ValueError(f"At least 10 records are required for {section} quantile calibration")
        q1, q2 = cfg["quantiles"]
        return (float(np.quantile(values, q1)), float(np.quantile(values, q2))), "dataset_quantile"

    def fit(self, records: Sequence[RawSceneRecord]) -> CalibrationState:
        bt, bs = self._thresholds(records, "brightness", "mid_brightness_ev",
                                  ("dark_to_medium_ev", "medium_to_bright_ev"))
        dt, ds = self._thresholds(records, "dynamic_range_proxy", "display_dr_proxy_ev",
                                  ("low_to_medium_ev", "medium_to_high_ev"))
        return CalibrationState(bt, bs, dt, ds)


class SceneRuleEngine:
    def __init__(self, config: Mapping[str, Any], calibration: CalibrationState) -> None:
        self.c = config
        self.cal = calibration

    def environment(self, r: RawSceneRecord) -> tuple[LabelDecision, List[str], Dict[str, float]]:
        seg, sem = r.segmentation_ratios, r.semantic_probabilities.get("environment", {})
        indoor = seg.get("indoor_structure", 0.0) + seg.get("furniture", 0.0) + 0.25 * seg.get("window_glass", 0.0)
        outdoor = seg.get("sky", 0.0) + seg.get("vegetation", 0.0) + seg.get("road_ground", 0.0) + seg.get("water", 0.0) + 0.2 * seg.get("vehicle", 0.0)
        region = normalize({"indoor": indoor, "outdoor": outdoor, "transition": 2 * min(indoor, outdoor)})
        sw, rw = self.c["fusion"]["environment_semantic_weight"], self.c["fusion"]["environment_region_weight"]
        fused = normalize({k: sw * sem.get(k, 1 / 3) + rw * region.get(k, 0.0) for k in ("indoor", "outdoor", "transition")})
        label, conf, margin = top(fused)
        cfg, reasons = self.c["classification"]["environment"], []
        if indoor >= cfg["transition_dual_evidence"] and outdoor >= cfg["transition_dual_evidence"]:
            label, conf = "transition", max(conf, 0.65)
        elif conf < cfg["min_confidence"] or margin < cfg["min_margin"]:
            label, reasons = "uncertain", ["low_environment_confidence"]
        evidence = {"indoor_region_evidence": float(indoor), "outdoor_region_evidence": float(outdoor), "margin": margin}
        return LabelDecision(label, conf, fused, evidence), reasons, evidence

    def time(self, r: RawSceneRecord, env: LabelDecision, ee: Mapping[str, float]) -> tuple[LabelDecision, List[str]]:
        cfg, seg = self.c["classification"]["time"], r.segmentation_ratios
        sky_ratio, outdoor = seg.get("sky", 0.0), ee.get("outdoor_region_evidence", 0.0)
        if env.label == "indoor" and outdoor < cfg["min_outdoor_evidence"] and sky_ratio < cfg["min_sky_ratio"]:
            return LabelDecision("unknown", 0.9, r.semantic_probabilities.get("time", {}),
                                 {"reason": "time_not_observable_in_pure_indoor_scene"}), []
        sem = r.semantic_probabilities.get("time", {})
        sky_mid = r.regional_photometry.get("sky", {}).get("mid_luminance")
        mid = float(r.photometric_features.get("mid_luminance") or 0.0)
        sky_mid = float(sky_mid) if sky_mid is not None else mid
        light = seg.get("light_display", 0.0)
        physical = normalize({
            "day": max(sky_mid - 0.2, 0) + 0.4 * max(mid - 0.2, 0),
            "twilight": max(0, 0.35 - abs(sky_mid - 0.18)),
            "night": max(0, 0.25 - sky_mid) + 0.6 * light + 0.3 * max(0, 0.15 - mid),
        })
        fused = normalize({k: 0.7 * sem.get(k, 1 / 3) + 0.3 * physical[k] for k in physical})
        label, conf, margin = top(fused)
        reasons: List[str] = []
        if conf < cfg["min_confidence"] or margin < cfg["min_margin"]:
            label, reasons = "uncertain", ["low_time_confidence"]
        return LabelDecision(label, conf, fused, {"sky_ratio": sky_ratio, "margin": margin, "time_is_observable": True}), reasons

    def content(self, r: RawSceneRecord) -> tuple[Dict[str, float], List[str], List[str]]:
        cfg, seg, sem = self.c["classification"]["content"], r.segmentation_ratios, r.semantic_probabilities.get("content", {})
        region = {
            "portrait": ratio(seg.get("person", 0.0), cfg["person_ratio_reference"]),
            "city": ratio(seg.get("architecture", 0.0) + seg.get("road_ground", 0.0) + seg.get("vehicle", 0.0), cfg["city_ratio_reference"]),
            "nature": ratio(seg.get("vegetation", 0.0) + seg.get("water", 0.0) + seg.get("sky", 0.0), cfg["nature_ratio_reference"]),
        }
        sw, rw = self.c["fusion"]["content_semantic_weight"], self.c["fusion"]["content_region_weight"]
        scores = {k: float(np.clip(sw * sem.get(k, 0.0) + rw * region[k], 0, 1)) for k in region}
        labels = [k for k, v in scores.items() if v >= cfg["threshold"]]
        reasons = ["no_confident_content_label"] if not labels else []
        return scores, labels, reasons

    def ordinal(self, value: float, thresholds: tuple[float, float], labels: tuple[str, str, str], proxy: bool = False) -> LabelDecision:
        lo, hi = thresholds
        label = labels[0] if value < lo else labels[1] if value < hi else labels[2]
        distance = min(abs(value - lo), abs(value - hi))
        confidence = float(np.clip(0.55 + 0.15 * distance, 0.55, 1.0))
        return LabelDecision(label, confidence, {}, {"value": value, "thresholds": [lo, hi], "not_physical_sensor_dynamic_range": proxy})

    def illumination(self, r: RawSceneRecord) -> tuple[LabelDecision, List[str]]:
        photo = r.photometric_features.get("apparent_illumination", {})
        sem = r.semantic_probabilities.get("illumination_semantic", {})
        if photo.get("label") == "unknown" and not sem:
            return LabelDecision("unknown", 0.0, {}, {"physical_cct_estimated": False}), ["insufficient_illumination_evidence"]
        if photo.get("label") == "mixed" and photo.get("confidence", 0) >= 0.6:
            label, conf = "mixed", float(photo["confidence"])
        else:
            pw, sw = self.c["fusion"]["illumination_photometric_weight"], self.c["fusion"]["illumination_semantic_weight"]
            pscore = {k: 0.0 for k in ("warm", "neutral", "cool", "mixed")}
            if photo.get("label") in pscore:
                pscore[photo["label"]] = photo.get("confidence", 0.0)
            fused = normalize({k: pw * pscore[k] + sw * sem.get(k, 0.25) for k in pscore})
            label, conf, _ = top(fused)
        return LabelDecision(label, conf, dict(sem), {"photometric": photo, "physical_cct_estimated": False}), []

    def exposure(self, r: RawSceneRecord) -> LabelDecision:
        f, cfg = r.photometric_features, self.c["classification"]["exposure"]
        shadow = float(f.get("non_sky_shadow_ratio") or f.get("deep_shadow_ratio") or 0)
        clip = float(f.get("non_sky_clip_ratio") or f.get("clip_ratio") or 0)
        ev = float(f.get("non_sky_mid_brightness_ev") or f.get("mid_brightness_ev") or -20)
        person = f.get("person_mid_brightness_ev")
        under = (shadow >= cfg["shadow_ratio_under"] and ev < cfg["non_sky_under_ev"]) or (person is not None and person < cfg["person_under_ev"])
        over = clip >= cfg["clip_ratio_over"] and ev > cfg["non_sky_over_ev"]
        dual = (shadow >= cfg["dual_shadow_ratio"] and clip >= cfg["dual_clip_ratio"]) or (under and over)
        label = "dual-risk" if dual else "underexposed" if under else "overexposed" if over else "normal"
        return LabelDecision(label, 0.75, {}, {"non_sky_shadow_ratio": shadow, "non_sky_clip_ratio": clip, "non_sky_mid_brightness_ev": ev, "person_mid_brightness_ev": person})

    def elements(self, r: RawSceneRecord) -> Dict[str, ElementDecision]:
        seg, ov, cfg = r.segmentation_ratios, r.open_vocab, self.c["classification"]["elements"]
        seg_on = r.model_metadata.get("segformer_model") is not None
        sky_ratio, light_ratio = seg.get("sky", 0.0), seg.get("light_display", 0.0)
        sky = ElementDecision("evaluated", sky_ratio >= cfg["sky_presence_ratio"], min(1.0, 0.6 + 3 * sky_ratio), sky_ratio, "segformer", ["sky"] if sky_ratio else []) if seg_on else ElementDecision("not_evaluated", None, None, None, "segformer_disabled")
        open_light = ov.get("light_source")
        if open_light:
            area = open_light.get("mask_area_ratio") or open_light.get("box_union_ratio") or 0.0
            light = ElementDecision("evaluated", bool(open_light.get("present")) or (seg_on and light_ratio >= cfg["semantic_light_presence_ratio"]), max(float(open_light.get("confidence", 0)), min(1.0, 0.5 + 5 * light_ratio)), max(float(area), light_ratio), str(open_light.get("detection_source", "grounding_dino")), list(open_light.get("types", [])))
        elif seg_on:
            light = ElementDecision("evaluated", light_ratio >= cfg["semantic_light_presence_ratio"], min(1.0, 0.5 + 5 * light_ratio), light_ratio, "segformer", ["semantic_light_or_display"] if light_ratio else [])
        else:
            light = ElementDecision("not_evaluated", None, None, None, "segformer_and_open_vocab_disabled")
        ref = ov.get("reflective_surface")
        reflection = ElementDecision("not_evaluated", None, None, None, "open_vocab_disabled", metadata={"reason": "reflection_requires_open_vocab"}) if not ref else ElementDecision("evaluated", bool(ref.get("present")), float(ref.get("confidence", 0)), float(ref.get("mask_area_ratio") or ref.get("box_union_ratio") or 0), str(ref.get("detection_source", "grounding_dino")), list(ref.get("types", [])))
        return {"sky": sky, "light_source": light, "reflective_surface": reflection}

    def build_profile(self, r: RawSceneRecord) -> SceneProfile:
        reasons: List[str] = []
        env, x, evidence = self.environment(r); reasons += x
        time, x = self.time(r, env, evidence); reasons += x
        content_scores, content_labels, x = self.content(r); reasons += x
        dr = self.ordinal(float(r.photometric_features["display_dr_proxy_ev"]), self.cal.dynamic_range_thresholds, ("low", "medium", "high"), True)
        bright = self.ordinal(float(r.photometric_features["mid_brightness_ev"]), self.cal.brightness_thresholds, ("dark", "medium", "bright"))
        illum, x = self.illumination(r); reasons += x
        exposure, elements = self.exposure(r), self.elements(r)
        min_valid = self.c["classification"]["uncertainty"]["min_segmentation_valid_ratio"]
        if r.segmentation_valid_ratio is not None and r.segmentation_valid_ratio < min_valid:
            reasons.append("low_segmentation_valid_ratio")
        limitations = ["dynamic_range_proxy_is_not_physical_sensor_dynamic_range", "apparent_illumination_is_not_physical_cct", "indoor_time_is_unknown_when_external_time_cues_are_not_observable"]
        if elements["reflective_surface"].status == "not_evaluated":
            limitations.append("reflective_surface_not_evaluated_without_open_vocab_backend")
        return SceneProfile(r.scene_id, r.image_path, env, time, content_scores, content_labels, dr, illum, bright, exposure, elements, r.photometric_features,
                            {"ratios": r.segmentation_ratios, "regional_photometry": r.regional_photometry, "segmentation_valid_ratio": r.segmentation_valid_ratio, "segmentation_mean_confidence": r.segmentation_mean_confidence},
                            {**r.model_metadata, "calibration": self.cal.to_dict()}, ReviewDecision(bool(reasons), sorted(set(reasons))), limitations)
