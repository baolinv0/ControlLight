from __future__ import annotations

import csv
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from PIL import Image

from .backends import OpenVocabularyDetector, SAM2BoxRefiner, SegFormerSceneParser, SigLIP2SceneClassifier
from .photometry import compute_global_photometry, estimate_apparent_illumination, load_rgb_image, robust_stats
from .rules import DatasetCalibrator, SceneRuleEngine
from .schema import RawSceneRecord, SceneProfile


class SceneProfilerPipeline:
    def __init__(self, config: Mapping[str, Any], load_models: bool = True) -> None:
        self.config = config
        runtime, models = config["runtime"], config["models"]
        device, dtype = str(runtime["device"]), str(runtime["dtype"])
        self.semantic_backend = SigLIP2SceneClassifier(models["siglip2"]["model_id"], config["semantic_axes"], device, dtype) if load_models and models["siglip2"]["enabled"] else None
        self.segmentation_backend = SegFormerSceneParser(models["segformer"]["model_id"], config["segmentation_groups"], models["segformer"]["min_pixel_confidence"], device, dtype) if load_models and models["segformer"]["enabled"] else None
        self.open_vocab_backend = OpenVocabularyDetector(models["open_vocab"]["model_id"], config["open_vocab_labels"], models["open_vocab"]["box_threshold"], models["open_vocab"]["text_threshold"], device, dtype) if load_models and models["open_vocab"]["enabled"] else None
        self.sam2_backend = SAM2BoxRefiner(models["sam2"]["model_id"], device, dtype) if load_models and models["sam2"]["enabled"] else None

    @staticmethod
    def resolve_input_directory(path: Path, reference: str) -> Path:
        return path / reference if (path / reference).is_dir() else path

    def discover_images(self, path: Path) -> List[Path]:
        root = self.resolve_input_directory(path, self.config["input"]["reference_level"])
        if not root.is_dir():
            raise FileNotFoundError(root)
        exts = {str(x).lower() for x in self.config["input"]["extensions"]}
        iterator = root.rglob("*") if self.config["input"]["recursive"] else root.glob("*")
        images = sorted(x for x in iterator if x.is_file() and x.suffix.lower() in exts)
        if not images:
            raise RuntimeError(f"No images found under {root}")
        return images

    def default_semantics(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, axis in self.config["semantic_axes"].items():
            labels = list(axis["classes"])
            out[name] = {x: (1 / len(labels) if axis.get("exclusive", True) else 0.0) for x in labels}
        return out

    @staticmethod
    def scene_id(path: Path, root: Path) -> str:
        return str(path.relative_to(root).with_suffix("")).replace("\\", "/")

    def process_image(self, path: Path, root: Path) -> tuple[RawSceneRecord, Dict[str, np.ndarray], Image.Image]:
        image, _, linear = load_rgb_image(path)
        semantic = self.semantic_backend.score(image) if self.semantic_backend else self.default_semantics()
        if self.segmentation_backend:
            seg = self.segmentation_backend.parse(image)
            masks, ratios, valid, conf = seg.group_masks, seg.group_ratios, seg.valid_pixel_ratio, seg.mean_confidence
        else:
            masks, ratios, valid, conf = {}, {k: 0.0 for k in self.config["segmentation_groups"]}, None, None
        ec = self.config["classification"]["exposure"]
        global_f, regional = compute_global_photometry(linear, masks, ec["deep_shadow_threshold"], ec["highlight_clip_threshold"])
        excluded = None
        for name in ("sky", "vegetation", "person", "light_display", "window_glass"):
            if name in masks:
                excluded = np.zeros_like(masks[name], bool) if excluded is None else excluded
                excluded |= masks[name]
        global_f["apparent_illumination"] = estimate_apparent_illumination(linear, excluded, self.config["classification"]["illumination"])
        ov, sam_masks = {}, {}
        if self.open_vocab_backend:
            ov = self.open_vocab_backend.detect(image)
            if self.sam2_backend:
                ov, sam_masks = self.sam2_backend.refine(image, ov)
        metadata = {
            "siglip2_model": self.semantic_backend.model_id if self.semantic_backend else None,
            "segformer_model": self.segmentation_backend.model_id if self.segmentation_backend else None,
            "open_vocab_model": self.open_vocab_backend.model_id if self.open_vocab_backend else None,
            "sam2_model": self.sam2_backend.model_id if self.sam2_backend else None,
            "reference_level": self.config["input"]["reference_level"],
        }
        record = RawSceneRecord(self.scene_id(path, root), str(path), image.width, image.height, semantic,
                                {k: float(v) for k, v in ratios.items()}, valid, conf, global_f, regional, ov, metadata)
        return record, {**masks, **{f"open_{k}": v for k, v in sam_masks.items()}}, image

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def csv_row(p: SceneProfile) -> Dict[str, Any]:
        e = p.elements
        return {
            "scene_id": p.scene_id, "image_path": p.image_path,
            "environment": p.environment.label, "environment_confidence": p.environment.confidence,
            "time": p.time.label, "time_confidence": p.time.confidence,
            "portrait_score": p.content_scores.get("portrait", 0), "city_score": p.content_scores.get("city", 0), "nature_score": p.content_scores.get("nature", 0),
            "content_labels": ";".join(p.content_labels), "dynamic_range_proxy": p.dynamic_range_proxy.label,
            "dynamic_range_proxy_ev": p.dynamic_range_proxy.evidence.get("value"), "apparent_illumination": p.apparent_illumination.label,
            "input_brightness": p.input_brightness.label, "mid_brightness_ev": p.input_brightness.evidence.get("value"),
            "exposure": p.exposure.label, "has_sky": e["sky"].present, "has_light_source": e["light_source"].present,
            "has_reflective_surface": e["reflective_surface"].present, "reflection_status": e["reflective_surface"].status,
            "review_required": p.review.required, "review_reasons": ";".join(p.review.reasons),
        }

    def write_csv(self, path: Path, profiles: List[SceneProfile]) -> None:
        rows = [self.csv_row(p) for p in profiles]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    @staticmethod
    def save_mask_overlay(image: Image.Image, masks: Mapping[str, np.ndarray], path: Path) -> None:
        if not masks:
            return
        overlay = np.asarray(image, dtype=np.float32).copy()
        for i, mask in enumerate(masks.values(), 1):
            if mask.shape != overlay.shape[:2]:
                continue
            color = np.array([(37*i+53)%255, (97*i+31)%255, (173*i+11)%255], dtype=np.float32)
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)

    def summary(self, profiles: List[SceneProfile], discovered: int, errors: List[Dict[str, Any]], calibration: Dict[str, Any]) -> Dict[str, Any]:
        def count(attr: str) -> Dict[str, int]:
            return dict(Counter(getattr(p, attr).label for p in profiles))
        content = {k: sum(k in p.content_labels for p in profiles) for k in ("portrait", "city", "nature")}
        return {
            "total_discovered": discovered, "valid_scenes": len(profiles), "failed_scenes": len(errors),
            "environment": count("environment"), "time": count("time"), "content_multilabel": content,
            "dynamic_range_proxy": count("dynamic_range_proxy"), "apparent_illumination": count("apparent_illumination"),
            "input_brightness": count("input_brightness"), "exposure": count("exposure"),
            "review_required": sum(p.review.required for p in profiles), "calibration": calibration,
            "numeric": {
                "mid_brightness_ev": robust_stats(p.photometric_features.get("mid_brightness_ev") for p in profiles),
                "display_dr_proxy_ev": robust_stats(p.photometric_features.get("display_dr_proxy_ev") for p in profiles),
            },
        }

    @staticmethod
    def load_raw(path: Path) -> List[RawSceneRecord]:
        rows = []
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try: rows.append(RawSceneRecord(**json.loads(line)))
                    except Exception as exc: raise ValueError(f"Invalid record at line {n}: {exc}") from exc
        if not rows: raise RuntimeError("No raw records")
        return rows

    def finish(self, records: List[RawSceneRecord], output: Path, discovered: int, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        calibration = DatasetCalibrator(self.config).fit(records)
        engine = SceneRuleEngine(self.config, calibration)
        profiles = [engine.build_profile(x) for x in records]
        self.write_jsonl(output / self.config["output"]["raw_features"], (x.to_dict() for x in records))
        self.write_jsonl(output / self.config["output"]["jsonl"], (x.to_dict() for x in profiles))
        self.write_jsonl(output / self.config["output"]["errors"], errors)
        self.write_csv(output / self.config["output"]["csv"], profiles)
        result = self.summary(profiles, discovered, errors, calibration.to_dict())
        with (output / self.config["output"]["summary"]).open("w", encoding="utf-8") as f: json.dump(result, f, indent=2, ensure_ascii=False)
        return result

    def relabel_from_raw_features(self, path: Path, output: Path) -> Dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        records = self.load_raw(path)
        result = self.finish(records, output, len(records), [])
        result["relabel_source"] = str(path)
        return result

    def run(self, input_path: Path, output: Path, max_images: int = 0) -> Dict[str, Any]:
        output.mkdir(parents=True, exist_ok=True)
        root = self.resolve_input_directory(input_path, self.config["input"]["reference_level"])
        images = self.discover_images(input_path)
        if max_images: images = images[:max_images]
        records, errors = [], []
        save_overlays, save_masks = int(self.config["runtime"]["save_overlays"]), bool(self.config["runtime"]["save_masks"])
        for i, path in enumerate(images):
            try:
                record, masks, image = self.process_image(path, root); records.append(record)
                rel = path.relative_to(root)
                if i < save_overlays: self.save_mask_overlay(image, masks, output / "overlays" / rel.with_suffix(".jpg"))
                if save_masks and masks:
                    target = output / "masks" / rel.with_suffix(".npz"); target.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(target, **masks)
            except Exception as exc:
                errors.append({"image": str(path), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                if not self.config["runtime"]["continue_on_error"]: raise
        if not records: raise RuntimeError("No images were successfully processed")
        return self.finish(records, output, len(images), errors)
