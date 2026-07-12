from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image


class BackendError(RuntimeError):
    """Raised when a model backend cannot be initialized or executed."""


def resolve_torch_device(device_spec: str):
    import torch

    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BackendError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_torch_dtype(device, dtype_spec: str):
    import torch

    if dtype_spec == "auto":
        if device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_spec not in mapping:
        raise BackendError(f"Unsupported dtype: {dtype_spec}")
    if device.type == "cpu" and dtype_spec in {"float16", "bfloat16"}:
        return torch.float32
    return mapping[dtype_spec]


def move_batch(batch: Mapping[str, Any], device, dtype) -> Dict[str, Any]:
    import torch

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if isinstance(value, torch.Tensor) and value.is_floating_point() and key == "pixel_values":
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved


class SigLIP2SceneClassifier:
    """One-pass prompt-ensemble zero-shot classifier for all semantic axes."""

    def __init__(
        self,
        model_id: str,
        semantic_axes: Mapping[str, Any],
        device_spec: str,
        dtype_spec: str,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = resolve_torch_device(device_spec)
        self.dtype = resolve_torch_dtype(self.device, dtype_spec)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).eval().to(self.device)
        self.model_id = model_id

        self.axis_specs: Dict[str, Dict[str, Any]] = {}
        self.prompts: List[str] = []
        self.prompt_mapping: List[Tuple[str, str]] = []

        for axis_name, axis in semantic_axes.items():
            classes = axis["classes"]
            self.axis_specs[axis_name] = {
                "exclusive": bool(axis.get("exclusive", True)),
                "classes": list(classes.keys()),
            }
            for class_name, class_prompts in classes.items():
                for prompt in class_prompts:
                    text = prompt.strip()
                    if not text.lower().startswith("this is a photo"):
                        text = f"This is a photo of {text}."
                    self.prompts.append(text)
                    self.prompt_mapping.append((axis_name, class_name))

        if not self.prompts:
            raise BackendError("No SigLIP2 prompts were configured")

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values - np.max(values)
        exp = np.exp(values)
        return exp / np.sum(exp)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-values))

    def score(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        inputs = self.processor(
            text=self.prompts,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = move_batch(inputs, self.device, self.dtype)

        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0].float().cpu().numpy()

        grouped_logits: Dict[str, Dict[str, List[float]]] = {}
        for logit, (axis_name, class_name) in zip(logits, self.prompt_mapping):
            grouped_logits.setdefault(axis_name, {}).setdefault(class_name, []).append(float(logit))

        result: Dict[str, Dict[str, float]] = {}
        for axis_name, spec in self.axis_specs.items():
            classes = spec["classes"]
            class_logits = np.asarray(
                [np.mean(grouped_logits[axis_name][class_name]) for class_name in classes],
                dtype=np.float64,
            )
            if spec["exclusive"]:
                probabilities = self._softmax(class_logits)
            else:
                probabilities = self._sigmoid(class_logits)
            result[axis_name] = {
                class_name: float(probability)
                for class_name, probability in zip(classes, probabilities)
            }
        return result


@dataclass
class SegmentationOutput:
    class_map: np.ndarray
    confidence_map: np.ndarray
    group_masks: Dict[str, np.ndarray]
    group_ratios: Dict[str, float]
    valid_pixel_ratio: float
    mean_confidence: float


class SegFormerSceneParser:
    def __init__(
        self,
        model_id: str,
        segmentation_groups: Mapping[str, Sequence[str]],
        min_pixel_confidence: float,
        device_spec: str,
        dtype_spec: str,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.torch = torch
        self.device = resolve_torch_device(device_spec)
        self.dtype = resolve_torch_dtype(self.device, dtype_spec)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(model_id, torch_dtype=self.dtype)
            .eval()
            .to(self.device)
        )
        self.model_id = model_id
        self.min_pixel_confidence = float(min_pixel_confidence)
        self.group_keywords = {
            group: tuple(self._normalize(keyword) for keyword in keywords)
            for group, keywords in segmentation_groups.items()
        }
        self.id2label = {
            int(key): str(value)
            for key, value in self.model.config.id2label.items()
        }
        self.class_to_group = {
            class_id: self._label_to_group(label)
            for class_id, label in self.id2label.items()
        }

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace(",", " ")
            .strip()
        )

    def _label_to_group(self, label: str) -> str:
        normalized = self._normalize(label)
        for group, keywords in self.group_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return group
        return "other"

    def parse(self, image: Image.Image) -> SegmentationOutput:
        import torch.nn.functional as F

        batch = self.processor(images=image, return_tensors="pt")
        batch = move_batch(batch, self.device, self.dtype)
        with self.torch.inference_mode():
            logits = self.model(**batch).logits
            logits = F.interpolate(
                logits.float(),
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = self.torch.softmax(logits, dim=1)
            confidence, class_map = probabilities.max(dim=1)

        class_map_np = class_map[0].cpu().numpy().astype(np.int16)
        confidence_np = confidence[0].cpu().numpy().astype(np.float32)
        valid = confidence_np >= self.min_pixel_confidence

        group_masks: Dict[str, np.ndarray] = {}
        for class_id in np.unique(class_map_np[valid]):
            group = self.class_to_group.get(int(class_id), "other")
            if group == "other":
                continue
            mask = valid & (class_map_np == class_id)
            if group in group_masks:
                group_masks[group] |= mask
            else:
                group_masks[group] = mask.copy()

        pixel_count = class_map_np.size
        group_ratios = {
            group: float(np.sum(mask)) / float(pixel_count)
            for group, mask in group_masks.items()
        }
        for group in self.group_keywords:
            group_masks.setdefault(group, np.zeros_like(valid, dtype=bool))
            group_ratios.setdefault(group, 0.0)

        return SegmentationOutput(
            class_map=class_map_np,
            confidence_map=confidence_np,
            group_masks=group_masks,
            group_ratios=group_ratios,
            valid_pixel_ratio=float(np.mean(valid)),
            mean_confidence=float(np.mean(confidence_np)),
        )


class OpenVocabularyDetector:
    def __init__(
        self,
        model_id: str,
        grouped_labels: Mapping[str, Sequence[str]],
        box_threshold: float,
        text_threshold: float,
        device_spec: str,
        dtype_spec: str,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = resolve_torch_device(device_spec)
        self.dtype = resolve_torch_dtype(self.device, dtype_spec)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id, torch_dtype=self.dtype)
            .eval()
            .to(self.device)
        )
        self.model_id = model_id
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.grouped_labels = {group: list(labels) for group, labels in grouped_labels.items()}
        self.flat_labels = [label for labels in self.grouped_labels.values() for label in labels]
        self.label_to_group = {
            label.lower(): group
            for group, labels in self.grouped_labels.items()
            for label in labels
        }

    def detect(self, image: Image.Image) -> Dict[str, Any]:
        text_labels = [self.flat_labels]
        batch = self.processor(images=image, text=text_labels, return_tensors="pt")
        batch = move_batch(batch, self.device, self.dtype)

        with self.torch.inference_mode():
            outputs = self.model(**batch)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            batch.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]

        labels = results.get("text_labels") or results.get("labels") or []
        boxes = results.get("boxes")
        scores = results.get("scores")

        grouped: Dict[str, Dict[str, Any]] = {
            group: {
                "present": False,
                "confidence": 0.0,
                "box_union_ratio": 0.0,
                "mask_area_ratio": None,
                "types": [],
                "boxes": [],
                "scores": [],
                "detection_source": "grounding_dino",
            }
            for group in self.grouped_labels
        }

        if boxes is None or scores is None:
            return grouped

        boxes_np = boxes.detach().float().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy()
        union_masks = {
            group: np.zeros((image.height, image.width), dtype=bool)
            for group in self.grouped_labels
        }

        for box, score, label_value in zip(boxes_np, scores_np, labels):
            label = str(label_value).lower().strip()
            group = self.label_to_group.get(label)
            if group is None:
                group = next(
                    (
                        candidate_group
                        for candidate_label, candidate_group in self.label_to_group.items()
                        if candidate_label in label or label in candidate_label
                    ),
                    None,
                )
            if group is None:
                continue

            x0, y0, x1, y1 = [float(value) for value in box]
            x0i = max(0, min(image.width, int(math.floor(x0))))
            y0i = max(0, min(image.height, int(math.floor(y0))))
            x1i = max(0, min(image.width, int(math.ceil(x1))))
            y1i = max(0, min(image.height, int(math.ceil(y1))))
            if x1i <= x0i or y1i <= y0i:
                continue

            union_masks[group][y0i:y1i, x0i:x1i] = True
            item = grouped[group]
            item["present"] = True
            item["confidence"] = max(float(item["confidence"]), float(score))
            item["types"].append(label)
            item["boxes"].append([x0, y0, x1, y1])
            item["scores"].append(float(score))

        for group, item in grouped.items():
            item["box_union_ratio"] = float(np.mean(union_masks[group]))
            item["types"] = sorted(set(item["types"]))
        return grouped


class SAM2BoxRefiner:
    def __init__(self, model_id: str, device_spec: str, dtype_spec: str) -> None:
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.torch = torch
        self.device = resolve_torch_device(device_spec)
        self.dtype = resolve_torch_dtype(self.device, dtype_spec)
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id, torch_dtype=self.dtype).eval().to(self.device)
        self.model_id = model_id

    def refine(self, image: Image.Image, grouped_detections: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        ordered: List[Tuple[str, List[float]]] = []
        for group, item in grouped_detections.items():
            for box in item.get("boxes", []):
                ordered.append((group, box))

        if not ordered:
            return grouped_detections, {}

        input_boxes = [[box for _, box in ordered]]
        batch = self.processor(images=image, input_boxes=input_boxes, return_tensors="pt")
        batch = move_batch(batch, self.device, self.dtype)

        with self.torch.inference_mode():
            outputs = self.model(**batch, multimask_output=False)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            batch["original_sizes"].cpu() if hasattr(batch["original_sizes"], "cpu") else batch["original_sizes"],
        )[0]
        masks_np = masks.numpy()
        while masks_np.ndim > 3:
            masks_np = np.squeeze(masks_np, axis=1)
        if masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        union_masks: Dict[str, np.ndarray] = {}
        for (group, _), mask in zip(ordered, masks_np):
            binary = np.asarray(mask > 0, dtype=bool)
            if group in union_masks:
                union_masks[group] |= binary
            else:
                union_masks[group] = binary.copy()

        for group, mask in union_masks.items():
            grouped_detections[group]["mask_area_ratio"] = float(np.mean(mask))
            grouped_detections[group]["detection_source"] = "grounding_dino+sam2"
        return grouped_detections, union_masks
