from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
from PIL import Image, ImageOps


EPS = 1e-6


def load_rgb_image(path: Path) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """Load an image and return PIL RGB, normalized sRGB, and linear RGB arrays."""
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        srgb = np.asarray(image, dtype=np.float32) / 255.0
    linear = srgb_to_linear(srgb)
    return image, srgb, linear


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * np.power(x, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * linear_rgb[..., 0]
        + 0.7152 * linear_rgb[..., 1]
        + 0.0722 * linear_rgb[..., 2]
    ).astype(np.float32)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def robust_stats(values: Iterable[float]) -> Dict[str, float | int | None]:
    arr = _finite_array(values)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def trimmed_mean(values: np.ndarray, low_percentile: float = 10.0, high_percentile: float = 90.0) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    low = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)
    selected = values[(values >= low) & (values <= high)]
    if selected.size == 0:
        return float(np.mean(values))
    return float(np.mean(selected))


def chromaticity_from_pixels(linear_rgb_pixels: np.ndarray) -> tuple[float, float]:
    pixels = np.asarray(linear_rgb_pixels, dtype=np.float64).reshape(-1, 3)
    if pixels.size == 0:
        return math.nan, math.nan
    means = np.mean(pixels, axis=0)
    denom = float(np.sum(means)) + EPS
    return float(means[0] / denom), float(means[1] / denom)


def _region_values(array: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return array.reshape(-1, *array.shape[2:]) if array.ndim > 2 else array.reshape(-1)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != array.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {array.shape[:2]}")
    return array[mask]


def compute_region_photometry(
    linear_rgb: np.ndarray,
    mask: Optional[np.ndarray] = None,
    deep_shadow_threshold: float = 0.01,
    clip_threshold: float = 0.99,
) -> Dict[str, float | int | None]:
    y = luminance(linear_rgb)
    y_values = _region_values(y, mask).astype(np.float64)
    rgb_values = _region_values(linear_rgb, mask).astype(np.float64)

    if y_values.size == 0:
        return {
            "pixels": 0,
            "mean_luminance": None,
            "mid_luminance": None,
            "mean_brightness_ev": None,
            "mid_brightness_ev": None,
            "p01_luminance": None,
            "p05_luminance": None,
            "p50_luminance": None,
            "p95_luminance": None,
            "p99_luminance": None,
            "dr_p95_p05_ev": None,
            "dr_p99_p01_ev": None,
            "deep_shadow_ratio": None,
            "dark_ratio": None,
            "highlight_ratio": None,
            "clip_ratio": None,
            "log_luminance_std": None,
            "chroma_r": None,
            "chroma_g": None,
        }

    p01, p05, p50, p95, p99 = np.percentile(y_values, [1, 5, 50, 95, 99])
    mean_y = float(np.mean(y_values))
    mid_y = trimmed_mean(y_values, 10.0, 90.0)
    log_y = np.log2(y_values + EPS)
    chroma_r, chroma_g = chromaticity_from_pixels(rgb_values)

    return {
        "pixels": int(y_values.size),
        "mean_luminance": mean_y,
        "mid_luminance": mid_y,
        "mean_brightness_ev": float(np.log2(mean_y + EPS)),
        "mid_brightness_ev": float(np.log2(mid_y + EPS)),
        "p01_luminance": float(p01),
        "p05_luminance": float(p05),
        "p50_luminance": float(p50),
        "p95_luminance": float(p95),
        "p99_luminance": float(p99),
        "dr_p95_p05_ev": float(np.log2(p95 + EPS) - np.log2(p05 + EPS)),
        "dr_p99_p01_ev": float(np.log2(p99 + EPS) - np.log2(p01 + EPS)),
        "deep_shadow_ratio": float(np.mean(y_values <= deep_shadow_threshold)),
        "dark_ratio": float(np.mean(y_values <= 0.02)),
        "highlight_ratio": float(np.mean(y_values >= 0.90)),
        "clip_ratio": float(np.mean(y_values >= clip_threshold)),
        "log_luminance_std": float(np.std(log_y)),
        "chroma_r": chroma_r,
        "chroma_g": chroma_g,
    }


def compute_global_photometry(
    linear_rgb: np.ndarray,
    group_masks: Mapping[str, np.ndarray] | None,
    deep_shadow_threshold: float,
    clip_threshold: float,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, float | int | None]]]:
    global_features = compute_region_photometry(
        linear_rgb,
        mask=None,
        deep_shadow_threshold=deep_shadow_threshold,
        clip_threshold=clip_threshold,
    )

    regional: Dict[str, Dict[str, float | int | None]] = {}
    if group_masks:
        for group, mask in group_masks.items():
            regional[group] = compute_region_photometry(
                linear_rgb,
                mask=mask,
                deep_shadow_threshold=deep_shadow_threshold,
                clip_threshold=clip_threshold,
            )

    sky_mask = group_masks.get("sky") if group_masks else None
    if sky_mask is not None and np.any(sky_mask):
        non_sky_mask = ~sky_mask
        non_sky = compute_region_photometry(
            linear_rgb,
            mask=non_sky_mask,
            deep_shadow_threshold=deep_shadow_threshold,
            clip_threshold=clip_threshold,
        )
    else:
        non_sky = dict(global_features)

    global_features["non_sky_mid_luminance"] = non_sky.get("mid_luminance")
    global_features["non_sky_mid_brightness_ev"] = non_sky.get("mid_brightness_ev")
    global_features["non_sky_shadow_ratio"] = non_sky.get("deep_shadow_ratio")
    global_features["non_sky_clip_ratio"] = non_sky.get("clip_ratio")

    person = regional.get("person", {})
    global_features["person_mid_luminance"] = person.get("mid_luminance")
    global_features["person_mid_brightness_ev"] = person.get("mid_brightness_ev")

    region_mid_values = [
        float(values["mid_brightness_ev"])
        for values in regional.values()
        if values.get("mid_brightness_ev") is not None
        and int(values.get("pixels") or 0) >= 100
    ]
    if region_mid_values:
        global_features["semantic_region_brightness_span_ev"] = float(
            max(region_mid_values) - min(region_mid_values)
        )
    else:
        global_features["semantic_region_brightness_span_ev"] = None

    # This is explicitly a display-domain difficulty proxy, not sensor dynamic range.
    dr_base = float(global_features["dr_p95_p05_ev"])
    region_span = global_features["semantic_region_brightness_span_ev"]
    dual_tail = min(
        float(global_features["deep_shadow_ratio"]),
        float(global_features["clip_ratio"]),
    )
    global_features["display_dr_proxy_ev"] = float(
        dr_base + 0.35 * (float(region_span) if region_span is not None else 0.0) + 2.0 * dual_tail
    )
    global_features["dual_tail_score"] = float(
        math.sqrt(
            max(float(global_features["deep_shadow_ratio"]), 0.0)
            * max(float(global_features["clip_ratio"]), 0.0)
        )
    )

    return global_features, regional


def _rgb_saturation(linear_rgb: np.ndarray) -> np.ndarray:
    max_c = np.max(linear_rgb, axis=-1)
    min_c = np.min(linear_rgb, axis=-1)
    return (max_c - min_c) / (max_c + EPS)


def _kmeans_1d_two_clusters(values: np.ndarray, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        return np.asarray([float(np.mean(values))]), np.zeros(values.size, dtype=np.int64)

    centers = np.asarray([np.percentile(values, 25), np.percentile(values, 75)], dtype=np.float64)
    labels = np.zeros(values.size, dtype=np.int64)

    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for index in range(2):
            selected = values[new_labels == index]
            if selected.size:
                new_centers[index] = float(np.mean(selected))
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    remap = np.zeros(2, dtype=np.int64)
    remap[order] = np.arange(2)
    return centers[order], remap[labels]


def estimate_apparent_illumination(
    linear_rgb: np.ndarray,
    exclusion_mask: Optional[np.ndarray],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Estimate apparent rendered illumination color from low-saturation patches.

    This is not a physical illuminant CCT estimator. It operates on rendered sRGB
    after inverse transfer-function conversion and should be interpreted as an
    apparent warm/neutral/cool/mixed appearance signal.
    """
    height, width = linear_rgb.shape[:2]
    y = luminance(linear_rgb)
    sat = _rgb_saturation(linear_rgb)

    valid = (
        (sat <= float(config["max_neutral_saturation"]))
        & (y >= float(config["min_luminance"]))
        & (y <= float(config["max_luminance"]))
    )
    if exclusion_mask is not None:
        valid &= ~np.asarray(exclusion_mask, dtype=bool)

    rows = int(config["grid_rows"])
    cols = int(config["grid_cols"])
    min_ratio = float(config["min_patch_valid_ratio"])

    patch_values: list[float] = []
    patch_weights: list[float] = []
    patch_locations: list[tuple[int, int]] = []

    for row in range(rows):
        y0 = int(round(row * height / rows))
        y1 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x0 = int(round(col * width / cols))
            x1 = int(round((col + 1) * width / cols))
            patch_valid = valid[y0:y1, x0:x1]
            if patch_valid.size == 0:
                continue
            ratio = float(np.mean(patch_valid))
            if ratio < min_ratio:
                continue
            pixels = linear_rgb[y0:y1, x0:x1][patch_valid]
            means = np.mean(pixels, axis=0)
            value = float(np.log2((float(means[2]) + EPS) / (float(means[0]) + EPS)))
            patch_values.append(value)
            patch_weights.append(ratio)
            patch_locations.append((row, col))

    min_patches = int(config["min_valid_patches"])
    if len(patch_values) < min_patches:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "reason": "insufficient_neutral_patches",
            "valid_patch_count": len(patch_values),
            "patch_values_log2_br": patch_values,
            "physical_cct_estimated": False,
        }

    values = np.asarray(patch_values, dtype=np.float64)
    weights = np.asarray(patch_weights, dtype=np.float64)
    centers, labels = _kmeans_1d_two_clusters(values)

    if centers.size == 1:
        shares = np.asarray([1.0])
        separation = 0.0
    else:
        shares = np.asarray([
            float(np.sum(weights[labels == index])) / float(np.sum(weights))
            for index in range(2)
        ])
        separation = float(abs(centers[1] - centers[0]))

    min_share = float(config["mixed_min_cluster_share"])
    min_separation = float(config["mixed_min_separation_log2_br"])
    mixed = (
        centers.size == 2
        and separation >= min_separation
        and float(np.min(shares)) >= min_share
    )

    if mixed:
        confidence = min(1.0, 0.5 + 0.5 * separation / max(min_separation, EPS))
        label = "mixed"
    else:
        weighted_mean = float(np.average(values, weights=weights))
        boundary = float(config["warm_cool_boundary_log2_br"])
        if weighted_mean < -boundary:
            label = "warm"
        elif weighted_mean > boundary:
            label = "cool"
        else:
            label = "neutral"
        distance = abs(weighted_mean) if label != "neutral" else max(0.0, boundary - abs(weighted_mean))
        confidence = float(min(1.0, 0.55 + 0.45 * distance / max(boundary, EPS)))

    return {
        "label": label,
        "confidence": float(confidence),
        "valid_patch_count": len(patch_values),
        "patch_values_log2_br": [float(value) for value in values],
        "patch_locations": [list(location) for location in patch_locations],
        "cluster_centers_log2_br": [float(value) for value in centers],
        "cluster_shares": [float(value) for value in shares],
        "cluster_separation_log2_br": separation,
        "physical_cct_estimated": False,
        "interpretation": "apparent_rendered_illumination_color",
    }
