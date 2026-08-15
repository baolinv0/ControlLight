import numpy as np


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float64)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        rgb /= np.iinfo(np.asarray(image).dtype).max
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_log_luminance(image: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    linear = srgb_to_linear(image)
    luminance = linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722
    return np.log2(luminance + epsilon)


def chromaticity_descriptor(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    rgb = np.asarray(image, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    sums = rgb.sum(axis=1)
    valid = sums > 0
    if not np.any(valid):
        return (0.0, 0.0)
    normalized = rgb[valid, :2] / sums[valid, None]
    median = np.median(normalized, axis=0)
    return (float(median[0]), float(median[1]))
