import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .models import RegionMasks, SamplePaths, SelectionResult
from .visualization import render_mask_overlay, render_selection_visualization


def _write_mask(path: Path, mask: np.ndarray | None, shape: tuple[int, int]) -> None:
    values = np.zeros(shape, np.uint8) if mask is None else np.asarray(mask, dtype=np.uint8) * 255
    Image.fromarray(values).save(path)


def _write_masks(output_dir: Path, sample: SamplePaths, result: SelectionResult, save_overlay: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sample.source) as source:
        source_shape = (source.height, source.width)
    with Image.open(sample.reference) as reference:
        reference_shape = (reference.height, reference.width)
    source_masks = result.source_masks
    reference_masks = result.reference_masks
    if source_masks is None or reference_masks is None:
        return
    for prefix, masks, shape in (
        ("source", source_masks, source_shape),
        ("ref", reference_masks, reference_shape),
    ):
        _write_mask(output_dir / f"{prefix}_person.png", masks.person, shape)
        _write_mask(output_dir / f"{prefix}_face.png", masks.face, shape)
        _write_mask(output_dir / f"{prefix}_local_bg.png", masks.local_background, shape)
    if save_overlay:
        render_mask_overlay(sample.source, source_masks, output_dir / "source_overlay.jpg")
        render_mask_overlay(sample.reference, reference_masks, output_dir / "ref_overlay.jpg")


def write_debug_overlays(result: SelectionResult, sample: SamplePaths, output_root: str | Path) -> None:
    if result.source_masks is None or result.reference_masks is None:
        return
    output_dir = Path(output_root) / "masks" / sample.sample_id
    render_mask_overlay(sample.source, result.source_masks, output_dir / "source_overlay.jpg")
    render_mask_overlay(sample.reference, result.reference_masks, output_dir / "ref_overlay.jpg")


def write_sample_artifacts(
    result: SelectionResult,
    sample: SamplePaths,
    output_root: str | Path,
    *,
    save_overlay: bool = False,
) -> dict[str, Path]:
    root = Path(output_root)
    score_path = root / "scores" / f"{sample.sample_id}.json"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {"score": score_path}
    if result.best_level is None or result.selected_path is None:
        return paths

    pseudo_path = root / "pseudo_gt" / f"{sample.sample_id}.png"
    pseudo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result.selected_path, pseudo_path)
    paths["pseudo_gt"] = pseudo_path

    visualization_path = root / "visualization" / f"{sample.sample_id}.jpg"
    render_selection_visualization(sample.source, sample.reference, sample.ladder, result, visualization_path)
    paths["visualization"] = visualization_path
    _write_masks(root / "masks" / sample.sample_id, sample, result, save_overlay)
    return paths
