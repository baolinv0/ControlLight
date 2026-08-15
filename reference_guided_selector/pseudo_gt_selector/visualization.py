from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .config import LEVEL_COEFFICIENTS
from .models import RegionMasks, SelectionResult


CELL_WIDTH = 220
CELL_HEIGHT = 190
IMAGE_HEIGHT = 145


@dataclass(frozen=True)
class RenderCell:
    kind: str
    image_key: str
    row: int
    column: int
    lines: tuple[str, ...]


def build_selection_render_model(result: SelectionResult) -> list[RenderCell]:
    cells = [
        RenderCell("source", "source", 0, 0, ("Source",)),
        RenderCell("reference", "reference", 0, 1, ("Reference", "person-centered tone target")),
        RenderCell(
            "best",
            "best",
            0,
            2,
            ("Aligned Pseudo-GT", result.best_level or "", f"Total={result.best_score:.4f}"),
        ),
    ]
    ranks = {
        level: rank
        for rank, level in enumerate(
            sorted(LEVEL_COEFFICIENTS, key=lambda name: result.levels[name].score), start=1
        )
    }
    for index, level in enumerate(LEVEL_COEFFICIENTS):
        score = result.levels[level]
        face = "NA" if score.face_error is None else f"{score.face_error:.3f}"
        background = "NA" if score.background_error is None else f"{score.background_error:.3f}"
        relation = "NA" if score.person_background_error is None else f"{score.person_background_error:.3f}"
        title = level + (" * BEST" if level == result.best_level else "")
        row, column = divmod(index, 5)
        cells.append(
            RenderCell(
                "candidate",
                level,
                row + 1,
                column,
                (
                    f"{title}  Rank #{ranks[level]}  Total={score.score:.3f}",
                    f"P={score.person_error:.3f} PB={relation}",
                    f"F={face} B={background}",
                ),
            )
        )
    return cells


def _thumbnail(image: Image.Image) -> Image.Image:
    result = image.convert("RGB").copy()
    result.thumbnail((CELL_WIDTH - 12, IMAGE_HEIGHT))
    canvas = Image.new("RGB", (CELL_WIDTH, IMAGE_HEIGHT), "white")
    canvas.paste(result, ((CELL_WIDTH - result.width) // 2, (IMAGE_HEIGHT - result.height) // 2))
    return canvas


def _cell(image: Image.Image, lines: list[str]) -> Image.Image:
    canvas = Image.new("RGB", (CELL_WIDTH, CELL_HEIGHT), "white")
    canvas.paste(_thumbnail(image), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines[:3]):
        draw.text((5, IMAGE_HEIGHT + 3 + index * 13), line, fill="black")
    return canvas


def render_selection_visualization(
    source_path: Path,
    reference_path: Path,
    ladder_paths: dict[str, Path],
    result: SelectionResult,
    output_path: Path,
) -> None:
    canvas = Image.new("RGB", (CELL_WIDTH * 5, CELL_HEIGHT * 3), (235, 235, 235))
    image_paths = {
        "source": source_path,
        "reference": reference_path,
        "best": ladder_paths[result.best_level],
        **ladder_paths,
    }
    for item in build_selection_render_model(result):
        with Image.open(image_paths[item.image_key]) as image:
            cell = _cell(image, list(item.lines))
        canvas.paste(cell, (item.column * CELL_WIDTH, item.row * CELL_HEIGHT))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _contour(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, 1)
    interior = mask.copy()
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        interior &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return mask & ~interior


def render_mask_overlay(image_path: Path, masks: RegionMasks, output_path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    if masks.local_background is not None:
        bg = masks.local_background
        image[bg] = 0.65 * image[bg] + 0.35 * np.asarray([0, 90, 255])
    image[_contour(masks.person)] = (255, 0, 0)
    if masks.face is not None:
        image[_contour(masks.face)] = (0, 255, 0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(output_path, quality=92)
