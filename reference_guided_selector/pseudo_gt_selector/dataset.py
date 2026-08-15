from pathlib import Path

from PIL import Image

from .config import LEVEL_COEFFICIENTS
from .models import SamplePaths


def _sample_from_root(root: Path) -> SamplePaths:
    ladder_root = root / "ladder"
    ladder = {
        level: ladder_root / f"{level}.png"
        for level in LEVEL_COEFFICIENTS
        if (ladder_root / f"{level}.png").is_file()
    }
    missing = tuple(level for level in LEVEL_COEFFICIENTS if level not in ladder)
    return SamplePaths(root.name, root, root / "source.png", root / "reference.png", ladder, missing)


def discover_samples(input_root: str | Path) -> list[SamplePaths]:
    root = Path(input_root)
    if (root / "source.png").is_file():
        return [_sample_from_root(root)]
    return [_sample_from_root(path) for path in sorted(root.iterdir()) if path.is_dir() and (path / "source.png").is_file()]


def validate_sample(sample: SamplePaths) -> list[str]:
    errors: list[str] = []
    if not sample.source.is_file():
        errors.append("missing source.png")
    if not sample.reference.is_file():
        errors.append("missing reference.png")
    if sample.missing_levels:
        errors.append(f"missing ladder levels: {', '.join(sample.missing_levels)}")
    if errors or not sample.source.is_file():
        return errors
    with Image.open(sample.source) as image:
        source_size = image.size
    for level in LEVEL_COEFFICIENTS:
        path = sample.ladder.get(level)
        if path is None:
            continue
        with Image.open(path) as image:
            size = image.size
        if size != source_size:
            errors.append(f"ladder {level} size {size} differs from source {source_size}")
    return errors
