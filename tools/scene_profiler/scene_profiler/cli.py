from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Tuple

from .config import apply_cli_overrides, load_config
from .pipeline import SceneProfilerPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile scene semantics and rendered photometric attributes for "
            "ControlLight-style datasets."
        )
    )
    parser.add_argument("--input", type=Path, default=None, help="Image directory or dataset root containing a_000.")
    parser.add_argument(
        "--relabel-from",
        type=Path,
        default=None,
        help="Recompute labels from an existing raw_features.jsonl without loading neural models.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML overriding default_config.yaml.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--reference-level", default=None, help="Reference folder when --input is a dataset root.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit images for debugging; 0 means all.")
    parser.add_argument("--save-masks", action="store_true", help="Save segmentation masks as compressed NPZ files.")
    parser.add_argument("--save-overlays", type=int, default=None, help="Number of mask overlays to save.")
    parser.add_argument("--disable-siglip2", action="store_true")
    parser.add_argument("--disable-segformer", action="store_true")
    parser.add_argument("--enable-open-vocab", action="store_true")
    parser.add_argument("--enable-sam2", action="store_true")
    parser.add_argument(
        "--dataset-relative-thresholds",
        action="store_true",
        help=(
            "Use dataset quantiles for brightness and DR bins. This is useful for exploratory "
            "stratification but should not be used to claim absolute coverage."
        ),
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    default_config = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    config = load_config(default_config, args.config)

    overrides: List[Tuple[str, Any]] = [
        ("runtime.device", args.device),
        ("runtime.dtype", args.dtype),
        ("input.reference_level", args.reference_level),
        ("runtime.save_overlays", args.save_overlays),
    ]
    if args.save_masks:
        overrides.append(("runtime.save_masks", True))
    if args.disable_siglip2:
        overrides.append(("models.siglip2.enabled", False))
    if args.disable_segformer:
        overrides.append(("models.segformer.enabled", False))
    if args.enable_open_vocab:
        overrides.append(("models.open_vocab.enabled", True))
    if args.enable_sam2:
        overrides.extend([
            ("models.open_vocab.enabled", True),
            ("models.sam2.enabled", True),
        ])
    if args.dataset_relative_thresholds:
        overrides.extend([
            ("classification.brightness.calibration_mode", "dataset_quantile"),
            ("classification.dynamic_range_proxy.calibration_mode", "dataset_quantile"),
        ])

    config = apply_cli_overrides(config, overrides)
    if args.relabel_from is not None:
        pipeline = SceneProfilerPipeline(config, load_models=False)
        summary = pipeline.relabel_from_raw_features(args.relabel_from, args.output)
    else:
        if args.input is None:
            parser.error("--input is required unless --relabel-from is provided")
        pipeline = SceneProfilerPipeline(config, load_models=True)
        summary = pipeline.run(args.input, args.output, max_images=args.max_images)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
