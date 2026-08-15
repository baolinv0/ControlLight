import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from .artifacts import write_debug_overlays, write_sample_artifacts
from .config import SelectorConfig
from .dataset import discover_samples
from .mask_provider import FileMaskProvider
from .models import SamplePaths, SelectionResult
from .selector import PseudoGTSelector
from .statistics import write_dataset_summaries


@dataclass(frozen=True)
class BatchRunReport:
    summary_results: list[SelectionResult]
    overlay_sample_ids: tuple[str, ...]
    max_mask_results_retained: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select person-tone-matched aligned pseudo-GT ladder levels")
    parser.add_argument("--input-root", required=True, help="Sample directory or directory containing sample folders")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument(
        "--mask-dir-name",
        default="provided_masks",
        help="Per-sample directory containing source/reference person and optional face masks",
    )
    parser.add_argument("--debug-overlay-limit", type=int, default=100)
    parser.add_argument("--debug-overlay-seed", type=int, default=0)
    return parser


def run_batch(
    input_root: str | Path,
    output_root: str | Path,
    mask_dir_name: str,
    config: SelectorConfig,
) -> BatchRunReport:
    output_root = Path(output_root)
    summary_results: list[SelectionResult] = []
    overlay_reservoir: list[tuple[SamplePaths, SelectionResult]] = []
    max_retained = 0
    valid_seen = 0
    rng = random.Random(config.debug_overlay_seed)

    for sample in discover_samples(input_root):
        provider = FileMaskProvider(sample.root / mask_dir_name, config)
        result = PseudoGTSelector(provider, config).select_paths(sample)
        write_sample_artifacts(result, sample, output_root, save_overlay=False)
        summary_results.append(result.without_masks())

        if result.best_level is not None:
            valid_seen += 1
            if len(overlay_reservoir) < config.debug_overlay_limit:
                overlay_reservoir.append((sample, result))
            elif config.debug_overlay_limit:
                replacement = rng.randrange(valid_seen)
                if replacement < config.debug_overlay_limit:
                    overlay_reservoir[replacement] = (sample, result)
            max_retained = max(max_retained, len(overlay_reservoir))

    for sample, result in overlay_reservoir:
        write_debug_overlays(result, sample, output_root)
    write_dataset_summaries(summary_results, output_root)
    return BatchRunReport(
        summary_results=summary_results,
        overlay_sample_ids=tuple(sample.sample_id for sample, _ in overlay_reservoir),
        max_mask_results_retained=max_retained,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SelectorConfig(
        debug_overlay_limit=args.debug_overlay_limit,
        debug_overlay_seed=args.debug_overlay_seed,
    )
    run_batch(args.input_root, args.output_root, args.mask_dir_name, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
