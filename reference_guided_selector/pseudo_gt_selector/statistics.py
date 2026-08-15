import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .config import LEVEL_COEFFICIENTS
from .models import SelectionResult


CSV_FIELDS = [
    "sample_id",
    "status",
    "best_level",
    "best_coefficient",
    "best_score",
    "second_best_level",
    "second_best_score",
    "margin",
    "person_error",
    "person_background_error",
    "face_error",
    "background_error",
    "person_valid",
    "face_valid",
    "background_valid",
]


def _percentile(values: list[float], q: float) -> float | None:
    return None if not values else float(np.percentile(np.asarray(values), q))


def write_dataset_summaries(
    results: list[SelectionResult], output_root: str | Path
) -> tuple[Path, Path]:
    summary_dir = Path(output_root) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "selection_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            components = result.best_components
            validity = result.region_validity
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "status": result.status.value,
                    "best_level": result.best_level,
                    "best_coefficient": result.best_coefficient,
                    "best_score": result.best_score,
                    "second_best_level": result.second_best_level,
                    "second_best_score": result.second_best_score,
                    "margin": result.margin,
                    "person_error": components.get("person_error"),
                    "person_background_error": components.get("person_background_error"),
                    "face_error": components.get("face_error"),
                    "background_error": components.get("background_error"),
                    "person_valid": validity.get("person", False),
                    "face_valid": validity.get("face", False),
                    "background_valid": validity.get("local_background", False),
                }
            )

    review_path = summary_dir / "review_list.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "status", "best_score", "margin", "missing_levels", "message"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if result.status.value == "ACCEPT":
                continue
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "status": result.status.value,
                    "best_score": result.best_score,
                    "margin": result.margin,
                    "missing_levels": ",".join(result.missing_levels),
                    "message": result.message,
                }
            )

    valid = [result for result in results if result.best_level is not None]
    histogram = Counter(result.best_level for result in valid)
    best_scores = [float(result.best_score) for result in valid]
    margins = [float(result.margin) for result in valid]
    count = len(results)
    status_counts = Counter(result.status.value for result in results)
    payload = {
        "num_samples": count,
        "status_counts": dict(sorted(status_counts.items())),
        "level_histogram": {level: histogram.get(level, 0) for level in LEVEL_COEFFICIENTS},
        "best_score": {
            "median": _percentile(best_scores, 50),
            "p90": _percentile(best_scores, 90),
            "p95": _percentile(best_scores, 95),
        },
        "margin": {"median": _percentile(margins, 50), "p10": _percentile(margins, 10)},
        "region_availability": {
            "person_valid_rate": 0.0 if not count else sum(r.region_validity.get("person", False) for r in results) / count,
            "face_valid_rate": 0.0 if not count else sum(r.region_validity.get("face", False) for r in results) / count,
            "local_background_valid_rate": (
                0.0 if not count else sum(r.region_validity.get("local_background", False) for r in results) / count
            ),
        },
    }
    json_path = summary_dir / "dataset_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path
