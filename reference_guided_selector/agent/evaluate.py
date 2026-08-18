import argparse
import csv
import json
from pathlib import Path

from pseudo_gt_selector.config import LEVEL_COEFFICIENTS


LEVELS = tuple(LEVEL_COEFFICIENTS)
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _goal_thresholds(goal: dict) -> dict[str, float]:
    values = goal.get("primary_metrics", goal)
    return {
        "top1_accuracy": float(values["top1_accuracy"]),
        "top1_pm1_accuracy": float(values["top1_pm1_accuracy"]),
    }


def evaluate_selection_summary(
    selection_summary: str | Path,
    labels_path: str | Path,
    goal: dict,
) -> tuple[dict, list[dict]]:
    labels = _read_rows(labels_path)
    predictions = _read_rows(selection_summary)
    if not labels:
        raise ValueError("labels file is empty; real-data acceptance cannot be evaluated")

    prediction_by_id = {row["sample_id"]: row for row in predictions}
    thresholds = _goal_thresholds(goal)

    exact = 0
    within_one = 0
    predicted = 0
    failures: list[dict] = []

    for label in labels:
        sample_id = label["sample_id"]
        target = label["target_level"]
        if target not in LEVEL_INDEX:
            raise ValueError(f"unknown target_level for {sample_id}: {target}")

        row = prediction_by_id.get(sample_id)
        selected = "" if row is None else row.get("best_level", "")
        status = "MISSING" if row is None else row.get("status", "")

        if selected in LEVEL_INDEX:
            predicted += 1
            delta = LEVEL_INDEX[selected] - LEVEL_INDEX[target]
            is_exact = delta == 0
            is_within_one = abs(delta) <= 1
            exact += int(is_exact)
            within_one += int(is_within_one)
            direction = "correct" if is_exact else ("too_dark" if delta < 0 else "too_bright")
        else:
            delta = None
            is_exact = False
            is_within_one = False
            direction = "missing_prediction"

        if not is_exact:
            failures.append(
                {
                    "sample_id": sample_id,
                    "target_level": target,
                    "selected_level": selected or None,
                    "status": status,
                    "delta_levels": delta,
                    "direction": direction,
                    "within_one": is_within_one,
                    "failure_category": label.get("failure_category", ""),
                }
            )

    denominator = len(labels)
    top1 = exact / denominator
    top1_pm1 = within_one / denominator
    coverage = predicted / denominator
    pass_top1 = top1 >= thresholds["top1_accuracy"]
    pass_top1_pm1 = top1_pm1 >= thresholds["top1_pm1_accuracy"]

    metrics = {
        "num_labeled": denominator,
        "num_predicted": predicted,
        "coverage": coverage,
        "top1_accuracy": top1,
        "top1_pm1_accuracy": top1_pm1,
        "thresholds": thresholds,
        "pass_top1": pass_top1,
        "pass_top1_pm1": pass_top1_pm1,
        "goal_reached": bool(pass_top1 and pass_top1_pm1),
        "num_top1_failures": len(failures),
    }
    return metrics, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate pseudo-GT level selection against frozen human labels")
    parser.add_argument("--selection-summary", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--failures-out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    goal = json.loads(Path(args.goal).read_text(encoding="utf-8"))
    metrics, failures = evaluate_selection_summary(args.selection_summary, args.labels, goal)

    metrics_out = Path(args.metrics_out)
    failures_out = Path(args.failures_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    failures_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    failures_out.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
