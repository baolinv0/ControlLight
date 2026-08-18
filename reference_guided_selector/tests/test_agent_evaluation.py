import csv
import json
from pathlib import Path

from agent.evaluate import evaluate_selection_summary, main


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_evaluator_counts_missing_predictions_as_wrong_and_tracks_neighbor_accuracy(tmp_path):
    labels = tmp_path / "labels.csv"
    summary = tmp_path / "selection_summary.csv"
    goal = {"top1_accuracy": 0.80, "top1_pm1_accuracy": 0.95}

    _write_csv(
        labels,
        ["sample_id", "target_level", "failure_category"],
        [
            {"sample_id": "scene001", "target_level": "a_000", "failure_category": ""},
            {"sample_id": "scene002", "target_level": "a_p050", "failure_category": "face_overbright"},
            {"sample_id": "scene003", "target_level": "a_m050", "failure_category": ""},
        ],
    )
    _write_csv(
        summary,
        ["sample_id", "status", "best_level"],
        [
            {"sample_id": "scene001", "status": "ACCEPT", "best_level": "a_000"},
            {"sample_id": "scene002", "status": "ACCEPT", "best_level": "a_p025"},
        ],
    )

    metrics, failures = evaluate_selection_summary(summary, labels, goal)

    assert metrics["num_labeled"] == 3
    assert metrics["num_predicted"] == 2
    assert metrics["coverage"] == 2 / 3
    assert metrics["top1_accuracy"] == 1 / 3
    assert metrics["top1_pm1_accuracy"] == 2 / 3
    assert metrics["goal_reached"] is False
    assert [item["sample_id"] for item in failures] == ["scene002", "scene003"]
    assert failures[0]["direction"] == "too_dark"
    assert failures[0]["failure_category"] == "face_overbright"
    assert failures[1]["direction"] == "missing_prediction"


def test_evaluator_reaches_goal_only_when_all_configured_thresholds_pass(tmp_path):
    labels = tmp_path / "labels.csv"
    summary = tmp_path / "selection_summary.csv"

    _write_csv(
        labels,
        ["sample_id", "target_level"],
        [
            {"sample_id": "scene001", "target_level": "a_000"},
            {"sample_id": "scene002", "target_level": "a_p025"},
        ],
    )
    _write_csv(
        summary,
        ["sample_id", "status", "best_level"],
        [
            {"sample_id": "scene001", "status": "ACCEPT", "best_level": "a_000"},
            {"sample_id": "scene002", "status": "REVIEW", "best_level": "a_p025"},
        ],
    )

    metrics, failures = evaluate_selection_summary(
        summary,
        labels,
        {"top1_accuracy": 0.80, "top1_pm1_accuracy": 0.95},
    )

    assert metrics["top1_accuracy"] == 1.0
    assert metrics["top1_pm1_accuracy"] == 1.0
    assert metrics["goal_reached"] is True
    assert failures == []


def test_evaluator_cli_writes_machine_readable_metrics_and_failures(tmp_path):
    labels = tmp_path / "labels.csv"
    summary = tmp_path / "selection_summary.csv"
    goal = tmp_path / "goal.json"
    metrics_out = tmp_path / "evaluation" / "metrics.json"
    failures_out = tmp_path / "evaluation" / "failures.json"

    _write_csv(
        labels,
        ["sample_id", "target_level"],
        [{"sample_id": "scene001", "target_level": "a_000"}],
    )
    _write_csv(
        summary,
        ["sample_id", "status", "best_level"],
        [{"sample_id": "scene001", "status": "ACCEPT", "best_level": "a_000"}],
    )
    goal.write_text(
        json.dumps({"primary_metrics": {"top1_accuracy": 0.80, "top1_pm1_accuracy": 0.95}}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--selection-summary",
            str(summary),
            "--labels",
            str(labels),
            "--goal",
            str(goal),
            "--metrics-out",
            str(metrics_out),
            "--failures-out",
            str(failures_out),
        ]
    )

    assert exit_code == 0
    assert json.loads(metrics_out.read_text(encoding="utf-8"))["goal_reached"] is True
    assert json.loads(failures_out.read_text(encoding="utf-8")) == []
