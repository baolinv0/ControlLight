import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pseudo_gt_selector.artifacts import write_sample_artifacts
from pseudo_gt_selector.config import LEVEL_COEFFICIENTS, SelectorConfig
from pseudo_gt_selector.dataset import discover_samples
from pseudo_gt_selector.mask_provider import FileMaskProvider
from pseudo_gt_selector.run import main, run_batch
from pseudo_gt_selector.selector import PseudoGTSelector
from pseudo_gt_selector.statistics import write_dataset_summaries
from pseudo_gt_selector.models import RegionMasks
from pseudo_gt_selector.visualization import build_selection_render_model


def _save_rgb(path: Path, person_value: int, background_value: int):
    image = np.full((20, 20, 3), background_value, np.uint8)
    image[7:13, 7:13] = person_value
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _make_sample(root: Path, sample_id="scene001"):
    sample = root / sample_id
    _save_rgb(sample / "source.png", 100, 100)
    _save_rgb(sample / "reference.png", 180, 80)
    for index, level in enumerate(LEVEL_COEFFICIENTS):
        _save_rgb(sample / "ladder" / f"{level}.png", 40 + index * 20, 80)
    _save_rgb(sample / "ladder" / "a_p025.png", 180, 80)
    masks = sample / "provided_masks"
    masks.mkdir()
    person = np.zeros((20, 20), np.uint8)
    person[7:13, 7:13] = 255
    face = np.zeros((20, 20), np.uint8)
    face[7:10, 8:12] = 255
    for stem in ("source", "reference"):
        Image.fromarray(person).save(masks / f"{stem}_person.png")
        Image.fromarray(face).save(masks / f"{stem}_face.png")
    return sample


def _select(sample_root: Path):
    sample = discover_samples(sample_root)[0]
    selector = PseudoGTSelector(FileMaskProvider(sample.root / "provided_masks"))
    return sample, selector.select_paths(sample)


def test_artifacts_copy_selected_file_byte_for_byte_and_write_schema(tmp_path):
    sample_root = _make_sample(tmp_path / "input")
    sample, result = _select(sample_root)
    output = tmp_path / "output"

    paths = write_sample_artifacts(result, sample, output, save_overlay=True)

    assert paths["pseudo_gt"].read_bytes() == result.selected_path.read_bytes()
    payload = json.loads(paths["score"].read_text(encoding="utf-8"))
    assert payload["best_level"] == result.best_level
    assert set(payload["levels"]) == set(LEVEL_COEFFICIENTS)
    assert "person_error" in payload["best_components"]
    mask_dir = output / "masks" / sample.sample_id
    required = {
        "source_person.png", "source_face.png", "source_local_bg.png",
        "ref_person.png", "ref_face.png", "ref_local_bg.png",
    }
    assert required <= {path.name for path in mask_dir.iterdir()}
    assert (mask_dir / "source_overlay.jpg").is_file()
    assert (mask_dir / "ref_overlay.jpg").is_file()


def test_visualization_contains_three_rows_of_content(tmp_path):
    sample_root = _make_sample(tmp_path / "input")
    sample, result = _select(sample_root)
    path = write_sample_artifacts(result, sample, tmp_path / "output")["visualization"]
    with Image.open(path) as image:
        width, height = image.size
    assert width >= 1000
    assert height >= 500


def test_visualization_render_model_contains_required_semantic_labels(tmp_path):
    sample_root = _make_sample(tmp_path / "input")
    _, result = _select(sample_root)

    cells = build_selection_render_model(result)

    assert [cell.image_key for cell in cells[:3]] == ["source", "reference", "best"]
    assert cells[0].lines == ("Source",)
    assert cells[1].lines[0] == "Reference"
    candidates = [cell for cell in cells if cell.kind == "candidate"]
    assert [cell.image_key for cell in candidates] == list(LEVEL_COEFFICIENTS)
    for cell in candidates:
        text = " ".join(cell.lines)
        assert "Rank #" in text
        assert "Total=" in text
        assert "P=" in text and "PB=" in text and "F=" in text and "B=" in text
    best = next(cell for cell in candidates if cell.image_key == result.best_level)
    assert "* BEST" in " ".join(best.lines)


def test_dataset_summaries_have_required_fields_and_percentiles(tmp_path):
    first_root = _make_sample(tmp_path / "input", "scene001")
    second_root = _make_sample(tmp_path / "input", "scene002")
    results = [_select(first_root)[1], _select(second_root)[1]]

    csv_path, json_path = write_dataset_summaries(results, tmp_path / "output")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {
        "sample_id", "best_level", "best_coefficient", "best_score", "second_best_level",
        "second_best_score", "margin", "person_error", "person_background_error", "face_error",
        "background_error", "person_valid", "face_valid", "background_valid",
    } <= set(rows[0])
    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(summary["level_histogram"]) == set(LEVEL_COEFFICIENTS)
    assert set(summary["best_score"]) == {"median", "p90", "p95"}
    assert set(summary["margin"]) == {"median", "p10"}
    assert summary["region_availability"]["person_valid_rate"] == 1.0
    assert (tmp_path / "output" / "summary" / "review_list.csv").is_file()


def test_dataset_availability_rates_follow_effective_empty_mask_fallback(tmp_path):
    empty = np.zeros((20, 20), bool)
    person = np.zeros((20, 20), bool)
    person[7:13, 7:13] = True
    result = PseudoGTSelector().select(
        np.full((20, 20, 3), 100, np.uint8),
        np.full((20, 20, 3), 120, np.uint8),
        {level: np.full((20, 20, 3), 100, np.uint8) for level in LEVEL_COEFFICIENTS},
        source_masks=RegionMasks(person, empty, empty),
        reference_masks=RegionMasks(person, empty, empty),
    )

    _, json_path = write_dataset_summaries([result], tmp_path / "output")
    availability = json.loads(json_path.read_text(encoding="utf-8"))["region_availability"]

    assert availability == {
        "person_valid_rate": 1.0,
        "face_valid_rate": 0.0,
        "local_background_valid_rate": 0.0,
    }


def test_cli_processes_batch_and_emits_all_output_groups(tmp_path):
    _make_sample(tmp_path / "input", "scene001")
    output = tmp_path / "output"

    exit_code = main([
        "--input-root", str(tmp_path / "input"),
        "--output-root", str(output),
        "--mask-dir-name", "provided_masks",
    ])

    assert exit_code == 0
    assert (output / "pseudo_gt" / "scene001.png").is_file()
    assert (output / "scores" / "scene001.json").is_file()
    assert (output / "visualization" / "scene001.jpg").is_file()
    assert (output / "summary" / "selection_summary.csv").is_file()
    assert (output / "summary" / "dataset_summary.json").is_file()


def test_cli_randomly_samples_requested_number_of_valid_debug_overlays(tmp_path):
    invalid_one = _make_sample(tmp_path / "input", "00_invalid")
    invalid_two = _make_sample(tmp_path / "input", "01_invalid")
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(
        invalid_one / "provided_masks" / "source_person.png"
    )
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(
        invalid_two / "provided_masks" / "source_person.png"
    )
    for sample_id in ("10_valid", "11_valid", "12_valid"):
        _make_sample(tmp_path / "input", sample_id)

    outputs = []
    for output_name in ("output_a", "output_b"):
        output = tmp_path / output_name
        assert main([
            "--input-root", str(tmp_path / "input"),
            "--output-root", str(output),
            "--mask-dir-name", "provided_masks",
            "--debug-overlay-limit", "2",
            "--debug-overlay-seed", "17",
        ]) == 0
        overlay_ids = {
            path.parent.name for path in (output / "masks").glob("*/source_overlay.jpg")
        }
        assert len(overlay_ids) == 2
        assert overlay_ids <= {"10_valid", "11_valid", "12_valid"}
        outputs.append(overlay_ids)

    assert outputs[0] == outputs[1]


def test_streaming_batch_retains_only_bounded_mask_results_and_complete_summaries(tmp_path):
    for index in range(9):
        _make_sample(tmp_path / "input", f"valid_{index:02d}")
    invalid = _make_sample(tmp_path / "input", "invalid")
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(
        invalid / "provided_masks" / "source_person.png"
    )
    config = SelectorConfig(debug_overlay_limit=3, debug_overlay_seed=23)

    first = run_batch(tmp_path / "input", tmp_path / "output_a", "provided_masks", config)
    second = run_batch(tmp_path / "input", tmp_path / "output_b", "provided_masks", config)

    assert first.max_mask_results_retained <= 3
    assert len(first.summary_results) == 10
    assert all(result.source_masks is None for result in first.summary_results)
    assert all(result.reference_masks is None for result in first.summary_results)
    assert len(first.overlay_sample_ids) == 3
    assert first.overlay_sample_ids == second.overlay_sample_ids
    assert len(list((tmp_path / "output_a" / "scores").glob("*.json"))) == 10
    summary = json.loads(
        (tmp_path / "output_a" / "summary" / "dataset_summary.json").read_text(encoding="utf-8")
    )
    assert summary["num_samples"] == 10
    assert sum(summary["level_histogram"].values()) == 9
    assert summary["status_counts"]["NO_PERSON"] == 1
    with (tmp_path / "output_a" / "summary" / "selection_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 10
