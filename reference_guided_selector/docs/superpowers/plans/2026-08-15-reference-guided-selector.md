# Reference-Guided Aligned Pseudo-GT Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, runnable V1 selector that chooses a person-tone-matched, source-aligned ladder image and emits all required per-sample and dataset artifacts.

**Architecture:** Implement a small Python package with injected mask providers, NumPy/OpenCV photometric and region operations, Pillow/Matplotlib-compatible artifact rendering, and a batch CLI. Keep reference analysis region-based and non-pixel-wise; copy the chosen candidate directly.

**Tech Stack:** Python 3.10+, NumPy, Pillow, OpenCV headless, pytest, standard-library argparse/json/csv/shutil.

## Global Constraints

- `project_sources/02-spec.txt` is authoritative; `project_sources/01-method.txt` is guidance.
- The nine level names and coefficients are exact and all are required.
- Reference/candidate pixel-wise L1, L2, PSNR, or SSIM must not influence selection.
- Source masks are computed once and reused by all ladder candidates.
- Selected pseudo-GT is a direct file copy with no resize, warp, generation, or color modification.
- Person and person/background relation dominate the score; color drift is debug-only with zero selection weight.
- No hash or SHA256 code.
- Implement only specification-required failures; do not add speculative security or extreme-case defenses.
- Rubric-driven outputs must remain meaningful, not mechanically over-expanded.

---

### Task 1: Package contracts, configuration, photometric descriptors, and scoring

**Files:** Create `pyproject.toml`, `pseudo_gt_selector/__init__.py`, `config.py`, `models.py`, `photometric.py`, `tone_descriptor.py`, `scorer.py`, and focused tests under `tests/`.

**Interfaces:** Define exact levels; immutable configuration; region/score/selection records; sRGB-to-linear log luminance; P10/P25/P50/P75/P90 descriptors; chromaticity debug distances; relation `(R1,R2,R3)`; weighted scoring with face/background fallback and normalized active weights.

- [ ] Write behavior-first tests for luminance conversion, descriptors, relation errors, default weights, face fallback, background fallback, and zero-weight color debug.
- [ ] Run each focused test group and confirm it fails because its production API is absent.
- [ ] Implement only enough production code to satisfy those tests.
- [ ] Run the focused and accumulated tests.

### Task 2: Masks, primary-person selection, and sample discovery

**Files:** Create `mask_provider.py`, `region_builder.py`, `dataset.py`, plus tests and compact test utilities.

**Interfaces:** `MaskProvider.get_person_masks`, `get_primary_person_mask`, `get_face_mask`; deterministic file-backed provider; primary-person score `0.50*normalized_area + 0.30*center_score + 0.20*face_confidence`; ambiguity signal; local background ring using 0.02/0.10 bbox-scale radii; required input discovery and missing-level reporting.

- [ ] Write failing tests for source-mask reuse contracts, primary person, ambiguity, local ring, expected sample layout, missing level, and size mismatch.
- [ ] Implement the interfaces without per-level segmentation or unrelated validation.
- [ ] Run focused and accumulated tests.

### Task 3: End-to-end selector and mandated invalid statuses

**Files:** Create `selector.py` and extend tests.

**Interfaces:** `PseudoGTSelector.select_paths(sample)` and array-level selection helper; valid results include every candidate's component scores, best/second/margin/coefficient/status/confidence/weights/region validity/color debug; invalid results explicitly cover `NO_PERSON`, `INVALID_REFERENCE_PERSON`, `AMBIGUOUS_PERSON`, and `INVALID` missing/unaligned ladder.

- [ ] Write at least five failing synthetic person-dominance tests in which global brightness favors A but person-centered scoring must favor B.
- [ ] Write failing tests for person/background discrimination, face fallback, deterministic repeatability, invalid statuses, complete score output, and direct selected-path identity.
- [ ] Implement selection, ranking, thresholds, margin, and confidence exactly from the spec.
- [ ] Run focused and accumulated tests.

### Task 4: Artifacts, visualization, summaries, CLI, and documentation

**Files:** Create `visualization.py`, `statistics.py`, `run.py`, `README.md`; extend tests.

**Interfaces:** Valid sample artifacts under `pseudo_gt`, `scores`, `visualization`, and `masks`; source/reference masks with exact filenames; selection visualization containing source/reference/best/all nine candidates and raw component scores/ranks; `selection_summary.csv`; `dataset_summary.json`; batch CLI.

- [ ] Write failing integration tests for byte-for-byte pseudo-GT copying, JSON schema, six masks, visualization existence/content dimensions, CSV fields, dataset percentiles/histogram/valid rates, and CLI output.
- [ ] Implement artifact writers, contact sheet, overlay option, aggregations, and CLI.
- [ ] Add README commands for installation, mask layout, single/batch execution, outputs, and provider injection.
- [ ] Run focused and accumulated tests.

### Task 5: Whole-feature verification

**Files:** Review all package, tests, and documentation files against `project_sources/02-spec.txt`.

- [ ] Run `python -m compileall pseudo_gt_selector tests`.
- [ ] Run `pytest -q` and report exact pass/fail counts.
- [ ] Run CLI help and a temporary synthetic end-to-end sample.
- [ ] Search production selection code to confirm no reference/candidate pixel-wise metrics, hashes, SHA256, or per-level segmentation.
- [ ] Record which final-delivery items require real user data: 100-sample human agreement, 50-100 real visualizations, and categorized real failure cases.

