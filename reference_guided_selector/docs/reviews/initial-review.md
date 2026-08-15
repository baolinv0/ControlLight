# Initial independent review

Reviewed commit: `d8a9d6d137cf7385c82ddcf43f24ff8083c151f0`

Baseline: `8af8493`
Authority: `project_sources/02-spec.txt`; `project_sources/01-method.txt` used only as supporting guidance.

## Verdict

**Changes requested — not ready for final acceptance.**

The core selection path is directionally correct: it computes source masks once, reuses them for all nine candidates, compares region-level log-luminance quantiles rather than reference/candidate pixels, applies the specified score formula with active-weight renormalization, preserves color as zero-weight debug data, and copies the selected ladder file byte-for-byte. No hash/SHA256 implementation or security-oriented defensive expansion was found.

There are no Critical findings. The following Important findings affect required multi-person behavior, validity reporting/statistics, and mandated debug sampling. The real-data acceptance package also remains outstanding because no real dataset or human labels are present in this repository.

## Important

### I-1 — File-backed multi-person selection assigns the same face confidence to every person

**Evidence:** `pseudo_gt_selector/mask_provider.py:81-95` discovers multiple `source_person_N.png` / `reference_person_N.png` masks, but sets `confidence = 1.0` for every detection whenever one generic `<stem>_face.png` exists. The generic face may overlap only one person, yet all people receive the same `0.20 * face_confidence` term. This defeats the specified primary-person rule in exactly the multi-person case it is meant to resolve and can select a larger edge person instead of the centered face-bearing subject. The existing test at `tests/test_masks_dataset.py:17-27` injects correct `PersonDetection.face_confidence` values directly, so it does not exercise the file provider's broken mapping.

**Minimal fix:** read the face mask once, derive each detection's confidence from whether/how strongly that face overlaps that person's mask (or support explicitly paired numbered face masks), then add a file-provider test in which only one of two people contains the face and the face term changes the primary-person result.

### I-2 — Empty or non-overlapping face/background masks are reported as valid although scoring falls back

**Evidence:** `ToneDescriptorExtractor._region()` correctly treats empty masks as unavailable (`pseudo_gt_selector/tone_descriptor.py:10-13`), so `ToneScorer` removes those components and renormalizes weights. However, `PseudoGTSelector.select()` reports validity using only `mask is not None` (`pseudo_gt_selector/selector.py:123-129`). A direct reproduction with empty face and background arrays produced:

```text
weights  = {'person': 1.0}
validity = {'person': True, 'face': True, 'local_background': True}
face_error/background_error/person_background_error = None
```

This makes per-sample JSON self-contradictory and inflates the required dataset-level face/background availability rates in `pseudo_gt_selector/statistics.py:102-107`. It is also reachable through `FileMaskProvider`: an empty face file, or a face belonging to another selected person, returns a non-`None` empty intersection at `mask_provider.py:101-107`.

**Minimal fix:** define validity from effective descriptor availability (or `np.any` on both source and reference masks), normalize an empty face intersection to `None`, and add regression tests that assert fallback weights, `region_validity`, JSON, and dataset rates agree for empty/non-overlapping masks.

### I-3 — Required debug overlays are neither randomly sampled nor guaranteed for 100 valid samples

**Evidence:** the specification requires at least a random 100-sample mask-overlay check. `pseudo_gt_selector/run.py:31-40` enables overlays only when the sample's overall sorted index is below the limit. `write_sample_artifacts()` returns before writing masks/overlays for invalid results (`pseudo_gt_selector/artifacts.py:51-52`). Therefore, if the first 100 sorted samples are invalid, later valid samples receive no overlays; even in the normal case the selected set is the first sorted samples, not a random sample. The current CLI test uses one valid sample and cannot expose either issue.

**Minimal fix:** determine the overlay set from valid results, using a reproducible random sample (fixed configurable seed) capped at 100, or perform a second artifact-writing pass once valid samples are known. Add a mixed invalid/valid batch test proving the requested number of valid overlays is produced.

### I-4 — Final real-data acceptance evidence is not delivered

**Evidence:** authoritative spec sections 28, 29, and 31 require a 100-sample human comparison, Top-1 and Top-1±1 accuracy, 50–100 real visualizations, and categorized real failure cases. The repository contains only synthetic tests. `README.md:55` acknowledges that the human study is still required, but no labeled real data or result artifacts are present.

**Minimal resolution:** this is not a request to fabricate data or complicate the selector. Run the existing CLI on the supplied real dataset, collect one human-selected level per 100 sampled cases, report Top-1 / adjacent-level agreement, and archive 50–100 visualizations plus the specified failure-category examples. If real data is intentionally outside this code milestone, explicitly mark these four items as an external acceptance gate rather than claiming the full specification is complete.

## Minor

### M-1 — Tests do not verify visualization semantics

**Evidence:** `tests/test_artifacts_cli.py:71-78` checks only that the JPEG is at least 1000×500. A regression that drops Source, Reference, a candidate, rank, or one raw component label could still pass, even though these are MUST-08 outputs. The implementation currently renders those items, but the test does not protect them.

**Minimal fix:** keep the test lightweight: expose the cell labels/rows as a small render model or monkeypatch the cell renderer and assert Source, Reference, all nine levels, BEST, rank, Total, P, PB, F, and B are supplied. No OCR or mechanical pixel rubric is needed.

## Verified requirements and evidence

- Exact nine levels and coefficients: `pseudo_gt_selector/config.py:6-16`.
- Linear-sRGB luminance followed by `log2(Y + 1e-6)`: `photometric.py:4-15`.
- P10/P25/P50/P75/P90 descriptors: `tone_descriptor.py:7-13`.
- Person/background relation and specified `1, 0.25, 0.25` error: `tone_descriptor.py:28-33`, `scorer.py:23-26`.
- Specified with-face and no-face weights, plus active-component renormalization: `config.py:26-33`, `scorer.py:28-38`.
- Source masks are obtained once and reused for all candidates: `selector.py:73-107`; there is no candidate-level mask-provider call.
- Reference/candidate selection uses region descriptors only; no pixel-wise L1/L2/PSNR/SSIM selection metric was found.
- Chromaticity is debug-only and absent from total score: `scorer.py:38-48`.
- All nine candidates are required; missing levels and size mismatches return explicit `INVALID`: `selector.py:65-71`, `dataset.py:27-47`.
- Required no-person/reference-person/ambiguous statuses exist: `selector.py:73-91`.
- Best, second, margin, threshold status, and confidence follow the supporting method: `selector.py:109-139`.
- Selected pseudo-GT is copied directly with `shutil.copyfile`: `artifacts.py:54-57`; a fresh synthetic end-to-end run confirmed byte equality with the selected ladder file.
- Score JSON, six masks, visualization, CSV, dataset JSON, and review list are implemented.
- No hash, SHA256, registration, warp, generated output, or candidate/reference pixel metric was found in production code.

## Verification run

- `python -m compileall pseudo_gt_selector tests` — passed.
- `python -m pseudo_gt_selector.run --help` — passed; required CLI arguments displayed.
- Fresh synthetic CLI end-to-end run — exit code 0; emitted pseudo-GT, score JSON with nine levels, six masks, two overlays, visualization, CSV/JSON summaries; selected file was byte-for-byte equal to its ladder source.
- Repository contains 33 test functions. The environment does not have the `pytest` package (`pytest: command not found`; the repository venv also lacks pip/dependencies), so the canonical `pytest -q` command could not be executed. As a secondary check, all 33 current test functions were executed with a minimal local fixture/`approx` compatibility runner and produced 33 passes, 0 failures. This does not replace a canonical pytest run after dependencies are available.
- `git diff --check 8af8493..d8a9d6d` — passed.
