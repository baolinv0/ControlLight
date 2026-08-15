# Review round 2

Reviewed remediation: `63d986c3ae2221cae34d2a3563f29fe3d5840d58`

Previous review: `61ec1f3cd6da0b7b99f84a25be60e370885a1773`

## Verdict

**Clean — ready for the code milestone.** I-5 is closed, the prior fixes remain intact, and no new specification regression or unnecessary defensive complexity was found. The real-data human study remains an explicitly documented external acceptance gate, not a claimed code result.

## I-5 verification

### Single-pass artifact processing

`run_batch()` discovers lightweight `SamplePaths`, then processes each sample once (`pseudo_gt_selector/run.py:48-64`). Per-sample JSON, pseudo-GT, visualization, and six masks are written immediately before the loop advances. It no longer accumulates all full-mask results before artifact emission.

### Mask-free complete summaries

`SelectionResult.without_masks()` uses `dataclasses.replace()` to clear only `source_masks` and `reference_masks` (`pseudo_gt_selector/models.py:77-78`). All scalar fields, status, region validity, best/second/margin, weights, component errors, nine candidate scores, selected path, and messages remain present. `run_batch()` appends only this mask-free copy to `summary_results` (`run.py:63`).

Fresh mixed-batch verification confirmed that all 13 input samples appeared in `selection_summary.csv`, `dataset_summary.json.num_samples` was 13, the histogram counted all 12 valid selections, and `NO_PERSON` counted the invalid sample. All returned summary records had both mask fields set to `None`.

### Bounded reproducible overlay reservoir

`run_batch()` retains full masks only in `overlay_reservoir` (`run.py:51-52, 65-74`). The implementation is standard one-pass reservoir sampling: the first `k` valid results fill the reservoir; each later valid result draws uniformly from the valid items seen so far and replaces only a slot below `k`. Invalid samples never enter the reservoir. The fixed `debug_overlay_seed` controls a local RNG, so repeated runs are reproducible without global random-state effects.

A fresh 12-valid/1-invalid run with limit 4 verified:

- `max_mask_results_retained == 4`;
- exactly four source/reference overlay pairs were emitted;
- all 13 score JSON files, 12 pseudo-GTs, 12 visualizations, and 12 six-mask groups were emitted;
- two runs with seed 31 selected the same overlay sample IDs;
- every pseudo-GT remained byte-for-byte equal to its selected ladder file.

The new repository regression test covers the same bounded-retention, mask-free-summary, deterministic-reservoir, mixed-status, score-count, histogram, status-count, and CSV-completeness contracts with small arrays. It does not rely on an oversized stress case.

## Regression and scope check

- Earlier fixes remain present: person-specific face confidence, empty-mask validity agreement, valid-only overlay selection, and semantic visualization testing.
- Core person-centric log-luminance score, source-mask reuse, active-weight normalization, zero-weight color debug, explicit invalid statuses, raw score outputs, and direct ladder copy are unchanged.
- The added `BatchRunReport`, mask-free copy helper, overlay writer, and reservoir are narrowly scoped to the identified batch-memory requirement.
- No hash/SHA256 code, reference/candidate pixel-wise selection metric, registration, warp, generated pseudo-GT, security-oriented checks, or speculative edge-case defenses were introduced.
- README accurately states bounded mask retention and keeps the unfulfilled real-data acceptance work explicit; it does not invent accuracy or failure-case evidence.

## Fresh verification

- `python -m compileall pseudo_gt_selector tests` — passed.
- CLI help — passed.
- All 40 current test functions executed with the local fixture/`approx` compatibility runner: **40 passed, 0 failed**.
- Canonical `python -m pytest -q` could not run because `pytest` is not installed in this environment. This environmental limitation is unchanged and should be resolved in the normal development environment.
- Fresh 12-valid/1-invalid batch E2E — passed all bounded retention, summary completeness, deterministic reservoir, artifact count, dataset statistics, and byte-for-byte copy assertions.
- `git diff --check 61ec1f3..63d986c` — passed.

## External acceptance gate

The code milestone is clean. Full product acceptance still requires the spec's supplied-real-data work: 100 human selections, Top-1 and Top-1±1 agreement, 50–100 archived real visualizations, and categorized failure examples.
