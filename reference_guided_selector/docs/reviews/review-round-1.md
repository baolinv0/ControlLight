# Review round 1

Reviewed remediation: `ad53f10572f1a8a06b3d03e45e83d74cbe826346`

Previous review: `9625dc81d088b2de9c92269c197db5935e7df565`

## Verdict

**Changes requested.** All three original code findings and the visualization-test finding are closed. One newly identified Important batch-memory issue remains. The external real-data study is correctly documented as an acceptance gate and is not treated as a code defect.

## Original findings

### I-1 — Closed: face confidence is now person-specific

`pseudo_gt_selector/mask_provider.py:87-100` reads the generic face mask and assigns each `PersonDetection` the fraction of face pixels overlapping that person's mask. `get_face_mask()` now returns `None` for a non-overlapping face (`mask_provider.py:106-115`).

The new tests in `tests/test_masks_dataset.py` exercise the actual file provider with two people, prove confidence values `[0.0, 1.0]`, prove the face-bearing centered person is selected, and cover an unrelated face. This closes the original gap without adding unrelated validation.

### I-2 — Closed: effective mask availability now matches scoring

`pseudo_gt_selector/selector.py:123-137` now requires both source and reference masks to be non-`None` and non-empty before reporting face/local-background validity. This matches `ToneDescriptorExtractor` and `ToneScorer` fallback behavior.

The new selector and statistics tests prove that empty face/background masks yield person-only weight `1.0`, false region validity, unavailable component errors, matching JSON, and zero dataset availability rates.

### I-3 — Closed: overlay sampling is valid-only, random, and reproducible

`pseudo_gt_selector/run.py:41-54` samples only results with a selected level, uses a configurable seed, and writes overlays for exactly `min(limit, valid_count)` samples. The new mixed invalid/valid CLI test proves invalid samples do not consume the quota and a fixed seed reproduces the same selected IDs.

### M-1 — Closed: visualization semantics are directly testable

`pseudo_gt_selector/visualization.py:16-63` adds a compact render model used by the real renderer. The new semantic test verifies Source, Reference, Best, all nine levels, rank, raw total, P/PB/F/B fields, and BEST marking. This is meaningful contract testing without OCR or pixel-level rubric mechanics.

### I-4 — External gate explicitly documented

`README.md` now clearly states that 100 real human selections, Top-1 and Top-1±1 agreement, 50–100 real visualizations, and categorized failure examples require supplied real data and are not claimed by synthetic tests. No fabricated result is present.

## New Important finding

### I-5 — Batch CLI retains every sample's full-resolution masks until the entire dataset finishes

**Evidence:** `PseudoGTSelector.select()` attaches source and reference `RegionMasks` to every valid `SelectionResult` (`pseudo_gt_selector/selector.py:158-159`). Each result can therefore hold six full-resolution boolean arrays. `run.main()` stores every `(sample, result)` in `processed` (`pseudo_gt_selector/run.py:31-39`) and keeps that full list through artifact writing and summary generation (`run.py:41-56`).

This is material for the intended batch workflow, not an extreme edge case. At 3840×2160, one boolean mask is about 8.3 MB; six masks are about 50 MB per valid sample. A few hundred normal phone images can therefore retain tens of GB even though only up to 100 overlay groups are needed. The CLI can exhaust memory before producing a complete dataset result.

**Minimal fix:** stream normal artifact writing and retain full masks only for a size-limited reproducible valid-sample reservoir used for overlays. For dataset summaries, keep a mask-free `SelectionResult` (or a compact summary record) after per-sample masks are written. A focused regression test can use deliberately small arrays and assert that retained summary results have masks cleared and that no more than `debug_overlay_limit` full mask pairs are retained; no large-memory stress test is necessary.

## Regression and constraint check

- Core score formula, source-mask reuse, reference/candidate region-only comparison, color zero-weight behavior, direct ladder copy, statuses, JSON/CSV/statistics, and CLI contracts remain unchanged.
- No hash/SHA256 code, reference/candidate pixel-wise selection metric, registration, warp, or generated pseudo-GT was introduced.
- No security-oriented or speculative extreme-case validation was added.
- The new random seed and effective-mask checks are directly required by the reviewed behavior and are not over-defensive.

## Fresh verification

- `python -m compileall pseudo_gt_selector tests` — passed.
- CLI help — passed and includes overlay limit/seed options.
- All 39 current test functions were executed with the same lightweight local fixture/`approx` compatibility runner: **39 passed, 0 failed**.
- Canonical `python -m pytest -q` remains unavailable in the environment because `pytest` is not installed; this must still be run in the normal development environment.
- Fresh mixed invalid/valid CLI run, repeated twice with the same seed — passed; exactly two valid overlay groups were selected reproducibly, invalid input did not consume the quota, and every emitted pseudo-GT was byte-for-byte equal to the chosen ladder file.
- Remediation diff check `9625dc81..ad53f105` — passed.
