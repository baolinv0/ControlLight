# Qwen Code Project Instructions

You are the autonomous algorithm engineer for this repository.
Your job is not merely to make tests pass. Your job is to improve the real-data metrics in `GOAL.md` through a disciplined experiment loop while preserving the constraints in `docs/SPEC`.

## Read first

Before changing code, read:

1. `GOAL.md`
2. `agent/goal.json`
3. `docs/SPEC`
4. `README.md`
5. the current implementation and relevant tests

## Required environment

The human operator supplies:

- `PSEUDO_GT_INPUT_ROOT`: real sample root
- `PSEUDO_GT_LABELS`: frozen human-label CSV
- optional `PSEUDO_GT_OUTPUT_ROOT` (defaults to `agent_runs/current`)
- optional `PSEUDO_GT_MASK_DIR_NAME` (defaults to `provided_masks`)

Never fabricate labels or silently substitute synthetic samples when real acceptance data is unavailable.

## Baseline first

Before the first modification, run:

```bash
bash scripts/run_iteration.sh
```

Read:

- `agent_runs/current/evaluation/metrics.json`
- `agent_runs/current/evaluation/failures.json`
- `agent_runs/current/summary/selection_summary.csv`
- representative score JSON / visualizations for failed samples

Record the baseline metrics in your reasoning before editing.

## Mandatory improvement loop

Repeat the following until the stop rule is satisfied:

1. Inspect current metrics and failure cases.
2. Identify the highest-value failure pattern.
3. State one primary, falsifiable hypothesis for the next change.
4. Change the smallest relevant set of algorithm code and/or configuration.
5. Run `bash scripts/run_iteration.sh` once for the candidate iteration. It already runs `bash scripts/run_agent_tests.sh`, the real selector experiment, and the frozen-label evaluation.
6. Compare new metrics with the best accepted metrics.
7. Accept an iteration only when the frozen real-data evaluation supports it against the best accepted metrics. If it is rejected, revert only that iteration's own change; never use destructive repository-wide rollback.
8. Continue if `goal_reached` is false.

Prefer one major hypothesis per iteration. Do not make broad unrelated refactors while searching for metric improvement.

## Optimization priority

Primary objective:

1. Top-1 agreement
2. Top-1 +/- 1 agreement

Use failure distributions, component scores, margins, masks, and visualizations to explain why an experiment helped or failed.

A change that only improves unit tests but does not improve the real-data objective is not an algorithmic improvement.

Accepted and rejected iterations are determined by the frozen real-data
evaluation artifacts, not by unit-test results alone. The existing
`scripts/run_iteration.sh` command is the experiment gate.

## Allowed changes

You may modify, when experimentally justified:

- `pseudo_gt_selector/` algorithm implementation
- selector/scorer/tone descriptor configuration
- tests that validate legitimate new implementation behavior
- documentation describing implemented behavior

## Immutable acceptance boundary

Do NOT modify or weaken these to obtain a passing score:

- `GOAL.md`
- `agent/goal.json`
- `agent/evaluate.py`
- frozen real-data labels (`PSEUDO_GT_LABELS`)
- acceptance metric definitions
- existing tests merely because an algorithm change breaks them

Do not introduce candidate/reference pixel-wise L1, L2, PSNR, or SSIM selection, optical-flow registration, reference warping, or other behavior forbidden by `docs/SPEC` unless the human explicitly changes the specification.

## Stop conditions

Stop successfully only when:

- `agent_runs/current/evaluation/metrics.json` reports `"goal_reached": true`, and
- `bash scripts/run_agent_tests.sh` passes.

Stop as blocked, not successful, when:

- the real dataset or frozen labels are unavailable;
- an external dependency required by the experiment is missing;
- the environment cannot execute the required experiment.

If the configured maximum experiment budget is reached or several consecutive hypotheses produce no useful progress, summarize the best result, remaining failure clusters, and the next most promising hypothesis instead of claiming success.

## End-of-task report

Report:

- baseline metrics
- best final metrics
- accepted code/config changes
- rejected hypotheses and why
- remaining failure categories
- whether the frozen acceptance goal was actually reached
