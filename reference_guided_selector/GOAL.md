# Autonomous Improvement Goal

## Objective

Improve the Reference-Guided Aligned Pseudo-GT Selector using real experimental feedback.
The selected pseudo-GT must remain one of the source-aligned ladder images while matching the generated reference at the person-tone and person/background-relation level defined by `docs/SPEC`.

## Acceptance metrics

The frozen machine-readable thresholds are stored in `agent/goal.json`:

- Top-1 agreement with human-selected ladder level: **>= 0.80**
- Top-1 +/- 1 level agreement: **>= 0.95**
- Existing unit tests must pass.

The real-data metrics are computed only by `agent/evaluate.py` from a frozen human-label CSV. Missing or invalid predictions count as incorrect.

## Required real-data label format

Provide a CSV with:

```text
sample_id,target_level,failure_category
scene001,a_p025,
scene002,a_000,face_overbright
```

`failure_category` is optional analysis metadata. `target_level` must be one of the nine ladder levels.

## Immutable acceptance assets

During autonomous improvement, do not modify any of the following to make the score easier to pass:

- `GOAL.md`
- `agent/goal.json`
- `agent/evaluate.py`
- the frozen human-label CSV supplied through `PSEUDO_GT_LABELS`
- acceptance tests
- the definition of Top-1 or Top-1 +/- 1

Changing the algorithm, descriptors, scoring logic, selector configuration, training parameters, or implementation tests is allowed when justified by an experiment.

## Completion rule

The task is complete only when `agent_runs/current/evaluation/metrics.json` contains:

```json
{
  "goal_reached": true
}
```

and the test command passes.

Synthetic/unit tests are engineering checks only and must never be reported as real-data acceptance results.
