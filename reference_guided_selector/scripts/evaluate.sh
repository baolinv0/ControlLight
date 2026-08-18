#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PSEUDO_GT_LABELS:?Set PSEUDO_GT_LABELS to the frozen human-label CSV before evaluation}"

PSEUDO_GT_OUTPUT_ROOT="${PSEUDO_GT_OUTPUT_ROOT:-$ROOT_DIR/agent_runs/current}"
PSEUDO_GT_GOAL_FILE="${PSEUDO_GT_GOAL_FILE:-$ROOT_DIR/agent/goal.json}"
SELECTION_SUMMARY="$PSEUDO_GT_OUTPUT_ROOT/summary/selection_summary.csv"
EVALUATION_DIR="$PSEUDO_GT_OUTPUT_ROOT/evaluation"

if [[ ! -f "$SELECTION_SUMMARY" ]]; then
  echo "Missing experiment summary: $SELECTION_SUMMARY" >&2
  exit 2
fi

python -m agent.evaluate \
  --selection-summary "$SELECTION_SUMMARY" \
  --labels "$PSEUDO_GT_LABELS" \
  --goal "$PSEUDO_GT_GOAL_FILE" \
  --metrics-out "$EVALUATION_DIR/metrics.json" \
  --failures-out "$EVALUATION_DIR/failures.json"
