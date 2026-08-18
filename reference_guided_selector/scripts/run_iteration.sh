#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

bash scripts/run_agent_tests.sh
bash scripts/run_experiment.sh
bash scripts/evaluate.sh

METRICS_FILE="${PSEUDO_GT_OUTPUT_ROOT:-$ROOT_DIR/agent_runs/current}/evaluation/metrics.json"
echo
echo "Iteration metrics: $METRICS_FILE"
cat "$METRICS_FILE"
echo
