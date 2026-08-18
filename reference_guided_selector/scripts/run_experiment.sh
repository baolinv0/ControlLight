#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PSEUDO_GT_INPUT_ROOT:?Set PSEUDO_GT_INPUT_ROOT to the real sample root before running experiments}"

PSEUDO_GT_OUTPUT_ROOT="${PSEUDO_GT_OUTPUT_ROOT:-$ROOT_DIR/agent_runs/current}"
PSEUDO_GT_MASK_DIR_NAME="${PSEUDO_GT_MASK_DIR_NAME:-provided_masks}"
PSEUDO_GT_DEBUG_OVERLAY_LIMIT="${PSEUDO_GT_DEBUG_OVERLAY_LIMIT:-100}"
PSEUDO_GT_DEBUG_OVERLAY_SEED="${PSEUDO_GT_DEBUG_OVERLAY_SEED:-0}"

rm -rf "$PSEUDO_GT_OUTPUT_ROOT"

python -m pseudo_gt_selector.run \
  --input-root "$PSEUDO_GT_INPUT_ROOT" \
  --output-root "$PSEUDO_GT_OUTPUT_ROOT" \
  --mask-dir-name "$PSEUDO_GT_MASK_DIR_NAME" \
  --debug-overlay-limit "$PSEUDO_GT_DEBUG_OVERLAY_LIMIT" \
  --debug-overlay-seed "$PSEUDO_GT_DEBUG_OVERLAY_SEED"
