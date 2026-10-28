#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/eval.sh {mfnet|pst|nyu} checkpoint.pth [eval.py arguments]" >&2
    exit 2
fi
dataset="$1"
checkpoint="$2"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$dataset" in mfnet|pst|nyu) ;; *) echo "Usage: bash scripts/eval.sh {mfnet|pst|nyu} checkpoint.pth [eval.py arguments]" >&2; exit 2 ;; esac
if [[ ! -f "$checkpoint" ]]; then echo "Checkpoint not found: $checkpoint" >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec python eval.py --dataset_name "$dataset" -d 0 -e "$checkpoint" "$@"
