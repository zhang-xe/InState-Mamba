#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset="${1:-mfnet}"
if [[ $# -gt 0 ]]; then shift; fi
case "$dataset" in mfnet|pst|nyu) ;; *) echo "Usage: bash scripts/train.sh {mfnet|pst|nyu} [train.py arguments]" >&2; exit 2 ;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
nproc="${NPROC_PER_NODE:-2}"
port="${MASTER_PORT:-16005}"
exec python -m torch.distributed.launch --nproc_per_node="$nproc" --master_port="$port" train.py --dataset_name "$dataset" --devices 0 --port "$port" "$@"
