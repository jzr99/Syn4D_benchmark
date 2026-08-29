#!/usr/bin/env bash
#SBATCH --job-name=syn4d-tracking-gt
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python}"
VARIANTS=(mixed mixed_no_bedlam og sim)
TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script with --array=0-3}"
VARIANT="${VARIANTS[$TASK_ID]}"

mkdir -p "$BENCHMARK_DIR/results/slurm"
cd "$BENCHMARK_DIR"
echo "job=$SLURM_JOB_ID host=$(hostname) variant=$VARIANT"

ARGS=(--variants "$VARIANT")
if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
  ARGS+=(--skip-convert)
fi
if [[ -n "${CONVERTER:-}" ]]; then
  ARGS+=(--converter "$CONVERTER")
fi
"$PYTHON" prepare_tracking_gt.py "${ARGS[@]}"
