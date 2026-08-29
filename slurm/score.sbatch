#!/usr/bin/env bash
#SBATCH --job-name=syn4d-score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python}"
MODEL="${MODEL:?set MODEL to a benchmark adapter name}"
TASKS="${TASKS:-tracking,depth,pose}"
PREDICTIONS="${PREDICTIONS:-$BENCHMARK_DIR/results/$MODEL/predictions}"
OUTPUT_FILE="${OUTPUT_FILE:-$BENCHMARK_DIR/results/$MODEL/summary.json}"
LIMIT="${LIMIT:-0}"

cd "$BENCHMARK_DIR"
EVAL_ARGS=(
  --predictions "$PREDICTIONS"
  --tasks "$TASKS"
  --strict
  --output "$OUTPUT_FILE"
)
if [[ "$LIMIT" -gt 0 ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi
"$PYTHON" evaluate.py "${EVAL_ARGS[@]}"
