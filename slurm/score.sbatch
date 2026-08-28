#!/usr/bin/env bash
#SBATCH --job-name=syn4d-score
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python}"
MODEL="${MODEL:?set MODEL to a benchmark adapter name}"
TASKS="${TASKS:-tracking,depth,pose}"

cd "$BENCHMARK_DIR"
"$PYTHON" evaluate.py \
  --predictions "$BENCHMARK_DIR/results/$MODEL/predictions" \
  --tasks "$TASKS" \
  --strict \
  --output "$BENCHMARK_DIR/results/$MODEL/summary.json"
