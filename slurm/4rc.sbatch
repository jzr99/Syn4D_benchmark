#!/usr/bin/env bash
#SBATCH --job-name=syn4d-4rc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/4rc/bin/python}"
TASKS="${TASKS:-tracking,depth,pose}"
SIZE="${SIZE:-512}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
OUTPUT="${OUTPUT:-$BENCHMARK_DIR/results/4rc/predictions}"

export HF_HOME="${HF_HOME:-$BENCHMARK_DIR/.hf_cache}"
mkdir -p "$BENCHMARK_DIR/results/slurm" "$HF_HOME"
cd "$BENCHMARK_DIR"

echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "tasks=$TASKS size=$SIZE shard=$SHARD_INDEX/$NUM_SHARDS limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" adapters/run_4rc.py \
  --tasks "$TASKS" \
  --size "$SIZE" \
  --limit "$LIMIT" \
  --output "$OUTPUT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
