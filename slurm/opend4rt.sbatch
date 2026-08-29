#!/usr/bin/env bash
#SBATCH --job-name=syn4d-opend4rt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python}"
TASKS="${TASKS:-tracking,depth,pose}"
DEPTH_GRID_SIZE="${DEPTH_GRID_SIZE:-128}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"
OUTPUT="${OUTPUT:-$BENCHMARK_DIR/results/opend4rt/predictions}"

mkdir -p "$BENCHMARK_DIR/results/slurm"
cd "$BENCHMARK_DIR"

echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "tasks=$TASKS depth_grid=$DEPTH_GRID_SIZE shard=$SHARD_INDEX/$NUM_SHARDS limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" adapters/run_opend4rt.py \
  --tasks "$TASKS" \
  --depth-grid-size "$DEPTH_GRID_SIZE" \
  --limit "$LIMIT" \
  --output "$OUTPUT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
