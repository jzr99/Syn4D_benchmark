#!/usr/bin/env bash
#SBATCH --job-name=syn4d-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=results/slurm/%x-%A_%a.out
#SBATCH --error=results/slurm/%x-%A_%a.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:?set MODEL to v-dpm, any4d, traceanything, st4rtrack, or spatrackerv2}"
PYTHON="${PYTHON:-python}"
LIMIT="${LIMIT:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${SHARD_INDEX:-0}}"

case "$MODEL" in
  v-dpm)
    RUNNER=adapters/run_vdpm.py
    DEFAULT_TASKS=tracking,depth,pose
    ;;
  any4d)
    RUNNER=adapters/run_any4d.py
    DEFAULT_TASKS=tracking,depth,pose
    ;;
  traceanything)
    RUNNER=adapters/run_traceanything.py
    DEFAULT_TASKS=tracking
    ;;
  st4rtrack)
    RUNNER=adapters/run_st4rtrack.py
    DEFAULT_TASKS=tracking
    ;;
  spatrackerv2)
    RUNNER=adapters/run_spatrackerv2.py
    DEFAULT_TASKS=tracking,depth,pose
    ;;
  *)
    echo "unknown MODEL=$MODEL" >&2
    exit 2
    ;;
esac

TASKS="${TASKS:-$DEFAULT_TASKS}"
OUTPUT="${OUTPUT:-$BENCHMARK_DIR/results/$MODEL/predictions}"
export HF_HOME="${HF_HOME:-$BENCHMARK_DIR/.hf_cache}"

mkdir -p "$BENCHMARK_DIR/results/slurm" "$HF_HOME"
cd "$BENCHMARK_DIR"
echo "job=$SLURM_JOB_ID host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "model=$MODEL tasks=$TASKS shard=$SHARD_INDEX/$NUM_SHARDS limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" "$RUNNER" \
  --tasks "$TASKS" \
  --limit "$LIMIT" \
  --output "$OUTPUT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
