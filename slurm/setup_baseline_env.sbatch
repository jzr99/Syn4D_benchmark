#!/usr/bin/env bash
#SBATCH --job-name=syn4d-setup
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

set -euo pipefail

BENCHMARK_DIR="${BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MODEL="${MODEL:?set MODEL to v-dpm, any4d, traceanything, st4rtrack, or spatrackerv2}"
BASE_PYTHON="${BASE_PYTHON:-/scratch/shared/beegfs/zeren/conda/envs/4rc/bin/python}"
ENV_ROOT="${ENV_ROOT:-$BENCHMARK_DIR/external/envs}"
ENV_DIR="$ENV_ROOT/$MODEL"

export HF_HOME="${HF_HOME:-$BENCHMARK_DIR/.hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BENCHMARK_DIR/.pip_cache}"
mkdir -p "$BENCHMARK_DIR/results/slurm" "$ENV_ROOT" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"
cd "$BENCHMARK_DIR"

inherited_env() {
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$ENV_DIR"
  fi
}

fresh_env() {
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$BASE_PYTHON" -m venv "$ENV_DIR"
  fi
}

download() {
  local url="$1"
  local destination="$2"
  if [[ ! -s "$destination" ]]; then
    mkdir -p "$(dirname "$destination")"
    curl --fail --location --retry 5 --retry-delay 5 \
      --output "$destination.partial" "$url"
    mv "$destination.partial" "$destination"
  fi
}

case "$MODEL" in
  v-dpm)
    inherited_env
    "$ENV_DIR/bin/pip" install einops omegaconf hydra-core roma jaxtyping
    "$ENV_DIR/bin/pip" install --no-deps \
      "git+https://github.com/facebookresearch/vggt.git@44b3afb"
    download \
      "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt" \
      "$TORCH_HOME/hub/checkpoints/vdpm_model.pt"
    ;;
  any4d)
    inherited_env
    "$ENV_DIR/bin/pip" install \
      jaxtyping natsort orjson pillow-heif plyfile python-box \
      'rerun-sdk>=0.23,<0.24'
    "$ENV_DIR/bin/pip" install --no-deps \
      "git+https://github.com/JayKarhade/UniCeption.git@dev/any4d" \
      "git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900"
    "$ENV_DIR/bin/pip" install --no-deps --editable "$BENCHMARK_DIR/external/any4d"
    download \
      "https://huggingface.co/airlabshare/any4d-checkpoint/resolve/main/any4d_4v_combined.pth" \
      "$BENCHMARK_DIR/external/any4d/checkpoints/any4d_4v_combined.pth"
    ;;
  traceanything)
    inherited_env
    "$ENV_DIR/bin/pip" install imageio
    download \
      "https://huggingface.co/depth-anything/trace-anything/resolve/main/trace_anything.pt" \
      "$BENCHMARK_DIR/external/traceanything/checkpoints/trace_anything.pt"
    ;;
  st4rtrack)
    inherited_env
    "$ENV_DIR/bin/pip" install evo prettytable scikit-image h5py pyro-ppl
    "$ENV_DIR/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("yupengchengg147/St4rTrack")
PY
    ;;
  spatrackerv2)
    fresh_env
    "$ENV_DIR/bin/pip" install \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
      --index-url https://download.pytorch.org/whl/cu124
    "$ENV_DIR/bin/pip" install \
      opencv-python-headless einops einx decord easydict moviepy==1.0.0 \
      safetensors scikit-learn hydra-core omegaconf matplotlib mediapy \
      scikit-image pycolmap==3.11.1 pillow-heif huggingface_hub \
      pyceres==2.4 kornia==0.8.1 kornia-rs==0.1.9 xformers==0.0.28 timm rich transformers \
      jaxtyping flow-vis
    "$ENV_DIR/bin/pip" install --no-deps \
      "git+https://github.com/EasternJournalist/utils3d.git@d3a577acf0a9ad7e513a1416449a07b6f47d967f"
    "$ENV_DIR/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Yuxihenry/SpatialTrackerV2_Front")
snapshot_download("Yuxihenry/SpatialTrackerV2-Offline")
PY
    ;;
  *)
    echo "unknown MODEL=$MODEL" >&2
    exit 2
    ;;
esac

echo "$MODEL environment ready: $ENV_DIR"
"$ENV_DIR/bin/python" --version
