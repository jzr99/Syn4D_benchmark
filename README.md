# Syn4D multi-task benchmark

This repository defines one reproducible benchmark over **every complete
sequence** in the public Syn4D evaluation release. It evaluates three outputs
from the same 32-frame input clip:

1. 3D point tracking;
2. monocular video depth;
3. camera pose.

The checked-in [manifest](manifests/syn4d_all.jsonl) contains 512 sequences:
4 render variants × 8 scenes × 8 base sequences × 2 cameras. The
`even_camera_png` convenience mirror is not indexed because it duplicates RGB
files and does not own independent ground truth.

## Quick start: download and evaluate your model

Clone the evaluator and install its lightweight scoring dependencies:

```bash
git clone https://github.com/jzr99/Syn4D_benchmark.git
cd Syn4D_benchmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the complete evaluation release from
[Hugging Face](https://huggingface.co/datasets/Syn4D/Syn4D_Benchmark). The
download is about 8.7 GiB and contains the 512 MP4 inputs, the fixed tracking
annotations, the 16,384 sampled depth frames, and camera-pose metadata:

```bash
python download_benchmark.py --output data/release
export SYN4D_DATA_ROOT="$PWD/data/release/challenge_eval"
export SYN4D_TRACKING_GT="$PWD/data/release/benchmark/data/tracking_gt"
```

The downloader is resumable and rejects incomplete releases. Recheck an
existing download without network access with:

```bash
python download_benchmark.py --output data/release --verify-only
```

For a task-specific download, pass for example `--tasks tracking` or
`--tasks tracking,pose`. Videos are always included because they are the model
inputs. Depth EXRs and camera CSVs are downloaded only when their corresponding
tasks are selected.

The resulting layout is:

```text
data/release/
├── benchmark/data/tracking_gt/<variant>/<scene>/<sequence>.npy
└── challenge_eval/<variant>/<scene>/
    ├── mp4/<sequence>.mp4
    ├── exr_layers/depth/<sequence>/<selected-frame>_depth.exr
    └── ground_truth/meta_exr_csv/<sequence>_camera.csv
```

Run your model on frames `0, 6, ..., 186` from each MP4 and write one prediction
file per sequence using the [prediction contract](#prediction-contract). The
common scorer is model-independent:

```bash
python evaluate.py \
  --predictions /path/to/your/predictions \
  --tasks tracking,depth,pose \
  --strict \
  --output results/your-model/summary.json
```

You can start with one sequence by adding `--limit 1`. For tracking-only
predictions, use `--tasks tracking`; no raw surface metadata, depth, or camera
CSV is read. The bundled baseline adapters also work with the MP4-only input
release: when source PNGs are absent they extract the 32 selected frames into
`data/frame_cache/` (override with `SYN4D_FRAME_CACHE`).

The evaluator and release are self-contained for scoring all three tasks.
Official baseline implementations and checkpoints are intentionally not
vendored; [setup_baselines.py](setup_baselines.py) retrieves their pinned
upstream revisions when you want to reproduce the baseline table.

## Protocol

All tasks use source frames `0, 6, 12, ..., 186` (32 frames spanning the full
192-frame render). Results are macro-averaged over sequences and also broken
down by render variant and scene. We deliberately do **not** combine tracking,
depth, and pose into one arbitrary scalar; each task has its own primary metric.
The machine-readable definition is [protocol.json](protocol.json).

### 3D tracking

- 512 deterministic frame-0-visible query pixels per sequence.
- Predictions are `[T,Q,3]` in OpenCV frame-0 camera coordinates.
- Invalid/occluded GT samples are ignored.
- One median scale aligns the entire predicted sequence.
- APD uses `{0.1, 0.3, 0.5, 1.0}` metre thresholds; EPE is also reported.
- Dynamic points have more than 1 cm of accumulated world-space motion across
  consecutive valid observations.
- Primary score: `0.5 * APD(all) + 0.5 * APD(dynamic)`.

This matches the useful part of the WorldTrack/Open-D4RT protocol while fixing
two ambiguities in the old evaluator: the dynamic subset uses the **same**
sequence scale as all points, and occluded intervals do not turn accumulated
motion into NaN.

### Video depth

- GT is the renderer's `Depth` EXR channel converted from centimetres to metres.
- Valid range is `(0.001, 300]` metres.
- One median scale is shared by every valid pixel of the full sequence.
- Primary metric: AbsRel. SqRel, RMSE, RMSE-log, SILog, and δ1/δ2/δ3 are also
  reported.
- `--depth-align metric` and `--depth-align scale_shift` are diagnostic modes;
  `scale` is the canonical monocular result.

### Camera pose

- Inputs/outputs are camera-to-world `[T,4,4]` matrices.
- Trajectories are normalized to frame 0 and aligned by one Sim(3).
- Primary metrics are ATE RMSE, RPE translation RMSE, and mean RPE rotation in
  degrees. Absolute rotation error and relative-pose AUC are also reported.

## Baseline results

Official checkpoints at the revisions in [baselines.json](baselines.json) are
evaluated on all 512 sequences. Values are macro-averages over sequences under
the canonical alignment rules above. `—` denotes a task that the official
method does not expose through a compatible inference interface.

| Method | Track score ↑ | APD ↑ | Dynamic APD ↑ | EPE (m) ↓ | Depth AbsRel ↓ | Depth δ1 ↑ | Pose ATE (m) ↓ | RPE trans. (m) ↓ | RPE rot. (°) ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Open-D4RT | 0.3043 | 0.3631 | 0.2317 | 2.3298 | 0.1290 | 0.8408 | 0.0676 | 0.0355 | 0.3872 |
| 4RC | 0.4763 | 0.5848 | 0.3417 | 1.5925 | 0.0775 | 0.9195 | 0.0370 | 0.0218 | 0.1593 |
| V-DPM | 0.4880 | 0.5774 | 0.3757 | 1.6371 | 0.0891 | 0.9022 | 0.0375 | 0.0265 | 0.1693 |
| Any4D | 0.2776 | 0.3676 | 0.1701 | 1.8578 | 0.1040 | 0.8874 | 0.1611 | 0.1382 | 0.7645 |
| TraceAnything | 0.2206 | 0.3240 | 0.1003 | 2.3577 | — | — | — | — | — |
| St4RTrack | 0.2575 | 0.3302 | 0.1667 | 2.2390 | — | — | — | — | — |
| SpaTrackerV2 | 0.4334 | 0.5452 | 0.2960 | 1.6163 | 0.0727 | 0.9210 | 0.0444 | 0.0332 | 0.1569 |

## Prediction contract

Every model adapter writes:

```text
results/<model>/predictions/<variant>/<scene>/<sequence>.npz
```

An NPZ may contain any subset of:

```text
tracking_xyz  float32 [32,512,3]
depth         float32 [32,H,W]
camera_c2w    float32 [32,4,4]
frame_indices int32   [32]
```

The common scorer—not a model repository—performs masking, alignment, metrics,
and aggregation. Model support and pinned official revisions are recorded in
[baselines.json](baselines.json).

## Tracking ground-truth contract

Each sequence has exactly one self-contained file:

```text
data/tracking_gt/<variant>/<scene>/<sequence>.npy
```

It is a structured NumPy scalar that loads with `allow_pickle=False` and embeds
the schema/version, sequence ID, trajectories, visibility, dynamic labels,
queries, frame indices, camera intrinsics, and source image size. In
particular, `evaluate.py --tasks tracking` reads no Syn4D camera CSV, surface
track, or other raw metadata. The raw surface metadata is needed only by the
dense conversion stage used to generate the portable NPY files.

## Rebuild or audit the manifest (maintainers)

```bash
cd /path/to/Syn4D_benchmark
python build_manifest.py --root /path/to/kaggle_eval
```

Discovery fails if a selected RGB frame, camera CSV, depth directory, or
tracking safetensor is missing. This keeps silent partial evaluations out of
the leaderboard.

## Prepare fixed tracking ground truth (maintainers)

Depth and pose GT are read directly from the raw dataset. Tracking needs one
preparation pass through the maintained Syn4D surface-track loader:

```bash
/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python prepare_tracking_gt.py \
  --converter /path/to/syn4d-kaggle/scripts/eval_open_d4rt/syn4d_to_worldtrack.py
```

This first creates ignored dense WorldTrack packs in `data/worldtrack_dense/`,
then fixed 512-query structured NPYs in `data/tracking_gt/`. The portable NPYs
are part of the benchmark release, so evaluation does not require raw
metadata; the command resumes any existing NPY output. To rebuild them from
existing dense packs without touching metadata, add `--skip-convert`; that path
also recovers source-resolution intrinsics from the dense pack and does not
need the converter. When this repository is checked out as `benchmark/` inside
the source kaggle repository, the converter path is discovered automatically.

On this cluster, run the full preparation as a four-variant CPU Slurm array:

```bash
mkdir -p results/slurm
sbatch --array=0-3 slurm/prepare_tracking_gt.sbatch
```

Set `SKIP_CONVERT=1` when all dense packs already exist. A complete preparation
must contain 512 NPY files; audit it with:

```bash
find data/tracking_gt -name '*.npy' | wc -l
```

For a standalone checkout, set `CONVERTER=/path/to/syn4d_to_worldtrack.py`
when submitting the conversion array. The released NPY files are already
checked in; regeneration is not required to evaluate tracking predictions.

## Evaluate Open-D4RT

GPU inference must be submitted through Slurm on this cluster. Start with a
one-sequence pose/depth integration smoke test:

```bash
mkdir -p results/slurm
TASKS=depth,pose DEPTH_GRID_SIZE=32 LIMIT=1 \
  OUTPUT="$PWD/results/opend4rt/smoke_predictions" \
  sbatch slurm/opend4rt.sbatch

# After the job completes:
/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python evaluate.py \
  --predictions results/opend4rt/smoke_predictions \
  --tasks depth,pose --limit 1 --strict \
  --output results/opend4rt/smoke_depth_pose.json
```

Tracking additionally requires the prepared fixed query GT. Its smoke job is:

```bash
TASKS=tracking LIMIT=1 OUTPUT="$PWD/results/opend4rt/smoke_predictions" \
  sbatch slurm/opend4rt.sbatch
```

Then run all tasks as a 16-worker array. Prediction files are disjoint because
array task `i` consumes manifest records `i, i+16, ...`; the runner resumes
existing files by default:

```bash
NUM_SHARDS=16 sbatch --array=0-15 slurm/opend4rt.sbatch

# After every array task completes:
/scratch/shared/beegfs/zeren/conda/envs/d4rt/bin/python evaluate.py \
  --predictions results/opend4rt/predictions \
  --tasks tracking,depth,pose --strict \
  --output results/opend4rt/summary.json
```

For unattended execution, submit the scorer with an `afterok` dependency on
the inference array:

```bash
MODEL=opend4rt sbatch --dependency=afterok:<inference-job-id> slurm/score.sbatch
```

Use `--variants`, `--scenes`, and `--cameras` to shard jobs. The default
Open-D4RT depth grid is 128×128 and is resized to GT resolution only for
scoring; increase `--depth-grid-size` for a resolution study.

## Evaluate 4RC

The adapter targets the official checkout pinned in
[adapters/4RC_REVISION](adapters/4RC_REVISION). It samples the released dense
world-space `track` field at the fixed benchmark queries, obtains depth from
the released `pts` point maps in each predicted camera, and consumes the
released camera-to-world `extrinsic` matrices.

Run one all-task smoke sequence through Slurm:

```bash
TASKS=tracking,depth,pose LIMIT=1 OUTPUT="$PWD/results/4rc/smoke_predictions" \
  sbatch slurm/4rc.sbatch

# After completion:
/scratch/shared/beegfs/zeren/conda/envs/4rc/bin/python evaluate.py \
  --predictions results/4rc/smoke_predictions \
  --tasks tracking,depth,pose --limit 1 --strict \
  --output results/4rc/smoke_all.json
```

Then use the same deterministic array sharding as Open-D4RT:

```bash
NUM_SHARDS=16 sbatch --array=0-15 slurm/4rc.sbatch
```

Likewise, use `MODEL=4rc` with `slurm/score.sbatch` after the 4RC array.

4RC and Open-D4RT intentionally use separate environments. The 4RC launcher
stores downloaded weights in the repository-level `.hf_cache/` and records the
model revision in every prediction NPZ.

## Additional 3D-tracking baselines

The benchmark includes adapters for the official V-DPM, Any4D,
TraceAnything, St4RTrack, and SpaTrackerV2 releases. They consume the same
fixed frame-0 queries and write the same canonical NPZ contract:

| Model | Tracking | Depth | Pose | Adapter |
| --- | ---: | ---: | ---: | --- |
| V-DPM | yes | yes | yes | `run_vdpm.py` |
| Any4D | yes | yes | yes | `run_any4d.py` |
| TraceAnything | yes | no | no | `run_traceanything.py` |
| St4RTrack | yes | no | no | `run_st4rtrack.py` |
| SpaTrackerV2 | yes | yes | yes | `run_spatrackerv2.py` |

“No” means the released inference interface does not expose that task in a
form compatible with this protocol; the adapter rejects it instead of
substituting ground truth or a different estimator.

Clone the official repositories and their submodules at the recorded commits:

```bash
python setup_baselines.py
```

Model environments and checkpoints remain separate because their CUDA/PyTorch
requirements conflict. On this cluster, prepare the isolated environments and
official checkpoints with CPU Slurm jobs:

```bash
for model in v-dpm any4d traceanything st4rtrack spatrackerv2; do
  MODEL="$model" sbatch slurm/setup_baseline_env.sbatch
done
```

The jobs create ignored environments under `external/envs/` and caches under
`.hf_cache/`; no GPU work runs during setup. Then submit inference through
Slurm with `PYTHON` pointing to that model's environment:

```bash
# One-sequence integration run
MODEL=traceanything PYTHON="$PWD/external/envs/traceanything/bin/python" LIMIT=1 \
  sbatch slurm/baseline.sbatch

# Full deterministic 16-way array, shown for SpaTrackerV2
MODEL=spatrackerv2 PYTHON="$PWD/external/envs/spatrackerv2/bin/python" NUM_SHARDS=16 \
  sbatch --array=0-15 slurm/baseline.sbatch

# Score only the tasks that adapter supports
MODEL=spatrackerv2 TASKS=tracking,depth,pose \
  sbatch slurm/score.sbatch
```

The generic launcher requests one GPU and 128 GB host memory. Adjust the Slurm
memory/time directives to local policy if needed; TraceAnything's official
release documents a 48 GB GPU for its examples. The adapters never launch GPU
work on a login node. Any4D keeps frame 0 in every forward pass and stitches
at most 16 views per pass in the frame-0 coordinate system; this is the
canonical policy used to fit a 32-frame sequence on a 48 GB GPU.

## Tests

```bash
python -m pytest -q
```

The tests cover perfect predictions, monocular scale invariance, invalid
tracking rows, pose Sim(3) invariance, portable tracking-NPY round trips, and
the baseline registry, including Any4D's anchor-preserving chunk stitching.
