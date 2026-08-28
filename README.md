# Syn4D multi-task benchmark

This directory defines one reproducible benchmark over **every complete
sequence** in `/work/kelvin/Syn4D/subsets/kaggle_eval`. It evaluates three
outputs from the same 32-frame input clip:

1. 3D point tracking;
2. monocular video depth;
3. camera pose.

The checked-in [manifest](manifests/syn4d_all.jsonl) contains 512 sequences:
4 render variants × 8 scenes × 8 base sequences × 2 cameras. The
`even_camera_png` convenience mirror is not indexed because it duplicates RGB
files and does not own independent ground truth.

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

## 1. Rebuild or audit the manifest

```bash
cd /path/to/Syn4D_benchmark
python build_manifest.py
```

Discovery fails if a selected RGB frame, camera CSV, depth directory, or
tracking safetensor is missing. This keeps silent partial evaluations out of
the leaderboard.

## 2. Prepare fixed tracking ground truth

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

## 3. Evaluate Open-D4RT first

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

## 4. Evaluate 4RC second

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

## 5. Additional 3D-tracking baselines

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
requirements conflict. Follow each official repository's installation and
checkpoint instructions, then submit inference through Slurm. `PYTHON` must
point to that model's environment:

```bash
# One-sequence integration run
MODEL=traceanything PYTHON=/path/to/traceanything/bin/python LIMIT=1 \
  sbatch slurm/baseline.sbatch

# Full deterministic 16-way array, shown for SpaTrackerV2
MODEL=spatrackerv2 PYTHON=/path/to/spatrackerv2/bin/python NUM_SHARDS=16 \
  sbatch --array=0-15 slurm/baseline.sbatch

# Score only the tasks that adapter supports
MODEL=spatrackerv2 TASKS=tracking,depth,pose \
  sbatch slurm/score.sbatch
```

The generic launcher requests one GPU and 128 GB host memory. Adjust the Slurm
memory/time directives to local policy if needed; TraceAnything's official
release documents a 48 GB GPU for its examples. The adapters never launch GPU
work on a login node.

## Tests

```bash
python -m pytest -q
```

The tests cover perfect predictions, monocular scale invariance, invalid
tracking rows, pose Sim(3) invariance, portable tracking-NPY round trips, and
the baseline registry.
