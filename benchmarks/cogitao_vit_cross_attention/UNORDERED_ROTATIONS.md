# Rotation composition holdout accounting for commutativity

This page records the preceding whole-grid experiment. The current object-local
experiment matching original `rot90` is in
[OBJECT_ANCHOR_ROTATIONS.md](OBJECT_ANCHOR_ROTATIONS.md).

The current experiment starts from random weights and a fresh optimizer,
training atomic 15°, 30°, 45°, 60°, 75°, and 90° rotations jointly with selected
compositions. It uses **1×1 patches and batch size 512**, as requested.

A held-out combination is excluded in both orders. No trained pair can be a
reversal of a held-out pair. Resulting angles remain seen atomically or through
other trained factorizations, as chosen by the user. This tests new unordered
factorizations of known rotations; it does not test unseen total angles.

| Entire held-out combination (both orders) | Net angle | Seen equivalent |
|---|---|---|
| 15° + 45° | 60° | Atomic 60°, or 30° + 30° |
| 15° + 75° | 90° | Atomic 90°, or 45° + 45° |
| 30° + 60° | 90° | Atomic 90°, or 45° + 45° |
| 30° + 90° | 120° | 60° + 60° |
| 45° + 75° | 120° | 60° + 60° |
| 60° + 90° | 150° | 75° + 75° |

These six unordered combinations produce 12 held-out ordered programs. The
remaining 24 ordered pairs (15 unique unordered combinations, including six
diagonals) are trained. Each angle appears four times in each trained pair
position and twice in each held-out position. Atomic tasks are additionally
trained, so each training source produces 30 examples (6 atomic + 24 pairs).

| Source split | Source grids | Atomic examples | Seen-pair examples | Held-out-pair examples | Total |
|---|---:|---:|---:|---:|---:|
| Train | 11,111 | 66,666 | 266,664 | 0 | 333,330 |
| Validation | 111 | 666 | 2,664 | 0 | 3,330 |
| Test | 111 | 666 | 2,664 | 1,332 | 4,662 |

All splits total 341,322 task examples derived from 11,333 source grids.
Training has 11,111 examples per program; test has 111 per program. Validation
for checkpoint selection contains only atomic and trained-pair programs. Source
input hashes are disjoint across train, validation, and test.

The earlier run's 99.22% score used an order-only holdout. It establishes
reverse-order generalization, not unseen unordered-composition generalization.
Its checkpoints and metrics remain available with an interpretation note.

## Run

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_degree_rotations \
  --config config/cogitao/degree_rotation_unordered_holdout.yaml \
  --output runs/cogitao_degree_rotation_unordered_holdout/run_000
```

The canvas, sampler, architecture, learning rate, warmup schedule, epoch limit,
and patience follow the earlier whole-grid experiment: a padded 24×24 canvas,
fixed-center CCW rotations, nearest-neighbor sampling, width 128, six layers,
four heads, AdamW at 0.001, BF16 mixed precision, 200 warmup steps, up to 100
epochs, and early stopping patience 15. No pretrained initialization or group-law
loss is used. Targets render the original input once at the combined angle.
Both train and evaluation batch sizes are 512. An epoch has 652 optimizer steps,
versus 2,605 at batch size 128 for this dataset size.

The L4 benchmark is saved in
`artifacts/cogitao_degree_rotation_unordered_holdout/batch_benchmark.json`.
Synthetic optimizer-step throughput was about 888 examples/s at batch size 128
and 834 examples/s at 512; the latter used about 9.93 GiB reserved memory.
Larger batches therefore do not establish faster epochs on this GPU. The
benchmark excludes data loading and validation, and does not predict the
number of epochs needed to converge.

Configuration, initial random weights and their hash, split lists, data
manifest, best/latest checkpoints, CSV metrics, live status, and final report
are saved in the output directory. `runs/cogitao_degree_rotation_current.json`
identifies the latest active experiment.

W&B logging follows the user's existing approval for this and future runs:
training metrics and final numeric evaluation scores in
`cogitao-compgen-slot-attention`; no grids or checkpoints are uploaded.

## Evaluation

The scorer reports atomic, seen-combination, and held-out-combination accuracy;
direct-versus-summed and direct-versus-sequential agreement; reversed-order
agreement; and oracle-intermediate and pure raster-resampling controls. It
classifies groups using the dataset's actual split rather than the legacy
order-only constants. Net-angle diagnostics above 90° are still extrapolation
for the single-angle branch, because those angles were seen only through
compositions. All actual held-out composition effects are known during training.

The optional `effect` protocol additionally excludes every factorization of
105°, 135°, and 165°, but is not the user's chosen current experiment. The
legacy `order` protocol remains available solely to reproduce earlier runs.
