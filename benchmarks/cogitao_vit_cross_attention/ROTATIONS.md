# Rotation-only ViT training

Train a fresh angle-conditioned ViT before studying composition or latent
homomorphisms:

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_rotations --dry-run
python -m benchmarks.cogitao_vit_cross_attention.run_rotations \
  --output runs/cogitao_rotation_token_vit/run_000
```

The config is `config/cogitao/rotation_token_vit.yaml`. The encoder reuses the
existing joint image/function-token ViT: 20×20 one-hot grids, 1×1 patches,
width 128, six layers, four attention heads, and a categorical grid decoder.
A separate four-angle vocabulary supplies the requested rotation. The angles
are 0°, 90°, 180°, and 270° counterclockwise. Identity (0°) is an explicit task
with an active token. The second token slot is padded during this phase.
No group-law regularizer or homomorphism objective is used.

The initial run uses setting 1, experiment 1. Only source rows with the exact
suite `[rot90]` are used. The existing object oracle generates a four-state
orbit from each raw input: each object's bounding-box patch rotates while its
top-left anchor stays fixed. This is the COGITAO object transformation, not a
whole-canvas rotation. The recorded 90° target must agree exactly with the
oracle. Orbits with clipping or overlap are excluded. Input and targets are
then resized to 20×20 using the existing nearest-neighbor convention. Repeated
rotation states retain object identities rather than re-inferring components.

Angles are balanced: each source contributes all four atomic tasks. Rotation
pairs in the original parquet files are excluded. Duplicate model-visible
inputs are removed within splits and against preceding splits (train, then
validation, then test). The initial audit retained 11,087 training sources
(44,348 examples), 111 validation sources (444 examples), and 111 test sources
(444 examples), with no oracle mismatches or cross-split input duplicates.
24 training sources had invalid orbits.

Training uses AdamW, learning rate 0.001, weight decay 0.001, 200-step warmup,
cosine decay, batch size 64, BF16 mixed precision, at most 100 epochs, and early
stopping after 15 epochs without better atomic validation cross entropy.
Checkpoint selection uses only atomic validation loss. The trainer writes local CSVs.
The active `run_000` is also mirrored to W&B project
`cogitao-compgen-slot-attention` by a separate metric relay, enabled after approval.
It backfills saved epochs and streams future metrics and final per-angle scores
without restarting training. It does not upload grid data or checkpoints.
The W&B run URL and backfill progress are saved in `wandb.json`.

Each run saves:

- `config.json` and `data_manifest.json`: configuration, retained row indices,
  hashes of model-visible inputs, and filtering counts.
- `status.json` and `metrics/version_0/metrics.csv`: progress and epoch metrics.
- `checkpoints/best.ckpt` and `checkpoints/last.ckpt`: best atomic validation
  checkpoint and resumable training state.
- `report.json`: after training, validation and test metrics for each angle,
  including exact-grid, pixel, foreground-union accuracy, cross entropy, and
  identity/background baselines. Accuracy values are percentages.

To resume an interrupted run, use the same config and overrides:

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_rotations \
  --output runs/cogitao_rotation_token_vit/run_000 \
  --ckpt-path runs/cogitao_rotation_token_vit/run_000/checkpoints/last.ckpt
```

For a smaller exploratory run, pass `--override data.train.init_args.max_sources=2048`
and use a fresh output directory. Sampling is seeded. The default uses every
valid source.

## Later compositional experiment

This phase trains **zero two-angle programs**. Every pair is therefore unseen
as a program at this stage, even when its summed angle was seen atomically.
A later phase can add a fixed subset of ordered pairs to training and hold out
other pairs; the atomic baseline checkpoint should remain available. Split
pairs by the ordered sequence, and report the resulting angle separately:
90°→180° and 180°→90° are distinct programs with the same 270° outcome.

For example, train on (90°,90°), (180°,180°), and (270°,270°), then hold out
(90°,180°), (180°,90°), (90°,270°), (270°,90°), (180°,270°), and (270°,180°).
This is a proposed future split, not data used in this run. Direct two-token
prediction and sequential application of atomic predictions should be scored
separately. The second task position is not trained in the atomic baseline,
so its direct two-token results cannot be treated as results from a model
trained on some compositions. Homomorphism analysis follows this evaluation.

Native rotation identities should be audited before resizing; the discrete
resize operation need not commute with rotations. A later arbitrary-angle
experiment also needs an explicit pivot, canvas, interpolation, and clipping
convention, since resampled grid rotations do not obey exact angle-addition
identities in general.

To attach the same optional W&B metric relay to another local run:

```bash
python -m benchmarks.cogitao_vit_cross_attention.rotation_wandb \
  --output runs/cogitao_rotation_token_vit/run_000
```

Run only one relay per training run. Reusing its output directory resumes the
same W&B run and skips already logged steps.

The selected-composition fine-tuning experiment and balanced seen/held-out
pair protocol are implemented in [ROTATION_COMPOSITIONS.md](ROTATION_COMPOSITIONS.md).
