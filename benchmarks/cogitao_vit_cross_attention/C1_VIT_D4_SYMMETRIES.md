# C1 trained-ViT D4 symmetry probe

`c1_vit_d4_symmetries.py` measures spatial symmetry in the image embeddings of
each trained C1 ViT checkpoint. It calls `model.encode_images` directly. It does
not construct or pass function tokens, and it does not call the function MLP,
cross-attention modules, output decoder, targets, or model predictions.

The probe supports C1 experiments 1 through 5. For each experiment it reads all
atomic function names from that experiment's setting-1 parquet metadata and
samples the same number of input images per function. The name is only a
stratification label. The ViT receives the image alone.

## Spatial actions

The code applies all eight elements of the square's dihedral group:

| Element | Image action |
| --- | --- |
| `e` | identity |
| `r90` | 90 degrees clockwise |
| `r180` | 180 degrees |
| `r270` | 270 degrees clockwise |
| `m` | left-right reflection |
| `r90_m` | reflection followed by 90 degrees clockwise |
| `r180_m` | reflection followed by 180 degrees |
| `r270_m` | reflection followed by 270 degrees clockwise |

This is a spatial-only analysis. Color permutations mentioned for C1-2 and C1-4
are intentionally outside this probe.

For every image layer, including the patch-plus-position input, each Transformer
block, and final layer norm, an independent orthogonal Procrustes matrix
`rho_l(g)` is fitted on atomic training images. Before fitting, the transformed
token map is moved back to the source spatial coordinates. The measured defect
is therefore

```text
|| align_g(H_l(g x)) - H_l(x) rho_l(g) || / || H_l(x) ||.
```

The validation and test images are excluded when they share a complete D4 orbit
with a fitting image. The report includes the all-token defect, foreground-only
defect, identity-channel baseline, no-spatial-alignment baseline, and the group
order closure for every element. It also checks all 64 D4 composition laws per
layer:

```text
rho_l(g) rho_l(h) ~= rho_l(h after g).
```

This includes `rho(R)^4 ~= I`, `rho(R^2)^2 ~= I`, every reflection square, and
the mixed rotation/reflection relations. Both operator-space and held-out
embedding-space errors are reported.

## Run one C1 experiment

The checkpoint path is inferred from `--experiment`.

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries \
  --experiment 1 \
  --mode all \
  --train-images-per-function 128 \
  --eval-images-per-function 128 \
  --batch-size 16 \
  --device auto \
  --wandb-mode online \
  --wandb-project cogitao-compgen-slot-attention \
  --wandb-group cogitao-c1-vit-d4 \
  --wandb-log-artifacts
```

To use a different checkpoint, add `--checkpoint /path/to/last.ckpt`. To keep
the W&B run local, use `--wandb-mode offline`; to disable W&B, use
`--wandb-mode disabled`.

## Run all five C1 experiments

```bash
for experiment in 1 2 3 4 5; do
  PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries \
    --experiment "$experiment" \
    --mode all \
    --train-images-per-function 128 \
    --eval-images-per-function 128 \
    --batch-size 16 \
    --device auto \
    --wandb-mode online \
    --wandb-project cogitao-compgen-slot-attention \
    --wandb-group cogitao-c1-vit-d4 \
    --wandb-log-artifacts
done
```

Each run writes to `artifacts/cogitao_c1_experiment_N_vit_d4/`:

- `frozen_d4_probes.npz`: fitted per-element, per-layer orthogonal matrices and
  fit metadata.
- `report.json`: full metrics, sampling coverage, D4 table, and protocol flags.
- `d4_element_metrics.csv`: compact per-action, per-layer defects.
- `d4_group_law_metrics.csv`: all 64 composition checks at every layer.
- `val_d4_examples.npz` and `test_d4_examples.npz`: per-example defects.

The command refuses to overwrite an existing frozen probe. Use a fresh
`--output-dir`, or reuse it with evaluation mode:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries \
  --experiment 1 \
  --mode eval \
  --output-dir artifacts/cogitao_c1_experiment_1_vit_d4 \
  --wandb-mode online
```

## Checks

```bash
PYTHONPATH=. pytest -q \
  tests/test_c1_vit_d4_symmetries.py \
  tests/test_c1_vit_spatial_symmetries.py
```
