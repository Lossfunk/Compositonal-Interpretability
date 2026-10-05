# C1 ViT image-only spatial symmetries

The implementation is `c1_vit_spatial_symmetries.py`. It calls the trained
checkpoint's `encode_images` method directly. It never constructs function
tokens and never calls the function MLP, cross-attention modules, decoder,
target path, or prediction path. `translate_up` is used only as dataset metadata
to select one consistent atomic ID image population.

For each image, the script creates a clockwise 90-degree rotation `R(x)` and a
left-right reflection `M(x)`. It captures the complete 20-by-20 token map at:

- the tokens entering the ViT after position embeddings;
- every trained Transformer image block;
- the encoder's final normalization.

For every layer and symmetry, transformed token positions are aligned back to
their source positions. An uncentered orthogonal Procrustes map is fitted only
on ID `train` images:

```text
rho_l(g) = argmin_Q ||align(H_l(gx)) - H_l(x) Q||_F,  Q.T Q = I
```

The frozen map is evaluated on held-out ID `val` and `test` images. Reported
metrics include all-token and foreground equivariance defects, identity-channel
and missing-spatial-permutation baselines, fit rank, and preprocessing mismatch.
Pure group identities are checked both on the operator and held-out features:

```text
rho_l(R)^4 ~= I
rho_l(M)^2 ~= I
```

Train/evaluation leakage is prevented at the full dihedral orbit level: if a
training image, any of its rotations, or any reflected rotation was used for
fitting, the entire orbit is excluded from evaluation.

## Run

From the repository root, run the full fit/evaluation with W&B:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_vit_spatial_symmetries \
  --mode all \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt \
  --data-root data/cogitao/files/CompGen \
  --output-dir artifacts/cogitao_c1_vit_spatial_symmetries \
  --train-images 256 --eval-images 128 --splits val test \
  --batch-size 16 --seed 42 --device auto \
  --wandb-mode online --wandb-project cogitao-compgen-slot-attention
```

Add `--wandb-log-artifacts` to upload the frozen operators, report, CSV, and
per-example arrays. Use `--wandb-mode disabled` for local-only evaluation.

Protocol checks:

```bash
PYTHONPATH=. pytest -q tests/test_c1_vit_spatial_symmetries.py
```

The main table is `spatial_symmetry_metrics.csv`. Each row contains one split,
symmetry, and trained ViT layer. `report.json` has the complete metric tree,
and `val_spatial_examples.npz`/`test_spatial_examples.npz` contain individual
defects. The implementation and checks were written without being executed.
