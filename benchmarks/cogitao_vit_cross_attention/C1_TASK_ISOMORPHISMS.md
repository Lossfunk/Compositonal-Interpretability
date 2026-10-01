# C1 rotation and translation: task-isomorphism geometry

This study uses the idea of testing structurally equivalent task variants from
[Schug and Lake, *Thought without systematicity?* (2026)](https://arxiv.org/abs/2609.13948).
It adapts that idea to the simple C1 transformation square below. It does not
reproduce the paper's rule-induction tasks or its evaluation protocol.

```text
                    rot90
        x --------------------> R(x)
        |                         |
  translate_up               translate_up
        |                         |
        v          rot90          v
       T(x) -----------------> T(R(x))
```

For valid objects, the bottom-right image is also `R(T(x))`. The script records
whether both paths are defined and whether their exact raw grids agree. Rotation
can leave the grid before translation would make room; those inputs remain in
the atomic representation data but are excluded from a two-path isomorphism
comparison. Exact transformations use the 15×15 raw C1 grids; each grid is
nearest-neighbor resized to 20×20 before the model receives it.

## Representations and comparisons

- **Image ViT chart:** foreground-token mean of the pretrained ViT encoder for
  `x`, exact `R(x)`, exact `T(x)`, exact `T(R(x))`, and exact `R(T(x))`. The script
  includes every atomic `rot90` and `translate_up` row from C1 experiment-1
  `train` and `val`. All available exact states from these rows fit one PCA.
- **Latent composition:** separate affine operators `A_R` and `A_T` are fitted
  on the corresponding *recorded atomic training pairs only*. Row-vector
  products `A_R @ A_T` and `A_T @ A_R` act on the ViT image vector for `x`.
  Validation rows are never used to fit these operators.
- **Full ViT + function MLP:** for each input, the unchanged checkpoint runs
  with `[rot90]`, `[translate_up]`, and `[rot90, translate_up]` function tokens.
  Its shared latents form a second, separately fitted PCA chart. The direct
  composition prediction is encoded again by the ViT and projected into the
  **same image PCA basis** as the exact output and latent composition.
- **Sequential model baseline:** the model's predicted rotation is passed back
  to the model with `[translate_up]`; that output is also re-encoded by the ViT.
- **Function MLP chart:** the five fixed MLP vectors for empty, `R`, `T`,
  `R→T`, and `T→R` are shown in their own PCA basis. Image, shared, and function
  vectors should not be overlaid as if they had a common coordinate system.

PCA is only for visualization. The report measures transformation-square edge
consistency, composition recovery, operator commutation, and exact-grid
accuracy in the original latent dimensions. The plots highlight up to 24 valid
validation squares for legibility; the PCA fit, point clouds, and saved arrays
use **all** eligible train and validation rows.

## Run

From the project root, using the completed experiment-1 checkpoint:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.plot_c1_task_isomorphisms \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt \
  --data-root data/cogitao/files/CompGen \
  --output-dir artifacts/cogitao_c1_task_isomorphisms/last \
  --batch-size 16 --ridge 1.0 --square-examples 24 --device auto
```

For the earlier OOD-validation-loss checkpoint, use the same command with:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.plot_c1_task_isomorphisms \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/epoch=021.ckpt \
  --output-dir artifacts/cogitao_c1_task_isomorphisms/epoch_021 \
  --batch-size 16 --ridge 1.0 --square-examples 24 --device auto
```

The script requires the project's existing PyTorch, PyArrow, SciPy, NumPy, and
Matplotlib dependencies. It does not train or change the checkpoint. Full-data
inference and 3D plotting can take substantial time; `--batch-size` controls
GPU memory without changing which examples are used.

## Output

- `exact_square_2d.png`, `exact_square_3d.png`: all exact state points and
  highlighted validation transformation squares.
- `recovery_2d.png`, `recovery_3d.png`: exact composition, fitted latent
  composition, full-model composition, and sequential-model output in the
  **same image PCA basis**.
- `atomic_recovery_2d.png`, `atomic_recovery_3d.png`: exact rotation and
  translation compared with the ViT+MLP atomic predictions re-encoded by ViT.
- `shared_2d.png`, `shared_3d.png`: task-conditioned shared latents.
- `function_mlp_2d.png`, `function_mlp_3d.png`: fixed task vectors in their
  separate PCA chart.
- `representations.npz`, `pca_coordinates.npz`, `pca_bases.npz`: full-width
  vectors, every projected point, PCA means/components/explained variance.
- `latent_operators.npz`, `model_prediction_grids.npz`, `examples.csv`, and
  `report.json`: fitted maps, direct/sequential predictions, row provenance,
  coverage and full-dimensional recovery metrics.

The code is in `plot_c1_task_isomorphisms.py`. This file and the script were
written without running the analysis.
