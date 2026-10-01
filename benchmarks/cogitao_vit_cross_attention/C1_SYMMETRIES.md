# C1 experiment 1: rotation, translation, and latent symmetry

The scripts analyze the trained C1 experiment-1 ViT image encoder with its
function-token MLP and cross-attention decoder. They also support the separate
joint image/function-token ViT after that model has completed training. Neither
analysis model uses Slot Attention or Slot Attention weights.

COGITAO applies `rot90` to each object's bounding-box patch counterclockwise,
keeping its top-left anchor, and `translate_up` subtracts one from the anchor
row. The raw C1 grids are 15×15; the model loader resizes them to 20×20 with
nearest-neighbor sampling. The exact matrices operate on the raw grids.

For homogeneous object attributes `[row, column, height, width, 1]` as a column
vector, `rotate` swaps height and width and `translate_up` subtracts one from
row. The script saves both 5×5 matrices and their two products. Pixel rotation
depends on each object's bounding box, so the 225×225 pixel matrices are
sample-specific. They map flattened foreground color values to new positions.
The saved sample includes separate rotation/translation matrices and both
composed orders. Matrix order is right-to-left: rotate then translate is
`T @ R`.

The learned latent probe uses the model's image-token embeddings. It aligns
source and target foreground pixels with the exact pixel operator and averages
all 20×20 model tokens sampled from each raw cell. A separate regularized
affine channel operator is fitted to atomic `rot90` and `translate_up` **training**
pairs only. These operators are composed without fitting on OOD pairs. The
reported mean squared error is compared with the identity baseline on held-out
atomic and composition examples. For the joint-token ViT, all probe encodings
use the same `rot90` function-token context; the direct prediction metrics use
the actual ordered function tokens.

The released C1 experiment-1 OOD validation and test suites contain
`translate_up → rot90`. `rot90 → translate_up` does not occur in the released
train, validation, or test suites. The script generates that reverse-order
counterfactual from the same OOD inputs with exact matrices, checks that its
result equals the recorded OOD target, and scores the model with reverse-order
function tokens. This tests function-order behavior while holding input and
target grids fixed. Model accuracy on the counterfactual is therefore a model
test, not an independently sampled dataset estimate.

## Run the trained function-MLP model

From the repository root:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries \
  --model mlp \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt \
  --train-pairs 256 --eval-pairs 128 --batch-size 32 --device auto \
  --output-dir artifacts/cogitao_c1_symmetries/mlp
```

Train C1 experiment 1's independent joint-token ViT, then analyze the
checkpoint produced by that run:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.run_joint_tokens \
  --experiment 1

TRAINED_CKPT=checkpoints/cogitao_joint_token_vit/cogitao_c1_experiment_1_joint_token_vit/run_XXX/last.ckpt
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries \
  --model joint --checkpoint "$TRAINED_CKPT" \
  --train-pairs 256 --eval-pairs 128 --batch-size 32 --device auto \
  --output-dir artifacts/cogitao_c1_symmetries/joint
```

Replace `run_XXX` with the completed run directory. The script requires at
least 1,000 optimizer steps by default. Current local joint-token checkpoints
have only 1, 1, and 538 steps, so none supports a scientific comparison yet.
`--allow-untrained` is available solely to check plumbing.

Output files:

- `report.json`: geometry coverage, commutation, direct grid/pixel accuracy,
  held-out latent probe errors, and checkpoint training step.
- `attribute_operators.npz`: exact 5×5 attribute matrices and products.
- `sample_*.npz`: exact sparse pixel operators and one raw example.
- `fitted_latent_operators.npz`: learned affine channel maps and compositions.

Interpretation: object geometry predicts that the two operations commute on
valid inputs. A nonzero latent commutator or high OOD composition residual
indicates that this checkpoint's chosen image-token representation does not
realize that symmetry with a single, global affine channel map. It does not
prove that no nonlinear or spatially conditional latent representation exists.
The 15→20 resize and the model's learned position embeddings are possible
sources of equivariance error. The direct reverse-order model score helps
separate latent-probe limitations from actual task prediction failures.

Load and compose the saved sample matrices with:

```python
from pathlib import Path
import numpy as np
from scipy import sparse

root = Path("artifacts/cogitao_c1_symmetries/mlp")
attributes = np.load(root / "attribute_operators.npz")
assert np.array_equal(attributes["rotate_then_up"],
                      attributes["translate_up"] @ attributes["rotate"])

rotation = sparse.load_npz(root / "sample_rotate.npz")
up_after_rotation = sparse.load_npz(root / "sample_translate_up_after_rotate.npz")
composed = sparse.load_npz(root / "sample_rotate_then_up.npz")
assert (composed != up_after_rotation @ rotation).nnz == 0
```

The second step matrix depends on the intermediate image; using the matrix
computed on the original image for both steps can give a wrong product.
