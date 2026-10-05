# Atomic ID first stage: translate up under a clockwise 90-degree symmetry

Implementation: [c1_atomic_equivariance.py](c1_atomic_equivariance.py).
Protocol checks: [test_c1_atomic_equivariance.py](../../tests/test_c1_atomic_equivariance.py).
The implementation and checks were written without executing them.

## Exact task isomorphism

Start with a recorded atomic ID example `y = T_up(x)`. Use a **whole-grid
clockwise** 90-degree rotation `g`, and transport both the input and task:

```text
original:      x   -- translate_up -->  y
               |                       |
               g                       g
               v                       v
transported:  g(x) -- translate_right -> g(y)
```

The induced task action is `alpha_g(T_up) = g T_up g^-1 = T_right`.
Consequently `g(T_up(x)) = T_right(g(x))`. Rotate first, then translate
**right** to reach `g(y)`. Keeping the literal up instruction after rotating
produces `T_up(g(x))`, a different task; it is an explicit control.

COGITAO's existing `rot90` token rotates each object counterclockwise in its
bounding box. It is not the symmetry `g` used here and is never inserted as a
function token in this experiment. The transformed branch explicitly receives
`translate_right`. This tests correspondence under task transport, rather than
whether the model autonomously changes an up instruction to right.

## Code and metrics

`read_atomic` reads experiment-1 atomic up/right rows, checks that their raw
recorded targets agree with one-cell translation, and rejects clipping,
wrapping, empty grids, and duplicate rotation orbits within each task. Fitting
uses only `train`. `val` and `test` exclude exact input grids and every rotation
or reflection in their dihedral orbit if the orbit was used for fitting. All filtering counts are reported.
Rows are sampled reproducibly using `--seed`; pair limits apply per task.

`LayerCollector` attaches temporary forward hooks to the unchanged pretrained
function-MLP model. It records every image Transformer block, the final image
normalization, every function-MLP module, the shared cross-attention block,
and the output cross-attention block. The **spatial image test** retains every
token entering the trained C1 ViT encoder after position embeddings, every
Transformer block, and its final normalization. A separate pooled analysis summarizes image blocks by their
foreground token mean, shared/output blocks by their query mean, and function
vectors directly. The spatial and pooled results have separate maps and metrics.

For spatial image layer `l`, let `H_l(x)[r,c,:]` be its feature vector at
position `(r,c)` on the checkpoint's 20-by-20 token grid. The code undoes the
clockwise rotation on `H_l(g(x))` to match every target token to the source
position, and fits an orthogonal **channel** map `Q_l` on atomic ID training
examples. In row-vector notation, the tested relation is:

```text
H_l(g(x))[g(r,c), :]  ~=  H_l(x)[r,c,:] @ Q_l
```

The reported spatial defect averages the per-example Frobenius norm of the
difference over **all positions and channels**, divided by the source norm.
It also reports the same ratio on the foreground union, an aligned identity
channel baseline, a baseline with no spatial permutation, map rank and
four-step closure. `input_rotation_resize_mismatch` gives the fraction of
input cells where raw rotation followed by nearest-neighbor resizing differs
from a rotation of the already resized input. This distinguishes preprocessing
effects from failures internal to the ViT.

The image encoder receives no task token, so this spatial test specifically
asks whether the trained ViT image representation realizes the rotation. The
pooled function/shared analysis separately asks whether the instructed task
changes from up to right inside the complete model.

## Pure group identities

The same atomic ID train/held-out split also fits a second per-layer channel
operator for a left-right reflection `M`. This reflection preserves the task:
`M(T_up(x)) = T_up(M(x))`, so the reflected branch uses the already-seen
`translate_up` token. After matching reflected token positions, the script
fits `rho_l(M)` independently at every trained ViT image layer.

For every image layer it reports both matrix-level and held-out data-relative
closure:

```text
rotation:    ||rho_l(R)^4 - I||_F / sqrt(D)
reflection:  ||rho_l(M)^2 - I||_F / sqrt(D)

data: mean_i ||H_l(x_i) rho_l(g)^order - H_l(x_i)||_F / ||H_l(x_i)||_F
```

Here `order=4` for rotation and `order=2` for reflection. Low one-step
equivariance defect together with low closure defect is the intended evidence
that a layer realizes the spatial group action. Each identity is measured at
the encoder input tokens, every Transformer block, and final normalization.
The report also records reflection/resize mismatch, analogous to the existing
rotation/resize diagnostic.

For each layer, `fit_orthogonal` learns an **uncentered**, bias-free orthogonal
Procrustes map from paired atomic training representations:

```text
X = H_l(x, translate_up)
Y = H_l(g(x), translate_right)
Q_l = argmin_Q ||X Q - Y||_F, subject to Q.T Q = I
```

If `X.T @ Y = U S V.T`, the saved row-vector map is `Q_l = U @ V.T`.
The screenshot's column-vector operator `rho_l(g)` is `Q_l.T`.
The held-out metric is computed per example and then averaged:

```text
E_eq(l) = mean_i ||Y_i - X_i Q_l||_2 / ||X_i||_2
```

The pooled report also includes the identity-map defect, median, 90th percentile,
standard error, and zero-norm coverage. A direction readout is independently
fitted to native atomic up/right training examples **if both are present**, then frozen. Its reported
metrics are native held-out balanced accuracy, right accuracy on transformed
representations, and right accuracy on `X Q_l`. The unchanged-up control on
rotated inputs also receives a direction readout where its up target is valid.
No direction readout is fitted on image-only layers, which receive no task.
If the C1 ID split contains no native right examples, the experiment still
fits and tests the spatial maps and right-task output; it records zero native
right coverage and omits the direction readout. The right task token is still
supplied explicitly on the transported branch.

Output checks compare actual predicted grids with the exact resized targets:

- Original up prediction against `y`.
- Transported right prediction against `g(y)`.
- Native up/right predictions on held-out ID rows.
- On inputs where both translations are valid, right prediction against the
  incorrect up target, and unchanged-up prediction against both targets.

Exact-grid, all-pixel, and foreground-union pixel accuracies are fractions in
`[0, 1]`. No output accuracy is assumed in advance.

For induced group actions, the code reports `||Q_l^4 - I||_F / sqrt(D_l)`
and data-relative four-step closure defects. These are diagnostics: uncentered
Procrustes does not impose `Q^4 = I`, and the up-to-right pair alone does not
validate every C4 task action. Function vectors are constant within each task,
making their learned maps underidentified; each layer reports the fitting
source rank. Low function-layer defect therefore has limited significance.

Geometry operates on the raw square grids. Model inputs and labels use the
checkpoint's existing nearest-neighbor resize. The 15-to-20 resize need not
commute with rotation, so this preprocessing can contribute to latent defects.

## Run commands

From the project root, fit the ID probes once:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_atomic_equivariance \
  --mode fit \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt \
  --data-root data/cogitao/files/CompGen \
  --output-dir artifacts/cogitao_c1_atomic_equivariance \
  --train-pairs 256 --batch-size 16 --seed 42 --device auto
```

Evaluate held-out atomic ID pairs with those frozen probes:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_atomic_equivariance \
  --mode eval \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt \
  --data-root data/cogitao/files/CompGen \
  --output-dir artifacts/cogitao_c1_atomic_equivariance \
  --probe-file artifacts/cogitao_c1_atomic_equivariance/frozen_probes.npz \
  --splits val test --eval-pairs 128 --batch-size 16 --seed 42 --device auto
```

Alternatively use `--mode all` in a fresh output directory for fit followed by
evaluation in a single process. Existing probe files cannot be overwritten by
fit/all. Evaluation checks the checkpoint SHA-256 against the saved probe
metadata. Both commands require the existing project dependencies and a trained
C1 experiment-1 function-MLP checkpoint; the helper requires at least 1,000
training steps.

## W&B logging

W&B is optional and disabled by default. It uses the existing
`cogitao-compgen-slot-attention` project and the
`cogitao-c1-atomic-equivariance` group; use `--wandb-project`,
`--wandb-entity`, and `--wandb-group` to select another destination.
`--wandb-mode offline` writes a local W&B run for later sync.
`--wandb-mode online` sends the run to the selected W&B project. Nothing is
sent by the commands above unless the mode is changed.

For one run that fits the maps and logs evaluation on `val` and `test`:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_atomic_equivariance \
  --mode all --output-dir artifacts/cogitao_c1_atomic_equivariance_wandb \
  --wandb-mode online --wandb-project cogitao-compgen-slot-attention \
  --train-pairs 256 --eval-pairs 128 --batch-size 16 --seed 42 --device auto
```

The output directory must be fresh for `--mode all`. To log the separate
fit/eval commands above instead, add `--wandb-mode online` to both. They form
two W&B runs in the same group, with the same checkpoint hash and probe path.
The fit run logs selected counts, geometry coverage, and per-layer fit rank.
The eval run logs every numeric report metric under split/layer paths,
including defect summaries, identity and four-step closure diagnostics,
two-step reflection closure, direction readouts, grid accuracies, and coverage.
It also logs a W&B table
with separate pooled and spatial rows for each applicable split/layer. The
checkpoint hash and local report/probe paths
are stored in run metadata.

By default, W&B receives these summary metrics and the compact layer table.
Add `--wandb-log-artifacts` if the selected project should also receive
`frozen_probes.npz`, the report, both layer-metric CSVs, and saved example arrays.
`wandb==0.28.2` is already in the project requirements. This integration has
not been executed.

Optional protocol checks, also not executed during implementation:

```bash
PYTHONPATH=. pytest -q tests/test_c1_atomic_equivariance.py
```

## Artifacts and scope

- `frozen_probes.npz`: pooled and spatial orthogonal maps and direction readouts, fitting input
  dihedral-orbit hashes and source indices, checkpoint hash, and protocol metadata.
- `fit_report.json`: fit coverage, rank, and checkpoint information.
- `report.json`: per-layer and output results for the held-out splits.
- `layer_metrics.csv`: pooled metrics by split/layer.
- `spatial_layer_metrics.csv`: full-token spatial metrics by split/ViT image layer.
- `spatial_group_identity_metrics.csv`: compact per-layer `R^4=I` and `M^2=I`
  operator and held-out data closure metrics.
- `val_examples.npz`, `test_examples.npz`: source/transformed representations,
  individual pooled and spatial defects, raw source/target grids, prediction grids,
  and source indices. Control grids carry separate pair and source indices.

This first set of tests deliberately uses atomic ID examples. Rotated copies
are symmetry counterfactuals sourced from ID; they are not the released C1
composition OOD set. The saved maps can be used in a later OOD protocol, after
specifying task transport for every operation in each composition. In particular,
the object-local `rot90` anchor convention prevents blindly substituting the
same token under whole-grid rotation. The present script accepts only `val`
and `test` and does not make OOD composition claims.
