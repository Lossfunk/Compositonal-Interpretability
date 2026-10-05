# COGITAO C1: direct patch transformation

This is a standalone, non-Slot-Attention C1 model. It logs to the existing
W&B project `cogitao-compgen-slot-attention` so its results remain beside the
previous runs, but uses distinct direct-decoder run names, tags, and checkpoint
directories.

## Architecture

The model receives only the native `20x20` one-hot input grid. A learned `1x1`,
stride-1 patch projection produces 400 tokens of width 128, and learned position
embeddings preserve cell identity. Six pre-LN self-attention decoder blocks use
four heads and 512-wide GELU feed-forward layers. A per-token linear output head
then predicts one of the ten grid symbols for every output cell.

```text
20x20 one-hot input
  -> 1x1 patch embeddings + learned positions
  -> 6-layer self-attention decoder
  -> per-patch categorical head
  -> 20x20 output logits
```

There are no function embeddings, task-token MLPs, learned shared queries,
cross-attention stages, or other functional conditioning. The C1 loaders do not
return task tokens. Transformation-suite metadata is retained only for logging
and diagnostics and is never passed through the model.

The output head also supports patch sizes larger than one by predicting all
categorical cells within each patch and unpatchifying them back to the configured
input resolution. The C1 launcher uses patch size 1.

## Checks and a single run

```bash
# Build the model and all C1 experiment-1 loaders without training.
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.run \
  --experiment 1 --dry-run

# Start one training run.
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.run \
  --experiment 1
```

## W&B C1 sweep

Register the five-experiment grid in the existing project:

```bash
wandb sweep \
  --project cogitao-compgen-slot-attention \
  benchmarks/cogitao_vit_cross_attention/sweep_setting1.yaml
```

Use the agent path printed by W&B in one or more terminals:

```bash
WANDB_AGENT_DISABLE_FLAPPING=true \
WANDB_AGENT_MAX_INITIAL_FAILURES=100 \
wandb agent --forward-signals \
  ENTITY/cogitao-compgen-slot-attention/SWEEP_ID
```

Do not add `--count 1` if a worker should automatically claim the next C1
experiment after its current run finishes or is stopped through W&B.

## Atomic C1 evaluation with the function-conditioned checkpoint

The evaluator restores the original ViT + function MLP + cross-attention model
from `function_conditioned_model.py`, loads the corresponding trained C1
checkpoint, and evaluates only one-function rows in `val`, `val_ood`, `test`,
and `test_ood`.

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 \
  --experiment 1 \
  --checkpoint checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt
```

To evaluate all five C1 experiments against their own function-conditioned
checkpoints:

```bash
for experiment in 1 2 3 4 5; do
  PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 \
    --experiment "$experiment"
done
```

Each run writes `metrics.csv` and `report.json` under
`artifacts/cogitao_c1_function_conditioned_atomic/experiment_N/`. Metrics
include exact-grid, all-pixel, and object-pixel accuracy plus cross entropy,
grouped by function, affected attribute, and observed fill color. An attribute
can include several functions, and a function can affect several attributes.
The parquet rows have no explicit function argument field; the fill-color group
is inferred from background-to-color changes in the input and target grids.

The released C1 `val_ood` and `test_ood` splits contain only two-function
compositions. This evaluator excludes them and records zero atomic coverage for
those splits; it does not report an accuracy from composition rows.

## One wide CSV across all C1 experiments

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.eval_atomic_c1_all \
  --output artifacts/cogitao_c1_function_conditioned_atomic/all_experiments.csv
```

The CSV has one column per exact atomic function (`rot90`, `pad_left`,
`pad_right`, `change_shape_color`, and so on). Rows are identified by `split`
and `metric`; the default `experiment=all` rows pool atomic results across the
five corresponding function-conditioned checkpoints. The `samples` metric
shows how many atomic rows contributed to each function. `val_ood` and
`test_ood` have zero atomic samples and blank accuracy/loss cells because the
released C1 OOD splits contain only compositions. Function attributes such as
left and right remain separate through their named columns (`pad_left` and
`pad_right`). The parquet files do not record additional function arguments.

Pass `--include-experiments` to add rows for each individual C1 checkpoint to
the same CSV.

## Function-conditioned ViT composition report

Run the trained ViT + function MLP + cross-attention checkpoints on every
ordered two-function composition in the C1 validation and test ID/OOD splits.
Atomic functions are evaluated on the ID validation split only. The default
uses the `run_000/last.ckpt` checkpoint for each of the five experiments.

```bash
python -m benchmarks.cogitao_vit_cross_attention.eval_c1_compositions \
  --batch-size 64 --device auto
```

To evaluate one experiment or choose another checkpoint run:

```bash
python -m benchmarks.cogitao_vit_cross_attention.eval_c1_compositions \
  --experiment 1 --run run_000 --batch-size 64 --device auto
```

The command writes three CSVs under
`artifacts/cogitao_c1_function_conditioned_compositions/`:

- `compositions.csv`: each ordered pair, with sample counts, exact-grid and
  pixel accuracies, object-pixel accuracy, and cross entropy for `val_id`,
  `val_ood`, `test_id`, and `test_ood`.
- `functions_val.csv`: the same metrics for each atomic function and all
  atomic functions combined, using only the `val` split.
- `coverage.csv`: the number of atomic and composition rows in each split.

The `experiment=all` rows pool raw results from the five corresponding
checkpoints. Blank accuracy cells mean that composition does not occur in that
split; `samples=0` records the same absence explicitly. Composition order is
preserved, so `A -> B` and `B -> A` remain distinct.

## Joint image/function-token ViT for C1

This is an independent ViT trained from scratch. It takes the native `20x20`
one-hot input grid as `1x1` image patches and appends up to two ordered
function tokens. Six pre-LN Transformer layers jointly encode image and
function tokens with width 128, four attention heads, and a 512-wide GELU
feed-forward layer. A padding mask excludes the empty second function token on
atomic tasks. A categorical head decodes the image-token embeddings into the
transformed grid. It does not instantiate Slot Attention or load its weights.

The standalone training config is `config/cogitao/joint_token_vit_c1.yaml`.
It uses the existing W&B project `cogitao-compgen-slot-attention`, trains each
C1 experiment for 10 epochs with batch size 64, AdamW at 1e-3 learning rate
and weight decay, 200 warmup steps, and cosine decay. Validation ID and OOD
are logged separately; the lowest OOD validation cross entropy selects the
checkpoint for final ID/OOD test evaluation.

Build the model and loaders without training:

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_joint_tokens \
  --experiment 1 --dry-run
```

Train C1 experiments 1 through 5 sequentially:

```bash
for experiment in 1 2 3 4 5; do
  python -m benchmarks.cogitao_vit_cross_attention.run_joint_tokens \
    --experiment "$experiment"
done
```

Register the five-job sweep once, then run the printed agent path in each
worker session:

```bash
wandb sweep \
  --project cogitao-compgen-slot-attention \
  benchmarks/cogitao_vit_cross_attention/sweep_joint_tokens_c1.yaml

WANDB_AGENT_DISABLE_FLAPPING=true \
WANDB_AGENT_MAX_INITIAL_FAILURES=100 \
wandb agent --forward-signals \
  ENTITY/cogitao-compgen-slot-attention/SWEEP_ID
```

Checkpoints go to `checkpoints/cogitao_joint_token_vit/`; W&B logs use the
`compgen-c1-joint-token-vit` group.

## C1 symmetry analysis

For exact object/attribute matrices, learned latent operators, direct model
scores on both rotation/translation orders, and the measured OOD structure,
see [C1_SYMMETRIES.md](C1_SYMMETRIES.md) and
[C1_SYMMETRY_FINDINGS.md](C1_SYMMETRY_FINDINGS.md).

## C1 task-isomorphism PCA

The all-atomic-train-and-validation 2D/3D latent geometry study, exact
rotation/translation squares, and run commands are in
[C1_TASK_ISOMORPHISMS.md](C1_TASK_ISOMORPHISMS.md).

## Progressive atomic evaluation sweep

The rot90 and translate_up checkpoint-progression evaluator, W&B sweep status,
run commands, and local metrics are in
[C1_ATOMIC_PROGRESSION.md](C1_ATOMIC_PROGRESSION.md).

## Atomic ID task equivariance

For the image-only per-layer study of clockwise rotation, left-right reflection,
and the pure identities `rho(R)^4 ~= I` and `rho(M)^2 ~= I`, see
[C1_VIT_SPATIAL_SYMMETRIES.md](C1_VIT_SPATIAL_SYMMETRIES.md). This path calls
the trained ViT image encoder directly and does not use function tokens or the
decoder.
