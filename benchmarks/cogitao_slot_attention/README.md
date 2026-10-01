# COGITAO CompGen with Slot Attention

This setup downloads and exposes all five COGITAO compositional-generalization
settings, all five experiments per setting, and their `train`, `val`,
`val_ood`, `test`, and `test_ood` splits. The source snapshot is pinned to
commit `25e347f9873acbaba98bdaef7e64ba430a2890c2`.

The launcher uses `config/slot_attention/clevr_2d_slot_attention.yaml` directly.
It does not modify or copy the hyperparameter file. At runtime it selects the
task-conditioned model and injects dataset, split-label, run-name, and
checkpoint wiring only. Additional `--override` flags are applied last, so
local experiments can still be tinkered with from the command line.

COGITAO CompGen grids remain at their native `20x20` resolution from input
through categorical output and metric computation. Following the paper's
backbone/tokenization setup, the COGITAO launcher applies these runtime-only
overrides before Slot Attention: `1x1` stride-1 cell patches, embedding width
128, six Transformer encoder layers, four attention heads, and a 512-wide GELU
feed-forward block. This yields 400 ordered grid tokens; the ordered task tokens
are appended before Slot Attention. A two-stage spatial broadcast path
(`5 -> 10 -> 20`) produces the native-sized categorical output. No padding
class is required because these CompGen grids are fixed at `20x20`, so the
output vocabulary remains the ten symbols `0-9`. The YAML itself remains
unchanged.

## Setup and checks (do not train)

```bash
bash scripts/setup_cogitao_compgen.sh

# Construct setting C1, experiment 1 without calling trainer.fit.
python -m benchmarks.cogitao_slot_attention.run \
  --setting 1 --experiment 1 --dry-run

# Inspect the fully expanded underlying command without running it.
python -m benchmarks.cogitao_slot_attention.run \
  --setting 1 --experiment 1 --print-command
```

Settings and experiments are both numbered 1 through 5. Change the two CLI
values to select any of the 25 benchmark runs.

## Training command for later

This command starts training; it is documented only and has not been run:

```bash
python -m benchmarks.cogitao_slot_attention.run --setting 1 --experiment 1
```

## Setting 1 W&B sweep (all five experiments)

`sweep_setting1.yaml` defines a five-run grid: Setting 1 is fixed and the
experiment number takes each value from 1 through 5 exactly once. It uses
`val/ood/grid_accuracy` as the sweep comparison metric and leaves the shared
Slot Attention hyperparameter YAML untouched.

Register the sweep once from the repository root (this creates the sweep but
does not start training):

```bash
wandb sweep \
  --project cogitao-compgen-slot-attention \
  benchmarks/cogitao_slot_attention/sweep_setting1.yaml
```

The command prints an agent path such as
`ENTITY/cogitao-compgen-slot-attention/SWEEP_ID`. In each tmux session, run the
same command using that path:

```bash
wandb agent --forward-signals \
  ENTITY/cogitao-compgen-slot-attention/SWEEP_ID
```

Each agent claims one unassigned experiment. When its run finishes, it
immediately claims the next available experiment; agents exit automatically
after all five grid entries complete. Do not pass `--count 1` if a tmux worker
should continue to the next run. You may start fewer or more than five agents,
but there are only five total jobs.

## Setting 2 W&B sweep (all five experiments)

Register the C2 grid once:

```bash
wandb sweep \
  --project cogitao-compgen-slot-attention \
  benchmarks/cogitao_slot_attention/sweep_setting2.yaml
```

Start a persistent worker with the path printed by that command:

```bash
WANDB_AGENT_DISABLE_FLAPPING=true \
WANDB_AGENT_MAX_INITIAL_FAILURES=100 \
wandb agent --forward-signals \
  ENTITY/cogitao-compgen-slot-attention/SWEEP_ID
```

Do not add `--count 1`. The agent cleans up a child run that succeeds, fails, or
is killed, then asks W&B for the next unassigned experiment. To kill only the
current training run while keeping the queue alive, stop that run in W&B or
send `SIGTERM` to the training child PID from another shell. Sending Ctrl-C or
SIGTERM to the agent process itself shuts down the worker, so it cannot claim
the next run.

To run every setting/experiment after reviewing the setup:

```bash
for setting in 1 2 3 4 5; do
  for experiment in 1 2 3 4 5; do
    python -m benchmarks.cogitao_slot_attention.run \
      --setting "$setting" --experiment "$experiment"
  done
done
```

## What this baseline measures

COGITAO rows are symbolic input/output grids plus a transformation suite. The
adapter one-hot encodes cell IDs 0--9 into `[10, H, W]` input tensors and
resizes grids with nearest-neighbor interpolation. Each ordered atomic
transformation is mapped to a learned task token with an order embedding.
Image-patch features and task-token features receive distinct modality
embeddings and are concatenated before Slot Attention. Padding task tokens are
masked out.

The spatial broadcast decoder emits ten categorical logits plus an alpha logit
per slot and cell. Alpha-normalized slot distributions are mixed into the
predicted output grid. Training is supervised with categorical negative
log-likelihood against `target_grid`; metrics are cell accuracy and exact-grid
match. In short:

```text
one-hot input grid + ordered task tokens -> Slot Attention -> output grid
```

Runs are sent to a separate W&B Cloud project named
`cogitao-compgen-slot-attention`, grouped by CompGen setting, with one indexed
run per setting/experiment execution. The launcher logs epoch-level pixel-wise
cross-entropy and the paper's accuracy suite (all percentages) under these
names:

```text
train/{cross_entropy_loss,total_loss,grid_accuracy,per_pixel_accuracy,object_per_pixel_accuracy}
val/{id,ood}/{cross_entropy_loss,total_loss,grid_accuracy,per_pixel_accuracy,object_per_pixel_accuracy}
test/{id,ood}/{cross_entropy_loss,total_loss,grid_accuracy,per_pixel_accuracy,object_per_pixel_accuracy}
val/grid_accuracy_id_minus_ood
test/grid_accuracy_id_minus_ood
```

W&B also receives qualitative tables at `outputs/train`, `outputs/val/id`,
`outputs/val/ood`, `outputs/test/id`, and `outputs/test/ood`. Each table contains
up to eight sampled rows with the transformation sequence, input grid,
ground-truth output, predicted output, exact-match result, and sample pixel
accuracy. The grids use the standard ARC color palette. Train/validation tables
follow the existing validation-plot interval; final test tables are always
logged once.

The same examples are written locally under the indexed run directory:

```text
runs/cogitao-compgen-slot-attention/<run-name>/run_NNN/qualitative_outputs/
├── train/sample_00.png
├── val/id/sample_00.png
├── val/ood/sample_00.png
├── test/id/sample_00.png
└── test/ood/sample_00.png
```

Each image is an `input | target | prediction` triptych. Up to four stable
filenames are used per split and overwritten at every epoch; epoch-numbered
copies are never accumulated. Test images are overwritten after final testing.

For a process that was started before this callback existed, attach the
CPU-only checkpoint watcher without restarting training:

```bash
python -m benchmarks.cogitao_slot_attention.watch_outputs \
  --checkpoint checkpoints/slot_attention/<run-name>/run_NNN/last.ckpt \
  --parent-pid <training-pid>
```

`grid_accuracy` is the paper's primary metric: the percentage of samples whose
entire output grid is correct. `per_pixel_accuracy` includes background cells;
`object_per_pixel_accuracy` scores only the union of predicted and target
non-background cells. The paper's diagnostic stubbornness metric is not logged:
it requires regenerating every OOD input under every training-seen rule, and the
released parquet rows do not contain those counterfactual targets.

Only `val` and `val_ood` are evaluated during fitting. Checkpoint selection
uses the lowest `val/ood/cross_entropy_loss`, matching the paper. After fitting,
the selected checkpoint is evaluated once on the distinct `test` and
`test_ood` loaders. A dry run builds and reports all loaders but neither opens a
W&B run nor trains/tests the model.

The existing hyperparameter YAML is not changed. In particular, its epoch and
optimizer settings remain whatever is currently in that file; change them only
after review if you want to match another training protocol.

The parquet files require `pyarrow`; downloading requires `huggingface_hub`.
Both are installed alongside the Hugging Face `datasets` package:

```bash
python -m pip install datasets
```
