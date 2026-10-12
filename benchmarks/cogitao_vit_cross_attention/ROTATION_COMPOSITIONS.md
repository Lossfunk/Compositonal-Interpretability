# Selected-composition rotation experiment

This experiment fine-tunes the saved atomic rotation ViT with a fresh optimizer
on four single-angle tasks and eight ordered two-angle programs. The atomic
checkpoint and its original experiment remain intact. The model still receives
ordered angle tokens; the sum is used to generate labels and scoring baselines,
not substituted for the two-token program in the trained model's input.

## Fixed pair split

The seen pairs have `(second - first) mod 360` equal to 0° or 90°. The other
pairs are held out. Each first angle, each second angle, and each resulting
angle occurs twice in both groups. Identity-containing pairs are included for
balance and also excluded in separate nonidentity summaries.

| Resulting angle | Seen ordered pairs | Held-out ordered pairs |
|---|---|---|
| 0° | 0°→0°, 180°→180° | 90°→270°, 270°→90° |
| 90° | 0°→90°, 180°→270° | 90°→0°, 270°→180° |
| 180° | 90°→90°, 270°→270° | 0°→180°, 180°→0° |
| 270° | 90°→180°, 270°→0° | 0°→270°, 180°→90° |

All resulting angles were previously trained as atomic tasks. This tests
unseen **programs**, rather than unseen net angles. In particular, 90°→180°
is trained while its reverse 180°→90° is held out.

Inputs use the same audited source splits as the atomic experiment. Objects
retain identity through the raw-grid rotation orbit. Targets are the canonical
raw-grid orbit state at `(first + second) mod 360`, resized once to 20×20.
The dataset contributes one example per source per configured program.

The full training split has 44,348 atomic and 88,696 composition examples from
11,087 source grids. Validation contains 444 atomic and 888 seen-pair examples
from 111 different source grids. Checkpoint selection uses only those programs
and validation cross entropy. Test contains 444 atomic, 888 seen-pair, and 888
held-out-pair examples, with 111 samples per program. Source hashes are checked
against both the fine-tuning splits and the atomic initialization manifests.

## Training

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_rotation_compositions --dry-run
python -m benchmarks.cogitao_vit_cross_attention.run_rotation_compositions \
  --output runs/cogitao_rotation_compositions_vit/run_000
```

The config is `config/cogitao/rotation_compositions_vit.yaml`. It initializes
from `runs/cogitao_rotation_token_vit/run_000/checkpoints/best.ckpt`, keeps the
six-layer width-128 ViT architecture, and uses AdamW at 0.0003 with 200-step
warmup and cosine decay. The run uses batch size 64, BF16 mixed precision,
a maximum of 60 epochs, and early stopping patience 12. It trains both ordered
token positions through seen compositions. No group-law loss is imposed.

The output includes `composition_split.json`, `data_manifest.json`, the exact
initialization checkpoint SHA256 in `initialization.json`, saved configuration,
CSV metrics, `status.json`, and best/latest checkpoints. To resume, use the
same configuration and pass `--ckpt-path` pointing at `checkpoints/last.ckpt`.

The existing W&B metric relay can mirror this run without affecting training:

```bash
python -m benchmarks.cogitao_vit_cross_attention.rotation_wandb \
  --output runs/cogitao_rotation_compositions_vit/run_000 \
  --name cogitao_rotation_compositions_vit_run_000 \
  --group compgen-c1-rotation-compositions \
  --tags rotations-only,selected-compositions,atomic-finetune
```

It streams metrics and final numeric scores to `cogitao-compgen-slot-attention`;
it does not upload grid data or checkpoints. The run link is saved in `wandb.json`.

## Test-time preservation scores

After selecting the checkpoint, `report.json` records each program and pooled
atomic, seen-pair, and held-out-pair scores, plus separate pair groups with both
angles nonzero. It evaluates:

- **Direct:** one prediction conditioned on the original two-angle program.
- **Summed angle:** one prediction conditioned on the equivalent atomic angle.
- **Sequential:** predict the first atomic rotation, then feed the predicted
  grid to the second atomic rotation.
- **Oracle intermediate:** feed the true resized first-rotation grid into the
  second atomic rotation, to distinguish first-stage prediction errors from
  difficulty caused by intermediate-grid distribution changes.

Each path is scored against the canonical target using exact-grid, pixel,
and foreground-union accuracy. Direct cross entropy and direct-versus-summed
and direct-versus-sequential grid/pixel agreement are also reported. Accuracies
and agreement values are percentages. Agreement alone does not establish
correctness; read it alongside target accuracy. The untouched atomic baseline
is evaluated with the same programs after fine-tuning's checkpoint selection.

Sequential execution operates on already resized grids. Resizing need not
commute with object rotation, so its errors can include this representation
shift; the oracle-intermediate diagnostic exposes that limitation. These are
behavioral tests of composition preservation. A latent homomorphism claim
requires a separate representation-level analysis.
