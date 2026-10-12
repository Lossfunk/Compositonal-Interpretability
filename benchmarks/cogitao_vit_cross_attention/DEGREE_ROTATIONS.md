# Fresh joint atomic/composition ViT, 15-degree increments

This page documents the earlier **order-only** experiment. Its 99.22% held-out
score measures reversed-order generalization. The corrected active experiment
holds out entire unordered pairs and is documented in
[UNORDERED_ROTATIONS.md](UNORDERED_ROTATIONS.md).

The earlier experiment trained **all weights and the optimizer from scratch**.
It does not initialize from either earlier checkpoint. Atomic tasks are 15°,
30°, 45°, 60°, 75°, and 90°. The same training run also includes 21 ordered
pairs from these angles and holds out the remaining 15 pairs.

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_degree_rotations \
  --config config/cogitao/degree_rotation_joint_scratch.yaml --dry-run
python -m benchmarks.cogitao_vit_cross_attention.run_degree_rotations \
  --config config/cogitao/degree_rotation_joint_scratch.yaml \
  --output runs/cogitao_degree_rotation_joint_scratch/run_000
```

Configuration: `config/cogitao/degree_rotation_joint_scratch.yaml`.
Earlier output: `runs/cogitao_degree_rotation_joint_scratch/run_000/`.

## Rotation convention

Per the user's choice, the **whole grid rotates around the canvas center**.
Raw COGITAO source grids are padded with background into a 24×24 canvas.
Sources must fit by their full diagonal with a margin, so the requested
rotations do not clip the original rectangular support. Padding may differ by
one cell on opposite sides; the pivot is always the canvas center at (11.5,11.5).
No resizing of source cells is performed. Rotation is counterclockwise, using
nearest-neighbor categorical sampling and background zero outside the canvas.
This is a new whole-grid task, distinct from the earlier object-at-anchor task.

Atomic targets rotate the padded source once by their requested angle. A pair
(a,b) has a canonical target rendered once from the original source at a+b.
Targets are not obtained by applying two raster rotations in succession.
Angles and targets are precomputed for all 15° increments through 180°.

## Seen versus held-out sequences

All six diagonal pairs are seen. Exactly one orientation of each of the 15
unordered off-diagonal pairs is seen; its reverse is held out. A circular
orientation rule balances each angle's appearances in the first and second
positions to within one. Examples:

| Seen during training | Held out | Result |
|---|---|---|
| 15°→30° | 30°→15° | 45° |
| 15°→60° | 60°→15° | 75° |
| 75°→15° | 15°→75° | 90° |
| 45°→90° | 90°→45° | 135° |
| 60°→90° | 90°→60° | 150° |
| 75°→90° | 90°→75° | 165° |

Every held-out pair has a resulting angle present in a seen pair. The endpoints
30° and 180° have only one possible pair each and therefore cannot occur on
both sides. This protocol specifically tests generalization to unseen ordered
programs and preservation under reversing rotation order. It does not test an
unseen unordered combination. The exact list is saved in `composition_split.json`.

Only the original atomic `[rot90]` rows supply raw input grids; their old
object-rotation output labels are not used. Each retained source contributes
six atomic and 21 seen-pair tasks to training. Model-visible source hashes are
checked for overlap across original train, validation, and test source splits.

The full audit has 11,111 training sources, yielding 299,997 examples: 66,666
atomic and 233,331 seen-pair examples. Validation has 111 separate sources,
yielding 666 atomic plus 2,331 seen-pair examples. Held-out pairs never enter
checkpoint-selection validation. Test has 111 sources and 4,662 examples:
666 atomic, 2,331 seen pairs, and 1,665 held-out pairs (111 per program).

## Model and training

The fresh ViT uses 1×1 categorical image patches, width 128, six layers, four
heads, feed-forward width 512, and two ordered angle-token positions. Tokens
encode sin/cos of their angle through a shared learned linear projection.
Their vocabulary includes the possible net angles through 180°, so diagnostics
can query an equivalent single-angle token without introducing random,
untrained embedding rows. Only 15°–90° single-angle tasks are actually trained.
The model is not given a summed angle for pair inputs and does not explicitly
add angles in its forward pass. No group-law regularizer is used.

AdamW uses learning rate 0.001, weight decay 0.001, 200-step warmup, cosine
decay, batch size 128, BF16 mixed precision, up to 100 epochs, and early stopping
patience 15. Best checkpoint selection uses atomic-plus-seen-pair validation
cross entropy. Both atomic and pair tasks are shuffled together from the start.

`initial_weights.pt` and its hash in `initialization.json` preserve the random
initial state; that file explicitly records `checkpoint: null`. Best and latest
checkpoints, config, data manifest, CSV metrics, live `status.json`, and the
final `report.json` are saved locally. Resume only the same output directory
with matching config and `--ckpt-path .../checkpoints/last.ckpt`.

## Composition evaluation

Final scores are separated into atomic, seen-pair, and held-out-pair groups.
Pair groups are further separated according to whether the net angle was
trained atomically (up to 90°) or exceeds the atomic training range.

The evaluator compares direct pair prediction against the target, the summed
single-angle prediction, sequential atomic predictions, and a second prediction
conditioned on the true first intermediate grid. It also measures agreement
with the reversed program. Exact-grid, pixel, and foreground-union accuracies
are reported per program, alongside agreement metrics. All are percentages.

Single-token queries above 90° are extrapolation beyond atomic training;
read those diagnostics separately. Two nearest-neighbor raster rotations need
not equal one rotation at the sum even for a perfect image transformation.
`raster_reference_*` scores explicitly measure that discrepancy, allowing
sequential prediction scores to be interpreted with this sampling limitation.
The canonical direct target represents the combined continuous angle followed
by a single sampling step. Behavioral success alone is not a latent
homomorphism result.

## W&B

The user explicitly authorized metrics-only W&B logging for this experiment
and future training runs they request in this project. The relay streams live
step and epoch metrics, then final numeric evaluation scores to
`cogitao-compgen-slot-attention`, with no grid or checkpoint uploads. The active
run link is in `wandb.json`.

```bash
python -m benchmarks.cogitao_vit_cross_attention.rotation_wandb \
  --output runs/cogitao_degree_rotation_joint_scratch/run_000 \
  --name cogitao_degree_rotation_joint_scratch_run_000 \
  --group compgen-degree-rotations-scratch \
  --tags rotations-only,15-degree-increments,atomic-and-compositions,from-scratch \
  --stream-steps
```

Run one relay per experiment. The saved W&B ID and CSV cursor let it resume
logging without restarting training.
