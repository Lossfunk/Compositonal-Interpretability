# Object-anchor rotations with held-out unordered compositions

This experiment starts from random weights and a fresh optimizer. Each original
eight-connected foreground object rotates separately; its foreground top-left
corner returns to its original anchor. This matches the original COGITAO
`rot90` placement, rather than a fixed geometric pivot at that corner.

Atomic angles are 15°, 30°, 45°, 60°, 75°, and 90°. Training also includes 24
ordered compositions. Both orders of these six unordered pairs are held out:
`(15,45)`, `(15,75)`, `(30,60)`, `(30,90)`, `(45,75)`, `(60,90)`.
Their resulting angles remain seen atomically or in other trained pairs. This
tests new factorizations of known effects; it does not hold out total angles.

The ViT uses 1×1 patches, width 128, six layers, four heads, and batch size 512
for both training and evaluation. Inputs retain native cell labels and are
center-padded to 24×24 without resizing. Object anchors receive only this fixed
padding offset. AdamW, learning rate 0.001, BF16, 200 warmup steps, a 100-epoch
limit, and early stopping patience 15 follow the previous configuration.

## Targets and audit

Multiples of 90° use exact `numpy.rot90` on each object's bounding-box patch.
Other angles use expanded `scipy.ndimage.rotate`, order 0, `grid-constant` zero
fill, and no prefilter, then re-anchor the foreground bounding box. Labels stay
categorical. Object identities are inferred once and retained in oracle states.

Every source's native-grid 90° target is checked against its recorded output
before padding. All 11,333 checks pass. Each retained source must render every
canonical angle from 15° through 180° without empty objects, clipping, or overlap.
The same rule covers all training, validation, and test programs; 365 sources
are excluded for overlap. Source hashes remain disjoint across splits.

| Split | Sources | Atomic | Seen pairs | Held-out pairs | Total |
|---|---:|---:|---:|---:|---:|
| Train | 10,753 | 64,518 | 258,072 | 0 | 322,590 |
| Validation | 109 | 654 | 2,616 | 0 | 3,270 |
| Test | 106 | 636 | 2,544 | 1,272 | 4,452 |

Overall there are 330,312 task examples from 10,968 source grids. Each training
source produces six atomic and 24 paired examples; each test source produces
six atomic and all 36 paired examples. Training has 631 batches per epoch.
Checkpoint selection uses atomic plus seen-pair validation only.

## Composition controls

Canonical pair targets rotate each original object once by the sum of the two
angles. They are invariant under token reversal and avoid repeated interpolation.
The model receives the two angle tokens separately; its forward pass does not
sum them or enforce a group law.

Small-angle raster rotations can change pixel counts and connected components.
Two samplings can differ from one sampling at the sum, even with a perfect
model. Returning a patch to its top-left anchor does not remove these effects.
The scorer reports direct, summed-angle, predicted sequential, and true
intermediate sequential predictions, plus reversal agreement. A pure raster
control rotates twice while retaining original object identities. Unrenderable
control states receive zero credit and their validity percentage is reported.

Evaluation additionally separates raster-exact versus raster-mismatched pairs,
preserved versus changed intermediate connected components, and the intersection
`raster_exact_and_identity_preserved`. The latter supports comparison without
these two measured confounds. On the full source audit, validation has three
unrenderable sequential controls and 645 task rows with changed intermediate
identities; test has 12 and 854 respectively. These are diagnostic counts, not
additional exclusions from canonical training or evaluation.

Single-angle diagnostics above 90° are extrapolations: their effects were seen
through trained compositions but those single tokens were never trained.

## Launch

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_degree_rotations \
  --config config/cogitao/degree_rotation_object_anchors.yaml \
  --output runs/cogitao_degree_rotation_object_anchors/run_000
```

The current-run pointer is `runs/cogitao_degree_rotation_current.json`. Run
outputs include source manifests, audits, initial random weights, checkpoints,
CSV metrics, live status, and a final report. W&B uploads training metrics and
final numeric scores only, under the user's existing authorization.

The preceding whole-grid run is preserved under
`runs/cogitao_degree_rotation_unordered_holdout/run_000`; it stopped during
epoch 15 and has no final held-out report. Its latest recorded validation exact
accuracy was 99.91%, and its best recorded exact validation accuracy was 100%.
Those scores refer to the previous whole-grid task.

Frozen hidden-state group-law tests and their post-training watcher are documented
in [ROTATION_GROUP_TESTS.md](ROTATION_GROUP_TESTS.md).
