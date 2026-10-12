# Object-anchor rotation experiment

Completed fresh ViT training on atomic 15°, 30°, 45°, 60°, 75°, and 90° rotations
and selected compositions, followed by behavioral and frozen-latent group tests.
Each object rotates separately and returns its top-left corner to its original
anchor, matching the original COGITAO rot90 placement.

Training: **45 epochs**, **28,395 optimizer steps**, **1×1 patches**,
**batch size 512**, **322,590 training examples**. All recorded native 90° targets
match the oracle. Whole sources that overlap at any canonical angle are excluded.

The split excludes both orders of (15,45), (15,75), (30,60), (30,90), (45,75),
and (60,90). Their effects remain known through atoms or other trained pairs.
Checkpoint selection uses atomic and seen-pair validation only.

## Behavioral test results

| Tasks | Samples | Exact-grid accuracy |
|---|---:|---:|
| Atomic rotations | 636 | 93.87% |
| Seen compositions | 2,544 | 94.65% |
| Held-out compositions | 1,272 | 11.24% |

For the held-out subset where raster composition is exact and intermediate object
identities remain recoverable, direct exact-grid accuracy is
**8.23%**
on 243 cases.
Atomic and seen-composition accuracy is high, while unseen unordered composition
generalization is poor. Pixel accuracy alone is misleading because most cells
are background.

## Group-theory tests

| Mean relative matrix residual (lower is better; zero is exact) | Trained | Initial random weights |
|---|---:|---:|
| Composition | 0.8589 | 0.1534 |
| Nontrivial commutativity | 0.4714 | 0.0296 |
| Finite-order cycles | 1.6822 | 0.5613 |

The tests fit affine probes to frozen, spatially pooled hidden states using
512 training sources only, then evaluate separate validation/test sources.
Equal-angle self-commutation is excluded from the main commutativity summary.
Identity and most inverse-angle queries are explicitly marked as extrapolation.

These probes do not show a clean learned affine homomorphism. The initial-weight
control has lower residuals, demonstrating why algebraic residuals alone are
insufficient evidence. This result concerns the fitted affine representation;
it does not rule out all nonlinear group structure. Sin/cos inputs already
contain angle information before training. Consult probe-fit quality, rank,
variance, direct latent consistency, and the initialization control together.

## Files

- [Behavioral report](report.json), [summary CSV](behavioral_summary.csv), and
  [per-program test scores](test_by_program.csv).
- [Group-theory report](group_tests/report.json), per-law CSV tables, and local
  [fitted probes](group_tests/probes.pt).
- [Initialization control](group_tests/initialization_control/report.json) and
  [matched comparison](group_tests/comparison.json).
- [Training metrics](training_metrics.csv), source manifests, configuration,
  composition split, and fresh-initialization metadata.
- [Rotation protocol](protocol/OBJECT_ANCHOR_ROTATIONS.md) and
  [group-test protocol and limitations](protocol/ROTATION_GROUP_TESTS.md).
- [Best checkpoint](checkpoints/best.ckpt), [last checkpoint](checkpoints/last.ckpt),
  and [initial weights](initial_weights.pt) are relative symlinks to the preserved
  local run; keep that run directory to resolve these links.
- `artifact_manifest.json` records source paths, copy/link kinds, and SHA-256 hashes.

Original run: `runs/cogitao_degree_rotation_object_anchors/run_000`.
Copied reports retain their original run paths as provenance.
[W&B run](https://wandb.ai/uaena/cogitao-compgen-slot-attention/runs/mu5gu3v0).
