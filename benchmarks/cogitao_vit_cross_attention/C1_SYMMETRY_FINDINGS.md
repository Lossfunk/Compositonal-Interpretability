# C1 experiment 1 symmetry findings

The exact operators follow the [COGITAO transformation implementation](https://github.com/yassinetb/COGITAO/blob/main/arcworld/transformations/shape_transformations.py): `rot90` rotates each object-local patch 90° counterclockwise at its current top-left anchor, and `translate_up` moves that anchor one row up. All results below use the local C1 experiment-1 data and the trained function-conditioned ViT + MLP checkpoint in `run_000/last.ckpt` (78,000 optimizer steps).

## Exact raw-grid structure

| Check | Validation | Test |
| --- | ---: | ---: |
| Recorded `translate_up → rot90` reconstructed by exact matrices | 1,000/1,000 | 1,000/1,000 |
| Reverse `rot90 → translate_up` valid and same target/matrix | 979/1,000 | 977/1,000 |
| Reverse invalid because rotation first leaves grid | 21/1,000 | 23/1,000 |

The 5×5 attribute matrices commute exactly. On the common valid domain, the sample-specific 225×225 pixel matrices also commute exactly. This is a **partial symmetry**: the order swap is valid only when both intermediate states fit the grid. A global pixel-rotation matrix cannot represent C1 `rot90`, because each object rotates around its own bounding box.

## Trained checkpoint without injected matrices

| Suite | Validation exact grids | Test exact grids |
| --- | ---: | ---: |
| Atomic `rot90` | 53/111 (47.75%) | 53/111 (47.75%) |
| Atomic `translate_up` | 103/112 (91.96%) | 100/112 (89.29%) |
| Recorded OOD `translate_up → rot90` | 0/1,000 | 0/1,000 |
| Reverse-order counterfactual `rot90 → translate_up` | 0/979 | 0/977 |

The reverse-order scores use precisely the OOD input grids for which both orders are valid and produce the same target. On that matched subset, object-pixel accuracy is 17.76% for the recorded order and 24.67% for the reverse order in validation; test is 18.48% and 23.85%. Both orders have zero exact-grid accuracy. The model therefore lacks useful **compositional prediction** for this pair even though it learned the individual operations to different degrees.

The saved OOD-validation-loss checkpoint `epoch=021.ckpt` (8,580 steps) also has zero exact grids on both OOD orders. Its atomic exact-grid accuracy is zero too, so the final checkpoint is the more informative check of composition after atomic learning.

## Image-token latent probe

An affine 129×129 homogeneous channel map was fitted separately for each atomic operation using 256 training pairs per operation and exact source/target foreground correspondences. No composition examples were used to fit these maps. On 128 OOD validation pairs, the composed `translate_up → rot90` map has mean squared error **0.318×** the identity-map error; reverse order has **0.309×** on 128 valid examples. The fitted products differ: their relative Frobenius commutator is **0.296**. The latent maps capture some atomic change, but do not exactly realize the geometry's commutation relation in this linear channel chart.

This probe examines image-token features after positional embeddings and self-attention. It does not prove that the model has no nonlinear symmetry representation. It also does not test a spatially conditional operator on the full latent tensor. The direct decoder result is the stronger evidence for the missing task behavior.

## OOD structure to add next

1. Use the native 15×15 grid, or apply exact raw operators before the existing 15→20 nearest-neighbor resize. This keeps the known operator and its boundary domain explicit.
2. Add both orderings on matched valid inputs as a controlled OOD set. Keep the 21/23 out-of-domain examples separate; they are boundary cases, not failures of commutation.
3. Factor held-out examples by object position, bounding-box height/width, color, and distance to the top/bottom grid edge. This reveals whether failures arise from position or shape changes rather than order alone.
4. Measure atomic accuracy, exact composition, and order-swap consistency together. A useful loss or inductive bias would preserve object-local rotation, translation of the anchor, and equality of the two orders where both are defined.
5. Repeat after the separate joint image/function-token ViT has a trained checkpoint. The current local joint-token checkpoints have 1, 1, and 538 steps, so they cannot support a scientific comparison yet.

Machine-readable results and matrices are in `artifacts/cogitao_c1_symmetries/mlp/`; the selected earlier checkpoint is in `artifacts/cogitao_c1_symmetries/mlp_best_ood/`. The script and commands are in `C1_SYMMETRIES.md`.
