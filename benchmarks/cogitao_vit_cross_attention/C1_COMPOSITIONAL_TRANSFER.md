# C1 compositional transfer

**Ordering: pair `(f,g)` denotes `fgx=f(g(x))`; the ordered task tokens are
`[g,f]`, written `g -> f`.** All models and their checkpoints stay frozen.

Run all five final C1 checkpoints with the requested 256 atomic fit examples,
128 matched evaluation examples per split/program, and all four splits:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m \
  benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer \
  --all-experiments \
  --data-root data/cogitao/files/CompGen \
  --output-dir artifacts/cogitao_c1_compositional_transfer \
  --fit-images-per-function 256 --eval-images-per-pair 128 \
  --splits val test val_ood test_ood --pool-grid 2 \
  --batch-size 16 --device auto --seed 42 --composition-upper-bound
```

The default checkpoint for experiment `i` is
`checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_i_vit6_function_mlp_cross_attention/run_000/last.ckpt`.
To run each experiment separately (replace the output root if already used):

```bash
for experiment in 1 2 3 4 5; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m \
    benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer \
    --experiment "$experiment" \
    --checkpoint "checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_${experiment}_vit6_function_mlp_cross_attention/run_000/last.ckpt" \
    --output-dir "artifacts/cogitao_c1_compositional_transfer_separate/experiment_${experiment}" \
    --fit-images-per-function 256 --eval-images-per-pair 128 \
    --splits val test val_ood test_ood --pool-grid 2 \
    --device auto --seed 42 --composition-upper-bound
done
```

Smallest end-to-end checkpoint smoke test (probes need at least two fit samples):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m \
  benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer \
  --experiment 1 --output-dir artifacts/cogitao_c1_compositional_transfer_smoke_minimal \
  --fit-images-per-function 2 --eval-images-per-pair 1 \
  --splits val_ood --pca-ranks 1 --batch-size 1 --bootstrap 0 --device auto
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. pytest -q tests
```

`--mode fit` saves atomic affine probes plus action PCA. `--mode eval` reuses
`--probe-file`, checking checkpoint SHA, experiment, train-only provenance,
resolution, pooling, ridge and seed. Fitting refuses to overwrite an existing
probe file. `--resume` verifies and reuses existing probes and fits missing
ones; it still reevaluates matched examples to rebuild complete reports. Use a fresh directory for a new fit. Repeated evaluation replaces
its reports and per-example files.

A trajectory can use an explicit checkpoint list or a quoted glob. Each
checkpoint has a separate probe file and output directory; no probe is reused
across checkpoints. Global training steps appear in every table, so order by
`global_step` rather than filename. Use `--allow-early-checkpoint` to evaluate
saved start/early checkpoints before step 1000; the default loader guard stays
in force otherwise. The trajectory table joins atomic/transfer/OOD behavior
with per-layer errors.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python -m \
  benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer \
  --experiment 1 \
  --checkpoint-glob 'checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/milestone_*pct.ckpt' \
  --output-dir artifacts/cogitao_c1_transfer_trajectory \
  --fit-images-per-function 256 --eval-images-per-pair 128 \
  --splits val_ood test_ood --device auto --seed 42
```

## What is measured

A is `M(x,[f])` against `fx`. B is `M(gx,[f])` against `fgx`. The first
step of C is `M(x,[g])` against `gx`; its detached argmax grid, already at model
resolution, is passed back into `M(...,[f])`, against `fgx`. D is `M(x,[g,f])`,
also against `fgx`. There is no ground-truth substitution in C. Predictions
receive only one-hot inputs, padded task tokens, masks, and a zero target
placeholder required by the existing forward interface. Oracle targets enter
only the subsequent metric computation.

All four conditions use the same selected examples. Required states are x,
fx, gx and fgx. The reverse gfx is optional; its absence excludes only reverse
and commuting analyses. Oracle reconstruction, support checks, persistent
object identity, exact recorded-target auditing, training-sample selection,
raw-grid overlap hashes, and resizing reuse the atomic-pair framework.
Distinct raw fit/evaluation grids that coincide after resizing are also
excluded. Counts distinguish examined, valid, overlapping, selected and
unexamined examples; `oracle_valid_count` counts examined valid candidates,
not an audit of all unexamined candidates. ID splits synthesize contexts on
atomic ID inputs; OOD splits use actual held-out `[g,f]` rows. Held-out suites
are discovered from both OOD splits.

Object accuracy uses the benchmark's prediction/target foreground union, so
extra predicted foreground is penalized. Accuracy is a fraction in [0,1]. NLL
is the sum across grid pixels and cross entropy is the mean. Confidence,
entropy and multiclass Brier scores are pixel means. ECE aggregates pixel-bin
counts across examples; `example_ece` is a separate mean of per-grid ECE.
This is pixel calibration, not calibration of exact-grid success.

Atomic affine maps use the existing homogeneous row-vector convention:
`apply_affine(apply_affine(z, A_g), A_f)`. Only atomic train pairs fit these
maps. `id_atomic_error` uses separately audited atomic val/test rows with fit
state exclusion; `matched_atomic_error` measures f on the same x used for
context transfer. The headline transfer gap subtracts the separate atomic
held-out baseline, using val for val_ood and test for test_ood. No OOD example
fits an atomic operator or an ID PCA basis. Fit sample counts, latent dimension,
effective source rank, training error and underidentification are reported.
Default pooled dimension is 512 for these checkpoints, exceeding 256 fit
samples; prioritize data-relative held-out defects over matrix distances.

Baselines are identity, zero, only A_g, only A_f, and shuffled/reversed atomic
operator assignment. `--composition-upper-bound` adds a **CHEATING** supervised
composition affine map: the first half of each matched split is calibration,
only the second half is scored, and fit indices are saved. It never affects
atomic maps, taxonomy or primary correlations. This optional result is a
supervised reference, not compositional generalization or a guaranteed bound.

PCA is centered and fitted only on atomic training action residuals. The
uncentered ID/OOD action residuals are projected onto its orthonormal directions;
the training residual mean norm is reported. Fixed ranks 1,2,4,8,16,32 and 90/95%
variance ranks are supported. Unavailable ranks are omitted, never padded with
null directions. Report projection residual, squared residual energy, atomic
held-out projection, and random orthonormal subspace baselines. An OOD PCA is
fitted only for descriptive principal angles; it never changes the ID projector.
Angles are omitted when the OOD rank is too small. Zero actions have projection
and energy errors zero and contribute no directional evidence. Rank-3 figures
are not used for quantitative claims.

Oracle-verified equalities among x/fx/gx/fgx/gfx and f²=I or f²=f are tested
against frozen predicted actions. Commutator defects and program-order
invariance are present only for exact oracle-equal endpoints. Existing raw
vector-displacement disagreement remains an exploratory diagnostic in the
older atomic-pair script; it is not a primary transfer criterion here.

## Hooks and localization

`ViTLayerCollector` captures image input tokens, every encoder block, and final
normalization through the established image-only path. `ExecutionCollector`
observes the original conditioned forward path with hooks:

| Stage | Existing module or hook |
|---|---|
| final image representation | `image_encoder` output |
| raw padded function embeddings | `function_embedding` output |
| ordered function-MLP output | `function_mlp` output |
| function-conditioned shared queries | `image_to_shared` pre-hook |
| first binding stage and attention | `image_to_shared` output tuple |
| output-grid stage and attention | `shared_to_output` output tuple |
| decoder logits | `output_head` output, before grid reshape |

For A/B/C/D and C's first step, compressed NPZ files save predictions,
probability metrics, and representations (float16 storage, float32/64
measurements). `--no-save-representations` is available for storage-constrained
runs; predictions, per-example metrics and path comparisons still persist.
The per-example NPZ files also preserve resized oracle grids, their raw shapes,
reverse predictions, and subsets used for path comparisons.

B vs D comparisons include all matched examples and the strict subset where
B is exact-correct and D exact-wrong. They report cosine distance, symmetric
normalized L2, centered linear CKA, and PCA principal angles with validated
rank capped at 32. CKA is undefined for constant function-only representations
or fewer than three examples; blank values are explicit. Attention maps receive
symmetric KL and JS; decoder logits receive probability KL and JS after softmax.
The same metrics define program-order invariance on commuting states, including
C1-1 `translate_up`/`rot90` when oracle equality is verified.

A seen training program `[h,f]` is selected when available, and a hook swaps
its first token embedding to the learned atomic embedding of g. The mask and
slot are preserved. This realizes the held-out program and is expected to
match direct `[g,f]`; equality is a useful intervention sanity check, not a
claim that swapping rescues performance. Swapped metrics and template details
are in `report.json`. Reverse `[f,g]` is evaluated only on its valid subset.

Earliest divergence and earliest binding divergence use an exposed normalized
L2 threshold. Image inputs and function programs differ by design, so those
stages may already diverge. Observational similarity does not establish a
causal failing module. Downstream divergence should be read alongside B/D
behavior and image transfer, not treated as a causal localization by itself.

## Artifacts and interpretation

Each experiment directory contains `behavioral_transfer.csv`,
`layerwise_transfer.csv`, `subspace_transfer.csv`, `program_path_similarity.csv`,
`atomic_heldout.csv`, `per_example_behavior.csv`, `per_example_latent.csv`,
`oracle_relations.csv`, `coverage.csv`, `failure_taxonomy.csv`,
`state_sufficiency_strata.csv`, `report.json`,
frozen probes, and one `*__per_example_transfer.npz` per split/program.
The root has `cross_c1_summary.csv`, `behavioral_transfer_summary.csv`,
`correlations.csv`, and this README. Trajectory mode adds
`checkpoint_trajectory.csv` and per-checkpoint directories.

All CSVs carry checkpoint path/SHA, git commit, experiment, split, ordering,
counts, layer, seed, fit counts, and relevant ridge/PCA settings. JSON holds
full probe diagnostics and support coverage. Use git diff alongside the
recorded commit to identify uncommitted analysis changes.

Failure taxonomy is multi-label and provisional. Every diagnosis exposes the
continuous A/B/C/D object accuracies, first-step accuracy, image-final transfer
error/gap, and all adjustable thresholds (`--good-accuracy`, `--bad-accuracy`,
`--transfer-gap-threshold`, `--low-latent-error`). Good B with poor D provides
behavioral evidence for a conditioning/binding issue; low frozen image transfer
error is required for the stronger Type B label. Inconclusive rows remain
inconclusive. Evidence cannot replace a causal intervention.

Per-example Pearson/Spearman, n and p-values are descriptive within one model;
examples sharing a checkpoint are not independent model replicates. Optional
percentile bootstrap intervals use example resampling within a checkpoint and
experiment-cluster resampling across checkpoint/program means. Training-time
checkpoints from one experiment share a run. Few C1 models severely limit
population inference; compare seeds when more become available. Principal-angle
correlations are group-level, never duplicated into per-example correlations.

Sanity audits include no-op rates, support/overlap counts, train vs atomic
held-out error, identity/zero/shuffled operator baselines, random subspaces,
effective ranks, resizing and foreground-union scoring. The resize/oracle
order defect compares transform-after-resize with resize-after-transform;
nonzero values can be expected because geometric operations use raw-grid
units. Predictions/targets follow the existing nearest-neighbor benchmark
resize convention. The intermediate reinference audit checks whether inferring
objects afresh from gx recovers the persistent-object oracle target fgx. If it
fails, a raster may have lost hidden object identity, weakening an image-only
transfer interpretation; these counts must accompany behavioral conclusions.
`state_sufficiency_strata.csv` reports A/B/C/D and final-image errors separately
for matching and nonmatching reinference endpoints. Structural failure claims
should persist on the raster-consistent subset.
