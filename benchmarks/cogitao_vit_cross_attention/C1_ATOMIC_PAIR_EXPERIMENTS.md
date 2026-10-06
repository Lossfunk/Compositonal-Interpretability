# Atomic-pair action transfer and oracle algebra

The runner is `c1_atomic_pair_experiments.py`. It handles C1 experiments 1–5
using each experiment's trained function-conditioned ViT checkpoint. Atomic
function names are discovered from that experiment's training parquet. Every
ordered pair is considered, including self-pairs. No composition is used to fit
the latent operators.

## Oracle states and validity

For each source image and pair `f, g`, the oracle constructs:

```text
x, fx=f(x), gx=g(x), fgx=f(g(x)), gfx=g(f(x)).
```

`fgx` corresponds to the dataset's ordered suite `g -> f`. The reverse endpoint
is `f -> g`.

The oracle reconstructs object actions from the [COGITAO generator source](https://github.com/yassinetb/COGITAO/blob/main/arcworld/transformations/shape_transformations.py).
These actions operate on individual objects; `rot90` rotates an object's patch
counterclockwise at its anchor, and the mirrors flip the object's patch. They
differ from the previous whole-image D4 perturbation experiment.

Objects are inferred once as eight-connected foreground components. Their
identities persist across both orders, even when cropping or doubling changes
connectivity. Explicit zero-valued crop points are preserved for subsequent
bounding-box operations, following the generator's point-cloud semantics.
Actions cover all 18 functions in the repository's vocabulary, including the
deterministic color actions. Padding uses the generator's direction-specific
colors: right 6, left 7, top 8.

Every retained sample must have all five valid states. Clipping, overlap,
empty objects, and invalid colors are rejected and counted. These are concrete
state-validity checks; the oracle does not impose the generator's original
shape-sampling restrictions on counterfactual intermediate states. Valid
no-ops, such as filling a shape without a hole, are retained and counted.

Training atomic oracle outputs must exactly equal recorded targets. Evaluation
samples also audit their recorded suite/target. A target mismatch stops the
experiment; it is evidence that inferred object identities or oracle semantics
do not reproduce the recorded data. The parquet does not supply object IDs, so
this target audit is necessary and does not prove an unobserved counterfactual
endpoint independently. A mismatch saves `oracle_audit_failure.npz` containing
the source, recorded target, reconstructed target, suite, split, and row index
before stopping. Raw oracle arrays can also be saved for inspection.

## Image representations and frozen operators

Representation collection calls only `model.encode_images`, capturing:

- Patch projection plus position embeddings.
- Every ViT Transformer block.
- Final image LayerNorm.

At each layer, the chart `z_l` concatenates ordered spatial-bin token means.
The default `--pool-grid 2` gives four bins and 512 coordinates for a width-128
ViT. This retains coarse spatial position and makes affine operator fitting
tractable. It is not the full flattened token map. Increasing the pool grid
keeps finer spatial structure but raises the operator dimension and fitting
cost. `--pool-grid 1` is a spatially averaged comparison condition.

For each atomic function and layer, fit a ridge-regularized affine map on
audited training pairs `z_l(x) -> z_l(f(x))`. The homogeneous row-vector matrix
`A_f` supports composition, including affine bias. Affine regression permits
noninvertible actions such as cropping and filling; an orthogonal group action
would impose an inappropriate constraint on these functions. Training fit
error, source rank, dimension, and identifiability are reported. Default sample
counts can yield deficient rank in the 512-dimensional chart; increase
`--fit-images-per-function` to study whether results depend on probe data.

Frozen probes store a checkpoint SHA-256 fingerprint, function list, pooling
settings, fit metadata, and every fit source/target hash. Evaluation rejects
samples if any of their five states matches a fit state.

## Four measurements

For every layer and example:

1. Action transfer for `f`: compare `z(fx)-z(x)` with `z(fgx)-z(gx)`.
2. Action transfer for `g`: compare `z(gx)-z(x)` with `z(gfx)-z(fx)`.
3. Commutator defect: `||A_f A_g - A_g A_f||`, scored only if the raw oracle
   endpoints `fgx` and `gfx` are exactly equal. Relative operator error and
   source-conditioned data error are also reported.
4. Oracle relation retention: detect every exact grid equality among the five
   states, then test the corresponding equality between the frozen operators'
   predicted states. Report relation count, mean/max defect, and the fraction
   within `--relation-tolerance` (default 0.05).

The action-transfer distances include cosine distance and symmetric relative L2:

```text
||delta_1 - delta_2|| / max((||delta_1|| + ||delta_2||)/2, epsilon).
```

Zero delta pairs are flagged. Their cosine distance is undefined and omitted
from averages. Additional active-action means exclude exact oracle no-ops in
either the source or context intervention. Relation and endpoint prediction
defects divide by `||z(x)||`.
No-relation samples have blank retention fields. Relation errors test learned
actions; encoding the same oracle image twice would give a trivial equality.
For self-pairs, `fx=gx` and `fgx=gfx` are structural tautologies, recorded but
excluded from retention averages. Self-pair commutators are also left blank;
they are identically zero for any fitted map. Genuine self-pair identities such
as observed idempotence still contribute to relation retention.
The operator table contains identity, absorption, idempotence, and commuting
relations whenever the five states exhibit them. This finite state set does
not test every possible higher-order algebraic law, such as rotation to the
fourth power; the existing D4 probe handles group order laws.

First-step prediction errors and errors when applying each atomic map to the
other function's actual intermediate embedding are included as probe-fidelity
checks. Low commutator error alone can result from collapsed fitted maps; it
must be interpreted together with these errors and fit rank.

## ID/OOD sampling and accuracy

`val` and `test` use held-out atomic source images to synthesize all pairs. This
tests transfer on ID source images; it does not establish that the generated
composition itself was seen in training. `val_ood` and `test_ood` use actual
recorded `g -> f` composition inputs. If that order is absent, the coverage row
records zero examples rather than labeling a synthesized pair as benchmark OOD.

The conditioned model is invoked separately for object-pixel accuracy on each
recorded composition. This pass uses the actual ordered function tokens and
recorded output; it does not provide representations to the probe. Object-pixel
accuracy uses the union of target and prediction foreground, so extra predicted
objects are penalized. It matches the repository's composition evaluator and
is stored as a fraction from 0 to 1.

Two accuracies are reported: all matching recorded composition rows, and the
retained oracle-valid subset. By default, accuracy evaluates every matching
composition row. `--max-accuracy-images N` caps that pass while including every
selected oracle example; it may exceed N if the oracle subset is larger.

For every OOD split and layer, Pearson and Spearman correlations join pair
metrics to each of these accuracies. The unit is one ordered composition per
checkpoint; per-image measurements are not treated as independent composition
units. Fewer than three usable compositions or constant values produce blank
correlations with an explicit status. The aggregation command also pools across
the five experiments while keeping validation, test, and layers separate.
These correlations describe association and do not establish causality.

## Commands

Run from the repository root. Default checkpoint paths are the existing
`cogitao_c1_experiment_N_vit6_function_mlp_cross_attention/run_000/last.ckpt`.

One experiment:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments \
  --experiment 1 \
  --mode all \
  --fit-images-per-function 256 \
  --eval-images-per-pair 128 \
  --pool-grid 2 \
  --batch-size 16 \
  --save-oracle-states \
  --wandb-mode online \
  --wandb-log-artifacts
```

All five, followed by pooled correlations:

```bash
for experiment in 1 2 3 4 5; do
  PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments \
    --experiment "$experiment" \
    --fit-images-per-function 256 \
    --eval-images-per-pair 128 \
    --pool-grid 2 \
    --batch-size 16 \
    --save-oracle-states \
    --wandb-mode online \
    --wandb-log-artifacts || break
done

PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.aggregate_c1_atomic_pairs \
  --wandb-mode online
```

Fit and evaluation can be separate. Run `--mode fit`, then reuse the checkpoint
and probe with `--mode eval`. The command refuses to overwrite fitted probes;
use a fresh `--output-dir` for a new fit. To supply a different checkpoint, add
`--checkpoint /path/to/checkpoint.ckpt`. Re-evaluation requires the same pooling
setting as the frozen probe.

Use `--wandb-mode offline` for local W&B storage, or `disabled` for no W&B.
Default W&B project is `cogitao-compgen-slot-attention`, with group
`cogitao-c1-atomic-pair-algebra` and a separate run per C1 checkpoint. Pair/layer
tables, OOD correlation tables, and optional result artifacts are logged.

## Results and tests

Each checkpoint writes `artifacts/cogitao_c1_experiment_N_atomic_pairs/`:

- `frozen_atomic_probes.npz`: atomic maps, metadata, and fitting state hashes.
- `pair_layer_metrics.csv`: all four measurements and accuracy by pair/layer.
- `per_example_metrics.csv`: individual action and relation scores.
- `oracle_relations.csv`: each observed equality and its latent defect.
- `coverage.csv`: availability, rejected states, no-ops, and selected counts.
- `composition_accuracy.csv`: recorded composition and oracle-subset accuracies.
- `correlations.csv`: Pearson/Spearman results and insufficient-data statuses.
- `report.json`: protocol, checkpoint, fit diagnostics, and evaluation results.
- Optional `SPLIT__g__f__states.npz`: all five raw oracle states, source indices,
  and `raw_shapes` (arrays are zero-padded if native grid dimensions vary).

The pooled report is written to `artifacts/cogitao_c1_atomic_pairs_all/`.

```bash
PYTHONPATH=. pytest -q tests/test_c1_atomic_pair_experiments.py
```

The implementation and these commands are provided without executing the
experiment or its tests.
