# Frozen-latent rotation homomorphism tests

These tests supplement output-level behavioral evaluation. They freeze the
selected checkpoint and analyze its final contextual image tokens. The group
is C24: multiples of 15° modulo 360°, rather than every continuous angle.

The ViT has no explicit action matrices. Independent affine probes test the
hypothesis of a shared linear action on compressed hidden states; they are not
the network's actual operators or proof that the nonlinear model is a homomorphism.

## Fitting

The reference `z(x)` uses the encoder with both task slots masked. This is an
untrained no-task reference, not a trained identity token. Contextual targets
`z_theta(x)` use canonical programs for each effect. Spatial pooling to 6×6
retains coarse position. Training-only PCA and whitening reduce the resulting
4,608 features to at most 64 dimensions, excluding near-zero PCA directions.

On 512 seeded training sources, ridge 0.01 fits independent affine homogeneous
maps `rho(theta) z(x) ~= z_theta(x)`, with unpenalized intercepts. Projection and
operator fitting use no validation/test sources or held-out pair programs.
Canonical 15°–90° programs are trained atomic tokens. Larger known effects use
trained compositions: 105° `[15,90]`, 120° `[60,60]`, 135° `[45,90]`,
150° `[75,75]`, 165° `[75,90]`, and 180° `[90,90]`.

Explicit queries at 0° and 195°–345° extend the existing shared sin/cos
projection during inference without changing weights. They are labeled
extrapolation. `rho(0)` is fitted from the explicit 0° query, not set to identity.
Inverse-angle maps are independently fitted, never manufactured by matrix inversion.

Validation and test use all 36 pairs after training: 109 and 106 source grids.
Their hashes are checked against the saved run and kept disjoint from probe fitting.

## Laws

With column vectors, program `[a,b]` acts as `rho(b) rho(a)`.

| Test | Comparison | Scope |
|---|---|---|
| Homomorphism | `rho(b)rho(a)` versus `rho(a+b)` | 24 seen and 12 held-out ordered pairs, known effects |
| Commutativity | `rho(b)rho(a)` versus `rho(a)rho(b)` | 18 nontrivial seen and 12 held-out ordered pairs |
| Identity | `rho(0)` versus `I`, left/right identity actions | Explicit 0° is untrained |
| Inverse | `rho(-a)rho(a)` and reverse versus `I` | Most inverse angles are untrained; 180° is known |
| Finite order | `rho(a)^(360/gcd(a,360))` versus `I` | Six trained atoms, including `rho(90)^4` |
| Factorization associativity | `rho(c)rho(a+b)` versus `rho(b+c)rho(a)`; each versus `rho(a+b+c)` | Atomic triples with total at most 180° |
| Full closure | Every `rho(b)rho(a)` versus `rho((a+b)%360)` | 576 laws, extrapolation separated |

Self-commutation of equal angles is tautological and excluded from primary
summaries, while retained as labeled controls in tables. Ordinary associativity
of matrix multiplication is also tautological; tests compare independently
fitted sum operators instead.

Direct hidden-state tests compare `[a,b]` versus `[b,a]`, canonical programs of
the same effect, and composed probe actions versus actual pair-program hidden
states. For 55 triples with both partial sums at most 90°, actual representations
of `[a+b,c]` and `[a,b+c]` are compared. This respects the two-slot interface,
rather than pretending the model supports three tokens.

Metrics include relative matrix residuals, latent/action relative L2 and RMSE,
cosine similarity, centered linear CKA, held-source fit R² against a training
target-mean baseline, operator rank/singular values, and representation variance.
Degenerate comparisons are unavailable rather than favorable scores. There is
no arbitrary pass/fail threshold. A zero map can commute and satisfy some
composition laws while failing identity/cycle and fit-quality controls.

Internal program checks separately report cases where the raster oracle composes
exactly and intermediate connected-component identities remain recoverable.
Rasterization itself need not preserve the continuous rotation group law.

## Initial-weight control

Sin/cos inputs already contain angle structure. By default the same analysis is
run on saved `initial_weights.pt`, with identical sources and probe settings.
Each checkpoint gets its own training-only PCA basis and operators.
`comparison.json` records trained minus initial-weight scores: lower error
differences are favorable; higher fit R² differences are favorable. This is a
protocol comparison, not a cross-checkpoint basis alignment or confidence interval.

Interpret residuals together with held-source fit, rank/variance, direct hidden
state consistency, and the random-weight control. A poor affine fit rejects
this particular linear explanation, not every possible nonlinear group action.

## Execution

```bash
python -m benchmarks.cogitao_vit_cross_attention.run_rotation_group_tests \
  --output runs/cogitao_degree_rotation_object_anchors/run_000 \
  --wait --log-wandb
```

The watcher requires a complete report and the training process to exit,
including its built-in behavioral evaluation. Failed or superseded training
stops the watcher. A lock prevents duplicate jobs. Without `--wait`, unfinished
training is refused. `--skip-initialization-control` is for manual debugging;
the queued experiment includes the control.

Local artifacts under `group_tests/` include job status/config/PID/log,
`report.json`, `summary.json`, per-split/per-law CSVs, `probes.pt`,
`initialization_control/`, and `comparison.json`.
Numeric summaries only go to the existing W&B run under `group_theory/trained`,
`group_theory/initialization_control`, and `group_theory/trained_minus_initialization`.
Grids, hidden states, bases, operators, and checkpoints remain local.

Validation includes exact rotations, noncommuting and collapsed counterexamples,
affine recovery on separate sources, equivalence of numeric and trained angle
queries, completion guards, and a CPU end-to-end workflow with the control.
