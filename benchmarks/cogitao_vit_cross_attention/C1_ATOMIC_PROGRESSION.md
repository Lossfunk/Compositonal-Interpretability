# C1 experiment-1 atomic checkpoint progression

This evaluation follows the trained function-conditioned ViT + function-MLP
checkpoint across all distinct saved optimizer steps. A single W&B grid sweep
contains two jobs, one for `rot90` and one for `translate_up`. Each job logs
validation and test metrics at every checkpoint with `checkpoint/global_step`
as its chart axis. Checkpoints with duplicate steps are collapsed; `last.ckpt`
is used for the final 78,000-step point.

Metrics per split and function are exact-grid accuracy, all-pixel accuracy,
object-pixel accuracy, cross entropy, sample count, changes from the initial
and preceding checkpoint, and the best validation exact-grid accuracy so far.
The atomic ID validation and test sets have 111 `rot90` examples and 112
`translate_up` examples each. The released C1 OOD splits contain only
compositions, so there is no atomic OOD score for either function.

## Local evaluation

From the repository root, evaluate both functions without W&B:

```bash
PYTHONPATH=. python -m benchmarks.cogitao_vit_cross_attention.eval_atomic_progression_c1 \
  --function both --wandb-mode disabled --device auto --batch-size 64
```

This writes `progressive_metrics.csv` and `report.json` to
`artifacts/cogitao_c1_atomic_progression/both/`. The completed local results
are also split into `rot90/` and `translate_up/`, with a readable table in
`artifacts/cogitao_c1_atomic_progression/PROGRESSIVE_RESULTS.md`.

## W&B sweep

The sweep has already been created in the existing project:

[View the C1 atomic progression sweep](https://wandb.ai/uaena/cogitao-compgen-slot-attention/sweeps/e93juwae)

After the destination/account is approved, run its two jobs with:

```bash
WANDB_AGENT_DISABLE_FLAPPING=true WANDB_AGENT_MAX_INITIAL_FAILURES=10 \
wandb agent --count 2 --forward-signals \
  uaena/cogitao-compgen-slot-attention/e93juwae
```

For a new sweep in a different authorized W&B project:

```bash
wandb sweep --project PROJECT_NAME \
  benchmarks/cogitao_vit_cross_attention/sweep_eval_atomic_progression_c1.yaml
wandb agent --count 2 ENTITY/PROJECT_NAME/SWEEP_ID
```

The YAML is `sweep_eval_atomic_progression_c1.yaml`; its two jobs call
`eval_atomic_progression_c1.py` with `--wandb-mode online` and distinct
`--function` values. W&B's `define_metric` is used to put validation and test
curves on the checkpoint global-step axis.

**Current status:** local progression metrics are complete and the sweep exists,
but no agent runs were uploaded. Automatic approval review rejected the agent
launch because it would export checkpoint-derived metrics to a W&B
account/project whose ownership was not verified by the user's request.
