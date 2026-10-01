# COGITAO C1 five-example ICL through Codex CLI

`icl.py` implements the C1 rows of the ICL table with **five labeled examples
per prompt**. It runs the logged-in `codex exec` CLI, not an API client. The
provided IID base instructions are in `prompts/iid.txt`; paste the separate OOD
base instructions into `prompts/ood.txt` before requesting an OOD split. Until
then OOD runs stop before any model call.

| Condition | Context per prompt | Target |
| --- | --- | --- |
| CompGen-ID1 | 2 single + 3 seen composed examples | Seen two-function suite |
| CompGen-ID2 | 5 seen composed examples | Seen two-function suite |
| CompGen-OOD1 | 2 single + 3 seen composed examples | Held-out two-function suite |
| CompGen-OOD2 | 5 seen composed examples | Held-out two-function suite |

The table specifies a mix of single and composed examples without a ratio; this
implementation uses 2 + 3. Context always comes from the C1 training split.
For OOD conditions the held-out pair never appears in context. One seen pair and
one held-out pair are evaluated per experiment, with 10 distinct target grids
per pair. The same targets and example-selection seed are used for both models.
By default the seen pair is the held-out pair in reverse order when that pair
was seen in training. Otherwise the script chooses a seen heterogeneous pair
sharing the held-out first function. The chosen target suite and five context
row IDs for every prompt are recorded in manifest JSON files. Pass
`--id-target-suites path/to/suites.json` to specify the ID pair for each
experiment, for example `{"1": ["translate_up", "mirror_horizontal"]}`.

C1 has two-function suites only, so the deeper-composition CompGen-OOD3
condition from the paper is unavailable in these splits.

The paper calls its two task-code regimes **explicit** (transformation names)
and **implicit** (opaque codes such as `t1` and `t2`). The existing runs use
explicit labels. Pass `--embedding implicit` for a new evaluation with stable
opaque codes within each experiment. Examples and test targets show ordered
code lists, while the same IID/OOD base instructions are used in both regimes.
The code map is recorded in each manifest for audit, but no true transformation
names appear in implicit prompts. The same sampled targets and five context
rows are used for both regimes. This implements the coded-label method in
[COGITAO Appendix C.1.1](https://arxiv.org/html/2509.05249v2).

## Commands

From the repository root, for GPT-6 Sol on IID validation and test:

```bash
python -m benchmarks.cogitao_codex_eval.icl \
  --models gpt-6-sol --splits val test --workers 4
```

After filling `prompts/ood.txt`, run OOD validation and test:

```bash
python -m benchmarks.cogitao_codex_eval.icl \
  --models gpt-6-sol --splits val_ood test_ood --workers 4
```

For the paper's implicit condition on GPT-6 Sol, run both commands:

```bash
python -m benchmarks.cogitao_codex_eval.icl \
  --models gpt-6-sol --embedding implicit --splits val test --workers 4
python -m benchmarks.cogitao_codex_eval.icl \
  --models gpt-6-sol --embedding implicit --splits val_ood test_ood --workers 4
```

Implicit results are saved separately under
`artifacts/cogitao_codex_c1_icl_implicit/`. The implicit base prompts are
`prompts/iid_implicit.txt` and `prompts/ood_implicit.txt`.

For GPT-6 Astra, replace `gpt-6-sol` with `gpt-6-astra`. `--workers` controls
the number of concurrent Codex CLI processes; each process uses an isolated
temporary directory. `--targets-per-experiment` defaults to 10. The default
commands create 50 prompts per condition per split, matching five C1
experiment combinations times ten targets. Use `--experiments 1` to narrow to
one experiment.

Results are written to
`artifacts/cogitao_codex_c1_icl/<model>/experiment_N/<split>/<condition>.jsonl`.
Each result is flushed as it finishes, and re-running the same command resumes
saved rows. The output directory also contains CSV reports:

| File | Rows |
| --- | --- |
| `analysis.csv` | One combined IID/OOD table for analysis, with explicit distribution, evaluation set, scope, result type, and transformation columns |
| `table3_c1_overall.csv` | Eight C1 task rows: IID/OOD validation and test totals, shown as exact hits / 50, with Exp. and Imp. columns per model |
| `table3_c1_breakdown.csv` | C1 pooled function and ordered-composition rows for each task and evaluation set, shown as exact hits / saved samples |
| `overall.csv` | Overall accuracy for each experiment, split, and ICL condition; pooled C1 rows use `experiment=all` |
| `by_function.csv` | Accuracy for targets containing each function, including either position in a composition |
| `by_composition.csv` | Accuracy for each ordered two-function target suite, such as `rot90 -> crop_bottom_side` |
| `summary.csv` | All three kinds of rows in one file, identified by `granularity` |

In `analysis.csv`, filter `scope=C1` for scores pooled across the five
experiments within each split and condition, or `scope=experiment` for the
individual suites. Filter `result_type` to `overall`, `function`, or
`composition`. The `evaluation_set` column is `val` or `test` for both IID and
OOD; `source_split` retains the dataset name (for example, `test_ood`).

Each report includes the model, split, condition, sample counts, exact-grid
accuracy, pixel accuracy, and separate timeout/error counts. Function rows count
each target once for each distinct function it contains. Timeouts and CLI errors
are excluded from accuracy denominators and remain visible in the counts.
Add `--retry-failures` to retry saved timeouts and CLI errors. The previous
full-split runner's results under
`artifacts/cogitao_codex_c1/` use a different protocol and are kept separate.

Codex CLI has no custom system-role flag, so the base prompt text is prepended
to each task as instructions. The script removes API-key environment variables
from child processes so it uses the existing Codex CLI login.

Both paper-style tables are regenerated when a standard explicit or implicit
run updates its summary. They cover all five C1 experiments and keep validation
and test separate. The overall table shows a score only once all 50 prompts for
that task and split have saved results; a timeout or CLI error counts as a miss
in its denominator. The breakdown table separates functions from ordered
compositions with `result_type`. Missing regimes stay blank. To rebuild the
tables from existing summary files without model calls, run:

```bash
python -m benchmarks.cogitao_codex_eval.table3_c1
```
