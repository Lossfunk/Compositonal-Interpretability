"""Frozen C1 compositional-transfer evaluation; never trains a base model.

ORDERING: pair (f, g) denotes f(g(x)); task tokens are [g, f], or g -> f.
A=M(x,[f]); B=M(gx,[f]); C=M(M(x,[g]),[f]); D=M(x,[g,f]).
B/C/D always share the same oracle target. Reverse support is optional and is
never required for inclusion in the primary behavioral comparison.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
import glob
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import resize_grids
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments import (
    SplitRows,
    collect_atomic_fit,
    encode_chart,
    fit_probes,
    sample_pair,
    grid_key,
    write_csv,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import (
    EPS,
    apply_affine,
    pair_measurements,
    relative_error,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle,
    ORACLE_SOURCE,
    oracle_sequence,
    oracle_states,
)
from benchmarks.cogitao_vit_cross_attention.c1_transfer_metrics import (
    bootstrap_correlation,
    diagnostic_evidence,
    distribution_divergence,
    principal_angle_defect,
    projection_metrics,
    random_subspace,
    rank_choices,
    residual_pca,
    similarity,
)
from benchmarks.cogitao_vit_cross_attention.c1_transfer_paths import (
    ExecutionCollector,
    STAGE_ORDER,
    execute,
    summarize_behavior,
    token_swap,
)
from benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries import (
    default_checkpoint,
    load_c1_model,
)
from benchmarks.cogitao_vit_cross_attention.c1_vit_spatial_symmetries import (
    ViTLayerCollector,
    checkpoint_digest,
)

PROTOCOL_VERSION = 1
CONVENTION = "pair (f,g): fgx=f(g(x)); task-token order [g,f] means g -> f"


CSV_PROVENANCE_FIELDS = (
    "checkpoint_path",
    "checkpoint_sha256",
    "git_commit",
    "experiment_id",
    "split",
    "f",
    "g",
    "program",
    "function_ordering",
    "oracle_valid_count",
    "model_sample_count",
    "sample_count",
    "layer",
    "seed",
    "global_step",
    "probe_fit_sample_count",
    "g_probe_fit_sample_count",
    "ridge",
    "pool_grid",
    "pca_settings",
    "source_kind",
)


def transfer_states(grid, f, g):
    """Use the shared persistent-object oracle with optional reverse support."""
    return oracle_states(grid, f, g, require_reverse=False)


def model_state_keys(grids, model):
    """Additionally prevent distinct raw states becoming identical after resize."""
    keys = set()
    groups = defaultdict(list)
    for grid in grids:
        groups[grid.shape].append(grid)
    for group in groups.values():
        for start in range(0, len(group), 128):
            values = resize_grids(
                group[start : start + 128], model.input_resolution, torch.device("cpu")
            ).numpy()
            keys.update(grid_key(grid) for grid in values)
    return keys


def resize_eligibility(model, excluded):
    def check(grids):
        return (
            "resized_fit_state_overlap"
            if model_state_keys(grids, model) & excluded
            else None
        )

    return check


def sample_transfer(
    data, split, f, g, args, excluded, model=None, resized_excluded=None
):
    """Extend shared selection with optional reverse support and resize overlap."""
    eligibility = (
        resize_eligibility(model, resized_excluded)
        if model is not None and resized_excluded
        else None
    )
    return sample_pair(
        data,
        split,
        f,
        g,
        args,
        excluded,
        require_reverse=False,
        eligibility=eligibility,
    )


def sample_atomic(data, function, args, excluded, model, resized_excluded):
    options = copy(args)
    options.fit_images_per_function = args.eval_images_per_pair
    selected, coverage, _ = collect_atomic_fit(
        data,
        [function],
        options,
        split=data.path.stem,
        excluded_state_keys=excluded,
        eligibility=resize_eligibility(model, resized_excluded),
        minimum_pairs=0,
    )
    return selected[function], coverage[function]


def fit_action_subspaces(
    model, collector, train, functions, args, device, arrays, metadata
):
    if train.path.stem != "train":
        raise ValueError("Action subspaces may only be fitted on atomic train rows")
    pairs, coverage, _ = collect_atomic_fit(train, functions, args)
    diagnostics = {}
    raw_fit_states = []
    for function, rows in pairs.items():
        if not np.array_equal(
            arrays[f"{function}__fit_indices"], [r["index"] for r in rows]
        ):
            raise ValueError("PCA fit rows must be identical to affine-probe fit rows")
        raw_fit_states.extend(
            v for row in rows for v in (row["input"], row["oracle_output"])
        )
        source = encode_chart(
            collector, [r["input"] for r in rows], model, args, device
        )
        target = encode_chart(
            collector, [r["oracle_output"] for r in rows], model, args, device
        )
        diagnostics[function] = {}
        for layer in metadata["layers"]:
            basis, variance, diagnostic = residual_pca(target[layer] - source[layer])
            arrays[f"{function}__{layer}__U"] = basis
            arrays[f"{function}__{layer}__variance"] = variance
            diagnostics[function][layer] = diagnostic
    arrays["resized_fit_state_keys"] = np.array(
        sorted(model_state_keys(raw_fit_states, model))
    )
    metadata["transfer_protocol_version"] = PROTOCOL_VERSION
    metadata["pca_fit_split"] = "train"
    metadata["pca_diagnostics"] = diagnostics
    metadata["pca_ranks"] = args.pca_ranks
    metadata["pca_variance"] = args.pca_variance
    metadata["pca_coverage"] = coverage
    arrays["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(args.probe_file, **arrays)


def common_fields(
    args,
    fingerprint,
    commit,
    global_step,
    metadata,
    split,
    f="",
    g="",
    count=0,
    coverage=None,
    layer="",
):
    coverage = coverage or {}
    return dict(
        checkpoint_path=str(args.checkpoint.resolve()),
        checkpoint_sha256=fingerprint,
        git_commit=commit,
        experiment_id=args.experiment,
        split=split,
        f=f,
        g=g,
        program=f"{g} -> {f}" if g else f,
        function_ordering=CONVENTION,
        oracle_valid_count=coverage.get("oracle_valid", count),
        model_sample_count=count,
        sample_count=count,
        layer=layer,
        seed=args.seed,
        global_step=global_step,
        probe_fit_sample_count=metadata["coverage"].get(f, {}).get("selected", ""),
        g_probe_fit_sample_count=metadata["coverage"].get(g, {}).get("selected", ""),
        ridge=args.ridge,
        pool_grid=args.pool_grid,
        pca_settings=json.dumps(
            dict(fixed_ranks=args.pca_ranks, variance_thresholds=args.pca_variance)
        ),
        source_kind="recorded_ood"
        if split.endswith("_ood")
        else "synthetic_context_on_atomic_id",
    )


def mean_or_none(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def evaluate_atomic(model, collector, arrays, metadata, args, device, base):
    excluded = set(arrays["fit_state_keys"].tolist())
    resized_excluded = set(arrays["resized_fit_state_keys"].tolist())
    baselines, records = {}, []
    for split in [
        s
        for s in ("val", "test")
        if (
            args.data_root
            / "exp_setting_1"
            / f"experiment_{args.experiment}"
            / f"{s}.parquet"
        ).exists()
    ]:
        data = SplitRows(args.data_root, args.experiment, split)
        for function in metadata["functions"]:
            rows, coverage = sample_atomic(
                data, function, args, excluded, model, resized_excluded
            )
            if not rows:
                continue
            source = encode_chart(
                collector, [r["input"] for r in rows], model, args, device
            )
            target = encode_chart(
                collector, [r["oracle_output"] for r in rows], model, args, device
            )
            for layer in metadata["layers"]:
                errors = relative_error(
                    apply_affine(source[layer], arrays[f"{function}__{layer}__A"]),
                    target[layer],
                )
                baselines[(split, function, layer)] = float(errors.mean())
                field = base(
                    split, function, count=len(rows), coverage=coverage, layer=layer
                )
                diagnostic = metadata["diagnostics"][function][layer]
                records.append(
                    {
                        **field,
                        "id_atomic_error": float(errors.mean()),
                        "train_atomic_error": diagnostic["training_relative_error"],
                        "effective_source_rank": diagnostic["rank"],
                        "latent_dimension": diagnostic["dimension"],
                        "underidentified": diagnostic["underidentified"],
                    }
                )
    write_csv(args.output_dir / "atomic_heldout.csv", records)
    return baselines


def behavioral_conditions(model, paths, rows, f, g, args, device):
    grids = {
        name: [row["states"][name] for row in rows] for name in ("x", "fx", "gx", "fgx")
    }
    conditions = {}
    conditions["A"] = execute(model, paths, grids["x"], grids["fx"], (f,), args, device)
    conditions["B"] = execute(
        model, paths, grids["gx"], grids["fgx"], (f,), args, device
    )
    conditions["C_first"] = execute(
        model, paths, grids["x"], grids["gx"], (g,), args, device
    )
    conditions["C"] = execute(
        model,
        paths,
        grids["x"],
        grids["fgx"],
        (f,),
        args,
        device,
        predicted_indices=conditions["C_first"]["predictions"],
    )
    conditions["D"] = execute(
        model, paths, grids["x"], grids["fgx"], (g, f), args, device
    )
    return conditions


def state_sufficiency_strata(saved, fields):
    """Separate raster-recoverable endpoints from persistent-object ambiguity."""
    consistent = saved["oracle_intermediate_reinference_target_match"].astype(bool)
    rows = []
    for name, mask in (
        ("raster_consistent", consistent),
        ("raster_inconsistent", ~consistent),
    ):
        count = int(mask.sum())
        row = dict(
            fields, state_sufficiency=name, model_sample_count=count, sample_count=count
        )
        if count:
            for condition in ("A", "B", "C_first", "C", "D"):
                metrics = {
                    key.split("__", 1)[1]: value[mask]
                    for key, value in saved.items()
                    if key.startswith(condition + "__")
                    and "representation" not in key
                    and not key.endswith("predictions")
                }
                row.update(
                    {
                        condition + "_" + key: value
                        for key, value in summarize_behavior(
                            dict(metrics=metrics)
                        ).items()
                    }
                )
            for metric in ("context_error", "transfer_gap", "composition_error"):
                key = "image_final_norm__" + metric
                row[metric] = mean_or_none(saved[key][mask])
        rows.append(row)
    return rows


def compare_paths(left, right, fields, comparison, indices, artifacts, prefix):
    records = []
    if len(indices) == 0:
        return records
    for stage in left["representations"]:
        x, y = (side["representations"][stage][indices] for side in (left, right))
        per_example, scalar = similarity(x, y)
        if stage.endswith("_attention") or stage == "decoder_logits":
            per_example.update(
                distribution_divergence(x, y, logits=stage == "decoder_logits")
            )
        for key, value in per_example.items():
            artifacts[f"{prefix}__{stage}__{key}"] = value
        records.append(
            {
                **fields,
                "comparison": comparison,
                "layer": stage,
                "model_sample_count": len(indices),
                "sample_count": len(indices),
                **{key: float(value.mean()) for key, value in per_example.items()},
                **scalar,
            }
        )
    pred_equal = (
        (left["predictions"][indices] == right["predictions"][indices])
        .reshape(len(indices), -1)
        .all(axis=1)
    )
    artifacts[f"{prefix}__prediction_equal"] = pred_equal
    for row in records:
        row["prediction_agreement"] = float(pred_equal.mean())
    return records


def action_relations(states, z, af, ag, tolerance):
    """Reuse the existing relation framework on reverse-valid examples only."""
    if "gfx" in states:
        metric, relations = pair_measurements(
            states,
            z,
            af,
            ag,
            tolerance,
            operator_defects=dict(absolute=None, relative=None),
        )
        return metric["commutator_data_defect"], relations
    predicted = dict(
        x=z["x"],
        fx=apply_affine(z["x"], af),
        gx=apply_affine(z["x"], ag),
        fgx=apply_affine(apply_affine(z["x"], ag), af),
    )
    relations = []
    for left, right in combinations(predicted, 2):
        if np.array_equal(states[left], states[right]):
            defect = float(
                np.linalg.norm(predicted[left] - predicted[right])
                / (np.linalg.norm(z["x"]) + EPS)
            )
            relations.append(
                dict(
                    left=left,
                    right=right,
                    defect=defect,
                    structural_tautology=0,
                    retained=int(defect <= tolerance),
                )
            )
    return None, relations


def layer_measurements(
    model,
    collector,
    rows,
    f,
    g,
    arrays,
    metadata,
    args,
    device,
    baselines,
    fields,
    saved,
):
    summaries, subspaces, samples, relations = [], [], [], []
    charts = {
        name: encode_chart(
            collector, [r["states"][name] for r in rows], model, args, device
        )
        for name in ("x", "fx", "gx", "fgx")
    }
    reverse = np.array(
        [i for i, r in enumerate(rows) if "gfx" in r["states"]], dtype=int
    )
    reverse_charts = (
        encode_chart(
            collector, [rows[i]["states"]["gfx"] for i in reverse], model, args, device
        )
        if len(reverse)
        else {}
    )
    reverse_positions = {index: i for i, index in enumerate(reverse)}
    baseline_split = "test" if fields["split"].startswith("test") else "val"
    for layer in metadata["layers"]:
        x, fx, gx, fgx = (charts[name][layer] for name in ("x", "fx", "gx", "fgx"))
        af, ag = (arrays[f"{function}__{layer}__A"] for function in (f, g))
        predicted_g = apply_affine(x, ag)
        predicted_f = apply_affine(x, af)
        composition = apply_affine(predicted_g, af)
        context = relative_error(apply_affine(gx, af), fgx)
        id_error = baselines.get((baseline_split, f, layer))
        g_id_error = baselines.get((baseline_split, g, layer))
        measured = dict(
            context_error=context,
            composition_error=relative_error(composition, fgx),
            matched_atomic_error=relative_error(predicted_f, fx),
            identity_error=relative_error(x, fgx),
            zero_error=relative_error(np.zeros_like(x), fgx),
            only_g_error=relative_error(predicted_g, fgx),
            only_f_error=relative_error(predicted_f, fgx),
            shuffled_operator_error=relative_error(apply_affine(predicted_f, ag), fgx),
        )
        measured["transfer_gap"] = (
            context - id_error if id_error is not None else np.full(len(rows), np.nan)
        )
        g_context = np.full(len(rows), np.nan)
        if len(reverse):
            g_context[reverse] = relative_error(
                apply_affine(fx[reverse], ag), reverse_charts[layer]
            )
        measured["g_context_error"] = g_context
        measured["g_transfer_gap"] = (
            g_context - g_id_error
            if g_id_error is not None
            else np.full(len(rows), np.nan)
        )
        # Optional supervised composition calibration: only the latter half is scored.
        if args.composition_upper_bound and len(rows) >= 4:
            from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import (
                fit_affine,
            )

            cut = len(rows) // 2
            upper, _ = fit_affine(x[:cut], fgx[:cut], args.ridge)
            upper_errors = np.full(len(rows), np.nan)
            upper_errors[cut:] = relative_error(apply_affine(x[cut:], upper), fgx[cut:])
            measured["cheating_composition_upper_bound_error"] = upper_errors
            saved[f"{layer}__cheating_composition_fit_indices"] = np.array(
                [r["index"] for r in rows[:cut]]
            )
        commutator = []
        for i, row in enumerate(rows):
            state_z = {name: value[layer][i] for name, value in charts.items()}
            if i in reverse_positions:
                state_z["gfx"] = reverse_charts[layer][reverse_positions[i]]
            defect, observed = action_relations(
                row["states"], state_z, af, ag, args.relation_tolerance
            )
            commutator.append(defect)
            relations.extend(
                {**fields, "layer": layer, "source_index": row["index"], **r}
                for r in observed
            )
            # f^2=I or f^2=f is tested only when the oracle endpoint establishes it.
            for function, action, once in ((f, af, "fx"), (g, ag, "gx")):
                try:
                    twice = oracle_sequence(row["input"], (function, function))
                except InvalidOracle:
                    continue
                for name in ("x", once):
                    if np.array_equal(twice, row["states"][name]):
                        estimate = apply_affine(apply_affine(x[i], action), action)
                        expected = x[i] if name == "x" else apply_affine(x[i], action)
                        value = float(
                            np.linalg.norm(estimate - expected)
                            / (np.linalg.norm(x[i]) + EPS)
                        )
                        relations.append(
                            {
                                **fields,
                                "layer": layer,
                                "source_index": row["index"],
                                "left": f"{function}^2",
                                "right": "I" if name == "x" else function,
                                "defect": value,
                                "structural_tautology": 0,
                                "retained": int(value <= args.relation_tolerance),
                            }
                        )
        measured["commutator_data_defect"] = np.array(
            [np.nan if v is None else v for v in commutator]
        )
        diagnostic = metadata["diagnostics"][f][layer]
        field = {**fields, "layer": layer}
        summaries.append(
            {
                **field,
                "id_atomic_error": id_error,
                "g_id_atomic_error": g_id_error,
                "atomic_baseline_split": baseline_split,
                **{key: mean_or_none(values) for key, values in measured.items()},
                "commuting_samples": sum(v is not None for v in commutator),
                "reverse_valid_samples": len(reverse),
                "effective_source_rank": diagnostic["rank"],
                "latent_dimension": diagnostic["dimension"],
                "underidentified": diagnostic["underidentified"],
                "probe_train_error": diagnostic["training_relative_error"],
                "upper_bound_label": "CHEATING; fit on first half of this split; score second half only",
                "upper_bound_fit_samples": len(rows) // 2
                if args.composition_upper_bound
                else 0,
            }
        )
        for key, value in measured.items():
            saved[f"{layer}__{key}"] = value
        for i, row in enumerate(rows):
            samples.append(
                {
                    **field,
                    "source_index": row["index"],
                    **{
                        key: float(value[i]) if np.isfinite(value[i]) else None
                        for key, value in measured.items()
                    },
                }
            )
        for function, context_name, residual, atomic_residual in (
            (f, "f_given_g", fgx - gx, fx - x),
            (
                g,
                "g_given_f",
                reverse_charts[layer] - fx[reverse] if len(reverse) else None,
                gx - x,
            ),
        ):
            if residual is None:
                continue
            basis = arrays[f"{function}__{layer}__U"]
            variance = arrays[f"{function}__{layer}__variance"]
            ood_basis, _, ood_diagnostic = residual_pca(residual)
            choices = rank_choices(basis, variance, args.pca_ranks, args.pca_variance)
            if not choices:
                subspaces.append(
                    {
                        **field,
                        "context": context_name,
                        "pca_status": "no_valid_rank",
                        "effective_id_rank": basis.shape[1],
                        "effective_ood_rank": ood_basis.shape[1],
                    }
                )
            for label, rank in choices:
                u = basis[:, :rank]
                proj, energy = projection_metrics(residual, u)
                id_proj, _ = projection_metrics(atomic_residual, u)
                random = random_subspace(
                    len(u), rank, np.random.default_rng(args.seed + rank)
                )
                random_proj, random_energy = projection_metrics(residual, random)
                angle = (
                    principal_angle_defect(u, ood_basis[:, :rank])
                    if ood_basis.shape[1] >= rank
                    else None
                )
                key = f"{layer}__{context_name}__{label}"
                saved[key + "__projection_error"] = proj
                saved[key + "__energy_error"] = energy
                if context_name == "f_given_g":
                    saved[key + "__source_index"] = np.array([r["index"] for r in rows])
                else:
                    saved[key + "__source_index"] = np.array(
                        [rows[i]["index"] for i in reverse]
                    )
                subspaces.append(
                    {
                        **field,
                        "context": context_name,
                        "rank_setting": label,
                        "pca_rank": rank,
                        "model_sample_count": len(residual),
                        "sample_count": len(residual),
                        "id_projection_error": float(id_proj.mean()),
                        "projection_error": float(proj.mean()),
                        "energy_error": float(energy.mean()),
                        "random_projection_error": float(random_proj.mean()),
                        "random_energy_error": float(random_energy.mean()),
                        "principal_angle_defect": angle,
                        "effective_id_rank": basis.shape[1],
                        "effective_ood_rank": ood_diagnostic["effective_rank"],
                        "ood_pca_role": "descriptive_only; never used for ID projection",
                        "pca_status": "ok"
                        if angle is not None
                        else "ood_rank_below_requested_rank",
                    }
                )
    return summaries, subspaces, samples, relations


def correlations_for_examples(samples, subspaces, behavior_by_index, fields, args):
    records = []
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["layer"]].append(row)
    for layer, rows in grouped.items():
        for metric in ("context_error", "composition_error", "transfer_gap"):
            for target in (
                "D_object_error",
                "D_nll",
                "D_confidence",
                "D_brier",
                "B_object_error",
            ):
                values = [
                    behavior_by_index[row["source_index"]][target] for row in rows
                ]
                result = bootstrap_correlation(
                    [r[metric] for r in rows], values, args.seed, args.bootstrap
                )
                records.append(
                    {
                        **fields,
                        "layer": layer,
                        "metric": metric,
                        "behavior_metric": target,
                        "correlation_level": "within_checkpoint_examples",
                        **result,
                        "inference_note": "Examples share a checkpoint; p-values and example bootstrap are descriptive.",
                    }
                )
    # Projection error is per-example. Principal angles are group-level only.
    for metric, value in subspaces.items():
        if not metric.endswith("__f_given_g__variance_0.9__projection_error"):
            continue
        layer = metric.split("__")[0]
        for target in ("D_object_error", "D_nll", "D_confidence", "D_brier"):
            rows = grouped[layer]
            result = bootstrap_correlation(
                value.tolist(),
                [behavior_by_index[r["source_index"]][target] for r in rows],
                args.seed,
                args.bootstrap,
            )
            records.append(
                {
                    **fields,
                    "layer": layer,
                    "metric": "subspace_projection_variance_0.9",
                    "behavior_metric": target,
                    "correlation_level": "within_checkpoint_examples",
                    **result,
                    "inference_note": "Per-example projections, not independent checkpoint replicates.",
                }
            )
    return records


def run_checkpoint(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.probe_file = args.probe_file or args.output_dir / "frozen_atomic_probes.npz"
    if args.mode != "eval" and args.probe_file.exists():
        raise ValueError(
            "Frozen probe file exists; use --mode eval or a fresh output directory"
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.manual_seed(args.seed)
    fingerprint = checkpoint_digest(args.checkpoint)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model, step = load_c1_model(
        args.checkpoint,
        args.experiment,
        device,
        minimum_global_step=0 if args.allow_early_checkpoint else 1000,
    )
    collector, paths = ViTLayerCollector(model), ExecutionCollector(model)
    try:
        train = SplitRows(args.data_root, args.experiment, "train")
        functions = sorted({s[0] for s in train.suites if len(s) == 1})
        # Discover held-out ordered programs from BOTH OOD files, independent of eval splits.
        programs = sorted(
            {
                suite
                for split in ("val_ood", "test_ood")
                for suite in SplitRows(args.data_root, args.experiment, split).suites
                if len(suite) == 2
            }
        )
        if args.mode != "eval":
            arrays, metadata = fit_probes(
                model, collector, train, functions, args, device, fingerprint
            )
            fit_action_subspaces(
                model, collector, train, functions, args, device, arrays, metadata
            )
        else:
            with np.load(args.probe_file, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            expected = dict(
                checkpoint_sha256=fingerprint,
                experiment=args.experiment,
                fit_split="train",
                pca_fit_split="train",
                transfer_protocol_version=PROTOCOL_VERSION,
                pool_grid=args.pool_grid,
                resolution=list(model.input_resolution),
                functions=functions,
                seed=args.seed,
                ridge=args.ridge,
            )
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"Frozen probe mismatch: {key}")

        def base(split, f="", g="", count=0, coverage=None, layer=""):
            return common_fields(
                args,
                fingerprint,
                commit,
                step,
                metadata,
                split,
                f,
                g,
                count,
                coverage,
                layer,
            )

        report = dict(
            protocol_version=PROTOCOL_VERSION,
            convention=CONVENTION,
            checkpoint_path=str(args.checkpoint.resolve()),
            checkpoint_sha256=fingerprint,
            git_commit=commit,
            global_step=step,
            experiment_id=args.experiment,
            fit=metadata,
            heldout_programs=programs,
            oracle_source=ORACLE_SOURCE,
            calibration_unit="pixel; NLL=sum over grid; cross_entropy=mean over grid",
            prediction_target_field="zeros; oracle labels only used after forward",
            foreground="prediction/target nonzero union",
            coverage=[],
            diagnoses=[],
            interpretation_limits=[
                "Observational localization does not establish module causality.",
                "PCA directions come from centered ID training residuals; action projections are uncentered.",
                "Underidentified affine operators are assessed by held-out data defects.",
                "Synthetic ID contexts are not recorded held-out composition rows.",
                "Within-checkpoint examples are not independent experimental replicates.",
                "Cheating composition baselines, when enabled, cannot support transfer claims.",
                "Resizing and oracle execution need not commute; resized labels follow the existing benchmark convention.",
                "Oracle object identity may not be reconstructible from an intermediate raster; reinference match is audited.",
            ],
        )
        if args.mode == "fit":
            (args.output_dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            return [], [], []
        baselines = evaluate_atomic(
            model, collector, arrays, metadata, args, device, base
        )
        behavior_rows, layer_rows, subspace_rows, path_rows = [], [], [], []
        per_behavior, per_latent, relation_rows, correlation_rows, diagnosis_rows = (
            [],
            [],
            [],
            [],
            [],
        )
        excluded = set(arrays["fit_state_keys"].tolist())
        resized_excluded = set(arrays["resized_fit_state_keys"].tolist())
        manifest = {}
        sufficiency_rows = []
        for split in args.splits:
            data = SplitRows(args.data_root, args.experiment, split)
            for g, f in programs:
                rows, coverage = sample_transfer(
                    data, split, f, g, args, excluded, model, resized_excluded
                )
                fields = base(split, f, g, len(rows), coverage)
                report["coverage"].append({**fields, **coverage})
                if not rows:
                    behavior_rows.append(
                        {**fields, "status": "no_eligible_oracle_examples"}
                    )
                    continue
                saved = dict(
                    source_index=np.array([r["index"] for r in rows]),
                    metadata=np.array(json.dumps(fields)),
                )
                conditions = behavioral_conditions(
                    model, paths, rows, f, g, args, device
                )
                behavioral = dict(fields, status="ok")
                for name in ("x", "fx", "gx", "fgx"):
                    raw_grids = [r["states"][name] for r in rows]
                    saved["oracle__" + name + "__raw_shapes"] = np.array(
                        [v.shape for v in raw_grids]
                    )
                    # Keep oracle grids separately from the forward path for auditability.
                    shapes = defaultdict(list)
                    resized = np.empty(
                        (len(rows), *model.input_resolution), dtype=np.int64
                    )
                    for i, value in enumerate(raw_grids):
                        shapes[value.shape].append(i)
                    for indices in shapes.values():
                        resized[indices] = resize_grids(
                            [raw_grids[i] for i in indices],
                            model.input_resolution,
                            torch.device("cpu"),
                        ).numpy()
                    saved["oracle__" + name + "__resized_grid"] = resized
                resize_defects, reinference = [], []
                for i, row in enumerate(rows):
                    try:
                        rederived = oracle_sequence(row["states"]["gx"], (f,))
                        reinference.append(
                            int(np.array_equal(rederived, row["states"]["fgx"]))
                        )
                    except InvalidOracle:
                        reinference.append(0)
                    try:
                        resized_endpoint = oracle_sequence(
                            saved["oracle__x__resized_grid"][i], (g, f)
                        )
                        resize_defects.append(
                            float(
                                np.mean(
                                    resized_endpoint
                                    != saved["oracle__fgx__resized_grid"][i]
                                )
                            )
                        )
                    except InvalidOracle:
                        resize_defects.append(np.nan)
                saved["oracle_intermediate_reinference_target_match"] = np.array(
                    reinference
                )
                saved["resize_oracle_order_pixel_defect"] = np.array(resize_defects)
                behavioral["oracle_intermediate_reinference_match_fraction"] = float(
                    np.mean(reinference)
                )
                behavioral["resize_oracle_order_pixel_defect"] = mean_or_none(
                    resize_defects
                )
                behavioral["resize_oracle_order_valid_samples"] = int(
                    sum(np.isfinite(resize_defects))
                )
                behavioral["input_resize_fraction"] = float(
                    np.mean(
                        [
                            tuple(r["input"].shape) != tuple(model.input_resolution)
                            for r in rows
                        ]
                    )
                )
                for label, result in conditions.items():
                    behavioral.update(
                        {
                            label + "_" + key: value
                            for key, value in summarize_behavior(result).items()
                        }
                    )
                    saved[label + "__predictions"] = result["predictions"]
                    for key, value in result["metrics"].items():
                        saved[label + "__" + key] = value
                    if args.save_representations:
                        for key, value in result["representations"].items():
                            # Float16 affects storage only; similarity always uses float32/64.
                            saved[label + "__representation__" + key] = value.astype(
                                np.float16
                            )
                behavior_rows.append(behavioral)
                behavior_index = {}
                for i, row in enumerate(rows):
                    record = {
                        **fields,
                        "source_index": row["index"],
                        "oracle_intermediate_reinference_target_match": reinference[i],
                    }
                    for label, result in conditions.items():
                        record.update(
                            {
                                label + "_" + key: float(value[i])
                                for key, value in result["metrics"].items()
                                if value.ndim == 1
                            }
                        )
                    per_behavior.append(record)
                    behavior_index[row["index"]] = dict(
                        D_object_error=1 - record["D_object_accuracy"],
                        D_nll=record["D_nll"],
                        D_confidence=record["D_confidence"],
                        D_brier=record["D_brier"],
                        B_object_error=1 - record["B_object_accuracy"],
                    )
                pair_layers, pair_subspaces, samples, relations = layer_measurements(
                    model,
                    collector,
                    rows,
                    f,
                    g,
                    arrays,
                    metadata,
                    args,
                    device,
                    baselines,
                    fields,
                    saved,
                )
                layer_rows.extend(pair_layers)
                subspace_rows.extend(pair_subspaces)
                per_latent.extend(samples)
                relation_rows.extend(relations)
                correlation_rows.extend(
                    correlations_for_examples(
                        samples, saved, behavior_index, fields, args
                    )
                )
                all_indices = np.arange(len(rows))
                path_rows.extend(
                    compare_paths(
                        conditions["B"],
                        conditions["D"],
                        fields,
                        "B_vs_D_all",
                        all_indices,
                        saved,
                        "B_vs_D_all",
                    )
                )
                successful = np.flatnonzero(
                    (conditions["B"]["metrics"]["exact_accuracy"] == 1)
                    & (conditions["D"]["metrics"]["exact_accuracy"] == 0)
                )
                path_rows.extend(
                    compare_paths(
                        conditions["B"],
                        conditions["D"],
                        fields,
                        "B_exact_success_D_exact_failure",
                        successful,
                        saved,
                        "B_success_D_failure",
                    )
                )
                saved["B_success_D_failure__source_index"] = np.array(
                    [rows[i]["index"] for i in successful]
                )
                reverse_indices = np.array(
                    [i for i, r in enumerate(rows) if "gfx" in r["states"]], dtype=int
                )
                if len(reverse_indices):
                    reversed_rows = [rows[i] for i in reverse_indices]
                    reverse_result = execute(
                        model,
                        paths,
                        [r["input"] for r in reversed_rows],
                        [r["states"]["gfx"] for r in reversed_rows],
                        (f, g),
                        args,
                        device,
                    )
                    saved["reverse__source_index"] = np.array(
                        [r["index"] for r in reversed_rows]
                    )
                    saved["reverse__predictions"] = reverse_result["predictions"]
                    for key, value in reverse_result["metrics"].items():
                        saved["reverse__" + key] = value
                    direct_subset = {
                        key: (
                            {
                                name: value[reverse_indices]
                                for name, value in val.items()
                            }
                            if isinstance(val, dict)
                            else val[reverse_indices]
                        )
                        for key, val in conditions["D"].items()
                    }
                    commuting = np.array(
                        [
                            i
                            for i, r in enumerate(reversed_rows)
                            if np.array_equal(r["states"]["fgx"], r["states"]["gfx"])
                        ],
                        dtype=int,
                    )
                    saved["commuting__source_index"] = np.array(
                        [reversed_rows[i]["index"] for i in commuting]
                    )
                    path_rows.extend(
                        compare_paths(
                            direct_subset,
                            reverse_result,
                            fields,
                            "oracle_verified_program_order_invariance",
                            commuting,
                            saved,
                            "order_invariance",
                        )
                    )
                    if args.save_representations:
                        for key, value in reverse_result["representations"].items():
                            saved["reverse__representation__" + key] = value.astype(
                                np.float16
                            )
                # Find a seen program [h,f], then swap its first embedding to g.
                template = next(
                    (
                        suite
                        for suite in sorted(set(train.suites))
                        if len(suite) == 2 and suite[1] == f and suite[0] != g
                    ),
                    None,
                )
                if template:
                    with token_swap(model, 0, g):
                        swapped = execute(
                            model,
                            paths,
                            [r["input"] for r in rows],
                            [r["states"]["fgx"] for r in rows],
                            template,
                            args,
                            device,
                        )
                    swap_summary = summarize_behavior(swapped)
                    path_rows.extend(
                        compare_paths(
                            conditions["D"],
                            swapped,
                            fields,
                            "direct_vs_seen_template_atomic_token_swap",
                            all_indices,
                            saved,
                            "token_swap",
                        )
                    )
                    saved["token_swap__template"] = np.array(template)
                    saved["token_swap__predictions"] = swapped["predictions"]
                    report.setdefault("token_swap", []).append(
                        {
                            **fields,
                            "seen_template": template,
                            "replacement_slot": 0,
                            "replacement": g,
                            **swap_summary,
                            "interpretation": "Swapped input embedding realizes the held-out program; observational behavior, not a rescue claim.",
                        }
                    )
                final_layer = next(
                    r for r in pair_layers if r["layer"] == "image_final_norm"
                )
                diagnosis = {
                    **fields,
                    **diagnostic_evidence(
                        behavioral,
                        final_layer,
                        args.good_accuracy,
                        args.bad_accuracy,
                        args.transfer_gap_threshold,
                        args.low_latent_error,
                    ),
                }
                localization = [
                    r
                    for r in path_rows
                    if r["split"] == split
                    and r["f"] == f
                    and r["g"] == g
                    and r["comparison"] == "B_exact_success_D_exact_failure"
                    and r["layer"] in STAGE_ORDER
                ]
                # Raw inputs/programs differ by design; earliest divergence is descriptive.
                diagnosis["earliest_divergent_stage"] = next(
                    (
                        stage
                        for stage in STAGE_ORDER
                        if any(
                            r["layer"] == stage
                            and r["normalized_l2"] >= args.path_defect_threshold
                            for r in localization
                        )
                    ),
                    None,
                )
                diagnosis["earliest_binding_divergent_stage"] = next(
                    (
                        stage
                        for stage in STAGE_ORDER[3:]
                        if any(
                            r["layer"] == stage
                            and r["normalized_l2"] >= args.path_defect_threshold
                            for r in localization
                        )
                    ),
                    None,
                )
                diagnosis["localization_interpretation"] = (
                    "Different inputs/programs can differ at early stages by design; divergence alone is not a causal failure site."
                )
                diagnosis["oracle_intermediate_reinference_match_fraction"] = (
                    behavioral["oracle_intermediate_reinference_match_fraction"]
                )
                diagnosis["path_defect_threshold"] = args.path_defect_threshold
                diagnosis["B_success_D_failure_samples"] = len(successful)
                diagnosis["state_sufficiency_confounded"] = (
                    behavioral["oracle_intermediate_reinference_match_fraction"] < 1
                )
                sufficiency_rows.extend(state_sufficiency_strata(saved, fields))
                diagnosis_rows.append(diagnosis)
                filename = f"{split}__{g}__{f}__per_example_transfer.npz"
                np.savez_compressed(args.output_dir / filename, **saved)
                manifest[f"{split}:{g}->{f}"] = filename
                print(
                    f"exp{args.experiment} {split} {g} -> {f}: n={len(rows)} "
                    f"A/B/C/D object={behavioral['A_object_accuracy']:.3f}/"
                    f"{behavioral['B_object_accuracy']:.3f}/{behavioral['C_object_accuracy']:.3f}/"
                    f"{behavioral['D_object_accuracy']:.3f}",
                    flush=True,
                )
        for filename, records in (
            ("behavioral_transfer.csv", behavior_rows),
            ("layerwise_transfer.csv", layer_rows),
            ("subspace_transfer.csv", subspace_rows),
            ("program_path_similarity.csv", path_rows),
            ("per_example_behavior.csv", per_behavior),
            ("per_example_latent.csv", per_latent),
            ("oracle_relations.csv", relation_rows),
            ("correlations.csv", correlation_rows),
            ("failure_taxonomy.csv", diagnosis_rows),
            ("state_sufficiency_strata.csv", sufficiency_rows),
            ("coverage.csv", report["coverage"]),
        ):
            empty_fields = list(CSV_PROVENANCE_FIELDS)
            if filename == "oracle_relations.csv":
                empty_fields += [
                    "source_index",
                    "left",
                    "right",
                    "defect",
                    "structural_tautology",
                    "retained",
                ]
            write_csv(
                args.output_dir / filename,
                records,
                fieldnames=empty_fields if not records else None,
            )
        report["diagnoses"] = diagnosis_rows
        report["state_sufficiency_strata"] = sufficiency_rows
        report["per_example_manifest"] = manifest
        report["behavioral_summary"] = behavior_rows
        report["representation_storage"] = (
            "float16 compressed NPZ; metrics computed from float32/64"
            if args.save_representations
            else "disabled"
        )
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        return behavior_rows, layer_rows, subspace_rows
    finally:
        collector.close()
        paths.close()


def pooled_provenance(rows):
    """Aggregate tables identify all contributing checkpoints, not the first one."""
    if not rows:
        return {}
    fields = dict(rows[0])
    fields.update(
        checkpoint_path=json.dumps(sorted({r["checkpoint_path"] for r in rows})),
        checkpoint_sha256=json.dumps(sorted({r["checkpoint_sha256"] for r in rows})),
        experiment_id=json.dumps(sorted({r["experiment_id"] for r in rows})),
        git_commit=json.dumps(sorted({r["git_commit"] for r in rows})),
        contributing_checkpoints=len({r["checkpoint_sha256"] for r in rows}),
        contributing_experiments=len({r["experiment_id"] for r in rows}),
        evaluated_example_count=sum(r["sample_count"] for r in rows),
        oracle_valid_count=sum(r["oracle_valid_count"] for r in rows),
        f=json.dumps(sorted({r["f"] for r in rows})),
        g=json.dumps(sorted({r["g"] for r in rows})),
        program=json.dumps(sorted({r["program"] for r in rows})),
        global_step=json.dumps(sorted({r["global_step"] for r in rows})),
    )
    if "pca_rank" in fields:
        fields["contributing_pca_ranks"] = json.dumps(
            sorted({r["pca_rank"] for r in rows})
        )
    return fields


def aggregate_runs(root, behavior, layers, subspaces, args):
    write_csv(
        root / "cross_c1_summary.csv",
        behavior,
        fieldnames=CSV_PROVENANCE_FIELDS if not behavior else None,
    )
    write_csv(
        root / "behavioral_transfer_summary.csv",
        behavior,
        fieldnames=CSV_PROVENANCE_FIELDS if not behavior else None,
    )
    records = []
    lookup = {(r["checkpoint_sha256"], r["split"], r["program"]): r for r in behavior}
    grouped = defaultdict(list)
    for row in layers:
        if row["split"].endswith("_ood"):
            grouped[(row["split"], row["layer"])].append(row)
    for (split, layer), rows in grouped.items():
        for metric in ("context_error", "composition_error", "transfer_gap"):
            for target in ("D_object_error", "D_nll", "D_confidence", "D_brier"):
                valid = [
                    r
                    for r in rows
                    if lookup.get(
                        (r["checkpoint_sha256"], split, r["program"]), {}
                    ).get("status")
                    == "ok"
                ]
                y = []
                for r in valid:
                    b = lookup[(r["checkpoint_sha256"], split, r["program"])]
                    y.append(
                        1 - b["D_object_accuracy"]
                        if target == "D_object_error"
                        else b[target]
                    )
                result = bootstrap_correlation(
                    [r[metric] for r in valid],
                    y,
                    args.seed,
                    args.bootstrap,
                    clusters=[f"experiment_{r['experiment_id']}" for r in valid],
                )
                records.append(
                    {
                        **pooled_provenance(valid),
                        "layer": layer,
                        "split": split,
                        "metric": metric,
                        "behavior_metric": target,
                        "correlation_level": "checkpoint_program_means",
                        **result,
                        "bootstrap_unit": "experiment_run",
                        "model_sample_count": len(valid),
                        "sample_count": len(valid),
                        "inference_note": "Cluster bootstrap by experiment; training-time checkpoints share a run. Few C1 models limit inference.",
                    }
                )
    # OOD principal angles are group-level; never broadcast them as example metrics.
    for split in args.splits:
        for layer in sorted({r["layer"] for r in subspaces}):
            chosen = [
                r
                for r in subspaces
                if r["split"] == split
                and r["layer"] == layer
                and r.get("context") == "f_given_g"
                and r.get("rank_setting") == "variance_0.9"
            ]
            if not chosen:
                continue
            means = [
                lookup[(r["checkpoint_sha256"], split, r["program"])] for r in chosen
            ]
            result = bootstrap_correlation(
                [r.get("principal_angle_defect") for r in chosen],
                [1 - r["D_object_accuracy"] for r in means],
                args.seed,
                args.bootstrap,
                clusters=[f"experiment_{r['experiment_id']}" for r in chosen],
            )
            records.append(
                {
                    **pooled_provenance(chosen),
                    "metric": "principal_angle_variance_0.9",
                    "behavior_metric": "D_object_error",
                    "correlation_level": "checkpoint_program_means",
                    **result,
                    "bootstrap_unit": "experiment_run",
                    "model_sample_count": len(chosen),
                    "sample_count": len(chosen),
                    "inference_note": "Descriptive OOD subspace fit; few model clusters.",
                }
            )
    write_csv(
        root / "correlations.csv",
        records,
        fieldnames=CSV_PROVENANCE_FIELDS if not records else None,
    )
    readme = Path(__file__).with_name("C1_COMPOSITIONAL_TRANSFER.md")
    (root / "README.md").write_text(
        readme.read_text() if readme.exists() else CONVENTION + "\n"
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--experiment", type=int, choices=range(1, 6))
    group.add_argument("--all-experiments", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="Training trajectory, one experiment",
    )
    parser.add_argument(
        "--checkpoint-glob", help="Training trajectory glob, one experiment"
    )
    parser.add_argument(
        "--allow-early-checkpoint",
        action="store_true",
        help="Permit saved checkpoints before step 1000 for trajectory/baseline evaluation",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/cogitao/files/CompGen")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cogitao_c1_compositional_transfer"),
    )
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--mode", choices=("fit", "eval", "all"), default="all")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing frozen probes; fit missing ones",
    )
    parser.add_argument("--fit-images-per-function", type=int, default=256)
    parser.add_argument("--eval-images-per-pair", type=int, default=128)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test", "val_ood", "test_ood"),
        default=["val", "test", "val_ood", "test_ood"],
    )
    parser.add_argument("--pool-grid", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument(
        "--pca-ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--pca-variance", nargs="*", type=float, default=[0.9, 0.95])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--save-representations", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--composition-upper-bound", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--relation-tolerance", type=float, default=0.05)
    parser.add_argument("--good-accuracy", type=float, default=0.9)
    parser.add_argument("--bad-accuracy", type=float, default=0.7)
    parser.add_argument("--transfer-gap-threshold", type=float, default=0.1)
    parser.add_argument("--low-latent-error", type=float, default=0.1)
    parser.add_argument("--path-defect-threshold", type=float, default=0.1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (
        min(
            args.fit_images_per_function,
            args.eval_images_per_pair,
            args.batch_size,
            args.pool_grid,
        )
        <= 0
        or args.fit_images_per_function < 2
    ):
        parser.error(
            "Positive image counts, pool grid and batch size required; fit count >=2"
        )
    if (
        args.ridge <= 0
        or args.bootstrap < 0
        or any(k < 1 for k in args.pca_ranks)
        or any(not 0 < t <= 1 for t in args.pca_variance)
    ):
        parser.error("Invalid ridge, bootstrap or PCA settings")
    if (
        not 0 <= args.bad_accuracy < args.good_accuracy <= 1
        or min(
            args.transfer_gap_threshold,
            args.low_latent_error,
            args.path_defect_threshold,
            args.relation_tolerance,
        )
        < 0
    ):
        parser.error("Invalid diagnostic thresholds")
    if (
        sum(bool(x) for x in (args.checkpoint, args.checkpoints, args.checkpoint_glob))
        > 1
    ):
        parser.error("Choose --checkpoint, --checkpoints, or --checkpoint-glob")
    if args.all_experiments and any(
        (args.checkpoint, args.checkpoints, args.checkpoint_glob, args.probe_file)
    ):
        parser.error("Explicit checkpoint and probe paths require one experiment")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.probe_file and args.probe_file.suffix != ".npz":
        parser.error("--probe-file must end in .npz")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    trajectories = bool(args.checkpoints or args.checkpoint_glob)
    if trajectories and args.probe_file:
        parser.error("Trajectory probes are stored separately for each checkpoint")
    all_behavior, all_layers, all_subspaces = [], [], []
    for experiment in range(1, 6) if args.all_experiments else [args.experiment]:
        paths = (
            args.checkpoints
            or [Path(p) for p in sorted(glob.glob(args.checkpoint_glob))]
            if trajectories
            else [args.checkpoint or default_checkpoint(experiment)]
        )
        if not paths:
            parser.error("Checkpoint glob matched no files")
        for checkpoint in paths:
            run = copy(args)
            run.experiment = experiment
            run.checkpoint = checkpoint
            run.output_dir = (
                root / f"experiment_{experiment}" if args.all_experiments else root
            )
            if trajectories:
                run.output_dir = run.output_dir / (
                    checkpoint.stem + "__" + checkpoint_digest(checkpoint)[:12]
                )
            if args.resume:
                run.mode = (
                    "eval"
                    if (
                        run.probe_file or run.output_dir / "frozen_atomic_probes.npz"
                    ).exists()
                    else "all"
                )
            b, l, s = run_checkpoint(run)
            all_behavior.extend(b)
            all_layers.extend(l)
            all_subspaces.extend(s)
    aggregate_runs(root, all_behavior, all_layers, all_subspaces, args)
    if trajectories:
        lookup = {
            (r["checkpoint_sha256"], r["split"], r["program"]): r for r in all_behavior
        }
        trajectory = []
        for row in all_layers:
            behavior = lookup[(row["checkpoint_sha256"], row["split"], row["program"])]
            trajectory.append(
                {
                    **row,
                    **{
                        key: value
                        for key, value in behavior.items()
                        if key.startswith(("A_", "B_", "C_", "D_"))
                    },
                }
            )
        write_csv(root / "checkpoint_trajectory.csv", trajectory)
    print(f"Wrote compositional-transfer artifacts to {root}", flush=True)


if __name__ == "__main__":
    main()
