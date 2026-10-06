"""Oracle atomic-pair action transfer and algebra retention at every C1 ViT layer.

Image representations use encode_images only. The conditioned prediction path
is called separately to measure composition object-pixel accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import resize_grids
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import (
    aggregate, correlation, fit_affine, operator_commutator_metrics,
    pair_measurements, predict_states, spatial_chart,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle, ORACLE_SOURCE, STATE_NAMES, oracle_sequence, oracle_states,
)
from benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries import (
    default_checkpoint, layer_names, load_c1_model,
)
from benchmarks.cogitao_vit_cross_attention.c1_vit_spatial_symmetries import (
    ViTLayerCollector, checkpoint_digest, numeric_metrics,
)
from data.cogitao.cogitao import TASK_TO_ID

PROTOCOL_VERSION = 1
CORRELATION_METRICS = (
    "action_transfer_f_relative_l2_mean", "action_transfer_g_relative_l2_mean",
    "action_transfer_f_cosine_distance_mean", "action_transfer_g_cosine_distance_mean",
    "commutator_data_defect_mean", "commutator_operator_relative_defect_mean",
    "oracle_relation_retention_mean", "oracle_relation_defect_mean",
    "fgx_prediction_defect_mean", "f_context_prediction_defect_mean",
    "action_transfer_f_relative_l2_active_mean", "action_transfer_g_relative_l2_active_mean",
)


def grid_key(grid):
    grid = np.asarray(grid, dtype="<i8")
    return hashlib.sha256(str(grid.shape).encode() + grid.tobytes()).hexdigest()


class SplitRows:
    def __init__(self, root, experiment, split):
        self.path = root / "exp_setting_1" / f"experiment_{experiment}" / f"{split}.parquet"
        self.table = pq.read_table(
            self.path, columns=["input", "output", "transformation_suite"], memory_map=True)
        self.suites = [tuple(suite) for suite in
                       self.table["transformation_suite"].to_pylist()]

    def row(self, index):
        return dict(index=int(index), suite=self.suites[index],
                    input=np.asarray(self.table["input"][index].as_py(), dtype=np.int64),
                    output=np.asarray(self.table["output"][index].as_py(), dtype=np.int64))


def audit_target(row, predicted, split, args):
    if np.array_equal(predicted, row["output"]):
        return
    path = args.output_dir / "oracle_audit_failure.npz"
    np.savez_compressed(path, source_index=np.array(row["index"]),
                        split=np.array(split), suite=np.array(row["suite"]),
                        input=row["input"], recorded_output=row["output"],
                        oracle_output=predicted)
    raise ValueError(
        f"Oracle audit failed in {split} row {row['index']}, suite {row['suite']}. "
        "Inferred objects/semantics differ from the recorded target. "
        f"Inspect {path}; no measurements from this mismatch are reported."
    )


def encode_chart(collector, grids, model, args, device):
    """Group raw grid sizes so resize_grids never stacks incompatible arrays."""
    groups = defaultdict(list)
    for index, grid in enumerate(grids):
        groups[grid.shape].append(index)
    result = {}
    for indices in groups.values():
        captured = collector.encode([grids[index] for index in indices], device,
                                    args.batch_size)
        for layer, tokens in captured.items():
            pooled = spatial_chart(tokens, model.input_resolution, args.pool_grid)
            if layer not in result:
                result[layer] = np.empty((len(grids), pooled.shape[1]), dtype=np.float64)
            result[layer][indices] = pooled
    return result


def collect_atomic_fit(data, functions, args):
    rng = np.random.default_rng(args.seed)
    selected, coverage, excluded = {}, {}, set()
    for function in functions:
        candidates = [index for index, suite in enumerate(data.suites)
                      if suite == (function,)]
        rng.shuffle(candidates)
        counts = Counter(available=len(candidates))
        pairs, seen = [], set()
        for index in candidates:
            row = data.row(index)
            key = grid_key(row["input"])
            if key in seen:
                counts["duplicate_input"] += 1
                continue
            try:
                output = oracle_sequence(row["input"], (function,))
            except InvalidOracle as error:
                counts["invalid:" + str(error)] += 1
                continue
            audit_target(row, output, "train", args)
            pairs.append(dict(row, oracle_output=output))
            seen.add(key)
            excluded.update((key, grid_key(output)))
            counts["audited_exact_targets"] += 1
            counts["no_op"] += int(np.array_equal(output, row["input"]))
            if len(pairs) >= args.fit_images_per_function:
                break
        counts["selected"] = len(pairs)
        if len(pairs) < 2:
            raise ValueError(f"Fewer than two audited atomic fit pairs for {function}")
        selected[function], coverage[function] = pairs, dict(counts)
    return selected, coverage, excluded


def fit_probes(model, collector, data, functions, args, device, fingerprint):
    selected, coverage, excluded = collect_atomic_fit(data, functions, args)
    arrays, diagnostics = {}, {}
    for function, pairs in selected.items():
        arrays[f"{function}__fit_indices"] = np.array([row["index"] for row in pairs])
        source = encode_chart(collector, [row["input"] for row in pairs], model, args, device)
        target = encode_chart(collector, [row["oracle_output"] for row in pairs],
                              model, args, device)
        diagnostics[function] = {}
        for layer in layer_names(model):
            operator, diagnostic = fit_affine(source[layer], target[layer], args.ridge)
            arrays[f"{function}__{layer}__A"] = operator
            diagnostics[function][layer] = diagnostic
        print(f"Fitted {function}: {len(pairs)} audited atomic training pairs", flush=True)
    metadata = dict(protocol_version=PROTOCOL_VERSION, experiment=args.experiment,
                    checkpoint_sha256=fingerprint, functions=functions,
                    layers=layer_names(model), pool_grid=args.pool_grid,
                    resolution=list(model.input_resolution), seed=args.seed,
                    ridge=args.ridge, coverage=coverage, diagnostics=diagnostics,
                    fit_split="train", oracle_source=ORACLE_SOURCE,
                    representation_path="encode_images", function_tokens_in_probe=False)
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["fit_state_keys"] = np.array(sorted(excluded))
    args.probe_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.probe_file, **arrays)
    return arrays, metadata


def sample_pair(data, split, f, g, args, excluded):
    """ID uses atomic inputs; OOD uses actual g -> f composition rows."""
    if split.endswith("_ood"):
        candidates = [index for index, suite in enumerate(data.suites) if suite == (g, f)]
    else:
        candidates = [index for index, suite in enumerate(data.suites) if len(suite) == 1]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)
    counts = Counter(available=len(candidates))
    selected, seen = [], set()
    for index in candidates:
        row = data.row(index)
        key = grid_key(row["input"])
        if key in seen:
            counts["duplicate_input"] += 1
            continue
        try:
            states = oracle_states(row["input"], f, g)
            recorded = (states["fgx"] if split.endswith("_ood") else
                        oracle_sequence(row["input"], row["suite"]))
        except InvalidOracle as error:
            counts["invalid:" + str(error)] += 1
            continue
        audit_target(row, recorded, split, args)
        if any(grid_key(value) in excluded for value in states.values()):
            counts["fit_state_overlap"] += 1
            continue
        selected.append(dict(row, states=states))
        seen.add(key)
        counts["oracle_commutes"] += int(np.array_equal(states["fgx"], states["gfx"]))
        counts["f_no_op"] += int(np.array_equal(states["x"], states["fx"]))
        counts["g_no_op"] += int(np.array_equal(states["x"], states["gx"]))
        if len(selected) >= args.eval_images_per_pair:
            break
    counts["selected"] = len(selected)
    counts["audited_exact_targets"] = len(selected) + counts["fit_state_overlap"]
    counts["unexamined_candidates"] = max(len(candidates) - sum(
        count for key, count in counts.items() if key.startswith("invalid:")) -
        counts["duplicate_input"] - counts["fit_state_overlap"] - len(selected), 0)
    return selected, dict(counts)


def composition_accuracy(model, data, indices, suite, args, device):
    """Benchmark object pixels use target/prediction foreground union."""
    if len(suite) > model.max_function_tokens:
        raise ValueError("Composition exceeds the checkpoint's function-token capacity")
    scores = {}
    groups = defaultdict(list)
    for index in indices:
        row = data.row(index)
        groups[(row["input"].shape, row["output"].shape)].append(row)
    for rows in groups.values():
        for offset in range(0, len(rows), args.batch_size):
            chunk = rows[offset:offset + args.batch_size]
            images = resize_grids([row["input"] for row in chunk],
                                  model.input_resolution, device)
            targets = resize_grids([row["output"] for row in chunk],
                                   model.input_resolution, device)
            ids = [TASK_TO_ID[name] for name in suite]
            ids += [0] * (model.max_function_tokens - len(ids))
            tokens = torch.tensor([ids] * len(chunk), device=device, dtype=torch.long)
            with torch.inference_mode():
                predictions = model(dict(
                    images=F.one_hot(images, num_classes=10).permute(0, 3, 1, 2).float(),
                    target_grid=targets, task_tokens=tokens,
                    task_token_mask=tokens != 0))["predictions"]
            correct = predictions == targets
            foreground = (predictions != 0) | (targets != 0)
            counts = foreground.flatten(1).sum(dim=1)
            accuracy = torch.where(counts > 0,
                (correct & foreground).flatten(1).sum(dim=1).float() / counts.clamp_min(1),
                correct.flatten(1).all(dim=1).float())
            for row, score in zip(chunk, accuracy.cpu().tolist()):
                scores[row["index"]] = float(score)
    return scores


def accuracy_fields(all_scores, selected):
    subset = [all_scores[row["index"]] for row in selected if row["index"] in all_scores]
    return dict(composition_accuracy_samples=len(all_scores),
                object_accuracy=float(np.mean(list(all_scores.values()))) if all_scores else None,
                oracle_valid_accuracy_samples=len(subset),
                oracle_valid_object_accuracy=float(np.mean(subset)) if subset else None)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_states(path, rows):
    shapes = np.array([row["states"]["x"].shape for row in rows])
    height, width = shapes.max(axis=0)
    saved = dict(source_index=np.array([row["index"] for row in rows]), raw_shapes=shapes)
    for name in STATE_NAMES:
        padded = np.zeros((len(rows), height, width), dtype=np.int64)
        for index, row in enumerate(rows):
            h, w = shapes[index]
            padded[index, :h, :w] = row["states"][name]
        saved[name] = padded
    np.savez_compressed(path, **saved)


def evaluate(model, collector, arrays, metadata, args, device):
    excluded = set(arrays["fit_state_keys"].tolist())
    summaries, coverage_rows, accuracy_rows = [], [], []
    sample_path = args.output_dir / "per_example_metrics.csv"
    relation_path = args.output_dir / "oracle_relations.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as samples_file, \
            relation_path.open("w", newline="", encoding="utf-8") as relations_file:
        sample_writer = None
        relation_writer = csv.DictWriter(relations_file, fieldnames=[
            "experiment", "split", "f", "g", "source_index", "layer",
            "left", "right", "defect", "structural_tautology", "retained"])
        relation_writer.writeheader()
        for split in args.splits:
            data = SplitRows(args.data_root, args.experiment, split)
            for f, g in product(metadata["functions"], repeat=2):
                selected, coverage = sample_pair(data, split, f, g, args, excluded)
                identity = dict(experiment=args.experiment, split=split, f=f, g=g,
                                composition=f"{g} -> {f}",
                                source_kind=("recorded_ood_composition" if split.endswith("_ood")
                                             else "synthetic_pair_on_id_atomic_input"))
                coverage_rows.append({**identity, **coverage})
                indices = [index for index, suite in enumerate(data.suites) if suite == (g, f)]
                if args.max_accuracy_images:
                    # Include every selected oracle row; add deterministic extras up to the cap.
                    required = {row["index"] for row in selected if row["suite"] == (g, f)}
                    extras = [index for index in indices if index not in required]
                    np.random.default_rng(args.seed).shuffle(extras)
                    indices = sorted(required) + extras[:max(
                        args.max_accuracy_images - len(required), 0)]
                scores = composition_accuracy(model, data, indices, (g, f), args, device)
                accuracy = accuracy_fields(scores, selected)
                accuracy_rows.append({**identity, **accuracy})
                if not selected:
                    print(f"{split}: {g} -> {f}: no eligible oracle examples", flush=True)
                    continue
                if args.save_oracle_states:
                    save_states(args.output_dir / f"{split}__{g}__{f}__states.npz", selected)
                per_layer = defaultdict(list)
                actions = {layer: tuple(arrays[f"{function}__{layer}__A"]
                                        for function in (f, g))
                           for layer in metadata["layers"]}
                operator_defects = {layer: None if f == g else
                                   operator_commutator_metrics(*actions[layer])
                                    for layer in metadata["layers"]}
                for offset in range(0, len(selected), args.batch_size):
                    chunk = selected[offset:offset + args.batch_size]
                    charts = {name: encode_chart(collector,
                        [row["states"][name] for row in chunk], model, args, device)
                              for name in STATE_NAMES}
                    for layer in metadata["layers"]:
                        af, ag = actions[layer]
                        batch_predictions = predict_states(
                            {name: charts[name][layer] for name in STATE_NAMES}, af, ag)
                        for index, row in enumerate(chunk):
                            embeddings = {name: charts[name][layer][index] for name in STATE_NAMES}
                            measurements, relations = pair_measurements(
                                row["states"], embeddings, af, ag, args.relation_tolerance,
                                operator_defects=operator_defects[layer], same_function=f == g,
                                predicted={name: value[index]
                                           for name, value in batch_predictions.items()})
                            fields = {**identity, "source_index": row["index"], "layer": layer}
                            sample = {**fields, **measurements,
                                      "object_accuracy": scores.get(row["index"])}
                            if sample_writer is None:
                                sample_writer = csv.DictWriter(samples_file, fieldnames=list(sample))
                                sample_writer.writeheader()
                            sample_writer.writerow(sample)
                            relation_writer.writerows({**{key: fields[key] for key in
                                ("experiment", "split", "f", "g", "source_index", "layer")},
                                **relation} for relation in relations)
                            per_layer[layer].append(measurements)
                for layer, measurements in per_layer.items():
                    summary = {**identity, "layer": layer, **aggregate(measurements), **accuracy}
                    summaries.append(summary)
                print(f"{split}: {g} -> {f}: {len(selected)} valid oracle examples", flush=True)
    write_csv(args.output_dir / "pair_layer_metrics.csv", summaries)
    write_csv(args.output_dir / "coverage.csv", coverage_rows)
    write_csv(args.output_dir / "composition_accuracy.csv", accuracy_rows)
    return summaries, coverage_rows, accuracy_rows


def compute_correlations(summaries, experiment):
    rows = []
    groups = defaultdict(list)
    for row in summaries:
        if row["split"].endswith("_ood"):
            groups[(row["split"], row["layer"])].append(row)
    for (split, layer), records in groups.items():
        for accuracy_metric in ("object_accuracy", "oracle_valid_object_accuracy"):
            for metric in CORRELATION_METRICS:
                result = correlation([row.get(metric) for row in records],
                                     [row.get(accuracy_metric) for row in records])
                rows.append(dict(experiment=experiment, split=split, layer=layer,
                                 metric=metric, accuracy_metric=accuracy_metric, **result))
    return rows


def log_wandb(report, summaries, correlations, args):
    if args.wandb_mode == "disabled":
        return
    import wandb

    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
        group=args.wandb_group, name=f"c1-exp{args.experiment}-atomic-pairs-{args.mode}",
        job_type="atomic_pair_algebra", mode=args.wandb_mode,
        tags=["c1", "vit", "atomic-pairs", f"experiment-{args.experiment}"],
        config={**{key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
                "checkpoint_sha256": report["fit"]["checkpoint_sha256"],
                "frozen_operator_ridge": report["fit"]["ridge"],
                "representation_path": "encode_images", "operator": "affine_ridge",
                "accuracy_path": "conditioned_model_forward"})
    try:
        def table(records):
            fields = list(dict.fromkeys(key for row in records for key in row))
            return wandb.Table(columns=fields,
                               data=[[row.get(key) for key in fields] for row in records])
        if summaries:
            run.log({"pairs/layer_table": table(summaries)})
        if correlations:
            run.log({"pairs/ood_correlations": table(correlations)})
        for row in summaries:
            prefix = f"{row['split']}/{row['g']}__then__{row['f']}/{row['layer']}"
            run.summary.update(numeric_metrics(row, prefix))
        run.summary["probe_function_tokens_used"] = False
        run.summary["oracle_source"] = ORACLE_SOURCE
        run.summary["global_step"] = report["global_step"]
        if args.wandb_log_artifacts:
            artifact = wandb.Artifact(
                f"c1-exp{args.experiment}-atomic-pairs-{report['fit']['checkpoint_sha256'][:12]}",
                type="analysis", metadata=dict(protocol_version=PROTOCOL_VERSION))
            report_name = "fit_report.json" if args.mode == "fit" else "report.json"
            names = (report_name,) if args.mode == "fit" else (
                report_name, "pair_layer_metrics.csv", "correlations.csv", "coverage.csv",
                "composition_accuracy.csv", "per_example_metrics.csv", "oracle_relations.csv")
            paths = [args.probe_file, *(args.output_dir / name for name in names)]
            for path in paths:
                if path.is_file():
                    artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--mode", choices=("fit", "eval", "all"), default="all")
    parser.add_argument("--fit-images-per-function", type=int, default=256)
    parser.add_argument("--eval-images-per-pair", type=int, default=128)
    parser.add_argument("--max-accuracy-images", type=int, default=0,
                        help="0 evaluates every matching recorded composition row")
    parser.add_argument("--splits", nargs="+", default=["val", "val_ood", "test", "test_ood"],
                        choices=("val", "val_ood", "test", "test_ood"))
    parser.add_argument("--pool-grid", type=int, default=2,
                        help="Ordered K by K spatial means define each layer's latent chart")
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--relation-tolerance", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-oracle-states", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="cogitao-c1-atomic-pair-algebra")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    args = parser.parse_args()
    if min(args.fit_images_per_function, args.eval_images_per_pair, args.batch_size,
           args.pool_grid) <= 0 or args.fit_images_per_function < 2:
        parser.error("Image counts, batch size, and pool grid must be positive; fit count >= 2")
    if (not np.isfinite(args.ridge) or not np.isfinite(args.relation_tolerance) or
            args.ridge <= 0 or args.relation_tolerance < 0 or args.max_accuracy_images < 0):
        parser.error("Ridge must be positive; tolerance and accuracy cap must be nonnegative")
    args.checkpoint = args.checkpoint or default_checkpoint(args.experiment)
    args.output_dir = args.output_dir or Path(
        f"artifacts/cogitao_c1_experiment_{args.experiment}_atomic_pairs")
    args.probe_file = args.probe_file or args.output_dir / "frozen_atomic_probes.npz"
    if args.probe_file.suffix != ".npz":
        parser.error("Probe file must end in .npz")
    if args.mode in ("all", "fit") and args.probe_file.exists():
        parser.error("Frozen probe exists; use --mode eval or a fresh --output-dir")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = checkpoint_digest(args.checkpoint)
    model, global_step = load_c1_model(args.checkpoint, args.experiment, device)
    if args.pool_grid > min(model.input_resolution):
        parser.error("Pool grid exceeds token resolution")
    collector = ViTLayerCollector(model)
    try:
        if args.mode in ("all", "fit"):
            data = SplitRows(args.data_root, args.experiment, "train")
            functions = sorted({suite[0] for suite in data.suites if len(suite) == 1})
            if not functions:
                raise ValueError("No atomic training functions")
            if any(name not in TASK_TO_ID for name in functions):
                raise ValueError("Unknown atomic function")
            arrays, metadata = fit_probes(model, collector, data, functions, args,
                                         device, fingerprint)
            del data
        else:
            with np.load(args.probe_file, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            if (metadata["protocol_version"] != PROTOCOL_VERSION or
                    metadata["checkpoint_sha256"] != fingerprint or
                    metadata["experiment"] != args.experiment or
                    metadata["pool_grid"] != args.pool_grid or
                    metadata["resolution"] != list(model.input_resolution) or
                    metadata["layers"] != layer_names(model)):
                raise ValueError(
                    "Frozen probe protocol, checkpoint, experiment, pooling, or layers differ")
        summaries, coverage, accuracy = evaluate(model, collector, arrays, metadata, args,
                                                 device) if args.mode != "fit" else ([], [], [])
    finally:
        collector.close()
    correlations = compute_correlations(summaries, args.experiment)
    write_csv(args.output_dir / "correlations.csv", correlations)
    report = dict(protocol_version=PROTOCOL_VERSION, experiment=args.experiment,
                  checkpoint=str(args.checkpoint), global_step=global_step,
                  fit=metadata, evaluation=dict(pair_layers=summaries, coverage=coverage,
                                                accuracy=accuracy, correlations=correlations),
                  evaluation_settings=dict(splits=args.splits, seed=args.seed,
                      eval_images_per_pair=args.eval_images_per_pair,
                      max_accuracy_images=args.max_accuracy_images,
                      accuracy_scale="fraction_0_to_1"),
                  convention="fgx=f(g(x)); recorded suite g -> f",
                  probe_path="encode_images", probe_function_tokens_used=False,
                  accuracy_path="conditioned_model_forward", oracle_source=ORACLE_SOURCE,
                  relation_tolerance=args.relation_tolerance,
                  limitations=["Objects inferred from eight-connected foreground components",
                               "Recorded targets audit the inferred object model",
                               "Operators act in a spatially pooled affine chart",
                               "Commutators scored only on oracle-commuting inputs",
                               "No-op actions and low operator fit rank are recorded"])
    report_path = args.output_dir / ("fit_report.json" if args.mode == "fit" else "report.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log_wandb(report, summaries, correlations, args)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
