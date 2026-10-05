"""Full per-layer D4 analysis of a trained C1 ViT image encoder.

The model path is strictly ``encode_images``. Dataset function names are used
only to stratify the ID image sample. No function token, function MLP,
cross-attention block, decoder, target, or prediction is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from benchmarks.cogitao_vit_cross_attention.c1_vit_spatial_symmetries import (
    EPS, ViTLayerCollector, checkpoint_digest, numeric_metrics, orbit_key,
    summarize,
)
from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 import (
    _checkpoint_config,
)
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import (
    resize_grids,
)


PROTOCOL_VERSION = 1
D4_ELEMENTS = {
    "e": (0, False),
    "r90": (1, False),
    "r180": (2, False),
    "r270": (3, False),
    "m": (0, True),
    "r90_m": (1, True),
    "r180_m": (2, True),
    "r270_m": (3, True),
}


def d4_transform(grid: np.ndarray, element: str) -> np.ndarray:
    """Apply R^k M^b: optional left-right M, then k clockwise rotations."""
    rotations, mirrored = D4_ELEMENTS[element]
    value = np.flip(grid, axis=1) if mirrored else grid
    return np.rot90(value, k=-rotations).copy()


def align_d4_tokens(tokens: np.ndarray, shape: tuple[int, int],
                    element: str) -> np.ndarray:
    """Apply (R^k M^b)^-1 to transformed token positions."""
    height, width = shape
    if height != width or tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("D4 requires square [batch, H*W, channels] tokens")
    rotations, mirrored = D4_ELEMENTS[element]
    value = np.rot90(tokens.reshape(len(tokens), height, width, -1),
                     k=rotations, axes=(1, 2))
    if mirrored:
        value = np.flip(value, axis=2)
    return value.copy()


def cayley_table() -> dict[tuple[str, str], str]:
    """Return h after g, matching row-vector products rho(g) @ rho(h)."""
    marker = np.arange(25, dtype=np.int64).reshape(5, 5)
    images = {name: d4_transform(marker, name) for name in D4_ELEMENTS}
    table = {}
    for first in D4_ELEMENTS:
        for second in D4_ELEMENTS:
            composed = d4_transform(d4_transform(marker, first), second)
            matches = [name for name, image in images.items()
                       if np.array_equal(composed, image)]
            if len(matches) != 1:
                raise AssertionError("D4 element definitions are not unique")
            table[(first, second)] = matches[0]
    return table


def read_atomic_id_images(root: Path, experiment: int, split: str,
                          per_function: int, seed: int,
                          excluded_orbits: set[str]) -> tuple[list[dict], dict]:
    """Stratify by atomic suite without passing the suite to the model."""
    path = root / "exp_setting_1" / f"experiment_{experiment}" / f"{split}.parquet"
    table = pq.read_table(path, columns=["input", "transformation_suite"])
    suites = [tuple(value) for value in table["transformation_suite"].to_pylist()]
    functions = sorted({suite[0] for suite in suites if len(suite) == 1})
    rng = np.random.default_rng(seed)
    rows, seen = [], set()
    coverage = Counter(total_rows=len(table), atomic_functions=len(functions))
    coverage["atomic_function_names"] = functions
    missing = []
    for function in functions:
        candidates = [index for index, suite in enumerate(suites)
                      if suite == (function,)]
        rng.shuffle(candidates)
        coverage[f"{function}|available"] = len(candidates)
        selected = 0
        for index in candidates:
            grid = np.asarray(table["input"][index].as_py(), dtype=np.int64)
            if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
                coverage[f"{function}|non_square"] += 1
                continue
            if not np.any(grid):
                coverage[f"{function}|empty"] += 1
                continue
            key = orbit_key(grid)
            if key in excluded_orbits or key in seen:
                coverage[f"{function}|excluded_dihedral_orbit"] += 1
                continue
            rows.append(dict(index=index, function=function, input=grid, orbit=key))
            seen.add(key)
            selected += 1
            if selected == per_function:
                break
        coverage[f"{function}|selected"] = selected
        if selected == 0:
            missing.append(function)
    coverage["selected_total"] = len(rows)
    if missing:
        raise ValueError(
            f"No eligible images for atomic functions {missing} in "
            f"experiment {experiment}/{split}"
        )
    if not rows:
        raise ValueError(f"No atomic ID images in experiment {experiment}/{split}")
    return rows, dict(coverage)


def load_c1_model(path: Path, experiment: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    train = config.get("data", {}).get("train", {}).get("init_args", {})
    if int(train.get("setting", -1)) != 1 or int(train.get("experiment", -1)) != experiment:
        raise ValueError(f"Checkpoint is not C1 experiment {experiment}")
    global_step = int(checkpoint.get("global_step", 0))
    if global_step < 1000:
        raise ValueError(f"Checkpoint has only {global_step} training steps")
    model = FunctionConditionedViTCrossAttentionModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, global_step


def layer_names(model) -> list[str]:
    return ["image_input_tokens", *(
        f"image_block_{index:02d}" for index in range(1, model.encoder_layers + 1)
    ), "image_final_norm"]


def fit_d4(model, collector, rows, device, batch_size):
    shape, dimension = tuple(model.input_resolution), model.embedding_dim
    if shape[0] != shape[1]:
        raise ValueError("D4 requires a square model token grid")
    layers = layer_names(model)
    cross = {(element, layer): np.zeros((dimension, dimension), dtype=np.float64)
             for element in D4_ELEMENTS for layer in layers}
    gram = {layer: np.zeros((dimension, dimension), dtype=np.float64)
            for layer in layers}
    token_count = 0
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        grids = [row["input"] for row in chunk]
        source = collector.encode(grids, device, batch_size)
        token_count += len(chunk) * shape[0] * shape[1]
        for layer in layers:
            x = source[layer].reshape(-1, dimension).astype(np.float64)
            gram[layer] += x.T @ x
            cross[("e", layer)] += x.T @ x
        for element in D4_ELEMENTS:
            if element == "e":
                continue
            transformed = collector.encode(
                [d4_transform(grid, element) for grid in grids], device, batch_size)
            for layer in layers:
                x = source[layer].reshape(-1, dimension).astype(np.float64)
                y = align_d4_tokens(transformed[layer], shape, element).reshape(
                    -1, dimension).astype(np.float64)
                cross[(element, layer)] += x.T @ y
    operators, ranks = {}, {}
    for layer in layers:
        rank = int(np.linalg.matrix_rank(gram[layer]))
        ranks[layer] = dict(source_gram_rank=rank, dimension=dimension,
                            underidentified=rank < dimension,
                            matched_training_tokens=token_count)
        operators[("e", layer)] = np.eye(dimension)
        for element in D4_ELEMENTS:
            if element == "e":
                continue
            left, _, right = np.linalg.svd(cross[(element, layer)], full_matrices=False)
            operators[(element, layer)] = left @ right
    return layers, operators, ranks


def evaluate_elements(model, collector, rows, operators, layers, ranks,
                      device, batch_size):
    """Evaluate full-token equivariance for all eight D4 elements."""
    shape = tuple(model.input_resolution)
    fields = ("equivariance", "foreground", "identity", "no_spatial", "closure")
    values = {(element, layer): {field: [] for field in fields}
              for element in D4_ELEMENTS for layer in layers}
    mismatch = {element: [] for element in D4_ELEMENTS}
    source_grams = {layer: np.zeros((model.embedding_dim, model.embedding_dim),
                                    dtype=np.float64) for layer in layers}
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        grids = [row["input"] for row in chunk]
        source = collector.encode(grids, device, batch_size)
        source_grids = resize_grids(grids, shape, device).cpu().numpy()
        for layer in layers:
            flat = source[layer].reshape(-1, model.embedding_dim).astype(np.float64)
            source_grams[layer] += flat.T @ flat
        for element, (_, mirrored) in D4_ELEMENTS.items():
            transformed_grids_raw = [d4_transform(grid, element) for grid in grids]
            transformed = collector.encode(transformed_grids_raw, device, batch_size)
            transformed_grids = resize_grids(
                transformed_grids_raw, shape, device).cpu().numpy()
            aligned_grids = align_d4_tokens(
                transformed_grids.reshape(len(chunk), -1, 1), shape, element
            )[..., 0]
            mismatch[element].extend((source_grids != aligned_grids).mean(
                axis=(1, 2)).tolist())
            foreground = (source_grids != 0) | (aligned_grids != 0)
            order = 1 if element == "e" else 2 if mirrored or element == "r180" else 4
            for layer in layers:
                x = source[layer].reshape(len(chunk), *shape, -1).astype(np.float64)
                y_native = transformed[layer].reshape(
                    len(chunk), *shape, -1).astype(np.float64)
                y = align_d4_tokens(transformed[layer], shape, element).astype(np.float64)
                q = operators[(element, layer)]
                residual = y - x @ q
                denominator = np.maximum(np.linalg.norm(
                    x.reshape(len(chunk), -1), axis=1), EPS)
                axes = (1, 2, 3)
                foreground_norm = np.maximum(np.sqrt(np.sum(
                    x * x * foreground[..., None], axis=axes)), EPS)
                measurements = {
                    "equivariance": np.linalg.norm(
                        residual.reshape(len(chunk), -1), axis=1) / denominator,
                    "foreground": np.sqrt(np.sum(
                        residual * residual * foreground[..., None], axis=axes
                    )) / foreground_norm,
                    "identity": np.linalg.norm(
                        (y - x).reshape(len(chunk), -1), axis=1) / denominator,
                    "no_spatial": np.linalg.norm(
                        (y_native - x).reshape(len(chunk), -1), axis=1) / denominator,
                    "closure": np.linalg.norm((
                        x @ np.linalg.matrix_power(q, order) - x
                    ).reshape(len(chunk), -1), axis=1) / denominator,
                }
                for field, measurement in measurements.items():
                    values[(element, layer)][field].extend(measurement.tolist())
    results, saved = {}, {}
    for element, (_, mirrored) in D4_ELEMENTS.items():
        order = 1 if element == "e" else 2 if mirrored or element == "r180" else 4
        results[element] = dict(order=order, resize_mismatch=summarize(mismatch[element]),
                                layers={})
        for layer in layers:
            result = values[(element, layer)]
            q = operators[(element, layer)]
            results[element]["layers"][layer] = dict(
                samples_scored=len(rows),
                equivariance_defect=summarize(result["equivariance"]),
                foreground_equivariance_defect=summarize(result["foreground"]),
                identity_channel_defect=summarize(result["identity"]),
                no_spatial_permutation_defect=summarize(result["no_spatial"]),
                operator_order_closure_error=float(np.linalg.norm(
                    np.linalg.matrix_power(q, order) - np.eye(len(q))) /
                    np.sqrt(len(q))),
                data_order_closure_defect=summarize(result["closure"]),
                fit_rank=ranks[layer])
            for field in ("equivariance", "foreground", "closure"):
                saved[f"{element}__{layer}__{field}"] = np.asarray(result[field])
    return results, saved, source_grams


def evaluate_group_law(operators, layers, source_grams):
    """Test rho(g)rho(h) ~= rho(h after g) for all 64 D4 products."""
    table = cayley_table()
    results, rows = {}, []
    for layer in layers:
        gram = source_grams[layer]
        energy = max(float(np.trace(gram)), EPS)
        operator_errors, data_errors = [], []
        worst = None
        for first in D4_ELEMENTS:
            for second in D4_ELEMENTS:
                product = table[(first, second)]
                difference = (operators[(first, layer)] @ operators[(second, layer)]
                              - operators[(product, layer)])
                operator_error = float(
                    np.linalg.norm(difference) / np.sqrt(difference.shape[0]))
                data_error = float(np.sqrt(max(
                    np.trace(difference.T @ gram @ difference), 0.0) / energy))
                row = dict(layer=layer, first=first, second=second,
                           product=product, operator_error=operator_error,
                           heldout_data_error=data_error)
                rows.append(row)
                operator_errors.append(operator_error)
                data_errors.append(data_error)
                if worst is None or data_error > worst["heldout_data_error"]:
                    worst = row
        results[layer] = dict(products=64,
                              operator_error=summarize(operator_errors),
                              heldout_data_error=summarize(data_errors),
                              worst_relation=worst)
    return results, rows, table


def fit(model, collector, args, device, fingerprint):
    rows, coverage = read_atomic_id_images(
        args.data_root, args.experiment, "train", args.train_images_per_function,
        args.seed, set())
    layers, operators, ranks = fit_d4(
        model, collector, rows, device, args.batch_size)
    arrays = {f"{element}__{layer}__rho": operator
              for (element, layer), operator in operators.items()}
    metadata = dict(
        protocol_version=PROTOCOL_VERSION, checkpoint_sha256=fingerprint,
        experiment=args.experiment, fit_split="train", fit_images=len(rows),
        function_tokens_used=False, decoder_used=False,
        seed=args.seed, resolution=list(model.input_resolution), layers=layers,
        elements={name: dict(rotation=value[0], reflected=value[1])
                  for name, value in D4_ELEMENTS.items()},
        atomic_functions=coverage["atomic_function_names"],
        ranks=ranks, coverage=coverage,
        convention="rho(g)rho(h) represents applying g then h")
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["fit_orbits"] = np.array(sorted(row["orbit"] for row in rows))
    arrays["fit_indices"] = np.array([row["index"] for row in rows])
    args.probe_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.probe_file, **arrays)
    return arrays, metadata


def evaluate(model, collector, arrays, metadata, args, device):
    operators = {(element, layer): arrays[f"{element}__{layer}__rho"]
                 for element in D4_ELEMENTS for layer in metadata["layers"]}
    excluded = set(arrays["fit_orbits"].tolist())
    report, element_rows, relation_rows = {}, [], []
    for split in args.splits:
        rows, coverage = read_atomic_id_images(
            args.data_root, args.experiment, split, args.eval_images_per_function,
            args.seed, excluded)
        if coverage["atomic_function_names"] != metadata["atomic_functions"]:
            raise ValueError(
                f"Atomic function set in {split} differs from the training set"
            )
        elements, saved, source_grams = evaluate_elements(
            model, collector, rows, operators, metadata["layers"], metadata["ranks"],
            device, args.batch_size)
        group_law, relations, table = evaluate_group_law(
            operators, metadata["layers"], source_grams)
        for row in relations:
            relation_rows.append({"split": split, **row})
        for element, element_result in elements.items():
            for layer, values in element_result["layers"].items():
                element_rows.append(dict(
                    split=split, element=element, order=element_result["order"],
                    layer=layer, samples=values["samples_scored"],
                    resize_mismatch_mean=element_result["resize_mismatch"]["mean"],
                    equivariance_defect_mean=values["equivariance_defect"]["mean"],
                    foreground_equivariance_defect_mean=values[
                        "foreground_equivariance_defect"]["mean"],
                    identity_channel_defect_mean=values[
                        "identity_channel_defect"]["mean"],
                    no_spatial_permutation_defect_mean=values[
                        "no_spatial_permutation_defect"]["mean"],
                    operator_order_closure_error=values[
                        "operator_order_closure_error"],
                    data_order_closure_defect_mean=values[
                        "data_order_closure_defect"]["mean"],
                    source_gram_rank=values["fit_rank"]["source_gram_rank"],
                    dimension=values["fit_rank"]["dimension"]))
        np.savez_compressed(args.output_dir / f"{split}_d4_examples.npz",
                            source_index=np.array([row["index"] for row in rows]),
                            raw_input=np.stack([row["input"] for row in rows]), **saved)
        report[split] = dict(coverage=coverage, elements=elements,
                             group_law=group_law,
                             cayley_table={f"{a} then {b}": c
                                           for (a, b), c in table.items()})
        print(f"Finished experiment {args.experiment} {split}: {len(rows)} images",
              flush=True)
    for filename, rows in (("d4_element_metrics.csv", element_rows),
                           ("d4_group_law_metrics.csv", relation_rows)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def log_wandb(report, report_path: Path, args):
    if args.wandb_mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=args.wandb_group, job_type="vit_d4_symmetry",
        name=f"c1-exp{args.experiment}-vit-d4-{args.mode}",
        tags=["cogitao", "c1", f"experiment-{args.experiment}", "vit", "d4"],
        mode=args.wandb_mode,
        config=dict(experiment=args.experiment, mode=args.mode,
                    checkpoint=str(args.checkpoint),
                    checkpoint_sha256=report["fit"]["checkpoint_sha256"],
                    function_tokens_used=False, function_mlp_used=False,
                    cross_attention_used=False, decoder_used=False,
                    color_permutations_used=False,
                    protocol_version=PROTOCOL_VERSION,
                    train_images_per_function=args.train_images_per_function,
                    eval_images_per_function=args.eval_images_per_function,
                    splits=list(args.splits), batch_size=args.batch_size, seed=args.seed))
    try:
        run.log({"checkpoint/global_step": report["global_step"],
                 **numeric_metrics(report["fit"]["coverage"], "fit/coverage"),
                 **numeric_metrics(report["evaluation"])})
        table_rows = []
        for split, split_result in report["evaluation"].items():
            for element, element_result in split_result["elements"].items():
                for layer, values in element_result["layers"].items():
                    table_rows.append(dict(
                        split=split, element=element, layer=layer,
                        equivariance_defect=values["equivariance_defect"]["mean"],
                        foreground_defect=values[
                            "foreground_equivariance_defect"]["mean"],
                        identity_baseline=values[
                            "identity_channel_defect"]["mean"],
                        no_spatial_baseline=values[
                            "no_spatial_permutation_defect"]["mean"],
                        operator_order_closure=values[
                            "operator_order_closure_error"],
                        data_order_closure=values[
                            "data_order_closure_defect"]["mean"]))
        if table_rows:
            columns = list(table_rows[0])
            run.log({"d4/layer_table": wandb.Table(
                columns=columns,
                data=[[row[column] for column in columns] for row in table_rows])})
        run.summary["function_tokens_used"] = False
        run.summary["function_mlp_used"] = False
        run.summary["cross_attention_used"] = False
        run.summary["decoder_used"] = False
        run.summary["color_permutations_used"] = False
        run.summary["local_report"] = str(report_path)
        if args.wandb_log_artifacts:
            artifact = wandb.Artifact(
                name=f"c1-exp{args.experiment}-vit-d4-{report['fit']['checkpoint_sha256'][:12]}",
                type="analysis", metadata=dict(experiment=args.experiment,
                                               protocol_version=PROTOCOL_VERSION))
            for path in (report_path, args.probe_file,
                         args.output_dir / "d4_element_metrics.csv",
                         args.output_dir / "d4_group_law_metrics.csv"):
                if path.is_file():
                    artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
    finally:
        run.finish()


def default_checkpoint(experiment: int) -> Path:
    return Path(
        "checkpoints/cogitao_vit_cross_attention/"
        f"cogitao_c1_experiment_{experiment}_vit6_function_mlp_cross_attention/"
        "run_000/last.ckpt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("fit", "eval", "all"), default="all")
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"),
                        default=["val", "test"])
    parser.add_argument("--train-images-per-function", type=int, default=128)
    parser.add_argument("--eval-images-per-function", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="cogitao-c1-vit-d4")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    args = parser.parse_args()
    if min(args.train_images_per_function, args.eval_images_per_function,
           args.batch_size) <= 0:
        parser.error("Image counts and batch size must be positive")
    args.checkpoint = args.checkpoint or default_checkpoint(args.experiment)
    args.output_dir = args.output_dir or Path(
        f"artifacts/cogitao_c1_experiment_{args.experiment}_vit_d4")
    args.probe_file = args.probe_file or args.output_dir / "frozen_d4_probes.npz"
    if args.probe_file.suffix != ".npz":
        parser.error("--probe-file must end in .npz")
    if args.mode in ("fit", "all") and args.probe_file.exists():
        parser.error("Probe file exists; use --mode eval or a fresh output directory")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    fingerprint = checkpoint_digest(args.checkpoint)
    model, global_step = load_c1_model(args.checkpoint, args.experiment, device)
    collector = ViTLayerCollector(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode in ("fit", "all"):
            arrays, metadata = fit(model, collector, args, device, fingerprint)
            print(f"Saved D4 probes to {args.probe_file}", flush=True)
        else:
            with np.load(args.probe_file, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            if metadata.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("Frozen probes use an incompatible protocol")
            if metadata["checkpoint_sha256"] != fingerprint:
                raise ValueError("Frozen probes were fitted on a different checkpoint")
            if metadata["experiment"] != args.experiment:
                raise ValueError("Frozen probes belong to a different C1 experiment")
        evaluation = evaluate(
            model, collector, arrays, metadata, args, device
        ) if args.mode != "fit" else {}
    finally:
        collector.close()
    report = dict(
        experiment=args.experiment, checkpoint=str(args.checkpoint),
        global_step=global_step, mode=args.mode, fit=metadata, evaluation=evaluation,
        scope="Trained ViT image encoder only; all eight D4 spatial actions",
        model_path_used="encode_images", function_tokens_used=False,
        function_mlp_used=False, cross_attention_used=False, decoder_used=False,
        color_permutations_used=False)
    report_path = args.output_dir / (
        "fit_report.json" if args.mode == "fit" else "report.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Wrote {report_path}")
    log_wandb(report, report_path, args)


if __name__ == "__main__":
    main()
