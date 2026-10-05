"""Per-layer spatial symmetries of the trained C1 ViT image encoder.

Only ``model.encode_images`` is called. Function tokens, the function MLP,
cross-attention, decoder, targets, and predictions are outside this protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import (
    load_model, resize_grids,
)


SOURCE_SUITE = "translate_up"  # Used only to select a consistent ID image subset.
EPS = 1e-12
PROTOCOL_VERSION = 1


def rotate_clockwise(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-1).copy()


def reflect_left_right(grid: np.ndarray) -> np.ndarray:
    return np.flip(grid, axis=1).copy()


def align_rotation(tokens: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if height != width or tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("Rotation requires square [batch, H*W, channels] tokens")
    return np.rot90(tokens.reshape(len(tokens), height, width, -1),
                    k=1, axes=(1, 2)).copy()


def align_reflection(tokens: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("Reflection requires [batch, H*W, channels] tokens")
    return np.flip(tokens.reshape(len(tokens), height, width, -1), axis=2).copy()


SYMMETRIES = {
    "rotation": dict(transform=rotate_clockwise, align=align_rotation,
                     order=4, identity="R^4=I"),
    "reflection": dict(transform=reflect_left_right, align=align_reflection,
                       order=2, identity="M^2=I"),
}


def orbit_key(grid: np.ndarray) -> str:
    """Canonical D4 hash prevents transformed fit/evaluation overlap."""
    variants = [np.rot90(grid, k) for k in range(4)]
    variants.extend(np.rot90(reflect_left_right(grid), k) for k in range(4))
    return min(hashlib.sha256(
        np.asarray(item, dtype="<i8").tobytes()
    ).hexdigest() for item in variants)


def read_id_images(root: Path, split: str, limit: int, seed: int,
                   excluded_orbits: set[str]) -> tuple[list[dict], dict]:
    path = root / "exp_setting_1" / "experiment_1" / f"{split}.parquet"
    table = pq.read_table(path, columns=["input", "transformation_suite"])
    suites = table["transformation_suite"].to_pylist()
    candidates = [index for index, suite in enumerate(suites)
                  if tuple(suite) == (SOURCE_SUITE,)]
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    coverage = Counter(total_rows=len(table), source_suite_available=len(candidates))
    rows, seen = [], set()
    for index in candidates:
        grid = np.asarray(table["input"][index].as_py(), dtype=np.int64)
        if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
            coverage["non_square"] += 1
            continue
        if not np.any(grid):
            coverage["empty"] += 1
            continue
        key = orbit_key(grid)
        if key in excluded_orbits or key in seen:
            coverage["excluded_duplicate_or_fit_dihedral_orbit"] += 1
            continue
        rows.append(dict(index=index, input=grid, orbit=key))
        seen.add(key)
        if len(rows) == limit:
            break
    coverage["selected"] = len(rows)
    if not rows:
        raise ValueError(f"No eligible ID images in {split}")
    return rows, dict(coverage)


class ViTLayerCollector:
    """Capture full token maps from the ViT image path only."""

    def __init__(self, model):
        self.model = model
        self.values: dict[str, torch.Tensor] = {}
        self.handles = [model.image_encoder.register_forward_pre_hook(
            self._input_hook)]
        for index, layer in enumerate(model.image_encoder.layers, 1):
            self.handles.append(layer.register_forward_hook(
                self._layer_hook(f"image_block_{index:02d}")))

    def _input_hook(self, _module, inputs):
        self.values["image_input_tokens"] = inputs[0].detach()

    def _layer_hook(self, name: str):
        def collect(_module, _inputs, output):
            self.values[name] = output.detach()
        return collect

    def encode(self, grids: list[np.ndarray], device: torch.device,
               batch_size: int) -> dict[str, np.ndarray]:
        if not grids:
            raise ValueError("Cannot encode an empty image list")
        collected: dict[str, list[np.ndarray]] = {}
        for offset in range(0, len(grids), batch_size):
            indices = resize_grids(
                grids[offset:offset + batch_size], self.model.input_resolution, device)
            images = F.one_hot(indices, num_classes=10).permute(0, 3, 1, 2).float()
            self.values.clear()
            with torch.inference_mode():
                final = self.model.encode_images(images)
            self.values["image_final_norm"] = final.detach()
            for name, value in self.values.items():
                collected.setdefault(name, []).append(value.float().cpu().numpy())
        return {name: np.concatenate(parts) for name, parts in collected.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def pair_batches(collector, rows, transform, device, batch_size):
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        grids = [row["input"] for row in chunk]
        yield (chunk,
               collector.encode(grids, device, batch_size),
               collector.encode([transform(grid) for grid in grids],
                                device, batch_size))


def fit_operators(model, collector, rows, device, batch_size):
    shape = tuple(model.input_resolution)
    if shape[0] != shape[1]:
        raise ValueError("The rotation test requires a square token grid")
    layers = ["image_input_tokens", *(
        f"image_block_{index:02d}" for index in range(1, model.encoder_layers + 1)
    ), "image_final_norm"]
    dimension = model.embedding_dim
    operators, ranks = {}, {}
    for symmetry, spec in SYMMETRIES.items():
        cross = {layer: np.zeros((dimension, dimension), dtype=np.float64)
                 for layer in layers}
        gram = {layer: np.zeros((dimension, dimension), dtype=np.float64)
                for layer in layers}
        token_count = 0
        for chunk, source, transformed in pair_batches(
            collector, rows, spec["transform"], device, batch_size
        ):
            token_count += len(chunk) * shape[0] * shape[1]
            for layer in layers:
                x = source[layer].reshape(-1, dimension).astype(np.float64)
                y = spec["align"](transformed[layer], shape).reshape(
                    -1, dimension).astype(np.float64)
                gram[layer] += x.T @ x
                cross[layer] += x.T @ y
        ranks[symmetry] = {}
        for layer in layers:
            left, _, right = np.linalg.svd(cross[layer], full_matrices=False)
            operators[(symmetry, layer)] = left @ right
            rank = int(np.linalg.matrix_rank(gram[layer]))
            ranks[symmetry][layer] = dict(
                source_gram_rank=rank, dimension=dimension,
                underidentified=rank < dimension,
                matched_training_tokens=token_count)
    return layers, operators, ranks


def summarize(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, 0.9)),
                standard_error=float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else None)


def evaluate_symmetry(model, collector, rows, symmetry, operators, ranks,
                      layers, device, batch_size):
    spec = SYMMETRIES[symmetry]
    shape = tuple(model.input_resolution)
    values = {layer: {name: [] for name in (
        "equivariance", "foreground", "identity", "no_spatial", "closure"
    )} for layer in layers}
    resize_mismatch = []
    for chunk, source, transformed in pair_batches(
        collector, rows, spec["transform"], device, batch_size
    ):
        grids = [row["input"] for row in chunk]
        source_grids = resize_grids(grids, shape, device).cpu().numpy()
        transformed_grids = resize_grids(
            [spec["transform"](grid) for grid in grids], shape, device
        ).cpu().numpy()
        aligned_grids = (np.rot90(transformed_grids, k=1, axes=(1, 2))
                         if symmetry == "rotation"
                         else np.flip(transformed_grids, axis=2))
        resize_mismatch.extend((source_grids != aligned_grids).mean(
            axis=(1, 2)).tolist())
        foreground = (source_grids != 0) | (aligned_grids != 0)
        for layer in layers:
            x = source[layer].reshape(len(chunk), *shape, -1).astype(np.float64)
            y_native = transformed[layer].reshape(
                len(chunk), *shape, -1).astype(np.float64)
            y = spec["align"](transformed[layer], shape).astype(np.float64)
            q = operators[(symmetry, layer)]
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
                    x @ np.linalg.matrix_power(q, spec["order"]) - x
                ).reshape(len(chunk), -1), axis=1) / denominator,
            }
            for name, measurement in measurements.items():
                values[layer][name].extend(measurement.tolist())
    results, saved = {}, {}
    for layer in layers:
        q = operators[(symmetry, layer)]
        result = values[layer]
        results[layer] = dict(
            samples_scored=len(rows), identity=spec["identity"],
            equivariance_defect=summarize(result["equivariance"]),
            foreground_equivariance_defect=summarize(result["foreground"]),
            identity_channel_defect=summarize(result["identity"]),
            no_spatial_permutation_defect=summarize(result["no_spatial"]),
            operator_closure_error=float(np.linalg.norm(
                np.linalg.matrix_power(q, spec["order"]) - np.eye(len(q))) /
                np.sqrt(len(q))),
            data_closure_defect=summarize(result["closure"]),
            fit_rank=ranks[symmetry][layer])
        for name in ("equivariance", "foreground", "closure"):
            saved[f"{symmetry}__{layer}__{name}"] = np.asarray(result[name])
    return results, saved, summarize(resize_mismatch)


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit(model, collector, args, device, fingerprint):
    rows, coverage = read_id_images(
        args.data_root, "train", args.train_images, args.seed, set())
    layers, operators, ranks = fit_operators(
        model, collector, rows, device, args.batch_size)
    arrays = {f"{symmetry}__{layer}__rho": operator
              for (symmetry, layer), operator in operators.items()}
    metadata = dict(
        protocol_version=PROTOCOL_VERSION, checkpoint_sha256=fingerprint,
        fit_split="train", image_selection_suite=SOURCE_SUITE,
        function_tokens_used=False, decoder_used=False,
        seed=args.seed, resolution=list(model.input_resolution),
        fit_images=len(rows), layers=layers, ranks=ranks, coverage=coverage,
        symmetries={name: dict(order=spec["order"], identity=spec["identity"])
                    for name, spec in SYMMETRIES.items()},
        convention="row vectors after inverse spatial alignment: H(gx) ~= H(x) @ rho(g)")
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["fit_orbits"] = np.array(sorted(row["orbit"] for row in rows))
    arrays["fit_indices"] = np.array([row["index"] for row in rows])
    args.probe_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.probe_file, **arrays)
    return arrays, metadata


def evaluate(model, collector, arrays, metadata, args, device):
    operators = {(symmetry, layer): arrays[f"{symmetry}__{layer}__rho"]
                 for symmetry in SYMMETRIES for layer in metadata["layers"]}
    excluded = set(arrays["fit_orbits"].tolist())
    report, csv_rows = {}, []
    for split in args.splits:
        rows, coverage = read_id_images(
            args.data_root, split, args.eval_images, args.seed, excluded)
        result = dict(coverage=coverage, symmetries={})
        saved = dict(source_index=np.array([row["index"] for row in rows]),
                     raw_input=np.stack([row["input"] for row in rows]))
        for symmetry in SYMMETRIES:
            layer_results, per_example, mismatch = evaluate_symmetry(
                model, collector, rows, symmetry, operators, metadata["ranks"],
                metadata["layers"], device, args.batch_size)
            result["symmetries"][symmetry] = dict(
                identity=SYMMETRIES[symmetry]["identity"],
                resize_mismatch=mismatch, layers=layer_results)
            saved.update(per_example)
            for layer, values in layer_results.items():
                csv_rows.append(dict(
                    split=split, symmetry=symmetry, identity=values["identity"],
                    layer=layer, samples=values["samples_scored"],
                    equivariance_defect_mean=values["equivariance_defect"]["mean"],
                    foreground_equivariance_defect_mean=values[
                        "foreground_equivariance_defect"]["mean"],
                    operator_closure_error=values["operator_closure_error"],
                    data_closure_defect_mean=values["data_closure_defect"]["mean"],
                    identity_channel_defect_mean=values["identity_channel_defect"]["mean"],
                    no_spatial_permutation_defect_mean=values[
                        "no_spatial_permutation_defect"]["mean"],
                    source_gram_rank=values["fit_rank"]["source_gram_rank"],
                    dimension=values["fit_rank"]["dimension"]))
        np.savez_compressed(args.output_dir / f"{split}_spatial_examples.npz", **saved)
        report[split] = result
        print(f"Finished {split}: {len(rows)} image-only examples", flush=True)
    with (args.output_dir / "spatial_symmetry_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return report


def numeric_metrics(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(numeric_metrics(item, f"{prefix}/{key}" if prefix else key))
        return result
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and np.isfinite(value):
        return {prefix: value.item() if isinstance(value, np.generic) else value}
    return {}


def wandb_rows(report):
    rows = []
    for split, split_result in report["evaluation"].items():
        for symmetry, symmetry_result in split_result["symmetries"].items():
            for layer, values in symmetry_result["layers"].items():
                rows.append(dict(
                    split=split, symmetry=symmetry, identity=values["identity"],
                    layer=layer, samples=values["samples_scored"],
                    equivariance_defect_mean=values["equivariance_defect"]["mean"],
                    foreground_equivariance_defect_mean=values[
                        "foreground_equivariance_defect"]["mean"],
                    operator_closure_error=values["operator_closure_error"],
                    data_closure_defect_mean=values["data_closure_defect"]["mean"]))
    return rows


def log_wandb(report, report_path: Path, args):
    if args.wandb_mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=args.wandb_group, job_type="vit_spatial_symmetry",
        name=f"c1-vit-spatial-symmetry-{args.mode}",
        tags=["cogitao", "c1", "vit", "spatial-symmetry", args.mode],
        mode=args.wandb_mode,
        config=dict(
            mode=args.mode, checkpoint=str(args.checkpoint),
            checkpoint_sha256=report["fit"]["checkpoint_sha256"],
            train_images=report["fit"]["fit_images"], eval_images=args.eval_images,
            splits=list(args.splits), batch_size=args.batch_size, seed=args.seed,
            function_tokens_used=False, decoder_used=False,
            protocol_version=PROTOCOL_VERSION))
    try:
        run.log({"checkpoint/global_step": report["global_step"],
                 **numeric_metrics(report["fit"]["coverage"], "fit/coverage"),
                 **numeric_metrics(report["evaluation"])})
        rows = wandb_rows(report)
        if rows:
            columns = list(rows[0])
            run.log({"spatial_symmetry/layer_table": wandb.Table(
                columns=columns,
                data=[[row[column] for column in columns] for row in rows])})
        run.summary["function_tokens_used"] = False
        run.summary["decoder_used"] = False
        run.summary["local_report"] = str(report_path)
        if args.wandb_log_artifacts:
            artifact = wandb.Artifact(
                name=f"c1-vit-spatial-{report['fit']['checkpoint_sha256'][:12]}-{args.mode}",
                type="analysis",
                metadata=dict(protocol_version=PROTOCOL_VERSION, mode=args.mode))
            artifact.add_file(str(report_path), name=report_path.name)
            artifact.add_file(str(args.probe_file), name=args.probe_file.name)
            for filename in ("spatial_symmetry_metrics.csv", *(
                f"{split}_spatial_examples.npz" for split in report["evaluation"]
            )):
                path = args.output_dir / filename
                if path.is_file():
                    artifact.add_file(str(path), name=filename)
            run.log_artifact(artifact)
    finally:
        run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/cogitao_vit_cross_attention/"
        "cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt"))
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/cogitao_c1_vit_spatial_symmetries"))
    parser.add_argument("--mode", choices=("fit", "eval", "all"), default="all")
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"),
                        default=["val", "test"])
    parser.add_argument("--train-images", type=int, default=256)
    parser.add_argument("--eval-images", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="cogitao-c1-vit-spatial-symmetries")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    args = parser.parse_args()
    if min(args.train_images, args.eval_images, args.batch_size) <= 0:
        parser.error("Image counts and batch size must be positive")
    args.probe_file = args.probe_file or args.output_dir / "frozen_spatial_probes.npz"
    if args.probe_file.suffix != ".npz":
        parser.error("--probe-file must have an .npz extension")
    if args.mode in ("fit", "all") and args.probe_file.exists():
        parser.error("Probe file exists; use --mode eval or a fresh output directory")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    fingerprint = checkpoint_digest(args.checkpoint)
    model, global_step = load_model(
        args.checkpoint, "mlp", device, allow_untrained=False)
    collector = ViTLayerCollector(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode in ("fit", "all"):
            arrays, metadata = fit(model, collector, args, device, fingerprint)
            print(f"Saved frozen spatial probes to {args.probe_file}", flush=True)
        else:
            with np.load(args.probe_file, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            if metadata.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("Frozen probes use an incompatible protocol")
            if metadata["checkpoint_sha256"] != fingerprint:
                raise ValueError("Frozen probes were fitted on a different checkpoint")
        evaluation = evaluate(
            model, collector, arrays, metadata, args, device
        ) if args.mode != "fit" else {}
    finally:
        collector.close()
    report = dict(
        checkpoint=str(args.checkpoint), global_step=global_step, mode=args.mode,
        fit=metadata, evaluation=evaluation,
        scope="Trained ViT image encoder only", model_path_used="encode_images",
        function_tokens_used=False, function_mlp_used=False,
        cross_attention_used=False, decoder_used=False,
        interpretation=("Per-layer full-token spatial equivariance under clockwise "
                        "rotation and left-right reflection, plus R^4=I and M^2=I."))
    report_name = "fit_report.json" if args.mode == "fit" else "report.json"
    report_path = args.output_dir / report_name
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Wrote {report_path}")
    log_wandb(report, report_path, args)


if __name__ == "__main__":
    main()
