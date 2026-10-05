"""Atomic C1 task isomorphism: whole-grid clockwise rotation sends up to right.

Fit orthogonal maps and direction readouts on ID train only, then freeze them.
This first-stage study evaluates atomic ID val/test and their rotated copies.
It does not treat the dataset's object-local rot90 as a global symmetry.
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

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import (
    load_model, model_batch, resize_grids,
)

UP = "translate_up"
RIGHT = "translate_right"
EPS = 1e-12


def rotate_clockwise(grid: np.ndarray) -> np.ndarray:
    """The external symmetry g, not COGITAO's object-local rot90 token."""
    return np.rot90(grid, k=-1).copy()


def reflect_left_right(grid: np.ndarray) -> np.ndarray:
    """Reflect across the vertical axis; this preserves translate_up."""
    return np.flip(grid, axis=1).copy()


def translate(grid: np.ndarray, direction: str) -> np.ndarray:
    """Move every foreground cell one cell without wrapping or clipping."""
    if direction not in (UP, RIGHT):
        raise ValueError(f"Unsupported direction: {direction}")
    if (direction == UP and np.any(grid[0] != 0)) or (
        direction == RIGHT and np.any(grid[:, -1] != 0)
    ):
        raise ValueError("Translation would leave the grid")
    output = np.zeros_like(grid)
    if direction == UP:
        output[:-1] = grid[1:]
    else:
        output[:, 1:] = grid[:, :-1]
    return output


def orbit_key(grid: np.ndarray) -> str:
    """Keep every rotation/reflection of a fit grid out of evaluation."""
    transforms = [np.rot90(grid, k) for k in range(4)]
    transforms.extend(np.rot90(reflect_left_right(grid), k) for k in range(4))
    return min(hashlib.sha256(
        np.asarray(transformed, dtype="<i8").tobytes()
    ).hexdigest() for transformed in transforms)


def read_atomic(root: Path, split: str, per_task: int, seed: int,
                excluded: set[str]) -> tuple[list[dict], dict]:
    path = root / "exp_setting_1" / "experiment_1" / f"{split}.parquet"
    table = pq.read_table(path, columns=["input", "output", "transformation_suite"])
    suites = table["transformation_suite"].to_pylist()
    rng = np.random.default_rng(seed)
    rows = []
    coverage = Counter(total_rows=len(table))
    for task in (UP, RIGHT):
        candidates = [i for i, suite in enumerate(suites) if tuple(suite) == (task,)]
        coverage[f"{task}|available"] = len(candidates)
        rng.shuffle(candidates)
        seen: set[str] = set()
        accepted = 0
        for index in candidates:
            grid = np.asarray(table["input"][index].as_py(), dtype=np.int64)
            target = np.asarray(table["output"][index].as_py(), dtype=np.int64)
            if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
                raise ValueError("This C4 experiment requires square raw grids")
            key = orbit_key(grid)
            if key in excluded or key in seen:
                coverage[f"{task}|excluded_duplicate_or_fit_dihedral_orbit"] += 1
                continue
            try:
                exact = translate(grid, task)
            except ValueError:
                coverage[f"{task}|invalid_boundary"] += 1
                continue
            if not np.array_equal(exact, target):
                coverage[f"{task}|recorded_target_mismatch"] += 1
                continue
            if not np.any(grid):
                coverage[f"{task}|empty"] += 1
                continue
            rows.append(dict(index=index, split=split, task=task, input=grid,
                             target=target, orbit=key))
            seen.add(key)
            accepted += 1
            if accepted == per_task:
                break
        coverage[f"{task}|selected"] = accepted
    return rows, dict(coverage)


def isomorphic_pair(row: dict) -> tuple[np.ndarray, np.ndarray]:
    if row["task"] != UP:
        raise ValueError("The first-stage isomorphism uses translate_up sources")
    gx, gy = rotate_clockwise(row["input"]), rotate_clockwise(row["target"])
    if not np.array_equal(translate(gx, RIGHT), gy):
        raise AssertionError("g(T_up(x)) != T_right(g(x))")
    return gx, gy


def reflected_pair(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """M commutes with translate_up for a left-right reflection."""
    if row["task"] != UP:
        raise ValueError("The reflection test uses translate_up sources")
    mx = reflect_left_right(row["input"])
    my = reflect_left_right(row["target"])
    if not np.array_equal(translate(mx, UP), my):
        raise AssertionError("M(T_up(x)) != T_up(M(x))")
    return mx, my


class LayerCollector:
    """Observe the existing forward pass; no checkpoint/model modifications.

    Pooled features support the task readout. Spatial capture retains the
    complete image-token map for a separate, position-aware equivariance test.
    """

    def __init__(self, model):
        self.model = model
        self.values: dict[str, torch.Tensor] = {}
        self.capture_spatial = False
        self.handles = []
        self.handles.append(model.image_encoder.register_forward_pre_hook(
            lambda module, inputs: self.hook("image_input_tokens", "image")(
                module, inputs, inputs[0])))
        for index, layer in enumerate(model.image_encoder.layers, 1):
            self.handles.append(layer.register_forward_hook(
                self.hook(f"image_block_{index:02d}", "image")))
        for index, layer in enumerate(model.function_mlp, 1):
            self.handles.append(layer.register_forward_hook(
                self.hook(f"function_mlp_{index:02d}", "vector")))
        self.handles.append(model.image_to_shared.register_forward_hook(
            self.hook("shared_cross_attention", "queries")))
        self.handles.append(model.shared_to_output.register_forward_hook(
            self.hook("output_cross_attention", "queries")))

    def hook(self, name: str, kind: str):
        def collect(_module, _inputs, output):
            tokens = output[0] if isinstance(output, tuple) else output
            if kind == "image":
                mask = self.foreground.to(tokens.dtype).unsqueeze(-1)
                value = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
                if self.capture_spatial:
                    self.values[f"{name}__spatial"] = tokens.detach()
            elif kind == "queries":
                value = tokens.mean(1)
            else:
                value = tokens
            self.values[name] = value.detach()
        return collect

    def run(self, rows: list[dict], device, batch_size: int, rotated: bool = False,
            mirrored: bool = False, spatial: bool = False):
        if rotated and mirrored:
            raise ValueError("Choose at most one spatial perturbation")
        self.capture_spatial = spatial
        features: dict[str, list[np.ndarray]] = {}
        predictions, targets = [], []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            if rotated:
                pairs = [isomorphic_pair(row) for row in chunk]
                inputs, labels = zip(*pairs)
                suites = [(RIGHT,)] * len(chunk)
            elif mirrored:
                pairs = [reflected_pair(row) for row in chunk]
                inputs, labels = zip(*pairs)
                suites = [(UP,)] * len(chunk)
            else:
                inputs = [row["input"] for row in chunk]
                labels = [row["target"] for row in chunk]
                suites = [(row["task"],) for row in chunk]
            batch = model_batch(self.model, list(inputs), list(labels), suites, device)
            self.foreground = batch["images"].argmax(1).flatten(1).ne(0)
            self.values.clear()
            with torch.inference_mode():
                output = self.model(batch)
                self.hook("image_final_norm", "image")(
                    None, None, output["image_latents"])
            for name, value in self.values.items():
                features.setdefault(name, []).append(value.float().cpu().numpy())
            predictions.append(output["predictions"].cpu().numpy())
            targets.append(batch["target_grid"].cpu().numpy())
        if not rows:
            raise ValueError("No eligible rows remain after geometry/leakage filters")
        self.capture_spatial = False
        return ({name: np.concatenate(parts) for name, parts in features.items()},
                np.concatenate(predictions), np.concatenate(targets))

    def close(self):
        for handle in self.handles:
            handle.remove()


def fit_orthogonal(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Uncentered Procrustes: min ||XQ-Y||_F with Q.T Q = I (row vectors)."""
    x, y = x.astype(np.float64), y.astype(np.float64)
    left, _, right = np.linalg.svd(x.T @ y, full_matrices=False)
    return left @ right, int(np.linalg.matrix_rank(x))


def align_clockwise_tokens(tokens: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Undo the image's clockwise rotation on its token positions."""
    height, width = shape
    if height != width or tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("Expected square [batch, H*W, channels] image tokens")
    return np.rot90(tokens.reshape(len(tokens), height, width, -1),
                    k=1, axes=(1, 2)).copy()


def align_reflected_tokens(tokens: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Undo a left-right reflection on token positions (M is self-inverse)."""
    height, width = shape
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError("Expected [batch, H*W, channels] image tokens")
    return np.flip(tokens.reshape(len(tokens), height, width, -1), axis=2).copy()


def spatial_pair_batches(collector, rows, device, batch_size):
    """Keep only one small batch of full image maps in memory at a time."""
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        source, _, _ = collector.run(chunk, device, batch_size, spatial=True)
        transported, _, _ = collector.run(
            chunk, device, batch_size, rotated=True, spatial=True)
        yield chunk, source, transported


def mirror_pair_batches(collector, rows, device, batch_size):
    """Yield source and reflected full image-token maps in bounded batches."""
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset:offset + batch_size]
        source, _, _ = collector.run(chunk, device, batch_size, spatial=True)
        reflected, _, _ = collector.run(
            chunk, device, batch_size, mirrored=True, spatial=True)
        yield chunk, source, reflected


def fit_spatial_image_maps(model, collector, rows, device, batch_size, layers):
    """Fit Q after matching each rotated image token to its source position."""
    shape = tuple(model.input_resolution)
    if shape[0] != shape[1]:
        raise ValueError("The C4 spatial-token test requires square model grids")
    dimension = model.embedding_dim
    cross = {name: np.zeros((dimension, dimension), dtype=np.float64) for name in layers}
    gram = {name: np.zeros((dimension, dimension), dtype=np.float64) for name in layers}
    token_count = 0
    for chunk, source, transported in spatial_pair_batches(
        collector, rows, device, batch_size
    ):
        token_count += len(chunk) * shape[0] * shape[1]
        for name in layers:
            x = source[f"{name}__spatial"].reshape(-1, dimension).astype(np.float64)
            y = align_clockwise_tokens(transported[f"{name}__spatial"], shape)
            y = y.reshape(-1, dimension).astype(np.float64)
            gram[name] += x.T @ x
            cross[name] += x.T @ y
    operators, ranks = {}, {}
    for name in layers:
        left, _, right = np.linalg.svd(cross[name], full_matrices=False)
        operators[name] = left @ right
        rank = int(np.linalg.matrix_rank(gram[name]))
        ranks[name] = dict(source_gram_rank=rank, dimension=dimension,
                           underidentified=rank < dimension,
                           matched_training_tokens=token_count)
    return operators, ranks


def fit_spatial_mirror_maps(model, collector, rows, device, batch_size, layers):
    """Fit rho(M) after matching left-right reflected token positions."""
    shape = tuple(model.input_resolution)
    dimension = model.embedding_dim
    cross = {name: np.zeros((dimension, dimension), dtype=np.float64) for name in layers}
    gram = {name: np.zeros((dimension, dimension), dtype=np.float64) for name in layers}
    token_count = 0
    for chunk, source, reflected in mirror_pair_batches(
        collector, rows, device, batch_size
    ):
        token_count += len(chunk) * shape[0] * shape[1]
        for name in layers:
            x = source[f"{name}__spatial"].reshape(-1, dimension).astype(np.float64)
            y = align_reflected_tokens(reflected[f"{name}__spatial"], shape)
            y = y.reshape(-1, dimension).astype(np.float64)
            gram[name] += x.T @ x
            cross[name] += x.T @ y
    operators, ranks = {}, {}
    for name in layers:
        left, _, right = np.linalg.svd(cross[name], full_matrices=False)
        operators[name] = left @ right
        rank = int(np.linalg.matrix_rank(gram[name]))
        ranks[name] = dict(source_gram_rank=rank, dimension=dimension,
                           underidentified=rank < dimension,
                           matched_training_tokens=token_count)
    return operators, ranks


def evaluate_spatial_image_maps(model, collector, rows, arrays, metadata,
                                device, batch_size):
    """Per-image Frobenius defects after spatially aligning all H*W tokens."""
    shape = tuple(model.input_resolution)
    layers = metadata["spatial_image_layers"]
    defects = {name: dict(all=[], foreground=[], identity=[], no_spatial=[], closure=[])
               for name in layers}
    mismatch = []
    for chunk, source, transported in spatial_pair_batches(
        collector, rows, device, batch_size
    ):
        original_grids = resize_grids(
            [row["input"] for row in chunk], shape, device).cpu().numpy()
        rotated_grids = resize_grids(
            [isomorphic_pair(row)[0] for row in chunk], shape, device).cpu().numpy()
        aligned_grids = np.rot90(rotated_grids, k=1, axes=(1, 2))
        mismatch.extend((original_grids != aligned_grids).mean(axis=(1, 2)).tolist())
        foreground = (original_grids != 0) | (aligned_grids != 0)
        for name in layers:
            x = source[f"{name}__spatial"].reshape(len(chunk), *shape, -1).astype(np.float64)
            y_native = transported[f"{name}__spatial"].reshape(
                len(chunk), *shape, -1).astype(np.float64)
            y = align_clockwise_tokens(transported[f"{name}__spatial"], shape)
            y = y.astype(np.float64)
            predicted = x @ arrays[f"{name}__spatial_rho"]
            residual = y - predicted
            axes = (1, 2, 3)
            denominator = np.maximum(np.linalg.norm(x.reshape(len(chunk), -1), axis=1), EPS)
            foreground_norm = np.maximum(np.sqrt(np.sum(
                x * x * foreground[..., None], axis=axes)), EPS)
            defects[name]["all"].extend((
                np.linalg.norm(residual.reshape(len(chunk), -1), axis=1) / denominator).tolist())
            defects[name]["foreground"].extend((np.sqrt(np.sum(
                residual * residual * foreground[..., None], axis=axes
            )) / foreground_norm).tolist())
            defects[name]["identity"].extend((np.linalg.norm(
                (y - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
            defects[name]["no_spatial"].extend((np.linalg.norm(
                (y_native - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
            cycle = np.linalg.matrix_power(arrays[f"{name}__spatial_rho"], 4)
            defects[name]["closure"].extend((np.linalg.norm(
                (x @ cycle - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
    results, saved = {}, {}
    for name in layers:
        values = defects[name]
        q = arrays[f"{name}__spatial_rho"]
        results[name] = dict(
            samples_scored=len(rows),
            equivariance_defect=summarize(np.array(values["all"])),
            foreground_equivariance_defect=summarize(np.array(values["foreground"])),
            identity_channel_defect=summarize(np.array(values["identity"])),
            no_spatial_permutation_defect=summarize(np.array(values["no_spatial"])),
            fit_rank=metadata["spatial_image_ranks"][name],
            c4_channel_closure_error=float(np.linalg.norm(
                np.linalg.matrix_power(q, 4) - np.eye(len(q))) / np.sqrt(len(q))),
            c4_data_closure_defect=summarize(np.array(values["closure"])))
        saved[f"{name}__spatial_per_example_defect"] = np.array(values["all"])
        saved[f"{name}__spatial_foreground_per_example_defect"] = np.array(
            values["foreground"])
    return results, dict(per_example=saved,
                         input_rotation_resize_mismatch=summarize(np.array(mismatch)))


def evaluate_spatial_mirror_maps(model, collector, rows, arrays, metadata,
                                 device, batch_size):
    """Score reflection equivariance and the pure identity rho(M)^2 ~= I."""
    shape = tuple(model.input_resolution)
    layers = metadata["spatial_image_layers"]
    defects = {name: dict(all=[], foreground=[], identity=[], no_spatial=[], closure=[])
               for name in layers}
    mismatch = []
    for chunk, source, reflected in mirror_pair_batches(
        collector, rows, device, batch_size
    ):
        original_grids = resize_grids(
            [row["input"] for row in chunk], shape, device).cpu().numpy()
        reflected_grids = resize_grids(
            [reflected_pair(row)[0] for row in chunk], shape, device).cpu().numpy()
        aligned_grids = np.flip(reflected_grids, axis=2)
        mismatch.extend((original_grids != aligned_grids).mean(axis=(1, 2)).tolist())
        foreground = (original_grids != 0) | (aligned_grids != 0)
        for name in layers:
            x = source[f"{name}__spatial"].reshape(len(chunk), *shape, -1).astype(np.float64)
            y_native = reflected[f"{name}__spatial"].reshape(
                len(chunk), *shape, -1).astype(np.float64)
            y = align_reflected_tokens(reflected[f"{name}__spatial"], shape).astype(
                np.float64)
            q = arrays[f"{name}__mirror_rho"]
            residual = y - x @ q
            axes = (1, 2, 3)
            denominator = np.maximum(np.linalg.norm(x.reshape(len(chunk), -1), axis=1), EPS)
            foreground_norm = np.maximum(np.sqrt(np.sum(
                x * x * foreground[..., None], axis=axes)), EPS)
            defects[name]["all"].extend((np.linalg.norm(
                residual.reshape(len(chunk), -1), axis=1) / denominator).tolist())
            defects[name]["foreground"].extend((np.sqrt(np.sum(
                residual * residual * foreground[..., None], axis=axes
            )) / foreground_norm).tolist())
            defects[name]["identity"].extend((np.linalg.norm(
                (y - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
            defects[name]["no_spatial"].extend((np.linalg.norm(
                (y_native - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
            square = np.linalg.matrix_power(q, 2)
            defects[name]["closure"].extend((np.linalg.norm(
                (x @ square - x).reshape(len(chunk), -1), axis=1) / denominator).tolist())
    results, saved = {}, {}
    for name in layers:
        values = defects[name]
        q = arrays[f"{name}__mirror_rho"]
        results[name] = dict(
            samples_scored=len(rows),
            equivariance_defect=summarize(np.array(values["all"])),
            foreground_equivariance_defect=summarize(np.array(values["foreground"])),
            identity_channel_defect=summarize(np.array(values["identity"])),
            no_spatial_permutation_defect=summarize(np.array(values["no_spatial"])),
            fit_rank=metadata["spatial_mirror_ranks"][name],
            m2_channel_closure_error=float(np.linalg.norm(
                np.linalg.matrix_power(q, 2) - np.eye(len(q))) / np.sqrt(len(q))),
            m2_data_closure_defect=summarize(np.array(values["closure"])))
        saved[f"{name}__mirror_per_example_defect"] = np.array(values["all"])
        saved[f"{name}__mirror_foreground_per_example_defect"] = np.array(
            values["foreground"])
    return results, dict(per_example=saved,
                         input_reflection_resize_mismatch=summarize(np.array(mismatch)))


def fit_direction_readout(x: np.ndarray, labels: np.ndarray, ridge: float):
    """Balanced ridge readout: positive means right; negative means up."""
    if not np.any(labels == 1) or not np.any(labels == -1):
        raise ValueError("Direction readout requires both up and right ID examples")
    mean = x.mean(0).astype(np.float64)
    scale = x.std(0).astype(np.float64)
    scale[scale < 1e-6] = 1.0
    design = np.column_stack(((x - mean) / scale, np.ones(len(x))))
    weights = np.where(labels == 1, 0.5 / (labels == 1).sum(),
                       0.5 / (labels == -1).sum())
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0
    coef = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                           design.T @ (weights * labels))
    return mean, scale, coef


def direction_scores(x, mean, scale, coef):
    return np.column_stack(((x - mean) / scale, np.ones(len(x)))) @ coef


def grid_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    correct = pred == target
    foreground = (pred != 0) | (target != 0)
    counts = foreground.sum(axis=(1, 2))
    object_accuracy = (correct & foreground).sum(axis=(1, 2)) / np.maximum(counts, 1)
    object_accuracy[counts == 0] = correct.all(axis=(1, 2))[counts == 0]
    return dict(samples=len(pred), exact_grid_accuracy=float(correct.all(axis=(1, 2)).mean()),
                pixel_accuracy=float(correct.mean()),
                object_pixel_accuracy=float(object_accuracy.mean()))


def summarize(values: np.ndarray) -> dict:
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, 0.9)),
                standard_error=float(values.std(ddof=1) / np.sqrt(len(values)))
                if len(values) > 1 else None)


def checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_metrics(value, prefix: str = "") -> dict[str, int | float]:
    """Flatten report counters and metrics into stable W&B scalar paths."""
    if isinstance(value, dict):
        metrics = {}
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            metrics.update(numeric_metrics(item, child))
        return metrics
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and np.isfinite(value):
        return {prefix: value.item() if isinstance(value, np.generic) else value}
    return {}


def wandb_metrics(report: dict) -> dict[str, int | float]:
    """Use split/layer paths for charts and preserve every numeric report metric."""
    metrics = {"checkpoint/global_step": report["global_step"]}
    metrics.update(numeric_metrics(report["fit"]["coverage"], "fit/coverage"))
    metrics["fit/up_pairs"] = report["fit"]["fit_up_pairs"]
    metrics["fit/native_pairs"] = report["fit"]["fit_native_pairs"]
    if "fit_native_right_pairs" in report["fit"]:
        metrics["fit/native_right_pairs"] = report["fit"]["fit_native_right_pairs"]
    for layer, rank in report["fit"]["ranks"].items():
        metrics.update(numeric_metrics(rank, f"fit/layers/{layer}"))
    for layer, rank in report["fit"].get("spatial_image_ranks", {}).items():
        metrics.update(numeric_metrics(rank, f"fit/spatial_image_layers/{layer}"))
    for layer, rank in report["fit"].get("spatial_mirror_ranks", {}).items():
        metrics.update(numeric_metrics(rank, f"fit/spatial_mirror_layers/{layer}"))
    for split, evaluation in report["evaluation"].items():
        metrics.update(numeric_metrics(evaluation, split))
    return metrics


def wandb_layer_rows(report: dict) -> list[dict]:
    """One compact row per evaluated layer for the W&B table."""
    rows = []
    for split, evaluation in report["evaluation"].items():
        for layer, values in evaluation["layers"].items():
            direction = values.get("direction_readout", {})
            rows.append(dict(
                split=split, representation="pooled", layer=layer,
                samples=values["samples_scored"],
                source_rank=values["fit_rank"]["source_rank"],
                dimension=values["fit_rank"]["dimension"],
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                equivariance_defect_median=values["equivariance_defect"]["median"],
                equivariance_defect_p90=values["equivariance_defect"]["p90"],
                identity_defect_mean=values["identity_defect"]["mean"],
                c4_operator_closure_error=values["c4_operator_closure_error"],
                native_balanced_accuracy=direction.get("native_balanced_accuracy"),
                transported_right_accuracy=direction.get("transported_right_accuracy"),
                rho_mapped_right_accuracy=direction.get("rho_mapped_right_accuracy"),
                unchanged_up_control_right_rate=direction.get(
                    "unchanged_up_control_right_rate"),
                foreground_equivariance_defect_mean=None,
                m2_operator_closure_error=None,
            ))
        for layer, values in evaluation.get("spatial_image_layers", {}).items():
            rows.append(dict(
                split=split, representation="spatial_tokens", layer=layer,
                samples=values["samples_scored"],
                source_rank=values["fit_rank"]["source_gram_rank"],
                dimension=values["fit_rank"]["dimension"],
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                equivariance_defect_median=values["equivariance_defect"]["median"],
                equivariance_defect_p90=values["equivariance_defect"]["p90"],
                identity_defect_mean=values["identity_channel_defect"]["mean"],
                c4_operator_closure_error=values["c4_channel_closure_error"],
                m2_operator_closure_error=None,
                native_balanced_accuracy=None, transported_right_accuracy=None,
                rho_mapped_right_accuracy=None, unchanged_up_control_right_rate=None,
                foreground_equivariance_defect_mean=values[
                    "foreground_equivariance_defect"]["mean"],
            ))
        for layer, values in evaluation.get("spatial_mirror_layers", {}).items():
            rows.append(dict(
                split=split, representation="spatial_reflection_tokens", layer=layer,
                samples=values["samples_scored"],
                source_rank=values["fit_rank"]["source_gram_rank"],
                dimension=values["fit_rank"]["dimension"],
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                equivariance_defect_median=values["equivariance_defect"]["median"],
                equivariance_defect_p90=values["equivariance_defect"]["p90"],
                identity_defect_mean=values["identity_channel_defect"]["mean"],
                c4_operator_closure_error=None,
                m2_operator_closure_error=values["m2_channel_closure_error"],
                native_balanced_accuracy=None, transported_right_accuracy=None,
                rho_mapped_right_accuracy=None, unchanged_up_control_right_rate=None,
                foreground_equivariance_defect_mean=values[
                    "foreground_equivariance_defect"]["mean"],
            ))
    return rows


def log_wandb(report: dict, report_path: Path, args) -> None:
    """Log summary metrics by default; full saved files need explicit opt-in."""
    if args.wandb_mode == "disabled":
        return
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type="atomic_equivariance",
        name=f"c1-exp1-up-to-right-{args.mode}",
        tags=["cogitao", "c1", "experiment-1", "atomic-equivariance", args.mode],
        mode=args.wandb_mode,
        config=dict(
            mode=args.mode, checkpoint=str(args.checkpoint),
            checkpoint_sha256=report["fit"]["checkpoint_sha256"],
            probe_file=str(args.probe_file), data_root=str(args.data_root),
            splits=list(args.splits) if args.mode != "fit" else [],
            train_pairs=report["fit"]["fit_up_pairs"],
            eval_pairs_per_task=args.eval_pairs, batch_size=args.batch_size,
            seed=args.seed, fit_ridge=report["fit"]["ridge"],
            symmetry=report["fit"]["symmetry"],
            protocol_version=report["fit"]["protocol_version"],
        ),
    )
    try:
        run.log(wandb_metrics(report))
        rows = wandb_layer_rows(report)
        if rows:
            columns = list(rows[0])
            table = wandb.Table(columns=columns,
                                data=[[row[column] for column in columns] for row in rows])
            run.log({"equivariance/layer_table": table})
        run.summary["local_report"] = str(report_path)
        run.summary["local_probe_file"] = str(args.probe_file)
        run.summary["checkpoint_sha256"] = report["fit"]["checkpoint_sha256"]
        if args.wandb_log_artifacts:
            artifact = wandb.Artifact(
                name=f"c1-atomic-equivariance-{report['fit']['checkpoint_sha256'][:12]}-{args.mode}",
                type="analysis",
                metadata=dict(mode=args.mode, protocol_version=report["fit"]["protocol_version"]),
            )
            artifact.add_file(str(report_path), name=report_path.name)
            artifact.add_file(str(args.probe_file), name=args.probe_file.name)
            filenames = ("layer_metrics.csv", "spatial_layer_metrics.csv",
                         "spatial_group_identity_metrics.csv", *(
                f"{split}_examples.npz" for split in report["evaluation"]
            )) if report["evaluation"] else ()
            for filename in filenames:
                path = args.output_dir / filename
                if path.is_file():
                    artifact.add_file(str(path), name=filename)
            run.log_artifact(artifact)
    finally:
        run.finish()


def fit_probes(model, collector, args, device, fingerprint):
    rows, coverage = read_atomic(args.data_root, "train", args.train_pairs, args.seed, set())
    up = [row for row in rows if row["task"] == UP]
    if not up:
        raise ValueError("Atomic translate_up ID training rows are required")
    source, _, _ = collector.run(up, device, args.batch_size)
    transported, _, _ = collector.run(up, device, args.batch_size, rotated=True)
    native, _, _ = collector.run(rows, device, args.batch_size)
    labels = np.array([1 if row["task"] == RIGHT else -1 for row in rows])
    arrays, ranks = {}, {}
    for name in source:
        q, rank = fit_orthogonal(source[name], transported[name])
        arrays[f"{name}__rho"] = q
        ranks[name] = dict(source_rank=rank, dimension=q.shape[0],
                           underidentified=rank < q.shape[0])
        if not name.startswith("image_") and np.any(labels == 1):
            mean, scale, coef = fit_direction_readout(native[name], labels, args.ridge)
            for key, value in (("mean", mean), ("scale", scale), ("coef", coef)):
                arrays[f"{name}__direction_{key}"] = value
    spatial_layers = [name for name in source if name.startswith("image_")]
    spatial_maps, spatial_ranks = fit_spatial_image_maps(
        model, collector, up, device, args.batch_size, spatial_layers)
    for name, operator in spatial_maps.items():
        arrays[f"{name}__spatial_rho"] = operator
    mirror_maps, mirror_ranks = fit_spatial_mirror_maps(
        model, collector, up, device, args.batch_size, spatial_layers)
    for name, operator in mirror_maps.items():
        arrays[f"{name}__mirror_rho"] = operator
    metadata = dict(protocol_version=3, checkpoint_sha256=fingerprint,
                    symmetry="whole_grid_clockwise_90",
                    source_task=UP, transported_task=RIGHT, fit_split="train",
                    seed=args.seed, resolution=list(model.input_resolution),
                    fit_up_pairs=len(up), fit_native_pairs=len(rows),
                    fit_native_right_pairs=int(np.sum(labels == 1)),
                    direction_readout_available=bool(np.any(labels == 1)),
                    layers=list(source), ranks=ranks, coverage=coverage,
                    spatial_image_layers=spatial_layers,
                    spatial_image_ranks=spatial_ranks,
                    spatial_mirror_ranks=mirror_ranks,
                    ridge=args.ridge, convention="row vector: H_transported ~= H_source @ Q")
    arrays["metadata"] = np.array(json.dumps(metadata))
    arrays["fit_orbits"] = np.array(sorted({row["orbit"] for row in rows}))
    arrays["fit_indices"] = np.array([row["index"] for row in rows])
    args.probe_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.probe_file, **arrays)
    return arrays, metadata


def evaluate(model, collector, arrays, metadata, args, device):
    results, layer_rows, spatial_rows, group_rows = {}, [], [], []
    excluded = set(arrays["fit_orbits"].tolist())
    for split in args.splits:
        rows, coverage = read_atomic(args.data_root, split, args.eval_pairs, args.seed, excluded)
        up = [row for row in rows if row["task"] == UP]
        if not up:
            raise ValueError(f"No eligible atomic translate_up rows in {split}")
        original, pred_up, target_up = collector.run(up, device, args.batch_size)
        transported, pred_right, target_right = collector.run(
            up, device, args.batch_size, rotated=True)
        native, pred_native, target_native = collector.run(rows, device, args.batch_size)
        native_labels = np.array([1 if row["task"] == RIGHT else -1 for row in rows])
        layers = {}
        saved = dict(predicted_up=pred_up, target_up=target_up,
                     predicted_right=pred_right, target_right=target_right,
                     raw_input=np.stack([row["input"] for row in up]),
                     raw_target=np.stack([row["target"] for row in up]),
                     source_index=np.array([row["index"] for row in up]))
        # An unchanged-task control is distinct from the isomorphic right task.
        control_rows = []
        control_indices = []
        for index, row in enumerate(up):
            gx, _ = isomorphic_pair(row)
            try:
                wrong_target = translate(gx, UP)
            except ValueError:
                continue
            control_indices.append(index)
            control_rows.append(dict(input=gx, target=wrong_target, task=UP))
        if control_rows:
            control_features, control_pred, control_target = collector.run(
                control_rows, device, args.batch_size)
            right_subset = target_right[control_indices]
            distinct = (control_target != right_subset).any(axis=(1, 2))
            saved.update(control_pair_index=np.array(control_indices),
                         control_source_index=np.array([up[i]["index"] for i in control_indices]),
                         predicted_unchanged_up=control_pred, target_unchanged_up=control_target)
            wrong_up_match = grid_metrics(pred_right[control_indices], control_target)
            control = dict(eligible_samples=len(control_rows),
                           distinct_up_right_targets=int(distinct.sum()),
                           right_prediction_matching_wrong_up=wrong_up_match,
                           model_with_up_on_rotated_input=grid_metrics(control_pred, control_target),
                           model_with_up_matching_right_target=grid_metrics(control_pred, right_subset))
        else:
            control = dict(eligible_samples=0)
        for name in metadata["layers"]:
            x = original[name].astype(np.float64)
            y = transported[name].astype(np.float64)
            q = arrays[f"{name}__rho"]
            denom = np.linalg.norm(x, axis=1)
            valid = denom > EPS
            if not valid.any():
                raise ValueError(f"Zero source norms at {split}/{name}")
            defects = np.linalg.norm(y - x @ q, axis=1)[valid] / denom[valid]
            identity = np.linalg.norm(y - x, axis=1)[valid] / denom[valid]
            # Low defect alone does not establish a group representation.
            cycle = np.linalg.matrix_power(q, 4)
            layer = dict(equivariance_defect=summarize(defects),
                         identity_defect=summarize(identity),
                         samples_scored=int(valid.sum()), zero_norm_samples=int((~valid).sum()),
                         fit_rank=metadata["ranks"][name],
                         orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(len(q)))),
                         c4_operator_closure_error=float(np.linalg.norm(cycle - np.eye(len(q))) /
                                                         np.sqrt(len(q))),
                         c4_data_closure_defect=summarize(
                             np.linalg.norm(x @ cycle - x, axis=1)[valid] / denom[valid]))
            if f"{name}__direction_coef" in arrays:
                params = [arrays[f"{name}__direction_{key}"] for key in ("mean", "scale", "coef")]
                native_scores = direction_scores(native[name], *params)
                right_scores = direction_scores(y, *params)
                mapped_scores = direction_scores(x @ q, *params)
                layer["direction_readout"] = dict(
                    native_balanced_accuracy=float(np.mean([
                        ((native_scores[native_labels == label] > 0) == (label == 1)).mean()
                        for label in (-1, 1)])) if np.any(native_labels == 1) else None,
                    transported_right_accuracy=float((right_scores > 0).mean()),
                    rho_mapped_right_accuracy=float((mapped_scores > 0).mean()))
                if control_rows:
                    control_scores = direction_scores(control_features[name], *params)
                    layer["direction_readout"]["unchanged_up_control_right_rate"] = float(
                        (control_scores > 0).mean())
            layers[name] = layer
            layer_rows.append(dict(split=split, layer=name, samples=int(valid.sum()),
                                   equivariance_defect_mean=defects.mean(),
                                   identity_defect_mean=identity.mean(),
                                   c4_operator_closure_error=layer["c4_operator_closure_error"],
                                   transported_right_accuracy=layer.get("direction_readout", {}).get(
                                       "transported_right_accuracy")))
            saved[f"{name}__source"] = original[name]
            saved[f"{name}__transported"] = transported[name]
            saved[f"{name}__per_example_defect"] = np.where(
                valid, np.linalg.norm(y - x @ q, axis=1) / np.maximum(denom, EPS), np.nan)
        spatial_layers, spatial_extra = evaluate_spatial_image_maps(
            model, collector, up, arrays, metadata, device, args.batch_size)
        saved.update(spatial_extra["per_example"])
        mirror_layers, mirror_extra = evaluate_spatial_mirror_maps(
            model, collector, up, arrays, metadata, device, args.batch_size)
        saved.update(mirror_extra["per_example"])
        for name, values in spatial_layers.items():
            spatial_rows.append(dict(
                split=split, layer=name, samples=len(up),
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                foreground_equivariance_defect_mean=values[
                    "foreground_equivariance_defect"]["mean"],
                identity_channel_defect_mean=values["identity_channel_defect"]["mean"],
                no_spatial_permutation_defect_mean=values[
                    "no_spatial_permutation_defect"]["mean"],
                c4_channel_closure_error=values["c4_channel_closure_error"],
            ))
            group_rows.append(dict(
                split=split, layer=name, identity="R^4=I",
                operator_closure_error=values["c4_channel_closure_error"],
                data_closure_defect_mean=values["c4_data_closure_defect"]["mean"],
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                foreground_equivariance_defect_mean=values[
                    "foreground_equivariance_defect"]["mean"],
            ))
        for name, values in mirror_layers.items():
            group_rows.append(dict(
                split=split, layer=name, identity="M^2=I",
                operator_closure_error=values["m2_channel_closure_error"],
                data_closure_defect_mean=values["m2_data_closure_defect"]["mean"],
                equivariance_defect_mean=values["equivariance_defect"]["mean"],
                foreground_equivariance_defect_mean=values[
                    "foreground_equivariance_defect"]["mean"],
            ))
        np.savez_compressed(args.output_dir / f"{split}_examples.npz", **saved)
        results[split] = dict(coverage=coverage, layers=layers,
                             spatial_image_layers=spatial_layers,
                             spatial_mirror_layers=mirror_layers,
                             input_rotation_resize_mismatch=spatial_extra[
                                 "input_rotation_resize_mismatch"],
                             input_reflection_resize_mismatch=mirror_extra[
                                 "input_reflection_resize_mismatch"],
                             original_up_output=grid_metrics(pred_up, target_up),
                             transported_right_output=grid_metrics(pred_right, target_right),
                             native_output=grid_metrics(pred_native, target_native),
                             unchanged_task_control=control)
        print(f"Finished {split}: {len(up)} paired up/right examples", flush=True)
    with (args.output_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    with (args.output_dir / "spatial_layer_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(spatial_rows[0]))
        writer.writeheader()
        writer.writerows(spatial_rows)
    with (args.output_dir / "spatial_group_identity_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/cogitao_vit_cross_attention/"
        "cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cogitao_c1_atomic_equivariance"))
    parser.add_argument("--mode", choices=("fit", "eval", "all"), default="all")
    parser.add_argument("--probe-file", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="cogitao-c1-atomic-equivariance")
    parser.add_argument("--wandb-log-artifacts", action="store_true")
    args = parser.parse_args()
    if min(args.train_pairs, args.eval_pairs, args.batch_size) <= 0 or args.ridge <= 0:
        parser.error("Pair counts, batch size and ridge must be positive")
    args.probe_file = args.probe_file or args.output_dir / "frozen_probes.npz"
    if args.probe_file.suffix != ".npz":
        parser.error("--probe-file must have an .npz extension")
    if args.mode in ("fit", "all") and args.probe_file.exists():
        parser.error("Probe file exists; use --mode eval or a new --probe-file to preserve frozen maps")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    fingerprint = checkpoint_digest(args.checkpoint)
    model, step = load_model(args.checkpoint, "mlp", device, allow_untrained=False)
    collector = LayerCollector(model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode in ("fit", "all"):
            arrays, metadata = fit_probes(model, collector, args, device, fingerprint)
            print(f"Saved ID-trained frozen probes to {args.probe_file}", flush=True)
        else:
            with np.load(args.probe_file, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(arrays["metadata"].item()))
            if metadata.get("protocol_version") != 3:
                raise ValueError("Unsupported frozen-probe protocol version")
            if metadata["checkpoint_sha256"] != fingerprint:
                raise ValueError("Frozen probes were fitted on a different checkpoint")
        results = evaluate(model, collector, arrays, metadata, args, device) if args.mode != "fit" else {}
    finally:
        collector.close()
    report = dict(checkpoint=str(args.checkpoint), global_step=step, mode=args.mode,
                  probe_file=str(args.probe_file), fit=metadata, evaluation=results,
                  evaluation_seed=args.seed, eval_pairs_per_task=args.eval_pairs,
                  protocol="Atomic ID first stage; transported copies are counterfactuals, not composition OOD",
                  raw_geometry="g(T_up(x)) = T_right(g(x)); g is whole-grid clockwise 90 degrees",
                  pooling="foreground mean image tokens; mean shared/output queries; function vectors",
                  spatial_image_metric="All ViT image tokens, matched by inverse 90-degree rotation; per-layer orthogonal channel map fitted on ID train",
                  group_identities="Per ViT image layer: rho(R)^4 ~= I for clockwise rotation and rho(M)^2 ~= I for left-right reflection",
                  resize="Raw-grid transforms precede existing nearest-neighbor model resizing; resize need not commute with g",
                  limitations=["Function vectors are constant per task, so their maps are underidentified",
                               "Image layers have no task context, so no direction readout is fitted there",
                               "A direction readout is omitted if native right examples are absent from ID train",
                               "The transported task token is explicitly right; this tests task transport, not automatic direction inference",
                               "Procrustes does not enforce Q^4=I; C4 closure is reported separately",
                               "Spatial-token metrics retain position but are affected by raw-grid resizing",
                               "C1 OOD compositions require a subsequent protocol"])
    name = "fit_report.json" if args.mode == "fit" else "report.json"
    path = args.output_dir / name
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    log_wandb(report, path, args)


if __name__ == "__main__":
    main()
