"""Plot C1 rot90/translate_up task-isomorphism geometry in 2-D and 3-D PCA.

All atomic rot90 and translate_up rows from C1 experiment-1 train and val
contribute to the exact-state PCA. Affine image-latent operators are fitted
only on the corresponding atomic train rows. No model weights are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import (
    fit_affine, load_model, read_split, resize_grids, task_tensors,
)
from benchmarks.cogitao_vit_cross_attention.c1_symmetry_geometry import ROT, UP, compose, step


STATE_NAMES = ("input", "rot", "up", "rot_then_up", "up_then_rot")
EXACT_PLOT_STATES = ("input", "rot", "up", "rot_then_up")
COLORS = {
    "input": "#525252", "rot": "#e6550d", "up": "#3182bd",
    "rot_then_up": "#31a354", "up_then_rot": "#756bb1",
    "latent_rot_then_up": "#ad494a", "latent_up_then_rot": "#8c6bb1",
    "model_rot": "#a63603", "model_up": "#08519c",
    "model_direct": "#d62728", "model_sequential": "#ff7f0e",
    "shared_rot": "#e6550d", "shared_up": "#3182bd",
    "shared_direct": "#31a354",
}
LABELS = {
    "input": "input", "rot": "exact R(x)", "up": "exact T(x)",
    "rot_then_up": "exact T(R(x))", "up_then_rot": "exact R(T(x))",
    "latent_rot_then_up": "latent A_T(A_R(z))",
    "latent_up_then_rot": "latent A_R(A_T(z))",
    "model_rot": "ViT+MLP [R], re-encoded",
    "model_up": "ViT+MLP [T], re-encoded",
    "model_direct": "ViT+MLP [R,T], re-encoded",
    "model_sequential": "ViT+MLP R then T, re-encoded",
    "shared_rot": "shared [R]", "shared_up": "shared [T]",
    "shared_direct": "shared [R,T]",
}


def maybe_step(grid: np.ndarray, operation: str) -> np.ndarray | None:
    try:
        return step(grid, operation)[0]
    except ValueError:
        return None


def maybe_compose(grid: np.ndarray, operations: tuple[str, str]) -> np.ndarray | None:
    try:
        return compose(grid, operations)[0]
    except ValueError:
        return None


def build_records(data_root: Path) -> tuple[list[dict], dict[str, int]]:
    """Keep every atomic row, including rows lacking a valid counterfactual."""
    records: list[dict] = []
    coverage: Counter[str] = Counter()
    for split in ("train", "val"):
        for row in read_split(data_root, split):
            if row["suite"] not in ((ROT,), (UP,)):
                continue
            raw = row["input"]
            rotation = maybe_step(raw, ROT)
            translation = maybe_step(raw, UP)
            observed = rotation if row["suite"] == (ROT,) else translation
            atomic_exact = observed is not None and np.array_equal(observed, row["target"])
            coverage[f"{split}|atomic_rows"] += 1
            coverage[f"{split}|{row['suite'][0]}|rows"] += 1
            coverage[f"{split}|{row['suite'][0]}|geometry_exact"] += atomic_exact
            # The observed target is retained for PCA even if our object
            # extraction cannot reconstruct a particular example.
            if row["suite"] == (ROT,):
                rotation = row["target"]
            else:
                translation = row["target"]
            rt = maybe_compose(raw, (ROT, UP)) if atomic_exact else None
            tr = maybe_compose(raw, (UP, ROT)) if atomic_exact else None
            if rt is not None:
                coverage[f"{split}|rotate_then_up_valid"] += 1
            if tr is not None:
                coverage[f"{split}|up_then_rotate_valid"] += 1
            if rt is not None and tr is not None:
                coverage[f"{split}|both_orders_valid"] += 1
                coverage[f"{split}|same_exact_target"] += np.array_equal(rt, tr)
            records.append({
                "split": split, "source_index": row["index"],
                "source_task": row["suite"][0], "atomic_geometry_exact": atomic_exact,
                "grids": {"input": raw, "rot": rotation, "up": translation,
                          "rot_then_up": rt, "up_then_rot": tr},
            })
    if not records:
        raise ValueError("No atomic rot90/translate_up rows in train or val")
    return records, dict(sorted(coverage.items()))


def one_hot(indices: torch.Tensor) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=10).permute(0, 3, 1, 2).float()


def pool_foreground(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """One image-state vector: mean ViT token on non-background cells."""
    foreground = indices.flatten(1).ne(0).to(tokens.dtype)
    counts = foreground.sum(dim=1, keepdim=True)
    selected = (tokens * foreground.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1)
    return torch.where(counts > 0, selected, tokens.mean(dim=1))


def encode_exact_states(model, records: list[dict], device: torch.device, batch_size: int):
    count, dimension = len(records), model.embedding_dim
    embeddings = {name: np.full((count, dimension), np.nan, dtype=np.float32)
                  for name in STATE_NAMES}
    for state in STATE_NAMES:
        valid = [i for i, record in enumerate(records) if record["grids"][state] is not None]
        for start in range(0, len(valid), batch_size):
            selected = valid[start:start + batch_size]
            grids = [records[i]["grids"][state] for i in selected]
            indices = resize_grids(grids, model.input_resolution, device)
            with torch.inference_mode():
                vectors = pool_foreground(model.encode_images(one_hot(indices)), indices)
            embeddings[state][selected] = vectors.float().cpu().numpy()
    return embeddings


def apply_affine(vectors: np.ndarray, operator: np.ndarray) -> np.ndarray:
    augmented = np.concatenate((vectors.astype(np.float64),
                                np.ones((len(vectors), 1))), axis=1)
    return (augmented @ operator)[:, :-1].astype(np.float32)


def fit_atomic_operators(records: list[dict], embeddings: dict, ridge: float):
    operators = {}
    train_count = {}
    for operation, state in ((ROT, "rot"), (UP, "up")):
        selected = np.array([
            i for i, record in enumerate(records)
            if record["split"] == "train" and record["source_task"] == operation
            and record["atomic_geometry_exact"]
            and np.isfinite(embeddings[state][i]).all()
        ], dtype=np.int64)
        if len(selected) == 0:
            raise ValueError(f"No exact training pairs for {operation}")
        operators[operation] = fit_affine(
            embeddings["input"][selected].astype(np.float64),
            embeddings[state][selected].astype(np.float64), ridge,
        )
        train_count[operation] = int(len(selected))
    return operators, train_count


def latent_compositions(embeddings: dict, operators: dict) -> dict[str, np.ndarray]:
    # Row-vector convention: rotate then up is z @ A_R @ A_T.
    source = embeddings["input"]
    return {
        "latent_rot_then_up": apply_affine(source, operators[ROT] @ operators[UP]),
        "latent_up_then_rot": apply_affine(source, operators[UP] @ operators[ROT]),
    }


def conditioned_forward(model, indices: torch.Tensor, suite: tuple[str, ...]):
    tokens, mask = task_tensors([suite] * len(indices), indices.device)
    return model({"images": one_hot(indices), "target_grid": indices,
                  "task_tokens": tokens, "task_token_mask": mask})


def model_representations(model, records: list[dict], device: torch.device, batch_size: int):
    """Run atomic, direct composite, and sequential model paths on all inputs."""
    count, dimension = len(records), model.embedding_dim
    shared = {name: np.empty((count, dimension), dtype=np.float32)
              for name in ("shared_rot", "shared_up", "shared_direct")}
    reencoded = {name: np.empty((count, dimension), dtype=np.float32)
                 for name in ("model_rot", "model_up", "model_direct", "model_sequential")}
    predictions = {name: np.empty((count, *model.input_resolution), dtype=np.uint8)
                   for name in ("model_rot", "model_up", "model_direct", "model_sequential")}
    for start in range(0, count, batch_size):
        selected = records[start:start + batch_size]
        source = resize_grids([row["grids"]["input"] for row in selected],
                              model.input_resolution, device)
        with torch.inference_mode():
            rot = conditioned_forward(model, source, (ROT,))
            up = conditioned_forward(model, source, (UP,))
            direct = conditioned_forward(model, source, (ROT, UP))
            sequential = conditioned_forward(model, rot["predictions"], (UP,))
            for name, output in (("shared_rot", rot), ("shared_up", up),
                                 ("shared_direct", direct)):
                shared[name][start:start + len(selected)] = (
                    output["shared_latents"].mean(dim=1).float().cpu().numpy())
            for name, output in (("model_rot", rot), ("model_up", up),
                                 ("model_direct", direct),
                                 ("model_sequential", sequential)):
                grid = output["predictions"]
                latent = model.encode_images(one_hot(grid))
                reencoded[name][start:start + len(selected)] = (
                    pool_foreground(latent, grid).float().cpu().numpy())
                predictions[name][start:start + len(selected)] = grid.byte().cpu().numpy()
    for shared_name, state in (("shared_rot", "rot"), ("shared_up", "up"),
                               ("shared_direct", "rot_then_up")):
        invalid = [i for i, row in enumerate(records) if row["grids"][state] is None]
        shared[shared_name][invalid] = np.nan
    return shared, reencoded, predictions


def function_vectors(model, device: torch.device) -> dict[str, np.ndarray]:
    suites = {"empty": (), "rot": (ROT,), "up": (UP,),
              "rot_then_up": (ROT, UP), "up_then_rot": (UP, ROT)}
    tokens, mask = task_tensors(list(suites.values()), device)
    with torch.inference_mode():
        vectors = model.encode_functions(tokens, mask).float().cpu().numpy()
    return dict(zip(suites, vectors))


def fit_pca(arrays: list[np.ndarray]):
    points = np.concatenate([array[np.isfinite(array).all(axis=1)] for array in arrays])
    if len(points) < 4:
        raise ValueError("PCA requires at least four finite vectors")
    mean = points.mean(axis=0, dtype=np.float64)
    centered = points.astype(np.float64) - mean
    covariance = centered.T @ centered / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    variance = np.maximum(eigenvalues[order], 0)
    components = eigenvectors[:, order[:3]].copy()
    # Fix arbitrary eigenvector signs for reproducible figure orientation.
    for column in range(3):
        peak = np.argmax(np.abs(components[:, column]))
        if components[peak, column] < 0:
            components[:, column] *= -1
    ratios = variance[:3] / max(variance.sum(), 1e-12)
    return mean, components, ratios


def project(array: np.ndarray, mean: np.ndarray, components: np.ndarray):
    result = np.full((len(array), 3), np.nan, dtype=np.float32)
    valid = np.isfinite(array).all(axis=1)
    result[valid] = ((array[valid].astype(np.float64) - mean) @ components).astype(np.float32)
    return result


def metric_distance(predicted: np.ndarray, target: np.ndarray, selected: np.ndarray):
    if not np.any(selected):
        return {"count": 0, "mean_l2": None, "mean_squared_error": None}
    error = predicted[selected] - target[selected]
    return {"count": int(selected.sum()),
            "mean_l2": float(np.linalg.norm(error, axis=1).mean()),
            "mean_squared_error": float(np.mean(error ** 2))}


def edge_consistency(first: np.ndarray, opposite: np.ndarray,
                     selected: np.ndarray) -> dict:
    """Compare corresponding edges of an exact transformation square."""
    selected = selected & np.isfinite(first).all(axis=1) & np.isfinite(opposite).all(axis=1)
    if not np.any(selected):
        return {"count": 0, "mean_l2": None, "mean_cosine": None,
                "mean_relative_l2": None}
    left, right = first[selected], opposite[selected]
    left_length = np.linalg.norm(left, axis=1)
    right_length = np.linalg.norm(right, axis=1)
    error = np.linalg.norm(left - right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(left_length * right_length, 1e-12)
    return {"count": int(selected.sum()), "mean_l2": float(error.mean()),
            "mean_cosine": float(cosine.mean()),
            "mean_relative_l2": float(np.mean(error / np.maximum(left_length, 1e-12)))}


def metrics(records, embeddings, composed, reencoded, predictions, operators):
    output = {}
    for split in ("train", "val"):
        split_mask = np.array([row["split"] == split for row in records])
        valid_rt = np.isfinite(embeddings["rot_then_up"]).all(axis=1) & split_mask
        valid_tr = np.isfinite(embeddings["up_then_rot"]).all(axis=1) & split_mask
        both = valid_rt & valid_tr
        target = embeddings["rot_then_up"]
        result = {"rows": int(split_mask.sum()), "valid_rotate_then_up": int(valid_rt.sum()),
                  "valid_both_orders": int(both.sum()),
                  "exact_order_swap_equal": int(sum(
                      np.array_equal(records[i]["grids"]["rot_then_up"],
                                     records[i]["grids"]["up_then_rot"])
                      for i in np.flatnonzero(both)))}
        reference_length = np.linalg.norm(target[valid_rt] - embeddings["input"][valid_rt],
                                          axis=1).mean() if np.any(valid_rt) else None
        for name, values in {**composed,
                             "model_direct": reencoded["model_direct"],
                             "model_sequential": reencoded["model_sequential"]}.items():
            result[name + "_vs_exact"] = metric_distance(values, target, valid_rt)
            distance = result[name + "_vs_exact"]["mean_l2"]
            result[name + "_vs_exact"]["l2_over_exact_change"] = (
                float(distance / reference_length)
                if distance is not None and reference_length is not None
                and reference_length > 1e-12 else None)
        result["image_rotation_edge_consistency"] = edge_consistency(
            embeddings["rot"] - embeddings["input"],
            embeddings["rot_then_up"] - embeddings["up"], valid_rt)
        result["image_translation_edge_consistency"] = edge_consistency(
            embeddings["up"] - embeddings["input"],
            embeddings["rot_then_up"] - embeddings["rot"], valid_rt)
        result["latent_order_commutator"] = metric_distance(
            composed["latent_rot_then_up"], composed["latent_up_then_rot"], both)
        for operation, state, model_name in ((ROT, "rot", "model_rot"),
                                             (UP, "up", "model_up")):
            atomic = split_mask & np.array([r["source_task"] == operation for r in records])
            result[model_name + "_vs_atomic_exact"] = metric_distance(
                reencoded[model_name], embeddings[state], atomic)
            if np.any(atomic):
                atomic_targets = np.stack([records[i]["grids"][state]
                                           for i in np.flatnonzero(atomic)])
                atomic_raw = torch.as_tensor(atomic_targets[:, None], dtype=torch.float32)
                atomic_resized = F.interpolate(
                    atomic_raw, size=predictions[model_name].shape[-2:],
                    mode="nearest")[:, 0].numpy().astype(np.uint8)
                result[model_name + "_exact_grid_accuracy"] = float(np.mean(np.all(
                    predictions[model_name][atomic] == atomic_resized, axis=(1, 2))))
        if np.any(valid_rt):
            target_grids = np.stack([records[i]["grids"]["rot_then_up"]
                                     for i in np.flatnonzero(valid_rt)])
            # Match the model's 15->20 nearest-neighbor preprocessing.
            raw = torch.as_tensor(target_grids[:, None], dtype=torch.float32)
            resized = F.interpolate(raw, size=predictions["model_direct"].shape[-2:],
                                    mode="nearest")[:, 0].numpy().astype(np.uint8)
            for name in ("model_direct", "model_sequential"):
                grid = predictions[name]
                result[name + "_exact_grid_accuracy"] = float(
                    np.mean(np.all(grid[valid_rt] == resized, axis=(1, 2))))
        output[split] = result
    commutator = operators[ROT] @ operators[UP] - operators[UP] @ operators[ROT]
    output["operators"] = {"relative_frobenius_commutator": float(
        np.linalg.norm(commutator) / max(np.linalg.norm(operators[ROT] @ operators[UP]),
                                        np.linalg.norm(operators[UP] @ operators[ROT]), 1e-12))}
    return output


def axes_for(dim: int, title: str):
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d" if dim == 3 else None)
    axis.set_title(title)
    axis.set_xlabel("PC 1")
    axis.set_ylabel("PC 2")
    if dim == 3:
        axis.set_zlabel("PC 3")
        axis.view_init(elev=24, azim=-58)
    return figure, axis


def draw_points(axis, points: np.ndarray, dim: int, *, color: str, label: str,
                size: float, alpha: float):
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return
    xyz = points[valid]
    if dim == 2:
        axis.scatter(xyz[:, 0], xyz[:, 1], s=size, c=color, alpha=alpha,
                     label=label, rasterized=True, linewidths=0)
    else:
        axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=size, c=color,
                     alpha=alpha, label=label, rasterized=True, linewidths=0)


def draw_path(axis, coordinates: dict, indices: list[int], states: tuple[str, ...],
              dim: int, color: str):
    for index in indices:
        path = np.stack([coordinates[state][index] for state in states])
        if not np.isfinite(path).all():
            continue
        if dim == 2:
            axis.plot(path[:, 0], path[:, 1], color=color, linewidth=0.7, alpha=0.55)
        else:
            axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color,
                      linewidth=0.7, alpha=0.55)


def save_figure(figure, axis, path: Path):
    axis.legend(loc="best", fontsize=8, markerscale=5)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_image_geometry(coordinates, records, output_dir: Path, square_examples: int):
    train = np.array([r["split"] == "train" for r in records])
    val = ~train
    square_indices = [i for i, r in enumerate(records)
                      if r["split"] == "val" and r["grids"]["rot_then_up"] is not None
                      and r["grids"]["up_then_rot"] is not None
                      and np.array_equal(r["grids"]["rot_then_up"],
                                         r["grids"]["up_then_rot"])][:square_examples]
    for dim in (2, 3):
        figure, axis = axes_for(dim, "Exact C1 transformation squares: all train and val atomic inputs")
        for state in EXACT_PLOT_STATES:
            draw_points(axis, coordinates[state][train], dim, color=COLORS[state],
                        label=LABELS[state] + " (train)", size=1.0, alpha=0.035)
            draw_points(axis, coordinates[state][val], dim, color=COLORS[state],
                        label=LABELS[state] + " (val)", size=9, alpha=0.8)
        draw_path(axis, coordinates, square_indices, ("input", "rot", "rot_then_up"),
                  dim, COLORS["rot"])
        draw_path(axis, coordinates, square_indices, ("input", "up", "up_then_rot"),
                  dim, COLORS["up"])
        save_figure(figure, axis, output_dir / f"exact_square_{dim}d.png")
        figure, axis = axes_for(dim, "Geometric recovery: exact, latent, direct, sequential")
        for state in ("rot_then_up", "latent_rot_then_up", "model_direct", "model_sequential"):
            valid_train = train & np.isfinite(coordinates["rot_then_up"]).all(axis=1)
            valid_val = val & np.isfinite(coordinates["rot_then_up"]).all(axis=1)
            draw_points(axis, coordinates[state][valid_train], dim,
                        color=COLORS[state], label=LABELS[state] + " (train)",
                        size=1.0, alpha=0.035)
            draw_points(axis, coordinates[state][valid_val], dim,
                        color=COLORS[state], label=LABELS[state] + " (val)",
                        size=9, alpha=0.8)
        for state in ("latent_rot_then_up", "model_direct", "model_sequential"):
            draw_path(axis, coordinates, square_indices, ("rot_then_up", state),
                      dim, COLORS[state])
        save_figure(figure, axis, output_dir / f"recovery_{dim}d.png")
        figure, axis = axes_for(dim, "Atomic recovery: exact and model-predicted states")
        for state, reference in (("rot", "rot"), ("model_rot", "rot"),
                                 ("up", "up"), ("model_up", "up")):
            eligible = np.isfinite(coordinates[reference]).all(axis=1)
            draw_points(axis, coordinates[state][train & eligible], dim,
                        color=COLORS[state], label=LABELS[state] + " (train)",
                        size=1.0, alpha=0.035)
            draw_points(axis, coordinates[state][val & eligible], dim,
                        color=COLORS[state], label=LABELS[state] + " (val)",
                        size=9, alpha=0.8)
        save_figure(figure, axis, output_dir / f"atomic_recovery_{dim}d.png")


def plot_shared_geometry(coordinates, records, output_dir: Path):
    train = np.array([r["split"] == "train" for r in records])
    for dim in (2, 3):
        figure, axis = axes_for(dim, "ViT + function-MLP shared latents")
        for state in ("shared_rot", "shared_up", "shared_direct"):
            draw_points(axis, coordinates[state][train], dim, color=COLORS[state],
                        label=LABELS[state] + " (train)", size=1.0, alpha=0.035)
            draw_points(axis, coordinates[state][~train], dim, color=COLORS[state],
                        label=LABELS[state] + " (val)", size=9, alpha=0.8)
        save_figure(figure, axis, output_dir / f"shared_{dim}d.png")


def plot_function_geometry(coordinates, output_dir: Path):
    labels = list(coordinates)
    points = np.stack([coordinates[name] for name in labels])
    for dim in (2, 3):
        figure, axis = axes_for(dim, "Function-MLP task vectors (separate PCA chart)")
        for name, point in zip(labels, points):
            draw_points(axis, point[None], dim, color=COLORS.get(name, "#555555"),
                        label=name, size=55, alpha=1.0)
            if dim == 2:
                axis.annotate(name, (point[0], point[1]), xytext=(5, 5),
                              textcoords="offset points", fontsize=9)
            else:
                axis.text(point[0], point[1], point[2], name, fontsize=9)
        save_figure(figure, axis, output_dir / f"function_mlp_{dim}d.png")


def write_metadata(records: list[dict], path: Path):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("row", "split", "source_index", "source_task",
                         "atomic_geometry_exact", "valid_rot", "valid_up",
                         "valid_rotate_then_up", "valid_up_then_rot", "same_exact_target"))
        for i, record in enumerate(records):
            grids = record["grids"]
            writer.writerow((i, record["split"], record["source_index"],
                             record["source_task"], int(record["atomic_geometry_exact"]),
                             int(grids["rot"] is not None), int(grids["up"] is not None),
                             int(grids["rot_then_up"] is not None),
                             int(grids["up_then_rot"] is not None),
                             int(grids["rot_then_up"] is not None and
                                 grids["up_then_rot"] is not None and
                                 np.array_equal(grids["rot_then_up"], grids["up_then_rot"]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/cogitao_vit_cross_attention/"
        "cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/cogitao_c1_task_isomorphisms"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--square-examples", type=int, default=24,
                        help="Number of val squares highlighted; PCA and plots still use all rows")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.ridge <= 0 or args.square_examples < 0:
        parser.error("batch-size and ridge must be positive; square-examples must be nonnegative")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model, global_step = load_model(args.checkpoint, "mlp", device, allow_untrained=False)
    records, coverage = build_records(args.data_root)
    print(f"Loaded {len(records)} atomic train+val examples", flush=True)
    exact = encode_exact_states(model, records, device, args.batch_size)
    operators, fitted_counts = fit_atomic_operators(records, exact, args.ridge)
    composed = latent_compositions(exact, operators)
    print("Encoded exact states and fitted train-only atomic operators", flush=True)
    shared, model_latents, predictions = model_representations(
        model, records, device, args.batch_size)
    functions = function_vectors(model, device)
    print("Encoded direct and sequential ViT+MLP outputs", flush=True)

    # Fit each PCA only on representations from its own space. Model recovery
    # points are projected into the exact-image PCA basis, never refitted there.
    image_mean, image_axes, image_ratio = fit_pca([exact[name] for name in EXACT_PLOT_STATES])
    image_coordinates = {name: project(values, image_mean, image_axes)
                         for name, values in {**exact, **composed, **model_latents}.items()}
    shared_mean, shared_axes, shared_ratio = fit_pca(list(shared.values()))
    shared_coordinates = {name: project(values, shared_mean, shared_axes)
                          for name, values in shared.items()}
    function_mean, function_axes, function_ratio = fit_pca([
        np.stack(list(functions.values()))])
    function_coordinates = {name: project(vector[None], function_mean, function_axes)[0]
                            for name, vector in functions.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_image_geometry(image_coordinates, records, args.output_dir, args.square_examples)
    plot_shared_geometry(shared_coordinates, records, args.output_dir)
    plot_function_geometry(function_coordinates, args.output_dir)
    write_metadata(records, args.output_dir / "examples.csv")
    np.savez_compressed(args.output_dir / "representations.npz",
                        **{f"image_{name}": value for name, value in exact.items()},
                        **composed, **model_latents, **shared,
                        **{f"function_{name}": value for name, value in functions.items()})
    np.savez_compressed(args.output_dir / "pca_coordinates.npz",
                        **{f"image_{name}": value for name, value in image_coordinates.items()},
                        **shared_coordinates,
                        **{f"function_{name}": value for name, value in function_coordinates.items()})
    np.savez(args.output_dir / "pca_bases.npz",
             image_mean=image_mean, image_components=image_axes, image_variance_ratio=image_ratio,
             shared_mean=shared_mean, shared_components=shared_axes, shared_variance_ratio=shared_ratio,
             function_mean=function_mean, function_components=function_axes,
             function_variance_ratio=function_ratio)
    np.savez(args.output_dir / "latent_operators.npz",
             rotate=operators[ROT], translate_up=operators[UP],
             rotate_then_up=operators[ROT] @ operators[UP],
             up_then_rotate=operators[UP] @ operators[ROT])
    np.savez_compressed(args.output_dir / "model_prediction_grids.npz", **predictions)
    result = {"checkpoint": str(args.checkpoint), "global_step": global_step,
              "source_splits": ["train", "val"], "examples": len(records),
              "coverage": coverage, "atomic_operator_fit_train_pairs": fitted_counts,
              "ridge": args.ridge,
              "pca_explained_variance_ratio": {
                  "image": image_ratio.tolist(), "shared": shared_ratio.tolist(),
                  "function_mlp": function_ratio.tolist()},
              "full_latent_metrics": metrics(records, exact, composed, model_latents,
                                             predictions, operators),
              "interpretation": "PCA is visualization only; distances and commutators use full latents."}
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
