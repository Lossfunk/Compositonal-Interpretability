"""Compare exact C1 object operators with ViT image latents and predictions.

Run from the project root with ``python -m``. Training fits only atomic
rot90/translate_up examples. Validation/test and OOD rows are never fitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import sparse
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.c1_symmetry_geometry import (
    ATTRIBUTE_ROT, ATTRIBUTE_UP, ROT, UP, compose, foreground_correspondence,
)
from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 import _checkpoint_config
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from benchmarks.cogitao_vit_cross_attention.joint_token_model import (
    JointTokenViTTransformationModel,
)
from data.cogitao.cogitao import TASK_TO_ID


def read_split(root: Path, split: str) -> list[dict]:
    path = root / "exp_setting_1" / "experiment_1" / f"{split}.parquet"
    table = pq.read_table(path, columns=["input", "output", "transformation_suite"])
    suites = table["transformation_suite"].to_pylist()
    wanted = {(ROT,), (UP,), (UP, ROT)}
    rows = []
    for i, suite in enumerate(suites):
        if tuple(suite) in wanted:
            rows.append({
                "index": i, "split": split, "suite": tuple(suite),
                "input": np.asarray(table["input"][i].as_py(), dtype=np.int64),
                "target": np.asarray(table["output"][i].as_py(), dtype=np.int64),
            })
    return rows


def load_model(path: Path, model_kind: str, device: torch.device, allow_untrained: bool):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    train = config.get("data", {}).get("train", {}).get("init_args", {})
    if int(train.get("setting", -1)) != 1 or int(train.get("experiment", -1)) != 1:
        raise ValueError("Checkpoint must be C1 experiment 1")
    global_step = int(checkpoint.get("global_step", 0))
    if global_step < 1000 and not allow_untrained:
        raise ValueError(
            f"Checkpoint has only {global_step} training steps. "
            "Pass --allow-untrained for a pipeline smoke check only."
        )
    cls = (FunctionConditionedViTCrossAttentionModel if model_kind == "mlp"
           else JointTokenViTTransformationModel)
    model = cls(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, global_step


def resize_grids(grids: list[np.ndarray], size: tuple[int, int], device: torch.device):
    tensor = torch.as_tensor(np.stack(grids), dtype=torch.float32, device=device)
    return F.interpolate(tensor[:, None], size=size, mode="nearest")[:, 0].long()


def task_tensors(suites: list[tuple[str, ...]], device: torch.device):
    tokens = torch.tensor(
        [[TASK_TO_ID[name] for name in suite] + [0] * (2 - len(suite)) for suite in suites],
        device=device,
    )
    return tokens, tokens != 0


def model_batch(model, raw_inputs, raw_targets, suites, device):
    input_grid = resize_grids(raw_inputs, model.input_resolution, device)
    target = resize_grids(raw_targets, model.input_resolution, device)
    tokens, mask = task_tensors(suites, device)
    return {
        "images": F.one_hot(input_grid, num_classes=10).permute(0, 3, 1, 2).float(),
        "target_grid": target,
        "task_tokens": tokens,
        "task_token_mask": mask,
    }


def encode(model, model_kind: str, grids: list[np.ndarray], device, batch_size: int):
    """Image tokens in a fixed chart; joint ViT uses rot90 context for all grids."""
    all_features = []
    for offset in range(0, len(grids), batch_size):
        chunk = grids[offset:offset + batch_size]
        image = resize_grids(chunk, model.input_resolution, device)
        image = F.one_hot(image, num_classes=10).permute(0, 3, 1, 2).float()
        with torch.inference_mode():
            if model_kind == "mlp":
                features = model.encode_images(image)
            else:
                tokens, mask = task_tensors([(ROT,)] * len(chunk), device)
                task_features = (
                    model.task_embedding(tokens)
                    + model.task_position_embedding
                    + model.task_modality_embedding
                ) * mask.unsqueeze(-1)
                features = model.encoder(image, task_features, mask)[:, :model.num_image_tokens]
        all_features.append(features.float().cpu().numpy())
    return np.concatenate(all_features, axis=0)


def raw_to_token_weights(raw_shape, model):
    raw_h, raw_w = raw_shape
    out_h, out_w = model.input_resolution
    patch = int(getattr(model, "patch_size", 1))
    tokens_h, tokens_w = out_h // patch, out_w // patch
    weights = np.zeros((raw_h * raw_w, tokens_h * tokens_w), dtype=np.float32)
    for out_r in range(out_h):
        for out_c in range(out_w):
            raw_r = out_r * raw_h // out_h
            raw_c = out_c * raw_w // out_w
            token = (out_r // patch) * tokens_w + (out_c // patch)
            weights[raw_r * raw_w + raw_c, token] += 1
    counts = weights.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Some raw grid cells are missing from model tokens")
    return weights / counts[:, None]


def add_geometry(rows: list[dict]) -> dict:
    coverage = Counter()
    for row in rows:
        suite = row["suite"]
        try:
            generated, matrix, _ = compose(row["input"], suite)
            row["matrix"] = matrix
            row["geometry_exact"] = bool(np.array_equal(generated, row["target"]))
            coverage[(row["split"], suite, "recorded_supported")] += 1
            coverage[(row["split"], suite, "recorded_exact")] += row["geometry_exact"]
            if suite == (UP, ROT):
                try:
                    reverse, reverse_matrix, _ = compose(row["input"], (ROT, UP))
                    row["reverse_target"] = reverse
                    row["reverse_matrix"] = reverse_matrix
                    row["commute"] = bool(np.array_equal(generated, reverse))
                    row["matrix_commute"] = bool((matrix != reverse_matrix).nnz == 0)
                    coverage[(row["split"], suite, "reverse_supported")] += 1
                    coverage[(row["split"], suite, "same_final_grid")] += row["commute"]
                    coverage[(row["split"], suite, "same_pixel_matrix")] += row["matrix_commute"]
                except ValueError as error:
                    row["reverse_error"] = str(error)
                    coverage[(row["split"], suite, "reverse_unsupported")] += 1
                    if "out of the grid" in str(error):
                        coverage[(row["split"], suite, "reverse_out_of_grid")] += 1
        except ValueError as error:
            row["geometry_error"] = str(error)
            coverage[(row["split"], suite, "recorded_unsupported")] += 1
    return {f"{split}|{' -> '.join(suite)}|{key}": value
            for (split, suite, key), value in sorted(coverage.items())}


def save_matrices(rows: list[dict], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "attribute_operators.npz", rotate=ATTRIBUTE_ROT,
             translate_up=ATTRIBUTE_UP, rotate_then_up=ATTRIBUTE_UP @ ATTRIBUTE_ROT,
             up_then_rotate=ATTRIBUTE_ROT @ ATTRIBUTE_UP)
    sample = next(row for row in rows if row["suite"] == (UP, ROT)
                  and row.get("geometry_exact") and row.get("commute"))
    _, up_then_rot, (up_first, rot_after_up) = compose(sample["input"], (UP, ROT))
    _, rot_then_up, (rot_first, up_after_rot) = compose(sample["input"], (ROT, UP))
    if (up_then_rot != rot_after_up @ up_first).nnz or (rot_then_up != up_after_rot @ rot_first).nnz:
        raise AssertionError("Composed pixel matrices do not equal step-matrix products")
    sparse.save_npz(output / "sample_rotate.npz", rot_first)
    sparse.save_npz(output / "sample_translate_up.npz", up_first)
    sparse.save_npz(output / "sample_rotate_after_up.npz", rot_after_up)
    sparse.save_npz(output / "sample_translate_up_after_rotate.npz", up_after_rot)
    sparse.save_npz(output / "sample_up_then_rotate.npz", up_then_rot)
    sparse.save_npz(output / "sample_rotate_then_up.npz", rot_then_up)
    np.savez(output / "sample_grids.npz", source=sample["input"],
             recorded_target=sample["target"], reverse_target=sample["reverse_target"])
    return {"split": sample["split"], "source_index": sample["index"],
            "raw_grid_shape": list(sample["input"].shape)}


def evaluate_outputs(model, rows, device, batch_size):
    groups = {}
    for split in ("val", "test", "val_ood", "test_ood"):
        for suite, paired in (((ROT,), False), ((UP,), False),
                              ((UP, ROT), False), ((UP, ROT), True),
                              ((ROT, UP), False)):
            selected = [row for row in rows if row["split"] == split and
                        row.get("geometry_exact") and
                        (not paired or row.get("matrix_commute")) and
                        (row["suite"] == suite or
                         suite == (ROT, UP) and row["suite"] == (UP, ROT)
                         and row.get("matrix_commute"))]
            if not selected:
                continue
            stats = Counter()
            for offset in range(0, len(selected), batch_size):
                batch_rows = selected[offset:offset + batch_size]
                targets = [row["reverse_target"] if suite == (ROT, UP)
                           else row["target"] for row in batch_rows]
                batch = model_batch(model, [row["input"] for row in batch_rows],
                                    targets, [suite] * len(batch_rows), device)
                with torch.inference_mode():
                    output = model(batch)
                correct = output["predictions"].eq(batch["target_grid"])
                object_mask = output["predictions"].ne(0) | batch["target_grid"].ne(0)
                object_count = object_mask.flatten(1).sum(dim=1).clamp_min(1)
                object_correct = (correct & object_mask).flatten(1).sum(dim=1)
                nll = F.nll_loss(output["output_log_probs"], batch["target_grid"],
                                 reduction="none")
                stats["samples"] += len(batch_rows)
                stats["exact"] += int(correct.flatten(1).all(dim=1).sum().item())
                stats["pixel_correct"] += int(correct.sum().item())
                stats["pixel_total"] += correct.numel()
                stats["object_accuracy_sum"] += float((object_correct / object_count).sum().item())
                stats["nll_sum"] += float(nll.sum().item())
            key = f"{split}|{' -> '.join(suite)}"
            if paired:
                key += "|paired_reverse_subset"
            groups[key] = {
                "samples": stats["samples"],
                "exact_grid_accuracy": stats["exact"] / stats["samples"],
                "pixel_accuracy": stats["pixel_correct"] / stats["pixel_total"],
                "object_pixel_accuracy": stats["object_accuracy_sum"] / stats["samples"],
                "cross_entropy": stats["nll_sum"] / stats["pixel_total"],
                "counterfactual": suite == (ROT, UP), "paired_reverse_subset": paired,
            }
    return groups


def feature_pairs(model, kind, rows, device, batch_size, weights):
    source_features = encode(model, kind, [row["input"] for row in rows], device, batch_size)
    target_features = encode(model, kind, [row["target"] for row in rows], device, batch_size)
    xs, ys = [], []
    for row, source, target in zip(rows, source_features, target_features):
        src, dst = foreground_correspondence(row["matrix"])
        xs.append((weights @ source)[src])
        ys.append((weights @ target)[dst])
    return np.concatenate(xs).astype(np.float64), np.concatenate(ys).astype(np.float64)


def fit_affine(x, y, ridge: float):
    xh = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    penalty = np.eye(xh.shape[1]) * ridge
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(xh.T @ xh + penalty, xh.T @ y)
    result = np.eye(xh.shape[1])
    result[:, :-1] = coefficients
    return result


def latent_metrics(x, y, operator):
    xh = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    predicted = (xh @ operator)[:, :-1]
    mse = float(np.mean((predicted - y) ** 2))
    identity = float(np.mean((x - y) ** 2))
    variance = float(np.var(y))
    return {"pixels": len(x), "mse": mse, "identity_mse": identity,
            "mse_over_identity": mse / identity if identity else None,
            "r2_global": 1 - mse / variance if variance else None}


def analyze_latents(model, kind, rows, device, batch_size, train_pairs, eval_pairs, ridge):
    weights = raw_to_token_weights(rows[0]["input"].shape, model)
    fitted = {}
    results = {}
    for operation in (ROT, UP):
        train = [r for r in rows if r["split"] == "train" and
                 r["suite"] == (operation,) and r.get("geometry_exact")][:train_pairs]
        if not train:
            raise ValueError(f"No exact atomic training pairs for {operation}")
        x, y = feature_pairs(model, kind, train, device, batch_size, weights)
        fitted[operation] = fit_affine(x, y, ridge)
        results[f"train|{operation}"] = {"pairs": len(train), **latent_metrics(x, y, fitted[operation])}
    # Row vectors: x @ A_R @ A_T corresponds to rotate, then translate.
    operators = {
        (ROT,): fitted[ROT], (UP,): fitted[UP],
        (ROT, UP): fitted[ROT] @ fitted[UP],
        (UP, ROT): fitted[UP] @ fitted[ROT],
    }
    for split in ("val", "test", "val_ood", "test_ood"):
        for suite in ((ROT,), (UP,), (UP, ROT), (ROT, UP)):
            selected = [r for r in rows if r["split"] == split and
                        r.get("geometry_exact") and
                        (r["suite"] == suite or
                         suite == (ROT, UP) and r["suite"] == (UP, ROT)
                         and r.get("matrix_commute"))][:eval_pairs]
            if not selected:
                continue
            # The reverse-order counterfactual has the same exact target only
            # on verified commuting examples.
            x, y = feature_pairs(model, kind, selected, device, batch_size, weights)
            results[f"{split}|{' -> '.join(suite)}"] = {
                "pairs": len(selected), **latent_metrics(x, y, operators[suite]),
                "counterfactual": suite == (ROT, UP),
            }
    commutator = operators[(ROT, UP)] - operators[(UP, ROT)]
    results["operator_diagnostics"] = {
        "dimension_homogeneous": int(commutator.shape[0]),
        "relative_commutator_frobenius": float(
            np.linalg.norm(commutator) /
            max(np.linalg.norm(operators[(ROT, UP)]), np.linalg.norm(operators[(UP, ROT)]), 1e-12)
        ),
        "ridge": ridge,
        "chart": "image token features; joint ViT uses fixed rot90 function context",
    }
    return results, operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "joint"), default="mlp")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "checkpoints/cogitao_vit_cross_attention/"
        "cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000/last.ckpt"))
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cogitao_c1_symmetries/mlp"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument("--eval-pairs", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-untrained", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.train_pairs, args.eval_pairs) <= 0 or args.ridge < 0:
        parser.error("Batch and pair counts must be positive; ridge must be nonnegative")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if not args.checkpoint.is_file():
        parser.error(f"Missing checkpoint: {args.checkpoint}")
    model, global_step = load_model(args.checkpoint, args.model, device, args.allow_untrained)
    rows = []
    for split in ("train", "val", "test", "val_ood", "test_ood"):
        selected = read_split(args.data_root, split)
        if split == "train":
            selected = ([r for r in selected if r["suite"] == (ROT,)][:args.train_pairs]
                        + [r for r in selected if r["suite"] == (UP,)][:args.train_pairs])
        rows.extend(selected)
    coverage = add_geometry(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    example = save_matrices(rows, args.output_dir)
    print("Geometry checked; evaluating direct model predictions", flush=True)
    direct = evaluate_outputs(model, rows, device, args.batch_size)
    print("Direct predictions done; fitting and scoring latent operators", flush=True)
    latent, operators = analyze_latents(model, args.model, rows, device, args.batch_size,
                                       args.train_pairs, args.eval_pairs, args.ridge)
    np.savez(args.output_dir / "fitted_latent_operators.npz",
             rotate=operators[(ROT,)], translate_up=operators[(UP,)],
             rotate_then_up=operators[(ROT, UP)], up_then_rotate=operators[(UP, ROT)])
    report = {
        "model": args.model, "checkpoint": str(args.checkpoint),
        "global_step": global_step, "trained_checkpoint": global_step >= 1000,
        "model_resolution": list(model.input_resolution), "raw_resolution": list(rows[0]["input"].shape),
        "geometry_coverage": coverage, "matrix_example": example,
        "attribute_matrices_commute": bool(np.array_equal(
            ATTRIBUTE_UP @ ATTRIBUTE_ROT, ATTRIBUTE_ROT @ ATTRIBUTE_UP)),
        "direct_model": direct, "latent_probe": latent,
        "probe_train_pairs_per_atomic": args.train_pairs,
        "probe_eval_pairs_per_suite": args.eval_pairs,
    }
    path = args.output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
