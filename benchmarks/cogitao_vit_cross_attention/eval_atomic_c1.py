"""Evaluate only atomic C1 transformations on the four validation/test splits.

Run from the repository root with ``python -m
benchmarks.cogitao_vit_cross_attention.eval_atomic_c1``. C1 OOD splits in the
released data contain no atomic rows; they are reported with zero coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from data.cogitao.cogitao import COGITAODataset, TASK_VOCABULARY


SPLITS = ("val", "val_ood", "test", "test_ood")
ATTRIBUTE_FAMILIES = {
    "change_shape_color": ("shape", "color"),
    "crop_bottom_side": ("size", "position"),
    "crop_contours": ("contour",),
    "crop_top_side": ("size", "position"),
    "double_down": ("size", "position"),
    "double_right": ("size", "position"),
    "empty_inside_pixels": ("interior",),
    "extend_contours_same_color": ("contour",),
    "fill_holes_different_color": ("interior", "color"),
    "fill_holes_same_color": ("interior", "color"),
    "mirror_horizontal": ("orientation",),
    "mirror_vertical": ("orientation",),
    "pad_left": ("position",),
    "pad_right": ("position",),
    "pad_top": ("position",),
    "rot90": ("orientation",),
    "translate_right": ("position",),
    "translate_up": ("position",),
}
if set(ATTRIBUTE_FAMILIES) != set(TASK_VOCABULARY):
    raise RuntimeError("Update ATTRIBUTE_FAMILIES for the COGITAO vocabulary.")


class Totals:
    def __init__(self) -> None:
        self.samples = 0
        self.exact = 0
        self.correct_pixels = 0
        self.pixels = 0
        self.object_accuracy_sum = 0.0
        self.nll_sum = 0.0

    def add(self, *, exact: bool, correct_pixels: int, pixels: int,
            object_accuracy: float, nll_sum: float) -> None:
        self.samples += 1
        self.exact += int(exact)
        self.correct_pixels += correct_pixels
        self.pixels += pixels
        self.object_accuracy_sum += object_accuracy
        self.nll_sum += nll_sum

    def merge(self, other: "Totals") -> None:
        self.samples += other.samples
        self.exact += other.exact
        self.correct_pixels += other.correct_pixels
        self.pixels += other.pixels
        self.object_accuracy_sum += other.object_accuracy_sum
        self.nll_sum += other.nll_sum

    def row(self, split: str, group_by: str, group: str) -> dict[str, Any]:
        return {
            "split": split,
            "group_by": group_by,
            "group": group,
            "samples": self.samples,
            "grid_accuracy_pct": 100 * self.exact / self.samples if self.samples else None,
            "per_pixel_accuracy_pct": 100 * self.correct_pixels / self.pixels if self.pixels else None,
            "object_per_pixel_accuracy_pct": 100 * self.object_accuracy_sum / self.samples if self.samples else None,
            "cross_entropy_loss": self.nll_sum / self.pixels if self.pixels else None,
        }


def _fill_color(input_grid: torch.Tensor, target: torch.Tensor) -> str:
    """Observed color written into background cells, not a dataset parameter."""
    added = target[(input_grid == 0) & (target != 0)].unique().tolist()
    if len(added) == 1:
        return str(added[0])
    return "none" if not added else "multiple"


def _atomic_indices(dataset: COGITAODataset) -> tuple[list[int], list[str]]:
    suites = dataset.table["transformation_suite"].slice(0, len(dataset)).to_pylist()
    indices = [i for i, suite in enumerate(suites) if len(suite) == 1]
    return indices, [suites[i][0] for i in indices]


def evaluate_split(
    model: FunctionConditionedViTCrossAttentionModel,
    dataset: COGITAODataset,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], Totals]]:
    indices, functions = _atomic_indices(dataset)
    coverage = {
        "split": split,
        "total_rows": len(dataset),
        "atomic_rows": len(indices),
        "composition_rows_skipped": len(dataset) - len(indices),
    }
    totals: dict[tuple[str, str], Totals] = defaultdict(Totals)
    if not indices:
        return coverage, totals

    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            count = batch["images"].size(0)
            batch_functions = functions[offset:offset + count]
            offset += count
            inputs = batch["images"].argmax(dim=1)
            targets_cpu = batch["target_grid"]
            output = model({
                "images": batch["images"].to(device, non_blocking=True),
                "target_grid": targets_cpu.to(device, non_blocking=True),
                "task_tokens": batch["task_tokens"].to(device, non_blocking=True),
                "task_token_mask": batch["task_token_mask"].to(device, non_blocking=True),
            })
            predictions = output["predictions"]
            target = output["target_grid"]
            correct = predictions.eq(target)
            object_mask = predictions.ne(0) | target.ne(0)
            object_pixels = object_mask.flatten(1).sum(dim=1)
            correct_object = (correct & object_mask).flatten(1).sum(dim=1)
            object_accuracy = torch.where(
                object_pixels > 0,
                correct_object.float() / object_pixels.clamp_min(1),
                correct.flatten(1).all(dim=1).float(),
            )
            nll = F.nll_loss(output["output_log_probs"], target, reduction="none")
            if not bool(torch.isfinite(nll).all()):
                raise FloatingPointError(f"Non-finite cross entropy in {split}.")
            exact = correct.flatten(1).all(dim=1).cpu().tolist()
            correct_counts = correct.flatten(1).sum(dim=1).cpu().tolist()
            object_scores = object_accuracy.cpu().tolist()
            nll_sums = nll.flatten(1).sum(dim=1).cpu().tolist()
            pixel_count = target.shape[-2] * target.shape[-1]

            for i, function in enumerate(batch_functions):
                groups = [("overall", "all_atomic"), ("function", function)]
                groups.extend(("attribute", name) for name in ATTRIBUTE_FAMILIES[function])
                if function.startswith("fill_holes_"):
                    fill_color = _fill_color(inputs[i], targets_cpu[i])
                    groups.append(("observed_fill_color", fill_color))
                    groups.append(("function_fill_color", f"{function}:color={fill_color}"))
                for key in groups:
                    totals[key].add(
                        exact=exact[i], correct_pixels=correct_counts[i],
                        pixels=pixel_count, object_accuracy=object_scores[i],
                        nll_sum=nll_sums[i],
                    )

    return coverage, totals


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("hyper_parameters", {}).get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint lacks hyper_parameters.config; use a C1 ViT checkpoint.")
    class_path = config.get("model", {}).get("class_path")
    expected = "benchmarks.cogitao_vit_cross_attention.model.FunctionConditionedViTCrossAttentionModel"
    if class_path != expected:
        raise ValueError(f"Checkpoint model is {class_path!r}; expected {expected!r}.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--checkpoint", type=Path, help="C1 function-conditioned .ckpt; defaults to run_000/last.ckpt")
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cogitao_c1_function_conditioned_atomic"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    checkpoint_path = args.checkpoint or Path(
        f"checkpoints/cogitao_vit_cross_attention/cogitao_c1_experiment_{args.experiment}_vit6_function_mlp_cross_attention/run_000/last.ckpt"
    )
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    train_args = config.get("data", {}).get("train", {}).get("init_args", {})
    if int(train_args.get("setting", -1)) != 1 or int(train_args.get("experiment", -1)) != args.experiment:
        parser.error("Checkpoint training split does not match C1 and --experiment")
    model = FunctionConditionedViTCrossAttentionModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    coverage: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        dataset = COGITAODataset({
            "root": args.data_root, "setting": 1, "experiment": args.experiment,
            "split": split, "image_size": list(model.input_resolution),
            "image_field": "input", "return_target": True,
            "representation": "one_hot", "return_task_tokens": True,
            "max_task_tokens": model.max_function_tokens,
            "return_metadata": True,
        })
        split_coverage, split_totals = evaluate_split(
            model, dataset, split, device, args.batch_size, args.num_workers,
        )
        coverage.append(split_coverage)
        if split_totals:
            rows.extend(
                total.row(split, kind, group)
                for (kind, group), total in sorted(split_totals.items())
            )
        else:
            rows.append(Totals().row(split, "overall", "all_atomic"))
        print(f"{split}: {split_coverage['atomic_rows']}/{split_coverage['total_rows']} atomic rows")

    output_dir = args.output_dir / f"experiment_{args.experiment}"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "setting": 1, "experiment": args.experiment,
        "checkpoint": str(checkpoint_path), "device": str(device),
        "coverage": coverage, "metrics": rows,
        "notes": [
            "Only transformation suites of length one were evaluated.",
            "Attribute groups describe the affected property; a function may belong to multiple groups.",
            "Observed fill color is inferred from target cells that were background in the input; it is not a recorded function argument.",
        ],
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
