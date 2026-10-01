"""Evaluate C1 experiment-1 rot90/translate_up across saved training checkpoints.

The W&B sweep launches one run per atomic function. Each run logs all distinct
checkpoint steps on val and test, so W&B plots show training progression.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 import _checkpoint_config
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from data.cogitao.cogitao import COGITAODataset


FUNCTIONS = ("rot90", "translate_up")
METRICS = ("samples", "grid_accuracy_pct", "per_pixel_accuracy_pct",
           "object_per_pixel_accuracy_pct", "cross_entropy_loss")
DEFAULT_CHECKPOINT_DIR = Path(
    "checkpoints/cogitao_vit_cross_attention/"
    "cogitao_c1_experiment_1_vit6_function_mlp_cross_attention/run_000"
)


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    global_step: int
    epoch: int

    @property
    def label(self) -> str:
        return self.path.stem


def priority(path: Path) -> int:
    name = path.name
    if name == "last.ckpt":
        return 4
    if name == "training_end.ckpt":
        return 3
    if name.startswith("milestone_"):
        return 2
    if name.startswith("epoch="):
        return 1
    return 0


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    train_args = config.get("data", {}).get("train", {}).get("init_args", {})
    if int(train_args.get("setting", -1)) != 1 or int(train_args.get("experiment", -1)) != 1:
        raise ValueError(f"Checkpoint is not C1 experiment 1: {path}")
    return checkpoint, config


def discover_checkpoints(directory: Path) -> list[Checkpoint]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    by_step: dict[int, Checkpoint] = {}
    for path in sorted(directory.glob("*.ckpt")):
        checkpoint, _ = load_checkpoint(path)
        candidate = Checkpoint(path, int(checkpoint.get("global_step", 0)),
                               int(checkpoint.get("epoch", -1)))
        existing = by_step.get(candidate.global_step)
        if existing is None or priority(path) > priority(existing.path):
            by_step[candidate.global_step] = candidate
        del checkpoint
    stages = sorted(by_step.values(), key=lambda item: item.global_step)
    if not stages:
        raise ValueError(f"No checkpoints in {directory}")
    if len(stages) < 2:
        raise ValueError("Progression needs at least two distinct checkpoint steps")
    return stages


def make_loader(data_root: Path, function: str, split: str,
                image_size: tuple[int, int], max_tokens: int,
                batch_size: int, num_workers: int, device: torch.device):
    dataset = COGITAODataset({
        "root": data_root, "setting": 1, "experiment": 1, "split": split,
        "image_size": list(image_size), "image_field": "input",
        "return_target": True, "representation": "one_hot",
        "return_task_tokens": True, "max_task_tokens": max_tokens,
        "return_metadata": False,
    })
    suites = dataset.table["transformation_suite"].to_pylist()
    selected = [i for i, suite in enumerate(suites) if suite == [function]]
    if not selected:
        raise ValueError(f"No atomic {function} examples in {split}")
    loader = DataLoader(
        Subset(dataset, selected), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    return loader, {"total_rows": len(dataset), "atomic_rows": len(selected)}


def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float | int]:
    samples = exact = pixel_correct = pixel_total = 0
    object_accuracy_sum = nll_sum = 0.0
    with torch.inference_mode():
        for batch in loader:
            output = model({
                "images": batch["images"].to(device, non_blocking=True),
                "target_grid": batch["target_grid"].to(device, non_blocking=True),
                "task_tokens": batch["task_tokens"].to(device, non_blocking=True),
                "task_token_mask": batch["task_token_mask"].to(device, non_blocking=True),
            })
            predictions = output["predictions"]
            target = output["target_grid"]
            correct = predictions.eq(target)
            object_mask = predictions.ne(0) | target.ne(0)
            object_counts = object_mask.flatten(1).sum(dim=1)
            object_correct = (correct & object_mask).flatten(1).sum(dim=1)
            object_accuracy = torch.where(
                object_counts > 0,
                object_correct.float() / object_counts.clamp_min(1),
                correct.flatten(1).all(dim=1).float(),
            )
            nll = F.nll_loss(output["output_log_probs"], target, reduction="sum")
            if not bool(torch.isfinite(nll)):
                raise FloatingPointError("Nonfinite C1 atomic cross entropy")
            samples += int(target.size(0))
            exact += int(correct.flatten(1).all(dim=1).sum().item())
            pixel_correct += int(correct.sum().item())
            pixel_total += int(correct.numel())
            object_accuracy_sum += float(object_accuracy.sum().item())
            nll_sum += float(nll.item())
    if samples == 0:
        raise ValueError("Atomic evaluation loader is empty")
    return {
        "samples": samples,
        "grid_accuracy_pct": 100.0 * exact / samples,
        "per_pixel_accuracy_pct": 100.0 * pixel_correct / pixel_total,
        "object_per_pixel_accuracy_pct": 100.0 * object_accuracy_sum / samples,
        "cross_entropy_loss": nll_sum / pixel_total,
    }


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def start_wandb(args):
    if args.wandb_mode == "disabled":
        return None, None
    import wandb
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group="cogitao-c1-atomic-checkpoint-progression",
        job_type="evaluation", name=f"c1-exp1-{args.function}-atomic-progression",
        tags=["cogitao", "c1", "experiment-1", "atomic-eval", "checkpoint-progression"],
        mode=args.wandb_mode,
        config={"function": args.function, "checkpoint_dir": str(args.checkpoint_dir),
                "batch_size": args.batch_size, "splits": ["val", "test"]},
    )
    run.define_metric("checkpoint/global_step")
    for split in ("val", "test"):
        for metric in METRICS:
            run.define_metric(f"{split}/{metric}", step_metric="checkpoint/global_step")
        for metric in ("grid_accuracy_pct", "object_per_pixel_accuracy_pct",
                       "cross_entropy_loss"):
            for delta in ("delta_from_start", "delta_from_previous"):
                run.define_metric(f"{split}/{metric}_{delta}",
                                  step_metric="checkpoint/global_step")
    run.define_metric("val/best_so_far_grid_accuracy_pct",
                      step_metric="checkpoint/global_step")
    return wandb, run


def append_progress(rows: list[dict]):
    """Add gains from baseline/previous and validation best-so-far."""
    for function in FUNCTIONS:
        subset = [row for row in rows if row["function"] == function]
        if not subset:
            continue
        baseline = subset[0]
        best_val = float("-inf")
        best_label = None
        for i, row in enumerate(subset):
            previous = subset[i - 1] if i else baseline
            for split in ("val", "test"):
                for metric in ("grid_accuracy_pct", "object_per_pixel_accuracy_pct",
                               "cross_entropy_loss"):
                    key = f"{split}_{metric}"
                    row[key + "_delta_from_start"] = row[key] - baseline[key]
                    row[key + "_delta_from_previous"] = row[key] - previous[key]
            if row["val_grid_accuracy_pct"] > best_val:
                best_val = row["val_grid_accuracy_pct"]
                best_label = row["checkpoint"]
            row["best_so_far_val_grid_accuracy_pct"] = best_val
            row["best_so_far_val_checkpoint"] = best_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=(*FUNCTIONS, "both"), default="both")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/cogitao_c1_atomic_progression"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                        default="disabled")
    parser.add_argument("--wandb-project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.function == "both" and args.wandb_mode != "disabled":
        parser.error("W&B logging uses one function per run; choose rot90 or translate_up")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    functions = FUNCTIONS if args.function == "both" else (args.function,)
    stages = discover_checkpoints(args.checkpoint_dir)
    first_checkpoint, first_config = load_checkpoint(stages[0].path)
    first_model = FunctionConditionedViTCrossAttentionModel(first_config)
    image_size = first_model.input_resolution
    max_tokens = first_model.max_function_tokens
    del first_checkpoint, first_model
    loaders = {}
    coverage = {}
    for function in functions:
        for split in ("val", "test"):
            loaders[(function, split)], coverage[f"{function}|{split}"] = make_loader(
                args.data_root, function, split, image_size, max_tokens,
                args.batch_size, args.num_workers, device)
    wandb_module, run = start_wandb(args)
    max_step = max(stage.global_step for stage in stages)
    rows: list[dict] = []
    try:
        for stage in stages:
            checkpoint, config = load_checkpoint(stage.path)
            model = FunctionConditionedViTCrossAttentionModel(config)
            if model.input_resolution != image_size or model.max_function_tokens != max_tokens:
                raise ValueError(f"Checkpoint preprocessing changed: {stage.path}")
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device).eval()
            for function in functions:
                row = {
                    "function": function, "checkpoint": stage.label,
                    "checkpoint_path": str(stage.path), "global_step": stage.global_step,
                    "epoch": stage.epoch,
                    "progress_pct": 100.0 * stage.global_step / max_step if max_step else 0.0,
                }
                for split in ("val", "test"):
                    result = evaluate(model, loaders[(function, split)], device)
                    row.update({f"{split}_{key}": value for key, value in result.items()})
                rows.append(row)
                append_progress(rows)
                print(f"{function} {stage.label} step={stage.global_step}: "
                      f"val exact={row['val_grid_accuracy_pct']:.2f}% "
                      f"test exact={row['test_grid_accuracy_pct']:.2f}%", flush=True)
                if run is not None:
                    payload = {
                        "checkpoint/global_step": stage.global_step,
                        "checkpoint/epoch": stage.epoch,
                        "checkpoint/progress_pct": row["progress_pct"],
                    }
                    for split in ("val", "test"):
                        payload.update({f"{split}/{metric}": row[f"{split}_{metric}"]
                                        for metric in METRICS})
                    for split in ("val", "test"):
                        for metric in ("grid_accuracy_pct", "object_per_pixel_accuracy_pct",
                                       "cross_entropy_loss"):
                            for delta in ("delta_from_start", "delta_from_previous"):
                                field = f"{split}_{metric}_{delta}"
                                payload[f"{split}/{metric}_{delta}"] = row[field]
                    payload["val/best_so_far_grid_accuracy_pct"] = (
                        row["best_so_far_val_grid_accuracy_pct"])
                    run.log(payload)
            del model, checkpoint
            if device.type == "cuda":
                torch.cuda.empty_cache()
        append_progress(rows)
        output = args.output_dir / args.function
        output.mkdir(parents=True, exist_ok=True)
        csv_path = output / "progressive_metrics.csv"
        write_csv(csv_path, rows)
        summaries = {}
        for function in functions:
            subset = [row for row in rows if row["function"] == function]
            best = max(subset, key=lambda row: row["val_grid_accuracy_pct"])
            summaries[function] = {
                "best_val_grid_checkpoint": best["checkpoint"],
                "best_val_grid_accuracy_pct": best["val_grid_accuracy_pct"],
                "test_grid_accuracy_at_val_best_pct": best["test_grid_accuracy_pct"],
                "final_checkpoint": subset[-1]["checkpoint"],
                "final_val_grid_accuracy_pct": subset[-1]["val_grid_accuracy_pct"],
                "final_test_grid_accuracy_pct": subset[-1]["test_grid_accuracy_pct"],
            }
            if run is not None and args.function != "both":
                run.summary.update(summaries[function])
        report = {
            "setting": 1, "experiment": 1, "checkpoint_dir": str(args.checkpoint_dir),
            "functions": list(functions), "checkpoint_count": len(stages),
            "coverage": coverage, "summary": summaries, "rows": rows,
            "notes": [
                "Each row evaluates the same atomic val/test examples at one checkpoint step.",
                "The released C1 OOD val/test splits have no atomic rot90 or translate_up rows.",
                "Duplicate files at the same global_step are collapsed; last.ckpt wins ties.",
            ],
        }
        json_path = output / "report.json"
        json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if run is not None:
            table = wandb_module.Table(columns=list(rows[0]),
                                       data=[[row[column] for column in rows[0]]
                                             for row in rows])
            run.log({"progressive/table": table})
            run.summary["local_csv"] = str(csv_path)
        print(f"Wrote {csv_path} and {json_path}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
