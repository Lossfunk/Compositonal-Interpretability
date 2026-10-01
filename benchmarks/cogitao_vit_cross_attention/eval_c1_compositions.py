"""Report ordered C1 compositions on ID/OOD splits and atomic functions on val.

Each experiment uses its own function-conditioned ViT checkpoint. The pooled
rows combine raw counts across experiments before computing percentages.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 import Totals
from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1_all import (
    _checkpoint_path,
    _dataset,
    _model_from_checkpoint,
)
from data.cogitao.cogitao import COGITAODataset, TASK_VOCABULARY


SPLITS = ("val", "val_ood", "test", "test_ood")
SPLIT_COLUMNS = {
    "val": "val_id",
    "val_ood": "val_ood",
    "test": "test_id",
    "test_ood": "test_ood",
}
METRICS = (
    "samples",
    "grid_accuracy_pct",
    "per_pixel_accuracy_pct",
    "object_per_pixel_accuracy_pct",
    "cross_entropy_loss",
)


def evaluate_split(
    model: torch.nn.Module,
    dataset: COGITAODataset,
    split: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Totals], dict[str, Totals], dict[str, int]]:
    suites = dataset.table["transformation_suite"].slice(0, len(dataset)).to_pylist()
    if any(len(suite) > model.max_function_tokens for suite in suites):
        raise ValueError(f"{split} contains a suite longer than the model supports")
    if any(not suite or any(name not in TASK_VOCABULARY for name in suite) for suite in suites):
        raise ValueError(f"{split} contains an empty or unknown transformation suite")

    # Atomic function results are requested only for val. All other splits
    # evaluate composition rows only.
    indices = [
        index for index, suite in enumerate(suites)
        if len(suite) == 2 or (split == "val" and len(suite) == 1)
    ]
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )
    compositions: dict[str, Totals] = defaultdict(Totals)
    functions: dict[str, Totals] = defaultdict(Totals)
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            count = batch["images"].size(0)
            batch_suites = [suites[index] for index in indices[offset:offset + count]]
            offset += count
            output = model({
                "images": batch["images"].to(device, non_blocking=True),
                "target_grid": batch["target_grid"].to(device, non_blocking=True),
                "task_tokens": batch["task_tokens"].to(device, non_blocking=True),
                "task_token_mask": batch["task_token_mask"].to(device, non_blocking=True),
            })
            target = output["target_grid"]
            predictions = output["predictions"]
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
                raise FloatingPointError(f"Non-finite cross entropy in {split}")
            exact = correct.flatten(1).all(dim=1).cpu().tolist()
            correct_counts = correct.flatten(1).sum(dim=1).cpu().tolist()
            object_scores = object_accuracy.cpu().tolist()
            nll_sums = nll.flatten(1).sum(dim=1).cpu().tolist()
            pixels = target.shape[-2] * target.shape[-1]

            for index, suite in enumerate(batch_suites):
                if len(suite) == 1:
                    groups = (functions[suite[0]], functions["all_functions"])
                else:
                    groups = (
                        compositions[" -> ".join(suite)],
                        compositions["all_compositions"],
                    )
                for totals in groups:
                    totals.add(
                        exact=exact[index], correct_pixels=correct_counts[index],
                        pixels=pixels, object_accuracy=object_scores[index],
                        nll_sum=nll_sums[index],
                    )

    coverage = {
        "total_rows": len(dataset),
        "composition_rows": sum(len(suite) == 2 for suite in suites),
        "function_rows": sum(len(suite) == 1 for suite in suites),
        "evaluated_rows": len(indices),
    }
    return compositions, functions, coverage


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=range(1, 6),
                        help="Evaluate one experiment; default: all five")
    parser.add_argument("--checkpoint-root", type=Path,
                        default=Path("checkpoints/cogitao_vit_cross_attention"))
    parser.add_argument("--run", default="run_000")
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/cogitao/files/CompGen"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/cogitao_c1_function_conditioned_compositions"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    experiments = [args.experiment] if args.experiment else list(range(1, 6))
    paths = {
        experiment: _checkpoint_path(args.checkpoint_root, experiment, args.run)
        for experiment in experiments
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error("Missing checkpoints: " + ", ".join(missing))

    compositions: dict[tuple[str, str, str], Totals] = defaultdict(Totals)
    functions: dict[tuple[str, str], Totals] = defaultdict(Totals)
    coverage_rows = []
    for experiment, path in paths.items():
        model = _model_from_checkpoint(path, experiment, device)
        for split in SPLITS:
            dataset = _dataset(args.data_root, model, experiment, split)
            split_compositions, split_functions, coverage = evaluate_split(
                model, dataset, split, device, args.batch_size, args.num_workers,
            )
            for name, totals in split_compositions.items():
                compositions[(str(experiment), name, split)].merge(totals)
                compositions[("all", name, split)].merge(totals)
            for name, totals in split_functions.items():
                functions[(str(experiment), name)].merge(totals)
                functions[("all", name)].merge(totals)
            coverage_rows.append({"experiment": experiment, "split": split, **coverage})
            print(f"experiment {experiment} {split}: "
                  f"{coverage['composition_rows']} compositions, "
                  f"{coverage['function_rows']} functions")
        del model

    labels = ["all", *(str(experiment) for experiment in experiments)]
    composition_names = sorted({name for _, name, _ in compositions if name != "all_compositions"})
    function_names = sorted({name for _, name in functions if name != "all_functions"})
    composition_rows = []
    for label in labels:
        for name in ["all_compositions", *composition_names]:
            for metric in METRICS:
                row = {"experiment": label, "composition": name, "metric": metric}
                for split, column in SPLIT_COLUMNS.items():
                    totals = compositions.get((label, name, split))
                    row[column] = (
                        totals.row(split, "composition", name)[metric] if totals
                        else 0 if metric == "samples" else ""
                    )
                composition_rows.append(row)
    function_rows = []
    for label in labels:
        for name in ["all_functions", *function_names]:
            totals = functions.get((label, name))
            if totals is None:
                continue
            values = totals.row("val", "function", name)
            function_rows.append({
                "experiment": label, "function": name,
                **{metric: values[metric] for metric in METRICS},
            })

    _write_csv(
        args.output_dir / "compositions.csv",
        ["experiment", "composition", "metric", *SPLIT_COLUMNS.values()],
        composition_rows,
    )
    _write_csv(
        args.output_dir / "functions_val.csv",
        ["experiment", "function", *METRICS], function_rows,
    )
    _write_csv(
        args.output_dir / "coverage.csv",
        ["experiment", "split", "total_rows", "composition_rows",
         "function_rows", "evaluated_rows"],
        coverage_rows,
    )


if __name__ == "__main__":
    main()
