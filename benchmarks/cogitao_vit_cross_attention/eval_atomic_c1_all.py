"""Write a wide, single-CSV atomic C1 report for function-conditioned ViTs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch

from benchmarks.cogitao_vit_cross_attention.eval_atomic_c1 import (
    SPLITS,
    Totals,
    _checkpoint_config,
    evaluate_split,
)
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from data.cogitao.cogitao import COGITAODataset, TASK_VOCABULARY


METRICS = (
    "samples",
    "grid_accuracy_pct",
    "per_pixel_accuracy_pct",
    "object_per_pixel_accuracy_pct",
    "cross_entropy_loss",
)


def _checkpoint_path(root: Path, experiment: int, run: str) -> Path:
    slug = f"cogitao_c1_experiment_{experiment}_vit6_function_mlp_cross_attention"
    return root / slug / run / "last.ckpt"


def _model_from_checkpoint(path: Path, experiment: int, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = _checkpoint_config(checkpoint)
    train_args = config.get("data", {}).get("train", {}).get("init_args", {})
    if int(train_args.get("setting", -1)) != 1 or int(train_args.get("experiment", -1)) != experiment:
        raise ValueError(f"Checkpoint was not trained on C1 experiment {experiment}: {path}")
    model = FunctionConditionedViTCrossAttentionModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _dataset(root: Path, model: FunctionConditionedViTCrossAttentionModel,
             experiment: int, split: str) -> COGITAODataset:
    return COGITAODataset({
        "root": root,
        "setting": 1,
        "experiment": experiment,
        "split": split,
        "image_size": list(model.input_resolution),
        "image_field": "input",
        "return_target": True,
        "representation": "one_hot",
        "return_task_tokens": True,
        "max_task_tokens": model.max_function_tokens,
        "return_metadata": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=Path("checkpoints/cogitao_vit_cross_attention"),
    )
    parser.add_argument("--run", default="run_000")
    parser.add_argument("--data-root", type=Path, default=Path("data/cogitao/files/CompGen"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/cogitao_c1_function_conditioned_atomic/all_experiments.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--include-experiments", action="store_true",
        help="Also add rows for each experiment; the default contains pooled C1 rows only.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    checkpoints = {
        experiment: _checkpoint_path(args.checkpoint_root, experiment, args.run)
        for experiment in range(1, 6)
    }
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        parser.error("Missing C1 function-conditioned checkpoints: " + ", ".join(missing))

    # Pool raw counts and loss sums, then compute each percentage once. This
    # weights every atomic example equally when a function spans experiments.
    pooled: dict[tuple[str, str, str], Totals] = defaultdict(Totals)
    functions: set[str] = set()
    for experiment, path in checkpoints.items():
        model = _model_from_checkpoint(path, experiment, device)
        for split in SPLITS:
            dataset = _dataset(args.data_root, model, experiment, split)
            coverage, totals = evaluate_split(
                model, dataset, split, device, args.batch_size, args.num_workers,
            )
            for (kind, function), source in totals.items():
                if kind != "function":
                    continue
                if function not in TASK_VOCABULARY:
                    raise ValueError(f"Unknown atomic function: {function}")
                functions.add(function)
                pooled[("all", split, function)].merge(source)
                if args.include_experiments:
                    pooled[(str(experiment), split, function)].merge(source)
            print(f"C1 experiment {experiment} {split}: {coverage['atomic_rows']} atomic rows")
        del model

    function_columns = sorted(functions)
    fieldnames = ["experiment", "split", "metric", *function_columns]
    rows = []
    experiment_labels = ["all", *(str(i) for i in range(1, 6))] if args.include_experiments else ["all"]
    for experiment in experiment_labels:
        for split in SPLITS:
            for metric in METRICS:
                row = {"experiment": experiment, "split": split, "metric": metric}
                for function in function_columns:
                    totals = pooled.get((experiment, split, function))
                    if metric == "samples":
                        row[function] = totals.samples if totals else 0
                    elif totals:
                        value = totals.row(split, "function", function)[metric]
                        row[function] = "" if value is None else value
                    else:
                        row[function] = ""
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
