"""Optional W&B metric relay for an existing rotation run; invoke after approval."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from pathlib import Path

from benchmarks.cogitao_vit_cross_attention.run_rotations import write_json


def complete_metric_rows(path):
    """Merge Lightning's train/validation rows before committing each step."""
    grouped = {}
    if not path.exists():
        return grouped
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                step = int(row["step"])
                values = {k: float(v) for k, v in row.items() if v and k != "step"}
            except (ValueError, TypeError, KeyError):
                continue
            grouped.setdefault(step, {}).update(values)
    return {
        step: values
        for step, values in grouped.items()
        if "train/cross_entropy_loss" in values
        and "val/id/cross_entropy_loss" in values
    }


def csv_metric_rows(path):
    """Read complete CSV lines, preserving later metrics at the same global step."""
    if not path.exists():
        return []
    value = path.read_text()
    if not value.endswith("\n"):
        value = value.rsplit("\n", 1)[0] + "\n" if "\n" in value else ""
    rows = []
    for index, row in enumerate(csv.DictReader(io.StringIO(value))):
        if None in row or any(v is None for v in row.values()):
            continue
        try:
            metrics = {key: float(v) for key, v in row.items() if v and key != "step"}
            step = int(row["step"])
        except (ValueError, TypeError, KeyError):
            continue
        if any(key != "epoch" for key in metrics):
            rows.append((index, step, metrics))
    return rows


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="cogitao-compgen-slot-attention")
    parser.add_argument("--name")
    parser.add_argument("--group", default="compgen-c1-rotations")
    parser.add_argument("--tags", default="rotations-only,atomic-baseline")
    parser.add_argument("--stream-steps", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    metadata_path = output / "wandb.json"
    old = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    run = wandb.init(
        project=args.project,
        id=old.get("id"),
        resume="allow" if old else None,
        name=args.name or f"cogitao_rotation_token_vit_{output.name}",
        group=args.group,
        tags=args.tags.split(","),
        dir=str(output),
        save_code=False,
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            disable_job_creation=True,
            x_disable_stats=True,
        ),
    )
    run.define_metric("trainer/global_step")
    run.define_metric("*", step_metric="trainer/global_step")
    state = {
        "id": run.id,
        "url": run.url,
        "project": run.project,
        "entity": run.entity,
        "status": "logging",
        "last_logged_step": old.get("last_logged_step", -1),
        "last_logged_row": old.get("last_logged_row", -1),
        "stream_steps": args.stream_steps,
    }
    write_json(metadata_path, state)
    print(json.dumps(state), flush=True)
    csv_path = output / "metrics" / "version_0" / "metrics.csv"
    while True:
        if args.stream_steps:
            for index, step, values in csv_metric_rows(csv_path):
                if index <= state["last_logged_row"]:
                    continue
                # W&B's internal step advances per CSV row. The explicit custom
                # training-step axis permits epoch metrics after step metrics.
                run.log({**values, "trainer/global_step": step + 1})
                state["last_logged_row"] = index
                state["last_logged_step"] = step
                write_json(metadata_path, state)
        else:
            for step, values in sorted(complete_metric_rows(csv_path).items()):
                if step <= state["last_logged_step"]:
                    continue
                run.log({**values, "trainer/global_step": step + 1}, step=step)
                state["last_logged_step"] = step
                write_json(metadata_path, state)
                print(f"Logged training step {step + 1}", flush=True)
        training = json.loads((output / "status.json").read_text())
        if training["status"] in ("complete", "failed", "superseded"):
            if training["status"] == "complete":
                report = json.loads((output / "report.json").read_text())
                run.summary["epochs_completed"] = report["epochs_completed"]
                run.summary["best_val_cross_entropy"] = report["best_val_cross_entropy"]
                for split in ("validation", "test"):
                    for angle, metrics in report.get(f"{split}_by_angle", {}).items():
                        for key, value in metrics.items():
                            run.summary[f"{split}/angle_{angle}/{key}"] = value
                for split in ("validation_seen", "test", "atomic_baseline_test"):
                    for kind in ("groups", "by_program"):
                        for label, metrics in (
                            report.get(split, {}).get(kind, {}).items()
                        ):
                            for key, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    run.summary[f"{split}/{kind}/{label}/{key}"] = value
            state["status"] = training["status"]
            write_json(metadata_path, state)
            run.finish(exit_code=0 if training["status"] == "complete" else 1)
            return
        try:
            os.kill(int((output / "pid.txt").read_text()), 0)
        except ProcessLookupError:
            state["status"] = "training_process_exited"
            write_json(metadata_path, state)
            run.finish(exit_code=1)
            return
        time.sleep(10)


if __name__ == "__main__":
    main()
