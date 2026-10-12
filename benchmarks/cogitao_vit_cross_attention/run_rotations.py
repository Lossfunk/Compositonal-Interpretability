"""Train only atomic rotation angles; reserve compositions for a later study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.rotation_model import RotationTokenViT
from src.train import (
    apply_override,
    build_dataloaders,
    build_named_evaluation_dataloaders,
    load_config,
    seed_everything,
)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Progress(L.Callback):
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = {
            name: float(value.detach().cpu())
            for name, value in trainer.callback_metrics.items()
        }
        state = {
            "status": "training",
            "epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "elapsed_seconds": time.monotonic() - self.started,
            "metrics": metrics,
        }
        write_json(self.output / "status.json", state)
        print(json.dumps(state), flush=True)

    def on_exception(self, trainer, module, exception):
        write_json(
            self.output / "status.json", {"status": "failed", "error": str(exception)}
        )
        trainer.save_checkpoint(str(self.output / "checkpoints" / "interrupted.ckpt"))


@torch.inference_mode()
def evaluate_angles(model, loader):
    model.eval()
    device = next(model.parameters()).device
    totals = {str(a): {"samples": 0} for a in model.angles}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(batch)
        for angle in model.angles:
            mask = batch["angle_degrees"] == angle
            count = int(mask.sum())
            if not count:
                continue
            selected = {
                "predictions": output["predictions"][mask],
                "target_grid": batch["target_grid"][mask],
            }
            metrics = {
                "cross_entropy_loss": F.nll_loss(
                    output["output_log_probs"][mask], selected["target_grid"]
                ),
                **model.compute_metrics(selected),
            }
            for baseline, prediction in (
                ("identity", batch["images"][mask].argmax(1)),
                ("background", torch.zeros_like(selected["target_grid"])),
            ):
                metrics.update(
                    {
                        f"{baseline}_{k}": v
                        for k, v in model.compute_metrics(
                            {
                                "predictions": prediction,
                                "target_grid": selected["target_grid"],
                            }
                        ).items()
                    }
                )
            row = totals[str(angle)]
            row["samples"] += count
            for name, value in metrics.items():
                row[name] = row.get(name, 0.0) + float(value) * count
    for row in totals.values():
        for name in list(row):
            if name != "samples":
                row[name] /= row["samples"]
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/cogitao/rotation_token_vit.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/cogitao_rotation_token_vit/run_000")
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(
        config["trainer"].get("float32_matmul_precision", "high")
    )
    seed_everything(config.get("seed", 42))
    # A single explicit vocabulary is shared by every dataset and the model.
    angles = config["model"]["config"]["angles"]
    for split in ("train", "val", "test"):
        if config["data"][split]["init_args"]["angles"] != angles:
            raise ValueError(f"Angle vocabulary mismatch for {split}.")
    model = RotationTokenViT(config)
    train, val = build_dataloaders(config)
    test = build_named_evaluation_dataloaders(config, "test_split_names")[0]
    datasets = {"train": train.dataset, "val": val.dataset, "test": test.dataset}
    manifests = {name: dataset.manifest() for name, dataset in datasets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(datasets[left].digests) & set(datasets[right].digests):
            raise ValueError(f"Input leakage between {left} and {right}.")
    summary = {name: manifest["audit"] for name, manifest in manifests.items()}
    print(
        json.dumps(
            {
                "dataset_audit": summary,
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        batch = next(iter(val))
        with torch.no_grad():
            result = model(batch)
        print(f"Verified forward: {list(result['output_logits'].shape)}", flush=True)
        return
    output = args.output.resolve()
    if (output / "config.json").exists():
        if args.ckpt_path is None:
            raise FileExistsError(
                f"Run exists: {output}; use a new --output or --ckpt-path to resume."
            )
        saved = json.loads((output / "config.json").read_text())
        if saved != config:
            raise ValueError(
                "Resume configuration must match the saved config exactly."
            )
        if json.loads((output / "data_manifest.json").read_text()) != manifests:
            raise ValueError("Resume data manifest must match exactly.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifests)
    write_json(output / "status.json", {"status": "starting", "dataset_audit": summary})
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", **config["callbacks"]["checkpoint"]
    )
    callbacks = [checkpoint, Progress(output)]
    patience = config["callbacks"].get("early_stopping_patience")
    if patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val/id/cross_entropy_loss", patience=int(patience), mode="min"
            )
        )
    trainer = L.Trainer(
        logger=CSVLogger(str(output), name="metrics", version=0),
        callbacks=callbacks,
        **config["trainer"]["config"],
    )
    trainer.fit(
        model,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path else None,
    )
    best = RotationTokenViT.load_from_checkpoint(
        checkpoint.best_model_path, map_location=model.device
    ).to(model.device)
    report = {
        "status": "complete",
        "scope": "atomic rotations only",
        "angles": angles,
        "rotation_convention": "object bounding-box CCW, fixed top-left anchor, before resize",
        "composition_training_examples": 0,
        "composition_evaluation": "deferred",
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_cross_entropy": float(checkpoint.best_model_score),
        "epochs_completed": trainer.current_epoch,
        "global_step": trainer.global_step,
        "dataset_audit": summary,
        "validation_by_angle": evaluate_angles(best, val),
        "test_by_angle": evaluate_angles(best, test),
    }
    write_json(output / "report.json", report)
    write_json(
        output / "status.json",
        {"status": "complete", "report": str(output / "report.json")},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
