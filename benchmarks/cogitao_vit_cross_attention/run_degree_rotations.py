"""Train a fresh ViT jointly on 15--90 degree atomic and selected paired rotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import (
    ATOMIC_ANGLES,
    ATOMIC_PROGRAMS,
    NET_ANGLES,
)
from benchmarks.cogitao_vit_cross_attention.degree_rotation_eval import evaluate_degrees
from benchmarks.cogitao_vit_cross_attention.degree_rotation_model import (
    DegreeRotationViT,
)
from benchmarks.cogitao_vit_cross_attention.run_rotations import write_json
from src.train import (
    apply_override,
    build_dataloaders,
    build_named_evaluation_dataloaders,
    load_config,
    seed_everything,
)


class DegreeProgress(L.Callback):
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()

    def save(self, trainer, epochs_completed):
        state = {
            "status": "training",
            "epochs_completed": epochs_completed,
            "current_epoch": trainer.current_epoch + 1,
            "global_step": trainer.global_step,
            "elapsed_seconds": time.monotonic() - self.started,
            "metrics": {
                k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items()
            },
        }
        write_json(self.output / "status.json", state)
        return state

    def on_train_start(self, trainer, module):
        self.save(trainer, trainer.current_epoch)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_index):
        if trainer.global_step % 50 == 0:
            self.save(trainer, trainer.current_epoch)

    def on_validation_end(self, trainer, module):
        if not trainer.sanity_checking:
            print(json.dumps(self.save(trainer, trainer.current_epoch + 1)), flush=True)

    def on_exception(self, trainer, module, exception):
        trainer.save_checkpoint(str(self.output / "checkpoints" / "interrupted.ckpt"))
        write_json(
            self.output / "status.json", {"status": "failed", "error": str(exception)}
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/cogitao/degree_rotation_object_anchors.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cogitao_degree_rotation_object_anchors/run_000"),
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--ckpt-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    if config.get("initialization"):
        raise ValueError(
            "This experiment must start fresh: pretrained initialization is forbidden."
        )
    if (
        tuple(config["model"]["config"]["angles"]) != NET_ANGLES
        or tuple(config["model"]["config"]["atomic_angles"]) != ATOMIC_ANGLES
    ):
        raise ValueError("Unexpected atomic/net angle vocabulary.")
    if config["data"]["validation_split_names"] != ["val"]:
        raise ValueError(
            "Only the seen-program validation split may select checkpoints."
        )
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(
        config["trainer"].get("float32_matmul_precision", "high")
    )
    seed_everything(config["seed"])
    model = DegreeRotationViT(config)
    # No checkpoint is loaded here. Optimizer and weights both start fresh.
    train, val = build_dataloaders(config)
    test = build_named_evaluation_dataloaders(config, "test_split_names")[0]
    if (
        train.dataset.programs != ATOMIC_PROGRAMS + train.dataset.seen_pairs
        or val.dataset.programs != train.dataset.programs
    ):
        raise ValueError(
            "Train/selection-validation must contain only atoms and seen pairs."
        )
    if set(train.dataset.held_out_pairs) & set(train.dataset.programs):
        raise ValueError("Held-out pair leakage into training.")
    datasets = {"train": train.dataset, "val": val.dataset, "test": test.dataset}
    if any(d.split_metadata != train.dataset.split_metadata for d in datasets.values()):
        raise ValueError("All source splits must share the same pair/effect protocol.")
    for name, dataset in datasets.items():
        if dataset.resolution != tuple(config["model"]["config"]["input_resolution"]):
            raise ValueError(f"Canvas/model resolution mismatch for {name}.")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(datasets[left].digests) & set(datasets[right].digests):
            raise ValueError(f"Source leakage between {left} and {right}.")
    manifests = {k: v.manifest() for k, v in datasets.items()}
    audit = {k: v["audit"] for k, v in manifests.items()}
    split = train.dataset.split_metadata
    print(
        json.dumps(
            {"fresh_initialization": True, "dataset_audit": audit, **split}, indent=2
        ),
        flush=True,
    )
    if args.dry_run:
        with torch.no_grad():
            result = model(next(iter(val)))
        print(
            f"Verified fresh-model forward: {list(result['output_logits'].shape)}",
            flush=True,
        )
        return
    output = args.output.resolve()
    if (output / "config.json").exists():
        if args.ckpt_path is None:
            raise FileExistsError(
                "Choose a fresh output directory; this run already exists."
            )
        if (
            json.loads((output / "config.json").read_text()) != config
            or json.loads((output / "data_manifest.json").read_text()) != manifests
        ):
            raise ValueError("Resume configuration and data must match the saved run.")
    elif args.ckpt_path is not None:
        raise ValueError(
            "Resume only an existing run with its matching saved configuration."
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifests)
    write_json(output / "composition_split.json", split)
    if args.ckpt_path is None:
        initial = output / "initial_weights.pt"
        torch.save(model.state_dict(), initial)
        write_json(
            output / "initialization.json",
            {
                "type": "random_from_scratch",
                "checkpoint": None,
                "seed": config["seed"],
                "initial_weights_sha256": hashlib.sha256(
                    initial.read_bytes()
                ).hexdigest(),
                "optimizer": "fresh AdamW",
            },
        )
    write_json(
        output / "status.json",
        {"status": "starting", "dataset_audit": audit, "fresh_initialization": True},
    )
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", **config["callbacks"]["checkpoint"]
    )
    trainer = L.Trainer(
        logger=CSVLogger(str(output), name="metrics", version=0),
        callbacks=[
            checkpoint,
            DegreeProgress(output),
            EarlyStopping(
                monitor="val/id/cross_entropy_loss",
                patience=config["callbacks"]["early_stopping_patience"],
                mode="min",
            ),
        ],
        **config["trainer"]["config"],
    )
    trainer.fit(
        model,
        train_dataloaders=train,
        val_dataloaders=val,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path else None,
    )
    write_json(
        output / "status.json",
        {"status": "evaluating", "epochs_completed": trainer.current_epoch},
    )
    # Lightning moves the fit model back to CPU after fit; explicitly choose the
    # evaluation device to avoid a long unintended CPU evaluation.
    device = torch.device(
        "cuda" if config["trainer"]["config"]["accelerator"] == "gpu" else "cpu"
    )
    best = DegreeRotationViT.load_from_checkpoint(
        checkpoint.best_model_path, map_location="cpu"
    ).to(device)
    report = {
        "status": "complete",
        "scope": "fresh joint atomic/selected-composition rotations",
        "fresh_initialization": True,
        "initialization_checkpoint": None,
        "atomic_angles": list(ATOMIC_ANGLES),
        "net_angles": list(NET_ANGLES),
        **split,
        "rotation_convention": manifests["train"]["rotation_convention"],
        "pair_target_convention": manifests["train"]["pair_target_convention"],
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_cross_entropy": float(checkpoint.best_model_score),
        "epochs_completed": trainer.current_epoch,
        "global_step": trainer.global_step,
        "dataset_audit": audit,
        "composition_training_examples": train.dataset.audit[
            "seen_composition_examples"
        ],
        "held_out_composition_training_examples": 0,
        "checkpoint_selection": "atomic plus seen-pair validation only",
        "summed_angle_diagnostic": "Single tokens for net angles above 90 are extrapolation beyond atomic training, reported separately.",
        "sequential_caveat": "Two nearest-neighbor samplings need not equal one sampling at the total angle. Raster-reference scores quantify this mismatch; invalid object states receive zero credit and their validity rate is reported. Object identities can also change when inferred again from an intermediate raster.",
        "validation_seen": evaluate_degrees(best, val),
        "test": evaluate_degrees(best, test),
    }
    write_json(output / "report.json", report)
    write_json(
        output / "status.json",
        {"status": "complete", "report": str(output / "report.json")},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
