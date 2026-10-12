"""Fine-tune atomic rotation ViT on selected pairs, then test held-out pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightning.pytorch as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from benchmarks.cogitao_vit_cross_attention.rotation_compositions import (
    ATOMIC_PROGRAMS,
    HELD_OUT_PAIRS,
    SEEN_PAIRS,
    evaluate_programs,
)
from benchmarks.cogitao_vit_cross_attention.rotation_model import RotationTokenViT
from benchmarks.cogitao_vit_cross_attention.run_rotations import Progress, write_json
from src.train import (
    apply_override,
    build_dataloaders,
    build_named_evaluation_dataloaders,
    load_config,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/cogitao/rotation_compositions_vit.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/cogitao_rotation_compositions_vit/run_000"),
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
    seed_everything(config["seed"])
    model = RotationTokenViT(config)
    initialization = Path(config["initialization"]["checkpoint"]).resolve()
    checkpoint_data = torch.load(initialization, map_location="cpu", weights_only=False)
    if (
        checkpoint_data["hyper_parameters"]["config"]["model"]["config"]["angles"]
        != config["model"]["config"]["angles"]
    ):
        raise ValueError("The atomic checkpoint has a different angle vocabulary.")
    model.load_state_dict(checkpoint_data["state_dict"], strict=True)
    train, val = build_dataloaders(config)
    test = build_named_evaluation_dataloaders(config, "test_split_names")[0]
    datasets = {"train": train.dataset, "val": val.dataset, "test": test.dataset}
    if (
        train.dataset.programs != ATOMIC_PROGRAMS + SEEN_PAIRS
        or val.dataset.programs != train.dataset.programs
    ):
        raise ValueError("Checkpoint selection must use only atomic and seen programs.")
    if set(HELD_OUT_PAIRS) & set(train.dataset.programs):
        raise ValueError("Held-out pair leakage into training.")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(datasets[left].digests) & set(datasets[right].digests):
            raise ValueError(f"Input leakage between {left} and {right}.")
    manifests = {name: dataset.manifest() for name, dataset in datasets.items()}
    # Check the pretraining sources too: fine-tuning split isolation alone does
    # not prove that held-out images were absent from the initialization run.
    atomic_manifests = json.loads(
        (initialization.parent.parent / "data_manifest.json").read_text()
    )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(atomic_manifests[left]["input_sha256"]) & set(datasets[right].digests):
            raise ValueError(
                f"Atomic-pretraining source leakage from {left} into {right}."
            )
    if any(
        set(dataset.digests) - set(atomic_manifests[name]["input_sha256"])
        for name, dataset in datasets.items()
    ):
        raise ValueError(
            "Composition sources must come from the audited atomic source split."
        )
    summary = {name: manifest["audit"] for name, manifest in manifests.items()}
    split = {
        "seen_pairs": [list(p) for p in SEEN_PAIRS],
        "held_out_pairs": [list(p) for p in HELD_OUT_PAIRS],
    }
    print(json.dumps({"dataset_audit": summary, **split}, indent=2), flush=True)
    if args.dry_run:
        batch = next(iter(val))
        result = model(batch)
        print(f"Verified forward: {list(result['output_logits'].shape)}", flush=True)
        return
    output = args.output.resolve()
    if (output / "config.json").exists():
        if args.ckpt_path is None:
            raise FileExistsError(
                f"Run exists: {output}; choose a fresh output or resume."
            )
        if (
            json.loads((output / "config.json").read_text()) != config
            or json.loads((output / "data_manifest.json").read_text()) != manifests
        ):
            raise ValueError(
                "Resume configuration and data must match the saved run exactly."
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifests)
    write_json(output / "composition_split.json", split)
    write_json(
        output / "initialization.json",
        {
            "checkpoint": str(initialization),
            "sha256": hashlib.sha256(initialization.read_bytes()).hexdigest(),
            "scope": "atomic rotations only; fresh optimizer for composition fine-tuning",
        },
    )
    write_json(output / "status.json", {"status": "starting", "dataset_audit": summary})
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", **config["callbacks"]["checkpoint"]
    )
    callbacks = [
        checkpoint,
        Progress(output),
        EarlyStopping(
            monitor="val/id/cross_entropy_loss",
            patience=config["callbacks"]["early_stopping_patience"],
            mode="min",
        ),
    ]
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
    write_json(
        output / "status.json",
        {"status": "evaluating", "epochs_completed": trainer.current_epoch},
    )
    best = RotationTokenViT.load_from_checkpoint(
        checkpoint.best_model_path, map_location=model.device
    ).to(model.device)
    report = {
        "status": "complete",
        "scope": "atomic rotations plus selected ordered pairs",
        "angles": list(best.angles),
        **split,
        "rotation_convention": "object bounding-box CCW, fixed top-left anchor, before resize",
        "initialization_checkpoint": str(initialization),
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_cross_entropy": float(checkpoint.best_model_score),
        "epochs_completed": trainer.current_epoch,
        "global_step": trainer.global_step,
        "composition_training_examples": train.dataset.audit[
            "seen_composition_examples"
        ],
        "held_out_composition_training_examples": 0,
        "dataset_audit": summary,
        "checkpoint_selection": "atomic plus seen-pair validation cross entropy; held-out pairs excluded",
        "validation_seen": evaluate_programs(best, val),
        "test": evaluate_programs(best, test),
        "sequential_caveat": "Sequential paths operate on resized grids; oracle-intermediate scores quantify the intermediate resize shift.",
    }
    baseline = RotationTokenViT.load_from_checkpoint(
        str(initialization), map_location=model.device
    ).to(model.device)
    report["atomic_baseline_test"] = evaluate_programs(baseline, test)
    write_json(output / "report.json", report)
    write_json(
        output / "status.json",
        {"status": "complete", "report": str(output / "report.json")},
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
