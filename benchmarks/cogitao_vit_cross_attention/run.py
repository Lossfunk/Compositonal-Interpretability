from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

from src.train import main as train_main


BASE_CONFIG = Path("config/slot_attention/clevr_2d_slot_attention.yaml")
PROJECT = "cogitao-compgen-slot-attention"
SETTING = 1
GRID_SIZE = (20, 20)
VALIDATION_SPLITS = ("val", "val_ood")
TEST_SPLITS = ("test", "test_ood")


def _override(name: str, value: Any) -> list[str]:
    return ["--override", f"{name}={value!r}"]


def _dataset(split: str, experiment: int) -> dict[str, Any]:
    return {
        "class_path": "cogitao.cogitao.v0Dataset",
        "init_args": {
            "root": "data/cogitao/files/CompGen",
            "setting": SETTING,
            "experiment": experiment,
            "split": split,
            "image_size": list(GRID_SIZE),
            "image_field": "input",
            "return_target": True,
            "representation": "one_hot",
            "return_task_tokens": False,
            "return_metadata": True,
        },
    }


def build_train_arguments(experiment: int, extra: list[str]) -> list[str]:
    slug = f"cogitao_c1_experiment_{experiment}_direct_patch_decoder"
    arguments = ["--config", str(BASE_CONFIG)]
    arguments += _override("experiment_name", slug)
    arguments += _override(
        "model.class_path",
        "benchmarks.cogitao_vit_cross_attention.model.DirectPatchTransformationModel",
    )
    arguments += _override("model.config.input_resolution", list(GRID_SIZE))
    arguments += _override("model.config.patch_embedding.patch_size", 1)
    arguments += _override("model.config.patch_embedding.hidden_dim", 128)
    arguments += _override("model.config.decoder.num_layers", 6)
    arguments += _override("model.config.decoder.num_heads", 4)
    arguments += _override("model.config.decoder.mlp_dim", 512)
    arguments += _override("model.config.decoder.dropout", 0.0)
    arguments += _override(
        "model.config.validation_split_names", list(VALIDATION_SPLITS)
    )
    arguments += _override("model.config.test_split_names", list(TEST_SPLITS))
    arguments += _override("data.validation_split_names", list(VALIDATION_SPLITS))
    arguments += _override("data.test_split_names", list(TEST_SPLITS))
    arguments += _override("data.train", _dataset("train", experiment))
    for split in (*VALIDATION_SPLITS, *TEST_SPLITS):
        arguments += _override(f"data.{split}", _dataset(split, experiment))
    arguments += _override("logging.project", PROJECT)
    arguments += _override("logging.name", slug)
    arguments += _override("logging.group", "compgen-c1-direct-patch-decoder")
    arguments += _override("logging.training_loss_names", ["cross_entropy_loss"])
    arguments += _override(
        "logging.tags",
        [
            "cogitao",
            "compgen",
            "c1",
            f"experiment-{experiment}",
            "vit-6-layer",
            "direct-transformation",
            "patch-decoder",
            "no-function-conditioning",
        ],
    )
    arguments += _override(
        "callbacks.checkpoint.dirpath",
        f"checkpoints/cogitao_direct_patch/{slug}",
    )
    arguments += _override("callbacks.checkpoint.filename", "{epoch:03d}")
    arguments += _override("callbacks.checkpoint.monitor", "val/ood/cross_entropy_loss")
    arguments += _override("callbacks.checkpoint.mode", "min")
    return arguments + extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run COGITAO C1 with a direct patch-to-output decoder."
    )
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved src/train.py command and exit.",
    )
    args, extra = parser.parse_known_args()
    train_arguments = build_train_arguments(args.experiment, extra)
    if args.print_command:
        print(shlex.join([sys.executable, "src/train.py", *train_arguments]))
        return
    sys.argv = ["src/train.py", *train_arguments]
    train_main()


if __name__ == "__main__":
    main()
