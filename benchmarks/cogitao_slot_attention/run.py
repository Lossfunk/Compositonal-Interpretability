from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

from src.train import main as train_main

BASE_CONFIG = Path("config/slot_attention/clevr_2d_slot_attention.yaml")
NATIVE_GRID_SIZE = (20, 20)
VALIDATION_SPLITS = ("val", "val_ood")
TEST_SPLITS = ("test", "test_ood")


def _override(name: str, value: Any) -> list[str]:
    return ["--override", f"{name}={value!r}"]


def _dataset(split: str, setting: int, experiment: int) -> dict[str, Any]:
    return {
        "class_path": "cogitao.cogitao.v0Dataset",
        "init_args": {
            "root": "data/cogitao/files/CompGen",
            "setting": setting,
            "experiment": experiment,
            "split": split,
            "image_size": list(NATIVE_GRID_SIZE),
            "image_field": "input",
            "return_target": True,
            "representation": "one_hot",
            "return_task_tokens": True,
            "max_task_tokens": 3,
            "return_metadata": True,
        },
    }


def build_train_arguments(setting: int, experiment: int, extra: list[str]) -> list[str]:
    slug = f"cogitao_c{setting}_experiment_{experiment}_slot_attention"
    arguments = ["--config", str(BASE_CONFIG)]
    arguments += _override("experiment_name", slug)
    arguments += _override(
        "model.class_path",
        "benchmarks.cogitao_slot_attention.model.TaskConditionedSlotAttentionModel",
    )
    arguments += _override("model.config.input_resolution", list(NATIVE_GRID_SIZE))
    arguments += _override("model.config.encoder.patch_size", 1)
    arguments += _override("model.config.encoder.hidden_dim", 128)
    arguments += _override("model.config.encoder.num_layers", 6)
    arguments += _override("model.config.encoder.num_heads", 4)
    arguments += _override("model.config.encoder.mlp_dim", 512)
    arguments += _override("model.config.encoder.bottleneck_hidden_dim", 128)
    # A 5 -> 10 -> 20 spatial broadcast path keeps decoder output native-sized.
    arguments += _override("model.config.decoder.channels", [64, 64])
    # Existing RGB-only figures cannot display ten-channel categorical grids.
    arguments += _override("model.config.eval_plots.enabled", [])
    arguments += _override(
        "model.config.validation_split_names", list(VALIDATION_SPLITS)
    )
    arguments += _override("model.config.test_split_names", list(TEST_SPLITS))
    arguments += _override("data.validation_split_names", list(VALIDATION_SPLITS))
    arguments += _override("data.test_split_names", list(TEST_SPLITS))
    arguments += _override("data.train", _dataset("train", setting, experiment))
    for split in (*VALIDATION_SPLITS, *TEST_SPLITS):
        arguments += _override(f"data.{split}", _dataset(split, setting, experiment))
    arguments += _override("logging.project", "cogitao-compgen-slot-attention")
    arguments += _override("logging.name", slug)
    arguments += _override("logging.group", f"compgen-c{setting}")
    arguments += _override(
        "logging.tags",
        [
            "cogitao",
            "compgen",
            f"c{setting}",
            f"experiment-{experiment}",
            "slot-attention",
            "task-conditioned-slot-attention",
        ],
    )
    arguments += _override(
        "callbacks.checkpoint.dirpath", f"checkpoints/slot_attention/{slug}"
    )
    arguments += _override("callbacks.checkpoint.filename", "{epoch:03d}")
    arguments += _override("callbacks.checkpoint.monitor", "val/ood/cross_entropy_loss")
    arguments += _override("callbacks.checkpoint.mode", "min")
    return arguments + extra


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing Slot Attention hyperparameters on one COGITAO "
            "CompGen setting/experiment."
        )
    )
    parser.add_argument("--setting", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved command and exit.",
    )
    args, extra = parser.parse_known_args()
    train_arguments = build_train_arguments(args.setting, args.experiment, extra)
    if args.print_command:
        print(shlex.join([sys.executable, "src/train.py", *train_arguments]))
        return
    sys.argv = ["src/train.py", *train_arguments]
    train_main()


if __name__ == "__main__":
    main()
