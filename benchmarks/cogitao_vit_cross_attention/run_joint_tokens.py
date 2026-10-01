"""Train the independent joint image/function-token ViT on one COGITAO C1 experiment."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any

from src.train import main as train_main


BASE_CONFIG = Path("config/cogitao/joint_token_vit_c1.yaml")
SPLITS = ("train", "val", "val_ood", "test", "test_ood")


def _override(name: str, value: Any) -> list[str]:
    return ["--override", f"{name}={value!r}"]


def build_train_arguments(experiment: int, extra: list[str]) -> list[str]:
    slug = f"cogitao_c1_experiment_{experiment}_joint_token_vit"
    arguments = ["--config", str(BASE_CONFIG)]
    arguments += _override("experiment_name", slug)
    arguments += _override("logging.name", slug)
    arguments += _override(
        "logging.tags",
        ["cogitao", "compgen", "c1", f"experiment-{experiment}",
         "joint-function-image-tokens", "independent-vit"],
    )
    arguments += _override(
        "callbacks.checkpoint.dirpath",
        f"checkpoints/cogitao_joint_token_vit/{slug}",
    )
    for split in SPLITS:
        arguments += _override(f"data.{split}.init_args.experiment", experiment)
    return arguments + extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--print-command", action="store_true",
                        help="Print the expanded training command and exit.")
    args, extra = parser.parse_known_args()
    train_arguments = build_train_arguments(args.experiment, extra)
    if args.print_command:
        print(shlex.join([sys.executable, "src/train.py", *train_arguments]))
        return
    sys.argv = ["src/train.py", *train_arguments]
    train_main()


if __name__ == "__main__":
    main()
