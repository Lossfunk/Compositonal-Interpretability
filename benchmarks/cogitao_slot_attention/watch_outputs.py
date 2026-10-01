from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import default_collate

from benchmarks.cogitao_slot_attention.model import TaskConditionedSlotAttentionModel
from src.train import build_dataset

SPLITS = {
    "train": ("train", None),
    "val": ("val", "id"),
    "val_ood": ("val", "ood"),
    "test": ("test", "id"),
    "test_ood": ("test", "ood"),
}


def _process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_model(
    checkpoint_path: Path,
) -> tuple[TaskConditionedSlotAttentionModel, dict[str, Any], int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("hyper_parameters", {}).get("config")
    if config is None:
        raise KeyError(f"Checkpoint has no saved experiment config: {checkpoint_path}")
    model = TaskConditionedSlotAttentionModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config, int(checkpoint.get("epoch", -1))


@torch.inference_mode()
def update_outputs(checkpoint_path: Path) -> int:
    model, config, epoch = _load_model(checkpoint_path)
    model._prediction_samples = {}
    sample_count = max(model.max_logged_outputs, model.max_local_outputs)
    for split_name, (stage, domain) in SPLITS.items():
        dataset_config = config.get("data", {}).get(split_name)
        if dataset_config is None:
            continue
        dataset = build_dataset(dataset_config)
        batch = default_collate(
            [dataset[index] for index in range(min(sample_count, len(dataset)))]
        )
        output = model(batch)
        model._cache_prediction_outputs(stage, domain, batch, output)
    for stage in ("train", "val", "test"):
        model._write_local_prediction_outputs(stage, require_trainer=False)
    return epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update stable COGITAO qualitative images whenever last.ckpt changes."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(2)
    last_signature: tuple[int, int] | None = None
    while True:
        if args.checkpoint.is_file():
            stat = args.checkpoint.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                try:
                    epoch = update_outputs(args.checkpoint)
                except (EOFError, OSError, RuntimeError):
                    # The trainer may be replacing last.ckpt; retry next poll.
                    pass
                else:
                    last_signature = signature
                    print(f"Updated qualitative outputs from epoch {epoch}", flush=True)
        if args.once or not _process_alive(args.parent_pid):
            return
        time.sleep(max(1.0, args.poll_seconds))


if __name__ == "__main__":
    main()
