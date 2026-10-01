from __future__ import annotations

import argparse
import ast
import importlib
import json
import random
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "data"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.checkpointing import TrainingProgressCheckpoint


def load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML configs require PyYAML: pip install pyyaml") from exc
            return yaml.safe_load(handle)
        if suffix == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {path}")


def parse_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "none" or lowered == "null":
            return None
        return value


def apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Overrides must be key=value, got: {override}")
    key, raw_value = override.split("=", 1)
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parse_value(raw_value)


def configure_run_identity(config: dict[str, Any]) -> None:
    logging_config = config.setdefault("logging", {})
    run_indexing = logging_config.get("run_indexing", True)
    if run_indexing is False:
        return

    index_config = run_indexing if isinstance(run_indexing, dict) else {}
    width = int(index_config.get("width", 3))
    base_name = index_config.get(
        "base_name",
        logging_config.get("name", config.get("experiment_name", "gaussian_ellipses")),
    )
    run_root = _run_index_root(config, base_name, index_config)
    run_index = index_config.get("index", logging_config.get("run_index"))
    if run_index is None:
        run_index = _next_run_index(run_root)
    run_index = int(run_index)
    run_id = f"run_{run_index:0{width}d}"

    logging_config.setdefault("group", base_name)
    logging_config["base_name"] = base_name
    logging_config["name"] = f"{base_name}_{run_id}"
    logging_config["run_index"] = run_index
    logging_config["run_id"] = run_id
    base_save_dir = Path(index_config.get("save_base_dir", logging_config.get("save_dir", "runs")))
    project = logging_config.get("project", "causal-dynamics")
    logging_config["base_save_dir"] = str(base_save_dir)
    logging_config["save_dir"] = str(base_save_dir / project / base_name / run_id)

    checkpoint_config = config.setdefault("callbacks", {}).get("checkpoint")
    if checkpoint_config and checkpoint_config.get("dirpath"):
        base_dir = Path(index_config.get("checkpoint_base_dir", checkpoint_config["dirpath"]))
        checkpoint_config["dirpath"] = str(base_dir / run_id)


def _run_index_root(
    config: dict[str, Any], base_name: str, index_config: dict[str, Any]
) -> Path:
    if "root" in index_config:
        return _resolve_path(index_config["root"])

    checkpoint_config = config.get("callbacks", {}).get("checkpoint")
    if checkpoint_config and checkpoint_config.get("dirpath"):
        return _resolve_path(index_config.get("checkpoint_base_dir", checkpoint_config["dirpath"]))

    logging_config = config.get("logging", {})
    save_dir = Path(logging_config.get("save_dir", "runs"))
    project = logging_config.get("project", "causal-dynamics")
    return _resolve_path(save_dir / project / base_name)


def _next_run_index(root: Path) -> int:
    if not root.exists():
        return 0
    run_pattern = re.compile(r"^run_(\d+)$")
    indices = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = run_pattern.match(child.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return REPO_ROOT / resolved


def import_from_path(class_path: str):
    module_name, _, attr = class_path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected dotted import path, got: {class_path}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    L.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_combination(combination: Any) -> tuple[str, str]:
    if isinstance(combination, str):
        color, _, shape = combination.strip().partition(" ")
        if color and shape:
            return color, shape
    elif isinstance(combination, dict):
        color = combination.get("color")
        shape = combination.get("shape")
        if color is not None and shape is not None:
            return str(color), str(shape)
    elif isinstance(combination, (list, tuple)) and len(combination) == 2:
        color, shape = combination
        return str(color), str(shape)
    raise ValueError(
        "Combinations must be 'color shape' strings, {'color': ..., 'shape': ...} "
        "dicts, or [color, shape] pairs."
    )


def _format_combination(combination: tuple[str, str]) -> str:
    color, shape = combination
    return f"{color} {shape}"


def _format_combinations(combinations: set[tuple[str, str]]) -> list[str]:
    return [_format_combination(pair) for pair in sorted(combinations)]


def _all_shape_color_combinations(init_args: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(color), str(shape))
        for color in init_args.get("colors", [])
        for shape in init_args.get("shapes", [])
    }


def _configured_combinations(init_args: dict[str, Any], key: str) -> set[tuple[str, str]]:
    return {_normalize_combination(combination) for combination in init_args.get(key, [])}


def _valid_shape_color_combinations(init_args: dict[str, Any]) -> set[tuple[str, str]]:
    all_combinations = _all_shape_color_combinations(init_args)
    included_config = init_args.get("included_combinations")
    if included_config is None:
        included = all_combinations
    else:
        included = {
            _normalize_combination(combination) for combination in included_config
        }
    excluded = _configured_combinations(init_args, "excluded_combinations")
    unknown = (included | excluded) - all_combinations
    if unknown:
        raise ValueError(
            "Unknown shape/color combinations: "
            + ", ".join(_format_combinations(unknown))
        )
    return included - excluded


def _set_combination_list(
    init_args: dict[str, Any], key: str, combinations: set[tuple[str, str]]
) -> None:
    init_args[key] = _format_combinations(combinations)


def _heldout_diagonal_combinations(
    train_args: dict[str, Any], offsets: Any, wrap: bool = False
) -> set[tuple[str, str]]:
    """Build color/shape matrix diagonals from their configured YAML order."""
    if offsets is None:
        return set()
    if isinstance(offsets, int):
        offsets = [offsets]
    if not isinstance(offsets, (list, tuple)) or not offsets:
        raise ValueError(
            "heldout_diagonal_offsets must be an integer or non-empty list."
        )
    colors = [str(color) for color in train_args.get("colors", [])]
    shapes = [str(shape) for shape in train_args.get("shapes", [])]
    if not colors or len(colors) != len(shapes):
        raise ValueError(
            "Held-out diagonals require equal, non-zero numbers of colors and "
            "shapes so every diagonal is one-to-one."
        )
    diagonal_offsets = []
    seen_offsets = set()
    for raw_offset in offsets:
        offset = int(raw_offset)
        identity = offset % len(shapes) if wrap else offset
        if identity in seen_offsets:
            raise ValueError(
                "heldout_diagonal_offsets contains duplicate diagonals."
            )
        seen_offsets.add(identity)
        diagonal_offsets.append(identity if wrap else offset)
    combinations = set()
    for offset in diagonal_offsets:
        for color_index, color in enumerate(colors):
            shape_index = color_index + offset
            if wrap:
                shape_index %= len(shapes)
            elif not 0 <= shape_index < len(shapes):
                continue
            combinations.add((color, shapes[shape_index]))
    return combinations


def _ensure_named_composition_split(data_config: dict[str, Any], split_name: str) -> None:
    if split_name in data_config:
        return
    if split_name != "val":
        raise ValueError(
            f"Composition split target '{split_name}' is not configured in data."
        )
    val_fraction = float(data_config.get("val_split", 0.0))
    if val_fraction <= 0.0:
        raise ValueError(
            "Composition splits need an explicit data.val section or data.val_split > 0."
        )

    data_config["val"] = deepcopy(data_config["train"])
    train_args = data_config["train"].setdefault("init_args", {})
    val_args = data_config["val"].setdefault("init_args", {})
    total_size = int(train_args["num_samples"])
    val_size = max(1, int(total_size * val_fraction))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError("val_split leaves no training samples.")
    train_args["num_samples"] = train_size
    val_args["num_samples"] = val_size
    data_config.pop("val_split", None)


def apply_composition_split(data_config: dict[str, Any]) -> None:
    split_config = data_config.get("composition_split") or data_config.get(
        "compositional_split"
    )
    if not split_config:
        return

    heldout_config = (
        split_config.get("heldout_combinations")
        or split_config.get("unseen_combinations")
        or split_config.get("test_combinations")
    )
    train_args = data_config["train"].setdefault("init_args", {})
    heldout = (
        {_normalize_combination(combination) for combination in heldout_config}
        if heldout_config
        else set()
    )
    heldout.update(
        _heldout_diagonal_combinations(
            train_args,
            split_config.get("heldout_diagonal_offsets"),
            wrap=bool(split_config.get("heldout_diagonal_wrap", False)),
        )
    )
    if not heldout:
        raise ValueError(
            "Composition split requires heldout_combinations, unseen_combinations, "
            "test_combinations, or heldout_diagonal_offsets."
        )

    all_train_combinations = _all_shape_color_combinations(train_args)
    unknown_heldout = heldout - all_train_combinations
    if unknown_heldout:
        raise ValueError(
            "Held-out combinations are not in the training concept grid: "
            + ", ".join(_format_combinations(unknown_heldout))
        )

    excluded = _configured_combinations(train_args, "excluded_combinations") | heldout
    simulated_train_args = deepcopy(train_args)
    _set_combination_list(simulated_train_args, "excluded_combinations", excluded)
    train_combinations = _valid_shape_color_combinations(simulated_train_args)
    train_shapes = {shape for _, shape in train_combinations}
    train_colors = {color for color, _ in train_combinations}
    missing_shapes = set(map(str, train_args.get("shapes", []))) - train_shapes
    missing_colors = set(map(str, train_args.get("colors", []))) - train_colors
    if missing_shapes or missing_colors:
        details = []
        if missing_shapes:
            details.append("shapes: " + ", ".join(sorted(missing_shapes)))
        if missing_colors:
            details.append("colors: " + ", ".join(sorted(missing_colors)))
        raise ValueError(
            "Held-out combinations remove concepts from training; every shape and "
            "color must appear in at least one training combination ("
            + "; ".join(details)
            + ")."
        )
    _set_combination_list(train_args, "excluded_combinations", excluded)

    target_splits = split_config.get("target_splits") or split_config.get(
        "test_splits", ["val"]
    )
    if isinstance(target_splits, str):
        target_splits = [target_splits]
    for split_name in target_splits:
        _ensure_named_composition_split(data_config, split_name)
        split_args = data_config[split_name].setdefault("init_args", {})
        _set_combination_list(split_args, "included_combinations", heldout)
        split_args["excluded_combinations"] = []

    iid_splits = split_config.get("iid_splits", [])
    if isinstance(iid_splits, str):
        iid_splits = [iid_splits]
    for split_name in iid_splits:
        _ensure_named_composition_split(data_config, split_name)
        split_args = data_config[split_name].setdefault("init_args", {})
        iid_excluded = (
            _configured_combinations(split_args, "excluded_combinations")
            | heldout
        )
        _set_combination_list(
            split_args, "excluded_combinations", iid_excluded
        )


def build_dataset(dataset_config: dict[str, Any]):
    class_path = dataset_config["class_path"]
    init_args = deepcopy(dataset_config.get("init_args", {}))
    dataset_cls = import_from_path(class_path)
    return dataset_cls(init_args)


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader | list[DataLoader] | None]:
    data_config = deepcopy(config["data"])
    apply_composition_split(data_config)
    loader_config = data_config.get("loader", {})
    train_dataset = build_dataset(data_config["train"])

    validation_split_names = data_config.get("validation_split_names")
    val_datasets = []
    if validation_split_names:
        for split_name in validation_split_names:
            if split_name not in data_config:
                raise KeyError(
                    f"Validation split {split_name!r} is not configured in data."
                )
            val_datasets.append(build_dataset(data_config[split_name]))
    elif "val" in data_config:
        val_datasets.append(build_dataset(data_config["val"]))
    elif data_config.get("val_split", 0.0) > 0:
        val_fraction = data_config["val_split"]
        val_size = max(1, int(len(train_dataset) * val_fraction))
        train_size = len(train_dataset) - val_size
        generator = torch.Generator().manual_seed(config.get("seed", 0))
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)
        val_datasets.append(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_config.get("batch_size", 32),
        shuffle=loader_config.get("shuffle", True),
        num_workers=loader_config.get("num_workers", 0),
        pin_memory=loader_config.get("pin_memory", False),
        drop_last=loader_config.get("drop_last", False),
        persistent_workers=loader_config.get("persistent_workers", False) and loader_config.get("num_workers", 0) > 0,
    )
    val_loaders = [
        DataLoader(
            val_dataset,
            batch_size=loader_config.get("val_batch_size", loader_config.get("batch_size", 32)),
            shuffle=False,
            num_workers=loader_config.get("num_workers", 0),
            pin_memory=loader_config.get("pin_memory", False),
            persistent_workers=loader_config.get("persistent_workers", False) and loader_config.get("num_workers", 0) > 0,
        )
        for val_dataset in val_datasets
    ]
    if not val_loaders:
        validation_loaders = None
    elif len(val_loaders) == 1:
        validation_loaders = val_loaders[0]
    else:
        validation_loaders = val_loaders
    return train_loader, validation_loaders


def build_named_evaluation_dataloaders(
    config: dict[str, Any], split_names_key: str
) -> list[DataLoader]:
    """Build explicitly named evaluation loaders without mixing them into val."""
    data_config = deepcopy(config["data"])
    split_names = data_config.get(split_names_key, [])
    if isinstance(split_names, str):
        split_names = [split_names]
    loader_config = data_config.get("loader", {})
    loaders = []
    for split_name in split_names:
        if split_name not in data_config:
            raise KeyError(
                f"Evaluation split {split_name!r} is not configured in data."
            )
        dataset = build_dataset(data_config[split_name])
        loaders.append(
            DataLoader(
                dataset,
                batch_size=loader_config.get(
                    "val_batch_size", loader_config.get("batch_size", 32)
                ),
                shuffle=False,
                num_workers=loader_config.get("num_workers", 0),
                pin_memory=loader_config.get("pin_memory", False),
                persistent_workers=loader_config.get("persistent_workers", False)
                and loader_config.get("num_workers", 0) > 0,
            )
        )
    return loaders


def build_logger(config: dict[str, Any]) -> WandbLogger:
    logging_config = config.get("logging", {})
    return WandbLogger(
        project=logging_config.get("project", "causal-dynamics"),
        name=logging_config.get("name", config.get("experiment_name", "gaussian_ellipses")),
        save_dir=logging_config.get("save_dir", "runs"),
        entity=logging_config.get("entity"),
        group=logging_config.get("group"),
        tags=logging_config.get("tags"),
        notes=logging_config.get("notes"),
        log_model=logging_config.get("log_model", False),
        offline=logging_config.get("offline", False),
        id=logging_config.get("id"),
        resume=logging_config.get("resume"),
        config=deepcopy(config),
    )


def build_callbacks(config: dict[str, Any]) -> list[Any]:
    callbacks_config = config.get("callbacks", {})
    checkpoint_config = callbacks_config.get("checkpoint")
    progress_config = deepcopy(callbacks_config.get("training_progress", {}))
    progress_enabled = bool(progress_config.pop("enabled", False))

    callbacks: list[Any] = []
    if checkpoint_config:
        best_checkpoint_config = deepcopy(checkpoint_config)
        if progress_enabled:
            # TrainingProgressCheckpoint owns last.ckpt and updates it every
            # epoch. Avoid letting the metric-based callback overwrite it.
            best_checkpoint_config["save_last"] = False
        callbacks.append(ModelCheckpoint(**best_checkpoint_config))

    if progress_enabled:
        dirpath = progress_config.pop("dirpath", None)
        if dirpath is None and checkpoint_config:
            dirpath = checkpoint_config.get("dirpath")
        if not dirpath:
            raise ValueError(
                "callbacks.training_progress needs dirpath, either directly or "
                "through callbacks.checkpoint.dirpath."
            )
        callbacks.append(TrainingProgressCheckpoint(dirpath=dirpath, **progress_config))
    return callbacks


def build_trainer(config: dict[str, Any]) -> L.Trainer:
    trainer_config = deepcopy(config.get("trainer", {}).get("config", {}))
    trainer_config.setdefault("max_epochs", 20)
    trainer_config.setdefault("accelerator", "auto")
    trainer_config.setdefault("devices", "auto")
    trainer_config.setdefault("log_every_n_steps", 10)
    return L.Trainer(
        **trainer_config,
        logger=build_logger(config),
        callbacks=build_callbacks(config),
    )


def initialize_model_weights(model: torch.nn.Module, config: dict[str, Any]) -> Path | None:
    initialization = config.get("initialization", {})
    checkpoint_value = initialization.get("checkpoint")
    if not checkpoint_value:
        return None
    checkpoint_path = _resolve_path(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Initialization checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    strict = bool(initialization.get("strict", True))
    include_prefixes = initialization.get("include_prefixes")
    if include_prefixes:
        if isinstance(include_prefixes, str):
            include_prefixes = [include_prefixes]
        include_prefixes = tuple(str(prefix) for prefix in include_prefixes)
        selected_state = {
            name: value
            for name, value in state_dict.items()
            if name.startswith(include_prefixes)
        }
        if not selected_state:
            raise KeyError(
                "No checkpoint tensors matched initialization.include_prefixes: "
                + ", ".join(include_prefixes)
            )
        model_state = model.state_dict()
        unknown = sorted(set(selected_state) - set(model_state))
        mismatched = sorted(
            name
            for name, value in selected_state.items()
            if name in model_state and value.shape != model_state[name].shape
        )
        if strict and (unknown or mismatched):
            details = []
            if unknown:
                details.append("unknown=" + ", ".join(unknown))
            if mismatched:
                details.append("shape_mismatch=" + ", ".join(mismatched))
            raise RuntimeError("Partial initialization failed: " + "; ".join(details))
        selected_state = {
            name: value
            for name, value in selected_state.items()
            if name in model_state and value.shape == model_state[name].shape
        }
        model.load_state_dict(selected_state, strict=False)
        print(
            "Initialized checkpoint prefixes "
            + ", ".join(include_prefixes)
            + f" ({len(selected_state)} tensors)"
        )
        print(f"Initialized model weights from: {checkpoint_path}")
        return checkpoint_path
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Missing initialization keys: {incompatible.missing_keys}")
        print(f"Unexpected initialization keys: {incompatible.unexpected_keys}")
    print(f"Initialized model weights from: {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gaussian ellipse compositional representation model.")
    parser.add_argument("--config", type=Path, default=Path("config/gaussian_ellipses/shapes_color.json"))
    parser.add_argument("--override", action="append", default=[], help="Override config values with dotted.path=value")
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help=(
            "Resume full Lightning state from this checkpoint, including the "
            "optimizer, loop counters, and global step."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Build model and dataloaders without calling trainer.fit")
    args = parser.parse_args()

    config = load_config(args.config)
    for override in args.override:
        apply_override(config, override)
    configure_run_identity(config)

    matmul_precision = config.get("trainer", {}).get(
        "float32_matmul_precision", "high"
    )
    torch.set_float32_matmul_precision(matmul_precision)

    seed_everything(config.get("seed"))
    model_cls = import_from_path(config.get("model", {}).get("class_path", "gaussian_box_model.HierarchicalGaussianBoxModel"))
    model = model_cls(config)
    initialization_checkpoint = initialize_model_weights(model, config)
    train_loader, val_loader = build_dataloaders(config)
    test_loaders = build_named_evaluation_dataloaders(config, "test_split_names")

    if args.dry_run:
        print(f"Built {model.__class__.__name__}")
        print(f"Run name: {config.get('logging', {}).get('name')}")
        print(f"W&B save dir: {config.get('logging', {}).get('save_dir')}")
        checkpoint_config = config.get("callbacks", {}).get("checkpoint")
        if checkpoint_config:
            print(f"Checkpoint dir: {checkpoint_config.get('dirpath')}")
        print(f"Train batches: {len(train_loader)}")
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        if initialization_checkpoint is not None:
            print(f"Initialization checkpoint: {initialization_checkpoint}")
        if isinstance(val_loader, list):
            split_names = config.get("data", {}).get(
                "validation_split_names", []
            )
            for index, loader in enumerate(val_loader):
                name = split_names[index] if index < len(split_names) else index
                print(f"Val batches ({name}): {len(loader)}")
        elif val_loader is not None:
            print(f"Val batches: {len(val_loader)}")
        test_split_names = config.get("data", {}).get("test_split_names", [])
        for index, loader in enumerate(test_loaders):
            name = test_split_names[index] if index < len(test_split_names) else index
            print(f"Test batches ({name}): {len(loader)}")
        return

    trainer = build_trainer(config)
    checkpoint_path = (
        str(_resolve_path(args.ckpt_path)) if args.ckpt_path is not None else None
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint_path,
    )
    if test_loaders:
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_model_path = getattr(checkpoint_callback, "best_model_path", "")
        trainer.test(
            model=model,
            dataloaders=test_loaders,
            ckpt_path="best" if best_model_path else None,
        )


if __name__ == "__main__":
    main()
