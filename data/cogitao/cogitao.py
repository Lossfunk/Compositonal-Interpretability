from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only without dependency.
    raise ImportError(
        "COGITAODataset requires pyarrow. Install it with `pip install pyarrow`."
    ) from exc


VALID_SPLITS = {"train", "val", "val_ood", "test", "test_ood"}

TASK_VOCABULARY = (
    "change_shape_color",
    "crop_bottom_side",
    "crop_contours",
    "crop_top_side",
    "double_down",
    "double_right",
    "empty_inside_pixels",
    "extend_contours_same_color",
    "fill_holes_different_color",
    "fill_holes_same_color",
    "mirror_horizontal",
    "mirror_vertical",
    "pad_left",
    "pad_right",
    "pad_top",
    "rot90",
    "translate_right",
    "translate_up",
)
TASK_TO_ID = {name: index + 1 for index, name in enumerate(TASK_VOCABULARY)}
PAD_TASK_ID = 0
NUM_GRID_CLASSES = 10
MAX_COMPOSITION_DEPTH = 3

# Conventional ARC-AGI rendering for integer cell values 0--9. COGITAO uses
# zero as background and values 1--9 for object colors.
ARC_PALETTE = torch.tensor(
    [
        [0x00, 0x00, 0x00],
        [0x00, 0x74, 0xD9],
        [0xFF, 0x41, 0x36],
        [0x2E, 0xCC, 0x40],
        [0xFF, 0xDC, 0x00],
        [0xAA, 0xAA, 0xAA],
        [0xF0, 0x12, 0xBE],
        [0xFF, 0x85, 0x1B],
        [0x7F, 0xDB, 0xFF],
        [0x87, 0x0C, 0x25],
    ],
    dtype=torch.float32,
).div(255.0)


def _positive_pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        value = (value, value)
    result = tuple(int(item) for item in value)
    if len(result) != 2 or min(result) <= 0:
        raise ValueError(f"{name} must be a positive integer or [height, width].")
    return result


class COGITAODataset(Dataset):
    """Read one COGITAO CompGen parquet split as model-ready grid tensors.

    The task-conditioned setup uses one-hot ``images``, categorical
    ``target_grid`` labels, and ordered, padded ``task_tokens``.
    Variable-length transformation suites are joined into a scalar metadata
    string so PyTorch's default batch collation remains valid.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.root = Path(config.get("root", "data/cogitao/files/CompGen"))
        self.setting = int(config.get("setting", 1))
        self.experiment = int(config.get("experiment", 1))
        self.split = str(config.get("split", "train"))
        if not 1 <= self.setting <= 5:
            raise ValueError("setting must be between 1 and 5.")
        if not 1 <= self.experiment <= 5:
            raise ValueError("experiment must be between 1 and 5.")
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(VALID_SPLITS)}, got {self.split!r}."
            )

        self.image_size = _positive_pair(
            config.get("image_size", [20, 20]), "image_size"
        )
        self.image_field = str(config.get("image_field", "input"))
        if self.image_field not in {"input", "output"}:
            raise ValueError("image_field must be either 'input' or 'output'.")
        self.return_target = bool(config.get("return_target", True))
        self.return_task_tokens = bool(config.get("return_task_tokens", True))
        self.max_task_tokens = int(config.get("max_task_tokens", MAX_COMPOSITION_DEPTH))
        if self.max_task_tokens <= 0:
            raise ValueError("max_task_tokens must be positive.")
        self.representation = str(config.get("representation", "rgb"))
        if self.representation not in {"rgb", "one_hot"}:
            raise ValueError("representation must be either 'rgb' or 'one_hot'.")
        self.return_metadata = bool(config.get("return_metadata", True))

        self.path = (
            self.root
            / f"exp_setting_{self.setting}"
            / f"experiment_{self.experiment}"
            / f"{self.split}.parquet"
        )
        if not self.path.is_file():
            raise FileNotFoundError(
                f"COGITAO split not found: {self.path}. Run "
                "`bash scripts/setup_cogitao_compgen.sh`."
            )

        columns = {self.image_field}
        if self.return_target:
            columns.add("output")
        if self.return_task_tokens:
            columns.add("transformation_suite")
        if self.return_metadata:
            columns.update(("transformation_suite", "task_key"))
        self.table = pq.read_table(self.path, columns=sorted(columns), memory_map=True)
        self.num_samples = len(self.table)
        configured_samples = config.get("max_samples")
        if configured_samples is not None:
            self.num_samples = min(self.num_samples, int(configured_samples))
        if self.num_samples <= 0:
            raise ValueError("max_samples must leave at least one sample.")

    def __len__(self) -> int:
        return self.num_samples

    def _grid_to_indices(self, grid: Any) -> torch.Tensor:
        indices = torch.as_tensor(grid, dtype=torch.long)
        if indices.ndim != 2:
            raise ValueError(
                f"Expected a 2-D COGITAO grid in {self.path}, "
                f"got {tuple(indices.shape)}."
            )
        if indices.numel() and (indices.min() < 0 or indices.max() >= len(ARC_PALETTE)):
            raise ValueError(
                "COGITAO grid contains a color outside the supported range 0--9."
            )
        if tuple(indices.shape) != self.image_size:
            indices = F.interpolate(
                indices[None, None].float(), size=self.image_size, mode="nearest"
            )[0, 0].long()
        return indices

    def _grid_to_image(self, grid: Any) -> torch.Tensor:
        indices = self._grid_to_indices(grid)
        return ARC_PALETTE[indices].permute(2, 0, 1)

    def _grid_to_one_hot(self, grid: Any) -> torch.Tensor:
        return (
            F.one_hot(self._grid_to_indices(grid), num_classes=NUM_GRID_CLASSES)
            .permute(2, 0, 1)
            .float()
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        image_grid = self.table[self.image_field][index].as_py()
        encode = (
            self._grid_to_one_hot
            if self.representation == "one_hot"
            else self._grid_to_image
        )
        sample: dict[str, Any] = {"images": encode(image_grid)}
        if self.return_target:
            output_grid = self.table["output"][index].as_py()
            sample["target_images"] = encode(output_grid)
            sample["target_grid"] = self._grid_to_indices(output_grid)
        suite = None
        if self.return_task_tokens or self.return_metadata:
            suite = self.table["transformation_suite"][index].as_py()
        if self.return_task_tokens:
            if len(suite) > self.max_task_tokens:
                raise ValueError(
                    f"Transformation suite depth {len(suite)} exceeds "
                    f"max_task_tokens={self.max_task_tokens}."
                )
            unknown = sorted(set(suite) - set(TASK_TO_ID))
            if unknown:
                raise ValueError(f"Unknown COGITAO transformations: {unknown}")
            task_ids = [TASK_TO_ID[name] for name in suite]
            padding = self.max_task_tokens - len(task_ids)
            sample["task_tokens"] = torch.tensor(
                task_ids + [PAD_TASK_ID] * padding, dtype=torch.long
            )
            sample["task_token_mask"] = torch.tensor(
                [True] * len(task_ids) + [False] * padding
            )
        if self.return_metadata:
            sample["metadata"] = {
                "source_index": index,
                "task_key": str(self.table["task_key"][index].as_py()),
                "transformation_suite": " -> ".join(map(str, suite)),
                "composition_depth": len(suite),
                "setting": self.setting,
                "experiment": self.experiment,
                "split": self.split,
            }
        return sample


# Match the versioned class naming convention used by repository configs.
v0Dataset = COGITAODataset
