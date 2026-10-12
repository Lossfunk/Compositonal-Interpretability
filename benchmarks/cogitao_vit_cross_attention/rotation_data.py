"""Exact object rotations derived from audited atomic COGITAO examples."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle,
    apply,
    infer_objects,
)


def validate_angles(angles):
    angles = tuple(int(a) for a in angles)
    if not angles or any(a not in (0, 90, 180, 270) for a in angles):
        raise ValueError("Exact grid rotations support 0, 90, 180, 270 degrees.")
    if len(set(angles)) != len(angles):
        raise ValueError("Rotation angles must be unique.")
    return angles


def resize_grid(grid, resolution):
    value = torch.as_tensor(np.asarray(grid).copy(), dtype=torch.long)
    if tuple(value.shape) != tuple(resolution):
        value = F.interpolate(
            value[None, None].float(), size=resolution, mode="nearest"
        )[0, 0].long()
    return value


def grid_digest(grid):
    value = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def rotation_orbit(grid):
    """Preserve object identities and anchors while applying exact CCW rotations."""
    grid = np.asarray(grid, dtype=np.int64)
    objects = infer_objects(grid)
    orbit = [grid]
    for _ in range(4):
        objects, value = apply(objects, "rot90", grid.shape)
        orbit.append(value)
    if not np.array_equal(orbit[0], orbit[4]):
        raise InvalidOracle("rotation_cycle_does_not_close")
    return orbit[:4]


class RotationDataset(Dataset):
    """One angle per example. No composition rows enter the training dataset.

    Rotations act on each object's bounding-box patch at its fixed top-left
    anchor, as in COGITAO, before the usual nearest-neighbor grid resize.
    All four states must render without clipping or overlap. The recorded
    atomic rot90 target must match the oracle exactly.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.angles = validate_angles(config.get("angles", (0, 90, 180, 270)))
        self.split = config.get("split", "train")
        if self.split not in ("train", "val", "test"):
            raise ValueError("Rotation-only splits are train, val, test.")
        self.resolution = tuple(config.get("image_size", (20, 20)))
        self.root = (
            Path(config.get("root", "data/cogitao/files/CompGen"))
            / f"exp_setting_{int(config.get('setting', 1))}"
            / f"experiment_{int(config.get('experiment', 1))}"
        )
        # Exclude duplicates against ALL prior atomic rotation rows, even when
        # max_sources limits the training subset. Compare model-visible inputs.
        excluded = set()
        previous = {"train": (), "val": ("train",), "test": ("train", "val")}
        for split in previous[self.split]:
            for _, source, _ in self._rows(split):
                excluded.add(grid_digest(resize_grid(source, self.resolution).numpy()))
        self.sources, self.targets, self.source_indices, self.digests = [], [], [], []
        self.audit = {
            "atomic_rows": 0,
            "invalid_orbits": 0,
            "duplicate_inputs": 0,
            "target_mismatches": 0,
        }
        seen = set()
        for index, source, recorded in self._rows(self.split):
            self.audit["atomic_rows"] += 1
            try:
                orbit = rotation_orbit(source)
            except InvalidOracle:
                self.audit["invalid_orbits"] += 1
                continue
            if not np.array_equal(orbit[1], np.asarray(recorded)):
                self.audit["target_mismatches"] += 1
                continue
            visible = resize_grid(source, self.resolution)
            digest = grid_digest(visible.numpy())
            if digest in seen or digest in excluded:
                self.audit["duplicate_inputs"] += 1
                continue
            seen.add(digest)
            self.sources.append(visible.to(torch.uint8))
            self.targets.append(
                torch.stack(
                    [resize_grid(orbit[a // 90], self.resolution) for a in self.angles]
                ).to(torch.uint8)
            )
            self.source_indices.append(index)
            self.digests.append(digest)
        if self.audit["target_mismatches"]:
            raise ValueError(
                f"Rotation oracle disagrees with recorded targets: {self.audit}"
            )
        maximum = config.get("max_sources")
        if maximum is not None:
            if int(maximum) <= 0:
                raise ValueError("max_sources must be positive.")
            # Seeded sampling avoids selecting only the parquet prefix.
            order = np.random.default_rng(int(config.get("seed", 42))).permutation(
                len(self.sources)
            )[: int(maximum)]
            for name in ("sources", "targets", "source_indices", "digests"):
                setattr(self, name, [getattr(self, name)[i] for i in order])
        if not self.sources:
            raise ValueError(
                f"No valid rotation sources in {self.root}/{self.split}.parquet"
            )
        self.sources = torch.stack(self.sources)
        self.targets = torch.stack(self.targets)
        self.audit.update(kept_sources=len(self.sources), examples=len(self))

    def _rows(self, split):
        table = pq.read_table(
            self.root / f"{split}.parquet",
            columns=["input", "output", "transformation_suite"],
        )
        for i, suite in enumerate(table["transformation_suite"].to_pylist()):
            if suite == ["rot90"]:
                yield i, table["input"][i].as_py(), table["output"][i].as_py()

    def __len__(self):
        return len(self.sources) * len(self.angles)

    def __getitem__(self, index):
        source, angle = divmod(index, len(self.angles))
        grid = self.sources[source].long()
        return {
            "images": F.one_hot(grid, num_classes=10).permute(2, 0, 1).float(),
            "target_grid": self.targets[source, angle].long(),
            "task_tokens": torch.tensor([angle + 1, 0]),
            "task_token_mask": torch.tensor([True, False]),
            "angle_degrees": self.angles[angle],
            "source_index": self.source_indices[source],
        }

    def manifest(self):
        return {
            "split": self.split,
            "angles": list(self.angles),
            "audit": self.audit,
            "source_indices": self.source_indices,
            "input_sha256": self.digests,
        }
