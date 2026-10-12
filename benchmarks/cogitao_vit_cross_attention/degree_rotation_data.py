"""Whole-canvas degree rotations with explicit sequence, pair, or effect splits."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from benchmarks.cogitao_vit_cross_attention.rotation_data import grid_digest

ATOMIC_ANGLES = (15, 30, 45, 60, 75, 90)
NET_ANGLES = tuple(range(15, 181, 15))
ALL_PAIRS = tuple(product(ATOMIC_ANGLES, repeat=2))
# All diagonals and one orientation of every off-diagonal pair. The circular
# tournament balances operand-position counts to within one occurrence.
SEEN_PAIRS = tuple(
    (a, b)
    for a, b in ALL_PAIRS
    if a == b
    or ((ATOMIC_ANGLES.index(b) - ATOMIC_ANGLES.index(a)) % 6) in (1, 2)
    or (
        ((ATOMIC_ANGLES.index(b) - ATOMIC_ANGLES.index(a)) % 6) == 3
        and ATOMIC_ANGLES.index(a) % 2 == 0
    )
)
HELD_OUT_PAIRS = tuple(p for p in ALL_PAIRS if p not in SEEN_PAIRS)
ATOMIC_PROGRAMS = tuple((a,) for a in ATOMIC_ANGLES)


UNORDERED_HOLDOUT = ((15, 45), (15, 75), (30, 60), (30, 90), (45, 75), (60, 90))
HELD_OUT_EFFECTS = (105, 135, 165)


def pair_split(protocol="order"):
    """Return pair sets and the equivalence class actually excluded from training.

    Legacy order-only mode is retained to reproduce the original experiment.
    New protocols group either (a,b)~(b,a), or all factorizations with the same
    total angle. Atomic tasks always remain the requested six angles.
    """
    if protocol == "order":
        seen, held = SEEN_PAIRS, HELD_OUT_PAIRS
        unit = "ordered sequence only; reversed pair may be trained"
    elif protocol == "unordered":
        held = tuple(p for p in ALL_PAIRS if tuple(sorted(p)) in UNORDERED_HOLDOUT)
        seen = tuple(p for p in ALL_PAIRS if p not in held)
        unit = "unordered angle pair, both orders excluded"
    elif protocol == "effect":
        held = tuple(p for p in ALL_PAIRS if sum(p) in HELD_OUT_EFFECTS)
        seen = tuple(p for p in ALL_PAIRS if p not in held)
        unit = "total rotation angle, all factorizations and both orders excluded"
    else:
        raise ValueError("pair_split must be 'order', 'unordered', or 'effect'.")
    observed = set(ATOMIC_ANGLES) | {sum(p) for p in seen}
    held_effects = {sum(p) for p in held}
    metadata = {
        "pair_split": protocol,
        "holdout_unit": unit,
        "commutativity_safe": protocol != "order",
        "training_effects": sorted(observed),
        "held_out_pair_effects": sorted(held_effects),
        "unseen_effects": sorted(held_effects - observed),
        "seen_pairs": [list(p) for p in seen],
        "held_out_pairs": [list(p) for p in held],
    }
    if protocol != "order" and {tuple(sorted(p)) for p in seen} & {
        tuple(sorted(p)) for p in held
    }:
        raise ValueError("Commutatively equivalent pairs cross the split.")
    if protocol == "effect" and held_effects & observed:
        raise ValueError("A held-out total angle appears in training.")
    return seen, held, metadata


def pad_grid(grid, resolution):
    value = torch.as_tensor(grid, dtype=torch.long)
    if value.ndim != 2 or value.numel() == 0 or value.min() < 0 or value.max() > 9:
        raise ValueError(
            "Expected a nonempty categorical grid with values 0 through 9."
        )
    h, w = value.shape
    height, width = resolution
    if height != width or math.hypot(h, w) + 1 > height:
        raise ValueError(
            "Canvas must be square and fit the input diagonal with a margin."
        )
    top, left = (height - h) // 2, (width - w) // 2
    return F.pad(value, (left, width - w - left, top, height - h - top))


def rotate_grids(grids, degrees):
    """Counterclockwise fixed-center rotation, nearest-neighbor labels, zero fill.

    Always render canonical targets from the original padded input. Sequential
    raster rotation is a separate diagnostic because interpolation is not an
    exact representation of the continuous rotation group.
    """
    if grids.ndim != 3 or grids.shape[1] != grids.shape[2]:
        raise ValueError("rotate_grids expects [batch, square_height, square_width].")
    angles = torch.as_tensor(degrees, dtype=torch.float32, device=grids.device)
    if angles.ndim == 0:
        angles = angles.expand(len(grids))
    if angles.shape != (len(grids),):
        raise ValueError("Expected one angle per grid or a scalar angle.")
    radians = angles * (math.pi / 180)
    cosine, sine = radians.cos(), radians.sin()
    theta = torch.zeros(len(grids), 2, 3, device=grids.device)
    theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
    theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
    images = grids[:, None].float()
    sampling = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(
        images, sampling, mode="nearest", padding_mode="zeros", align_corners=False
    )[:, 0].long()


class DegreeRotationDataset(Dataset):
    def __init__(self, config):
        self.config = dict(config)
        self.split = config.get("split", "train")
        if self.split not in ("train", "val", "test"):
            raise ValueError("Expected train, val, or test source split.")
        selection = config.get("program_selection", "seen")
        if selection not in ("seen", "all") or (
            self.split == "train" and selection != "seen"
        ):
            raise ValueError(
                "Training must use seen programs; evaluation may use seen or all."
            )
        self.seen_pairs, self.held_out_pairs, self.split_metadata = pair_split(
            config.get("pair_split", "order")
        )
        self.programs = ATOMIC_PROGRAMS + (
            self.seen_pairs if selection == "seen" else ALL_PAIRS
        )
        self.resolution = tuple(config.get("image_size", (24, 24)))
        self.root = (
            Path(config.get("root", "data/cogitao/files/CompGen"))
            / f"exp_setting_{int(config.get('setting', 1))}"
            / f"experiment_{int(config.get('experiment', 1))}"
        )
        prior = {"train": (), "val": ("train",), "test": ("train", "val")}[self.split]
        excluded = set()
        for split in prior:
            for _, raw in self.rows(split):
                excluded.add(grid_digest(pad_grid(raw, self.resolution).numpy()))
        grids, indices, digests, seen = [], [], [], set()
        self.audit = {"source_atomic_rot90_rows": 0, "duplicate_inputs": 0}
        for index, raw in self.rows(self.split):
            self.audit["source_atomic_rot90_rows"] += 1
            grid = pad_grid(raw, self.resolution)
            digest = grid_digest(grid.numpy())
            if digest in excluded or digest in seen:
                self.audit["duplicate_inputs"] += 1
                continue
            seen.add(digest)
            grids.append(grid.to(torch.uint8))
            indices.append(index)
            digests.append(digest)
        maximum = config.get("max_sources")
        if maximum is not None:
            if int(maximum) <= 0:
                raise ValueError("max_sources must be positive.")
            order = np.random.default_rng(config.get("seed", 42)).permutation(
                len(grids)
            )[: int(maximum)]
            grids, indices, digests = (
                [items[i] for i in order] for items in (grids, indices, digests)
            )
        if not grids:
            raise ValueError("No isolated rotation sources remain.")
        self.sources = torch.stack(grids)
        self.source_indices, self.digests = indices, digests
        self.targets = torch.empty(
            len(grids), len(NET_ANGLES), *self.resolution, dtype=torch.uint8
        )
        for start in range(0, len(grids), 512):
            source = self.sources[start : start + 512]
            for i, angle in enumerate(NET_ANGLES):
                self.targets[start : start + 512, i] = rotate_grids(source, angle).to(
                    torch.uint8
                )
        # Targets of pairs are never produced by repeated raster sampling.
        self.audit.update(
            kept_sources=len(grids),
            examples=len(self),
            atomic_examples=len(grids) * len(ATOMIC_PROGRAMS),
            seen_composition_examples=len(grids) * len(self.seen_pairs),
            held_out_composition_examples=len(grids) * len(self.held_out_pairs)
            if selection == "all"
            else 0,
        )

    def rows(self, split):
        table = pq.read_table(
            self.root / f"{split}.parquet", columns=["input", "transformation_suite"]
        )
        for index, suite in enumerate(table["transformation_suite"].to_pylist()):
            if suite == ["rot90"]:
                yield index, table["input"][index].as_py()

    def __len__(self):
        return len(self.sources) * len(self.programs)

    def __getitem__(self, index):
        source, program_index = divmod(index, len(self.programs))
        program = self.programs[program_index]
        total = sum(program)
        ids = [NET_ANGLES.index(a) + 1 for a in program]
        return {
            "images": F.one_hot(self.sources[source].long(), 10)
            .permute(2, 0, 1)
            .float(),
            "target_grid": self.targets[source, NET_ANGLES.index(total)].long(),
            "intermediate_grid": self.targets[
                source, NET_ANGLES.index(program[0])
            ].long(),
            "task_tokens": torch.tensor(ids + [0] * (2 - len(ids))),
            "task_token_mask": torch.tensor(
                [True] * len(ids) + [False] * (2 - len(ids))
            ),
            "program_index": program_index,
            "angle_degrees": total,
            "source_index": self.source_indices[source],
        }

    def manifest(self):
        return {
            "split": self.split,
            "atomic_angles": list(ATOMIC_ANGLES),
            "net_angles": list(NET_ANGLES),
            "resolution": list(self.resolution),
            "program_selection": self.config.get("program_selection", "seen"),
            "programs": [list(p) for p in self.programs],
            "seen_pairs": [list(p) for p in self.seen_pairs],
            "held_out_pairs": [list(p) for p in self.held_out_pairs],
            **(
                {"split_metadata": self.split_metadata}
                if self.config.get("pair_split", "order") != "order"
                else {}
            ),
            "source_indices": self.source_indices,
            "input_sha256": self.digests,
            "rotation_convention": "whole canvas, fixed center, CCW, nearest-neighbor, zero padding",
            "pair_target_convention": "rotate original grid once by summed angle",
            "audit": self.audit,
        }
