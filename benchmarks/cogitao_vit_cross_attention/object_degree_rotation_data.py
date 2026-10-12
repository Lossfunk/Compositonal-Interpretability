"""COGITAO object-patch rotations, keeping each original top-left anchor.

Objects are inferred once. Canonical pair targets rotate the original object
patches once by the sum, avoiding accumulated raster resampling. The separate
two-step raster control retains identities and reports unrenderable states.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy import ndimage

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle,
    from_patch,
    infer_objects,
    render,
)
from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import (
    ALL_PAIRS,
    ATOMIC_PROGRAMS,
    NET_ANGLES,
    DegreeRotationDataset,
    pad_grid,
    pair_split,
)
from benchmarks.cogitao_vit_cross_attention.rotation_data import grid_digest


def rotate_object(obj, angle):
    angle = int(angle) % 360
    patch = obj.patch()
    if angle % 90 == 0:
        changed = np.rot90(patch, angle // 90)
    else:
        changed = ndimage.rotate(
            patch, angle, reshape=True, order=0, mode="grid-constant",
            cval=0, prefilter=False,
        )
    return from_patch(changed, obj.anchor, move_to_anchor=True)


def rotate_objects(objects, angle, shape):
    changed = tuple(rotate_object(obj, angle) for obj in objects)
    return changed, render(changed, shape)


def identity_preserved(objects, grid):
    """Whether re-inferring eight-connected components recovers these identities."""
    expected = {frozenset(obj.points) for obj in objects}
    actual = {frozenset(obj.points) for obj in infer_objects(grid)}
    return expected == actual


class ObjectAnchorDegreeRotationDataset(DegreeRotationDataset):
    rotation_convention = (
        "each object's bounding-box patch, CCW, expanded nearest-neighbor "
        "sampling (scipy order=0, grid-constant zero fill), then foreground "
        "top-left re-anchored; multiples of 90 use exact numpy.rot90"
    )
    pair_target_convention = (
        "infer objects once from the original source, rotate each original "
        "patch once by the total angle, and re-anchor at its original top-left"
    )

    def __init__(self, config):
        self.config = dict(config)
        self.split = config.get("split", "train")
        if self.split not in ("train", "val", "test"):
            raise ValueError("Expected train, val, or test source split.")
        selection = config.get("program_selection", "seen")
        if selection not in ("seen", "all") or (
            self.split == "train" and selection != "seen"
        ):
            raise ValueError("Training must use seen programs; evaluation may use seen or all.")
        self.seen_pairs, self.held_out_pairs, self.split_metadata = pair_split(
            config.get("pair_split", "unordered")
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
        excluded = set()
        for split in {"train": (), "val": ("train",), "test": ("train", "val")}[self.split]:
            for _, raw, _ in self.object_rows(split):
                excluded.add(grid_digest(pad_grid(raw, self.resolution).numpy()))
        self.audit = {
            "source_atomic_rot90_rows": 0, "duplicate_inputs": 0,
            "target_mismatches": 0, "recorded_rot90_verified": 0,
            "invalid_sources": 0,
        }
        failures = Counter()
        sources, targets, indices, digests, original_objects = [], [], [], [], []
        seen = set()
        for index, raw, recorded in self.object_rows(self.split):
            self.audit["source_atomic_rot90_rows"] += 1
            raw = np.asarray(raw, dtype=np.int64)
            try:
                raw_objects = infer_objects(raw)
                _, ninety = rotate_objects(raw_objects, 90, raw.shape)
            except InvalidOracle as error:
                self.audit["invalid_sources"] += 1
                failures["recorded_rot90:" + str(error)] += 1
                continue
            if not np.array_equal(ninety, np.asarray(recorded)):
                self.audit["target_mismatches"] += 1
                continue
            self.audit["recorded_rot90_verified"] += 1
            source = pad_grid(raw, self.resolution)
            digest = grid_digest(source.numpy())
            if digest in excluded or digest in seen:
                self.audit["duplicate_inputs"] += 1
                continue
            objects = infer_objects(source.numpy())
            try:
                orbit = [rotate_objects(objects, a, self.resolution)[1] for a in NET_ANGLES]
            except InvalidOracle as error:
                self.audit["invalid_sources"] += 1
                failures["canonical_target:" + str(error)] += 1
                continue
            seen.add(digest)
            sources.append(source.to(torch.uint8))
            targets.append(torch.from_numpy(np.stack(orbit).astype(np.uint8)))
            indices.append(index)
            digests.append(digest)
            original_objects.append(objects)
        if self.audit["target_mismatches"]:
            raise ValueError(f"Object rotation disagrees with recorded rot90 targets: {self.audit}")
        maximum = config.get("max_sources")
        if maximum is not None:
            if int(maximum) <= 0:
                raise ValueError("max_sources must be positive.")
            order = np.random.default_rng(config.get("seed", 42)).permutation(len(sources))[:int(maximum)]
            sources, targets, indices, digests, original_objects = (
                [items[i] for i in order]
                for items in (sources, targets, indices, digests, original_objects)
            )
        if not sources:
            raise ValueError(f"No valid object rotation sources remain: {self.audit}")
        self.sources, self.targets = torch.stack(sources), torch.stack(targets)
        self.source_indices, self.digests = indices, digests
        self.audit.update(
            invalid_reasons=dict(failures), kept_sources=len(sources), examples=len(self),
            atomic_examples=len(sources) * len(ATOMIC_PROGRAMS),
            seen_composition_examples=len(sources) * len(self.seen_pairs),
            held_out_composition_examples=len(sources) * len(self.held_out_pairs) if selection == "all" else 0,
        )
        if self.split != "train":
            self.raster_references = torch.zeros(
                len(sources), len(self.programs), *self.resolution, dtype=torch.uint8
            )
            self.raster_valid = torch.ones(len(sources), len(self.programs), dtype=torch.bool)
            self.identities_preserved = torch.ones_like(self.raster_valid)
            raster_failures = Counter()
            for i, objects in enumerate(original_objects):
                first_states = {
                    a: rotate_objects(objects, a, self.resolution)
                    for a, in ATOMIC_PROGRAMS
                }
                for j, program in enumerate(self.programs):
                    changed, grid = first_states[program[0]]
                    self.identities_preserved[i, j] = identity_preserved(changed, grid)
                    try:
                        if len(program) == 2:
                            _, grid = rotate_objects(changed, program[1], self.resolution)
                        self.raster_references[i, j] = torch.from_numpy(grid.astype(np.uint8))
                    except InvalidOracle as error:
                        self.raster_valid[i, j] = False
                        raster_failures[str(error)] += 1
            self.audit["sequential_raster_invalid_examples"] = int((~self.raster_valid).sum())
            self.audit["sequential_raster_invalid_reasons"] = dict(raster_failures)
            self.audit["intermediate_identity_changed_examples"] = int((~self.identities_preserved).sum())

    def object_rows(self, split):
        table = pq.read_table(
            self.root / f"{split}.parquet", columns=["input", "output", "transformation_suite"]
        )
        for i, suite in enumerate(table["transformation_suite"].to_pylist()):
            if suite == ["rot90"]:
                yield i, table["input"][i].as_py(), table["output"][i].as_py()

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.split != "train":
            source, program = divmod(index, len(self.programs))
            sample.update(
                raster_reference_grid=self.raster_references[source, program].long(),
                raster_reference_valid=self.raster_valid[source, program],
                intermediate_identity_preserved=self.identities_preserved[source, program],
            )
        return sample

    def manifest(self):
        result = super().manifest()
        result.update(
            rotation_convention=self.rotation_convention,
            pair_target_convention=self.pair_target_convention,
            split_metadata=self.split_metadata,
            recorded_rot90_target_check="exact on native grid before padding",
            object_identity_convention="infer eight-connected components once; retain identities in oracle states",
            invalid_source_policy="exclude whole source if any canonical angle 15 through 180 is empty, clipped, or overlapping",
            raster_reference_convention="two object-patch samplings with persistent identities; invalid states score zero and validity is reported",
        )
        return result
