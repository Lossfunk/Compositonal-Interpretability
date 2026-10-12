import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle, apply, infer_objects,
)
from benchmarks.cogitao_vit_cross_attention.degree_rotation_eval import evaluate_degrees
from benchmarks.cogitao_vit_cross_attention.degree_rotation_model import DegreeRotationViT
from benchmarks.cogitao_vit_cross_attention.object_degree_rotation_data import (
    ObjectAnchorDegreeRotationDataset, rotate_objects,
)


def test_cardinal_rotations_match_original_oracle_and_keep_separate_anchors():
    grid = np.zeros((15, 15), dtype=np.int64)
    grid[1:4, 1:5] = [[1, 2, 3, 0], [4, 0, 0, 0], [5, 6, 7, 8]]
    grid[9:11, 10:13] = [[9, 9, 0], [9, 9, 9]]
    objects = infer_objects(grid)
    expected_objects = objects
    for angle in (90, 180, 270, 360):
        expected_objects, expected = apply(expected_objects, "rot90", grid.shape)
        changed, actual = rotate_objects(objects, angle, grid.shape)
        np.testing.assert_array_equal(actual, expected)
        assert [obj.anchor for obj in changed] == [obj.anchor for obj in objects]
    for angle in (15, 30, 45, 60, 75, 105, 120, 135, 150, 165):
        changed, actual = rotate_objects(objects, angle, grid.shape)
        assert [obj.anchor for obj in changed] == [obj.anchor for obj in objects]
        assert set(np.unique(actual)) <= set(np.unique(grid))


def test_expanded_object_rotation_rejects_overlap_instead_of_overwriting():
    grid = np.zeros((10, 10), dtype=np.int64)
    grid[1:4, 1] = 1
    grid[1, 3] = 2
    with pytest.raises(InvalidOracle, match="object_overlap"):
        rotate_objects(infer_objects(grid), 90, grid.shape)


@pytest.fixture
def object_root(tmp_path):
    folder = tmp_path / "exp_setting_1" / "experiment_1"
    folder.mkdir(parents=True)
    for split, color in (("train", 1), ("val", 2), ("test", 3)):
        grid = np.zeros((7, 7), dtype=np.int64)
        grid[1:4, 2:5] = [[color, 0, 0], [color, color, 0], [color, color, color]]
        _, target = apply(infer_objects(grid), "rot90", grid.shape)
        pq.write_table(pa.Table.from_pylist([{
            "input": grid.tolist(), "output": target.tolist(),
            "transformation_suite": ["rot90"],
        }]), folder / f"{split}.parquet")
    return tmp_path


class PerfectObjectModel(torch.nn.Module):
    compute_metrics = staticmethod(DegreeRotationViT.compute_metrics)

    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        grids = batch["images"].argmax(1)
        angles = batch["task_tokens"].sum(1) * 15
        results = []
        for grid, angle in zip(grids, angles):
            try:
                _, result = rotate_objects(infer_objects(grid.numpy()), int(angle), grid.shape)
            except InvalidOracle:
                result = np.zeros(grid.shape, dtype=np.int64)
            results.append(torch.from_numpy(result))
        predictions = torch.stack(results)
        probabilities = F.one_hot(predictions, 10).permute(0, 3, 1, 2).float() * 0.999 + 0.0001
        return {"predictions": predictions, "output_log_probs": probabilities.log()}


def test_object_dataset_targets_and_scorer_use_local_rotations(object_root):
    train = ObjectAnchorDegreeRotationDataset({"root": str(object_root), "split": "train"})
    test = ObjectAnchorDegreeRotationDataset({
        "root": str(object_root), "split": "test", "program_selection": "all",
    })
    assert len(train) == 30 and len(test) == 42
    assert train.audit["target_mismatches"] == 0
    assert train.audit["recorded_rot90_verified"] == 1
    assert not set(train.digests) & set(test.digests)
    assert set(train.held_out_pairs).isdisjoint(train.programs)
    objects = infer_objects(train.sources[0].numpy())
    for i, program in enumerate(train.programs):
        _, expected = rotate_objects(objects, sum(program), train.resolution)
        np.testing.assert_array_equal(train[i]["target_grid"].numpy(), expected)
    result = evaluate_degrees(PerfectObjectModel(), DataLoader(test, batch_size=8))
    assert result["groups"]["held_out_pairs"]["samples"] == 12
    assert result["groups"]["seen_pairs/raster_exact_and_identity_preserved"]["direct_grid_accuracy"] == 100
    for row in result["by_program"].values():
        assert row["direct_grid_accuracy"] == 100
        assert row["direct_vs_summed_grid_agreement"] == 100
        assert row["reverse_order_grid_agreement"] == 100
        assert "raster_reference_valid_percent" in row
        assert "intermediate_identity_preserved_percent" in row
    assert any(row["raster_reference_grid_accuracy"] < 100 for row in result["by_program"].values())


def test_recorded_rot90_mismatch_blocks_training(object_root):
    path = object_root / "exp_setting_1" / "experiment_1" / "train.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["output"] = np.zeros((7, 7), dtype=np.int64).tolist()
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="disagrees with recorded rot90"):
        ObjectAnchorDegreeRotationDataset({"root": str(object_root), "split": "train"})
