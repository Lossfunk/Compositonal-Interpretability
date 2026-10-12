from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks.cogitao_vit_cross_attention.rotation_data import (
    RotationDataset,
    rotation_orbit,
)
from benchmarks.cogitao_vit_cross_attention.rotation_model import RotationTokenViT
from src.train import load_config


def test_rotation_cycle_preserves_object_anchor_and_colors():
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[2:4, 3:6] = [[1, 0, 2], [3, 4, 0]]
    orbit = rotation_orbit(grid)
    for k, state in enumerate(orbit):
        expected = np.zeros_like(grid)
        patch = np.rot90(grid[2:4, 3:6], k)
        expected[2 : 2 + patch.shape[0], 3 : 3 + patch.shape[1]] = patch
        np.testing.assert_array_equal(state, expected)
    np.testing.assert_array_equal(rotation_orbit(orbit[1])[3], grid)


def test_dataset_excludes_compositions_and_cross_split_duplicates(tmp_path):
    root = tmp_path / "exp_setting_1" / "experiment_1"
    root.mkdir(parents=True)
    grids = []
    for color in (1, 2, 3):
        grid = np.zeros((8, 8), dtype=np.int64)
        grid[2:4, 3:6] = [[color, 0, color], [color, color, 0]]
        grids.append(grid)
    for name, indices in (("train", [0]), ("val", [0, 1]), ("test", [0, 1, 2])):
        rows = [
            {
                "input": grids[i].tolist(),
                "output": rotation_orbit(grids[i])[1].tolist(),
                "transformation_suite": ["rot90"],
            }
            for i in indices
        ]
        rows.append(
            {
                "input": grids[2].tolist(),
                "output": grids[2].tolist(),
                "transformation_suite": ["rot90", "rot90"],
            }
        )
        pq.write_table(pa.Table.from_pylist(rows), root / f"{name}.parquet")
    datasets = [
        RotationDataset({"root": str(tmp_path), "split": split, "image_size": [8, 8]})
        for split in ("train", "val", "test")
    ]
    assert [len(d) for d in datasets] == [4, 4, 4]
    for d in datasets:
        for a in range(4):
            sample = d[a]
            assert sample["angle_degrees"] == a * 90
            assert sample["task_tokens"].tolist() == [a + 1, 0]
            assert sample["task_token_mask"].tolist() == [True, False]
    assert set(datasets[0].digests).isdisjoint(datasets[1].digests)
    assert set(datasets[1].digests).isdisjoint(datasets[2].digests)
    path = root / "train.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0]["output"] = np.zeros((8, 8), dtype=int).tolist()
    pq.write_table(pa.Table.from_pylist(rows), path)
    with pytest.raises(ValueError, match="disagrees"):
        RotationDataset({"root": str(tmp_path), "image_size": [8, 8]})


def test_angle_conditioning_receives_gradient_without_target_leakage():
    config = load_config(Path("config/cogitao/rotation_token_vit.yaml"))
    config["model"]["config"]["input_resolution"] = [4, 4]
    config["model"]["config"]["encoder"].update(hidden_dim=16, num_layers=1, mlp_dim=32)
    model = RotationTokenViT(config).eval()
    grid = torch.randint(0, 10, (4, 4, 4))
    batch = {
        "images": torch.nn.functional.one_hot(grid, 10).permute(0, 3, 1, 2).float(),
        "target_grid": grid,
        "task_tokens": torch.tensor([[1, 0], [2, 0], [3, 0], [4, 0]]),
        "task_token_mask": torch.tensor([[True, False]] * 4),
    }
    output = model(batch)
    changed = model({**batch, "target_grid": torch.zeros_like(grid)})
    torch.testing.assert_close(output["output_logits"], changed["output_logits"])
    torch.nn.functional.nll_loss(output["output_log_probs"], grid).backward()
    assert model.task_embedding.weight.grad[1:].abs().sum() > 0
    assert model.encoder.patch_embedding.weight.grad.abs().sum() > 0
