from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmarks.cogitao_vit_cross_attention.rotation_compositions import (
    ALL_PAIRS,
    ATOMIC_PROGRAMS,
    HELD_OUT_PAIRS,
    SEEN_PAIRS,
    RotationProgramDataset,
    evaluate_programs,
)
from benchmarks.cogitao_vit_cross_attention.rotation_data import rotation_orbit
from benchmarks.cogitao_vit_cross_attention.rotation_model import RotationTokenViT


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "exp_setting_1" / "experiment_1"
    folder.mkdir(parents=True)
    for split, color in (("train", 1), ("val", 2), ("test", 3)):
        grid = np.zeros((8, 8), dtype=np.int64)
        grid[2:4, 3:6] = [[color, 0, color], [color, color, 0]]
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "input": grid.tolist(),
                        "output": rotation_orbit(grid)[1].tolist(),
                        "transformation_suite": ["rot90"],
                    }
                ]
            ),
            folder / f"{split}.parquet",
        )
    return tmp_path


def test_pair_split_balances_operands_and_sums():
    assert len(SEEN_PAIRS) == len(HELD_OUT_PAIRS) == 8
    assert set(SEEN_PAIRS).isdisjoint(HELD_OUT_PAIRS)
    assert set(SEEN_PAIRS) | set(HELD_OUT_PAIRS) == set(ALL_PAIRS)
    for pairs in (SEEN_PAIRS, HELD_OUT_PAIRS):
        for axis in (0, 1):
            assert Counter(p[axis] for p in pairs) == {0: 2, 90: 2, 180: 2, 270: 2}
        assert Counter(sum(p) % 360 for p in pairs) == {0: 2, 90: 2, 180: 2, 270: 2}


def test_only_seen_pairs_train_and_composition_tokens_are_active(root):
    train = RotationProgramDataset(
        {"root": str(root), "split": "train", "image_size": [8, 8]}
    )
    test = RotationProgramDataset(
        {
            "root": str(root),
            "split": "test",
            "program_selection": "all",
            "image_size": [8, 8],
        }
    )
    assert train.programs == ATOMIC_PROGRAMS + SEEN_PAIRS
    assert len(train) == 12 and len(test) == 20
    for i, program in enumerate(train.programs):
        sample = train[i]
        assert sample["task_token_mask"].tolist() == (
            [True, False] if len(program) == 1 else [True, True]
        )
        assert sample["task_tokens"].tolist() == [a // 90 + 1 for a in program] + [
            0
        ] * (2 - len(program))
        torch.testing.assert_close(
            sample["target_grid"], train.targets[0, sum(program) % 360 // 90].long()
        )
    assert train.audit["held_out_composition_examples"] == 0
    assert test.audit["held_out_composition_examples"] == 8
    with pytest.raises(ValueError, match="never enter training"):
        RotationProgramDataset(
            {"root": str(root), "split": "train", "program_selection": "all"}
        )


class PerfectRotationModel(torch.nn.Module):
    compute_metrics = staticmethod(RotationTokenViT.compute_metrics)

    def __init__(self):
        super().__init__()
        self.device_marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        grids = batch["images"].argmax(1)
        predictions = []
        for grid, tokens in zip(grids, batch["task_tokens"]):
            quarter_turns = sum(int(t) - 1 for t in tokens if t != 0) % 4
            predictions.append(
                torch.from_numpy(rotation_orbit(grid.numpy())[quarter_turns].copy())
            )
        predictions = torch.stack(predictions).long()
        probabilities = (
            F.one_hot(predictions, 10).permute(0, 3, 1, 2).float() * 0.999 + 0.0001
        )
        return {"predictions": predictions, "output_log_probs": probabilities.log()}


def test_composition_scorer_groups_pairs_and_checks_preservation(root):
    dataset = RotationProgramDataset(
        {
            "root": str(root),
            "split": "test",
            "program_selection": "all",
            "image_size": [8, 8],
        }
    )
    result = evaluate_programs(
        PerfectRotationModel(), DataLoader(dataset, batch_size=7)
    )
    assert result["groups"]["atomic"]["samples"] == 4
    assert result["groups"]["seen_pairs"]["samples"] == 8
    assert result["groups"]["held_out_pairs"]["samples"] == 8
    assert result["groups"]["seen_pairs_without_identity"]["samples"] == 5
    assert result["groups"]["held_out_pairs_without_identity"]["samples"] == 4
    assert len(result["by_program"]) == 20
    for row in result["by_program"].values():
        assert row["direct_grid_accuracy"] == 100
        assert row["summed_angle_grid_accuracy"] == 100
        assert row["sequential_grid_accuracy"] == 100
        assert row["direct_vs_summed_grid_agreement"] == 100
        assert row["direct_vs_sequential_grid_agreement"] == 100
