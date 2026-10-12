from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import (
    ALL_PAIRS,
    ATOMIC_PROGRAMS,
    HELD_OUT_PAIRS,
    NET_ANGLES,
    SEEN_PAIRS,
    DegreeRotationDataset,
    pad_grid,
    rotate_grids,
)
from benchmarks.cogitao_vit_cross_attention.degree_rotation_eval import evaluate_degrees
from benchmarks.cogitao_vit_cross_attention.degree_rotation_model import (
    DegreeRotationViT,
)
from src.train import load_config


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "exp_setting_1" / "experiment_1"
    folder.mkdir(parents=True)
    for split, color in (("train", 1), ("val", 2), ("test", 3)):
        grid = np.zeros((7, 7), dtype=np.int64)
        grid[1:4, 2:5] = [[color, 0, 0], [color, color, 0], [color, color, color]]
        pq.write_table(
            pa.Table.from_pylist(
                [{"input": grid.tolist(), "transformation_suite": ["rot90"]}]
            ),
            folder / f"{split}.parquet",
        )
    return tmp_path


def test_renderer_ccw_center_and_categorical_labels():
    grid = torch.zeros(1, 24, 24, dtype=torch.long)
    grid[0, 6:10, 8:13] = torch.arange(20).reshape(4, 5) % 10
    for angle in (0, 90, 180, 270):
        torch.testing.assert_close(
            rotate_grids(grid, angle), torch.rot90(grid, angle // 90, (-2, -1))
        )
    for angle in NET_ANGLES:
        result = rotate_grids(grid, angle)
        assert set(result.unique().tolist()) <= set(grid.unique().tolist())
    with pytest.raises(ValueError, match="diagonal"):
        pad_grid(np.ones((20, 20)), (24, 24))


def test_split_holds_out_reversals_with_seen_net_angles():
    assert len(SEEN_PAIRS) == 21 and len(HELD_OUT_PAIRS) == 15
    assert set(SEEN_PAIRS).isdisjoint(HELD_OUT_PAIRS)
    assert set(SEEN_PAIRS) | set(HELD_OUT_PAIRS) == set(ALL_PAIRS)
    for a, b in HELD_OUT_PAIRS:
        assert (b, a) in SEEN_PAIRS
        assert sum((a, b)) in {sum(p) for p in SEEN_PAIRS}
    for axis in (0, 1):
        counts = Counter(p[axis] for p in SEEN_PAIRS)
        assert max(counts.values()) - min(counts.values()) == 1


def test_dataset_pairs_use_single_canonical_rotation_and_no_holdout_leakage(root):
    train = DegreeRotationDataset({"root": str(root), "split": "train"})
    test = DegreeRotationDataset(
        {"root": str(root), "split": "test", "program_selection": "all"}
    )
    assert train.programs == ATOMIC_PROGRAMS + SEEN_PAIRS
    assert len(train) == 27 and len(test) == 42
    assert not set(train.digests) & set(test.digests)
    for i, program in enumerate(train.programs):
        sample = train[i]
        expected = rotate_grids(train.sources.long(), sum(program))[0]
        torch.testing.assert_close(sample["target_grid"], expected)
        assert sample["task_token_mask"].tolist() == [True, len(program) == 2]
    with pytest.raises(ValueError, match="Training must use seen"):
        DegreeRotationDataset(
            {"root": str(root), "split": "train", "program_selection": "all"}
        )


def test_fresh_numeric_model_has_shared_angle_parameters_and_gradients():
    config = load_config(Path("config/cogitao/degree_rotation_joint_scratch.yaml"))
    assert not config.get("initialization")
    config["model"]["config"]["input_resolution"] = [8, 8]
    config["model"]["config"]["encoder"].update(hidden_dim=16, num_layers=1, mlp_dim=32)
    torch.manual_seed(42)
    model = DegreeRotationViT(config).eval()
    grid = torch.randint(0, 10, (2, 8, 8))
    batch = {
        "images": F.one_hot(grid, 10).permute(0, 3, 1, 2).float(),
        "target_grid": grid,
        "task_tokens": torch.tensor([[1, 2], [6, 0]]),
        "task_token_mask": torch.tensor([[True, True], [True, False]]),
    }
    output = model(batch)
    altered = model({**batch, "target_grid": torch.zeros_like(grid)})
    torch.testing.assert_close(output["output_logits"], altered["output_logits"])
    F.nll_loss(output["output_log_probs"], grid).backward()
    assert model.task_embedding.projection.weight.grad.abs().sum() > 0
    assert model.task_position_embedding.grad[:, 1].abs().sum() > 0
    assert model.task_embedding(torch.tensor([7, 12])).shape == (2, 16)


class PerfectCanonicalModel(torch.nn.Module):
    compute_metrics = staticmethod(DegreeRotationViT.compute_metrics)

    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        grids = batch["images"].argmax(1)
        degrees = batch["task_tokens"].sum(1) * 15
        predictions = rotate_grids(grids, degrees)
        probability = (
            F.one_hot(predictions, 10).permute(0, 3, 1, 2).float() * 0.999 + 0.0001
        )
        return {"predictions": predictions, "output_log_probs": probability.log()}


def test_scorer_exposes_raster_mismatch_and_net_angle_extrapolation(root):
    dataset = DegreeRotationDataset(
        {"root": str(root), "split": "test", "program_selection": "all"}
    )
    result = evaluate_degrees(
        PerfectCanonicalModel(), DataLoader(dataset, batch_size=8)
    )
    assert len(result["by_program"]) == 42
    assert result["groups"]["atomic"]["samples"] == 6
    assert result["groups"]["seen_pairs"]["samples"] == 21
    assert result["groups"]["held_out_pairs"]["samples"] == 15
    for row in result["by_program"].values():
        assert row["direct_grid_accuracy"] == 100
        assert row["direct_vs_summed_grid_agreement"] == 100
        assert row["reverse_order_grid_agreement"] == 100
    assert any(
        r["raster_reference_grid_accuracy"] < 100 for r in result["by_program"].values()
    )
    assert "held_out_pairs/net_angle_beyond_atomic_training" in result["groups"]


def test_unordered_holdout_excludes_both_orders_and_balances_operands():
    from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import pair_split

    seen, held, metadata = pair_split("unordered")
    assert len(seen) == 24 and len(held) == 12
    assert {tuple(sorted(p)) for p in seen}.isdisjoint({tuple(sorted(p)) for p in held})
    assert metadata["commutativity_safe"] is True
    assert metadata["unseen_effects"] == []
    for a, b in held:
        assert (b, a) in held
    for axis in (0, 1):
        assert set(Counter(p[axis] for p in seen).values()) == {4}
        assert set(Counter(p[axis] for p in held).values()) == {2}


def test_effect_holdout_excludes_every_factorization_and_atomic_equivalent():
    from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import (
        ATOMIC_ANGLES,
        pair_split,
    )

    seen, held, metadata = pair_split("effect")
    assert len(seen) == 24 and len(held) == 12
    observed = set(ATOMIC_ANGLES) | {sum(p) for p in seen}
    held_effects = {sum(p) for p in held}
    assert held_effects == {105, 135, 165}
    assert observed.isdisjoint(held_effects)
    assert metadata["unseen_effects"] == [105, 135, 165]
    for a, b in held:
        assert (b, a) in held and (a, b) not in seen
    assert set(seen) | set(held) == set(ALL_PAIRS)


@pytest.mark.parametrize("protocol", ["unordered", "effect"])
def test_new_dataset_protocol_is_used_by_training_and_scorer(root, protocol):
    from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import pair_split

    seen, held, metadata = pair_split(protocol)
    config = {"root": str(root), "pair_split": protocol}
    train = DegreeRotationDataset({**config, "split": "train"})
    val = DegreeRotationDataset({**config, "split": "val"})
    test = DegreeRotationDataset(
        {**config, "split": "test", "program_selection": "all"}
    )
    assert len(train) == len(val) == 30
    assert train.programs == ATOMIC_PROGRAMS + seen
    assert not set(held) & set(train.programs)
    assert train.split_metadata == metadata
    assert test.manifest()["split_metadata"]["commutativity_safe"]
    result = evaluate_degrees(PerfectCanonicalModel(), DataLoader(test, batch_size=7))
    assert result["groups"]["seen_pairs"]["samples"] == 24
    assert result["groups"]["held_out_pairs"]["samples"] == 12
    if protocol == "effect":
        assert result["groups"]["held_out_pairs/net_effect_unseen"]["samples"] == 12
        assert "held_out_pairs/net_effect_seen" not in result["groups"]
        for program in held:
            label = "_then_".join(map(str, program))
            assert result["by_program"][label]["net_effect_seen_in_training"] is False
    else:
        assert result["groups"]["held_out_pairs/net_effect_seen"]["samples"] == 12
