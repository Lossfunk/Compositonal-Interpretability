"""Protocol checks for atomic task transport; no checkpoint or real data needed."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks.cogitao_vit_cross_attention.c1_atomic_equivariance import (
    RIGHT, UP, LayerCollector, align_clockwise_tokens, align_reflected_tokens,
    direction_scores, evaluate_spatial_image_maps, evaluate_spatial_mirror_maps,
    fit_direction_readout, fit_orthogonal, fit_spatial_image_maps,
    fit_spatial_mirror_maps, isomorphic_pair, log_wandb, orbit_key, read_atomic,
    reflect_left_right, reflected_pair, rotate_clockwise, translate,
    wandb_layer_rows, wandb_metrics,
)
from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import model_batch
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)


def asymmetric_grid():
    grid = np.zeros((7, 7), dtype=np.int64)
    grid[2, 2:4] = [1, 2]
    grid[3, 2] = 3
    grid[4, 4] = 4
    return grid


def test_clockwise_task_transport_and_unchanged_task_are_distinct():
    grid = asymmetric_grid()
    gx, gy = isomorphic_pair(dict(input=grid, target=translate(grid, UP), task=UP))
    np.testing.assert_array_equal(gy, translate(gx, RIGHT))
    assert not np.array_equal(gy, translate(gx, UP))
    restored = grid
    for _ in range(4):
        restored = rotate_clockwise(restored)
    np.testing.assert_array_equal(restored, grid)


def test_left_right_reflection_preserves_up_and_squares_to_identity():
    grid = asymmetric_grid()
    mx, my = reflected_pair(dict(input=grid, target=translate(grid, UP), task=UP))
    np.testing.assert_array_equal(my, translate(mx, UP))
    np.testing.assert_array_equal(reflect_left_right(mx), grid)


def test_boundary_transport_does_not_wrap_or_clip():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[0, 2] = 1
    with pytest.raises(ValueError, match="leave the grid"):
        translate(grid, UP)
    with pytest.raises(ValueError, match="leave the grid"):
        translate(rotate_clockwise(grid), RIGHT)


def test_procrustes_recovers_known_row_vector_action_on_heldout_data():
    rng = np.random.default_rng(12)
    q = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                  [0., 0., 0., 1.], [0., 0., -1., 0.]])
    train = rng.normal(size=(40, 4))
    fitted, rank = fit_orthogonal(train, train @ q)
    heldout = rng.normal(size=(11, 4))
    assert rank == 4
    np.testing.assert_allclose(heldout @ fitted, heldout @ q, atol=1e-10)
    np.testing.assert_allclose(fitted.T @ fitted, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(np.linalg.matrix_power(fitted, 4), np.eye(4), atol=1e-10)


def test_constant_function_vectors_flag_underidentified_map():
    x = np.tile(np.array([1., 2., 3., 4.]), (20, 1))
    y = np.tile(np.array([4., 3., 2., 1.]), (20, 1))
    q, rank = fit_orthogonal(x, y)
    assert rank == 1
    np.testing.assert_allclose(x @ q, y, atol=1e-10)


def test_full_image_token_map_recovers_spatial_rotation_and_channel_action():
    rng = np.random.default_rng(9)
    source = rng.normal(size=(2, 9, 3)).astype(np.float32)
    q = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    rotated = np.rot90(source.reshape(2, 3, 3, 3), k=-1,
                       axes=(1, 2)).reshape(2, 9, 3) @ q
    np.testing.assert_allclose(align_clockwise_tokens(rotated, (3, 3)),
                               (source @ q).reshape(2, 3, 3, 3), atol=1e-6)

    class Collector:
        def run(self, rows, _device, _batch_size, rotated=False, spatial=False):
            assert spatial
            values = rotated_tokens if rotated else source
            return {"image_block_01__spatial": values[:len(rows)]}, None, None

    rotated_tokens = rotated.astype(np.float32)
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = 2
    rows = [dict(input=grid, target=translate(grid, UP), task=UP) for _ in range(2)]
    model = SimpleNamespace(input_resolution=(3, 3), embedding_dim=3)
    device = torch.device("cpu")
    operators, ranks = fit_spatial_image_maps(
        model, Collector(), rows, device, 2, ["image_block_01"])
    assert ranks["image_block_01"]["source_gram_rank"] == 3
    np.testing.assert_allclose(operators["image_block_01"], q, atol=1e-6)
    metadata = dict(spatial_image_layers=["image_block_01"],
                    spatial_image_ranks=ranks)
    results, extra = evaluate_spatial_image_maps(
        model, Collector(), rows,
        {"image_block_01__spatial_rho": operators["image_block_01"]},
        metadata, device, 2)
    assert results["image_block_01"]["equivariance_defect"]["mean"] < 1e-6
    assert extra["input_rotation_resize_mismatch"]["mean"] == 0


def test_full_image_token_map_recovers_reflection_and_m2_identity():
    rng = np.random.default_rng(17)
    source = rng.normal(size=(2, 9, 3)).astype(np.float32)
    q = np.diag([1., -1., 1.])
    reflected = np.flip(source.reshape(2, 3, 3, 3), axis=2).reshape(2, 9, 3) @ q
    np.testing.assert_allclose(align_reflected_tokens(reflected, (3, 3)),
                               (source @ q).reshape(2, 3, 3, 3), atol=1e-6)

    class Collector:
        def run(self, rows, _device, _batch_size, rotated=False, mirrored=False,
                spatial=False):
            assert spatial and not rotated
            values = reflected_tokens if mirrored else source
            return {"image_block_01__spatial": values[:len(rows)]}, None, None

    reflected_tokens = reflected.astype(np.float32)
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 1] = 2
    rows = [dict(input=grid, target=translate(grid, UP), task=UP) for _ in range(2)]
    model = SimpleNamespace(input_resolution=(3, 3), embedding_dim=3)
    device = torch.device("cpu")
    operators, ranks = fit_spatial_mirror_maps(
        model, Collector(), rows, device, 2, ["image_block_01"])
    np.testing.assert_allclose(operators["image_block_01"], q, atol=1e-6)
    metadata = dict(spatial_image_layers=["image_block_01"],
                    spatial_mirror_ranks=ranks)
    results, extra = evaluate_spatial_mirror_maps(
        model, Collector(), rows,
        {"image_block_01__mirror_rho": operators["image_block_01"]},
        metadata, device, 2)
    assert results["image_block_01"]["equivariance_defect"]["mean"] < 1e-6
    assert results["image_block_01"]["m2_channel_closure_error"] < 1e-6
    assert results["image_block_01"]["m2_data_closure_defect"]["mean"] < 1e-6
    assert extra["input_reflection_resize_mismatch"]["mean"] == 0


def test_direction_readout_handles_constant_channels_and_class_imbalance():
    labels = np.array([-1] * 10 + [1] * 3)
    x = np.column_stack((labels * 2., np.full(len(labels), 7.)))
    params = fit_direction_readout(x, labels, ridge=0.01)
    scores = direction_scores(np.array([[-3., 7.], [3., 7.]]), *params)
    assert scores[0] < 0 < scores[1]


def test_reader_excludes_fit_orbits_and_checks_recorded_targets(tmp_path):
    grid = asymmetric_grid()
    bad = grid.copy()
    bad[5, 3] = 7
    table = pa.Table.from_pylist([
        dict(input=grid.tolist(), output=translate(grid, UP).tolist(),
             transformation_suite=[UP]),
        dict(input=rotate_clockwise(grid).tolist(), output=translate(
            rotate_clockwise(grid), RIGHT).tolist(), transformation_suite=[RIGHT]),
        dict(input=bad.tolist(), output=bad.tolist(), transformation_suite=[UP]),
    ])
    directory = tmp_path / "exp_setting_1" / "experiment_1"
    directory.mkdir(parents=True)
    pq.write_table(table, directory / "val.parquet")
    rows, coverage = read_atomic(tmp_path, "val", 20, 42, {orbit_key(grid)})
    assert rows == []
    assert coverage[f"{UP}|excluded_duplicate_or_fit_dihedral_orbit"] == 1
    assert coverage[f"{RIGHT}|excluded_duplicate_or_fit_dihedral_orbit"] == 1
    assert coverage[f"{UP}|recorded_target_mismatch"] == 1


def test_collection_observes_each_block_without_changing_predictions():
    config = dict(model=dict(config=dict(
        input_resolution=[7, 7],
        encoder=dict(hidden_dim=8, num_layers=2, num_heads=2, mlp_dim=16, dropout=0.),
        function_mlp=dict(max_tokens=2, embedding_dim=8, hidden_dim=16),
        cross_attention=dict(num_shared_latents=3, num_heads=2, mlp_dim=16, dropout=0.),
    )))
    model = FunctionConditionedViTCrossAttentionModel(config).eval()
    grid = asymmetric_grid()
    target = translate(grid, UP)
    device = torch.device("cpu")
    batch = model_batch(model, [grid], [target], [(UP,)], device)
    with torch.inference_mode():
        expected = model(batch)["predictions"].numpy()
    collector = LayerCollector(model)
    try:
        features, predictions, _ = collector.run(
            [dict(input=grid, target=target, task=UP)], device, 1)
    finally:
        collector.close()
    np.testing.assert_array_equal(predictions, expected)
    assert {"image_input_tokens", "image_block_01", "image_block_02", "image_final_norm",
            "function_mlp_01", "function_mlp_02", "function_mlp_03",
            "shared_cross_attention", "output_cross_attention"} == set(features)
    assert features["image_block_01"].shape == (1, 8)
    assert features["function_mlp_01"].shape == (1, 16)
    assert features["shared_cross_attention"].shape == (1, 8)


def test_wandb_summary_keeps_split_and_layer_metrics_separate():
    layer = dict(
        samples_scored=5, fit_rank=dict(source_rank=4, dimension=8),
        equivariance_defect=dict(mean=0.2, median=0.1, p90=0.4),
        identity_defect=dict(mean=0.8), c4_operator_closure_error=0.3,
        direction_readout=dict(native_balanced_accuracy=0.9,
                               transported_right_accuracy=0.8,
                               rho_mapped_right_accuracy=0.7),
    )
    report = dict(global_step=123, fit=dict(
        coverage={"translate_up|selected": 5}, fit_up_pairs=5,
        fit_native_pairs=10, ranks={"shared_cross_attention": dict(
            source_rank=4, dimension=8, underidentified=True)}),
        evaluation={"val": dict(layers={"shared_cross_attention": layer},
                                transported_right_output=dict(exact_grid_accuracy=0.6))})
    metrics = wandb_metrics(report)
    assert metrics["checkpoint/global_step"] == 123
    assert metrics["fit/coverage/translate_up|selected"] == 5
    assert metrics["val/layers/shared_cross_attention/equivariance_defect/mean"] == 0.2
    assert metrics["val/transported_right_output/exact_grid_accuracy"] == 0.6
    assert "fit/layers/shared_cross_attention/underidentified" not in metrics
    rows = wandb_layer_rows(report)
    assert len(rows) == 1
    assert rows[0]["split"] == "val"
    assert rows[0]["rho_mapped_right_accuracy"] == 0.7


def test_wandb_logging_uses_opt_in_and_finishes_run(monkeypatch, tmp_path):
    run = MagicMock()
    fake_wandb = SimpleNamespace(init=MagicMock(return_value=run), Table=MagicMock())
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    args = SimpleNamespace(
        wandb_mode="disabled", wandb_project="project", wandb_entity=None,
        wandb_group="group", mode="fit", checkpoint=tmp_path / "model.ckpt",
        probe_file=tmp_path / "frozen_probes.npz", data_root=tmp_path,
        splits=["val"], eval_pairs=8, batch_size=4, seed=42,
        wandb_log_artifacts=False, output_dir=tmp_path,
    )
    report = dict(global_step=1000, fit=dict(
        checkpoint_sha256="a" * 64, fit_up_pairs=8, fit_native_pairs=16,
        ridge=0.01, symmetry="whole_grid_clockwise_90", protocol_version=1,
        coverage={"translate_up|selected": 8}, ranks={}), evaluation={})
    log_wandb(report, tmp_path / "fit_report.json", args)
    fake_wandb.init.assert_not_called()

    args.wandb_mode = "online"
    log_wandb(report, tmp_path / "fit_report.json", args)
    assert fake_wandb.init.call_args.kwargs["project"] == "project"
    assert run.log.call_args.args[0]["fit/up_pairs"] == 8
    run.log_artifact.assert_not_called()
    run.finish.assert_called_once()
