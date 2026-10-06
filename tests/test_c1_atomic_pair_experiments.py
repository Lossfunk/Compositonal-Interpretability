"""Oracle geometry, intervention metrics, and split isolation checks.

These tests are deliberately small and do not require trained checkpoints.
"""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments import (
    SplitRows, audit_target, composition_accuracy, grid_key, sample_pair,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import (
    apply_affine, correlation, delta_distance, fit_affine, pair_measurements,
    spatial_chart,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle, oracle_sequence, oracle_states,
)
from benchmarks.cogitao_vit_cross_attention.c1_symmetry_geometry import compose
from data.cogitao.cogitao import TASK_TO_ID


def asymmetric_grid():
    grid = np.zeros((12, 12), dtype=np.int64)
    grid[4:7, 4:6] = [[3, 3], [3, 0], [3, 3]]
    return grid


@pytest.mark.parametrize("suite", [
    ("rot90",), ("translate_up",), ("rot90", "translate_up"),
    ("translate_up", "rot90"),
])
def test_oracle_matches_existing_exact_pixel_geometry(suite):
    grid = asymmetric_grid()
    expected, _, _ = compose(grid, suite)
    np.testing.assert_array_equal(oracle_sequence(grid, suite), expected)


def test_object_identity_survives_fragmentation_by_crop():
    grid = np.zeros((16, 16), dtype=np.int64)
    grid[3:8, 3] = grid[3:8, 7] = 2
    grid[7, 3:8] = 2  # A single connected U; cropping splits its visible legs.
    result = oracle_sequence(grid, ("crop_bottom_side", "double_right"))
    expected = np.zeros_like(grid)
    expected[3:6, [3, 7, 8, 12]] = 2
    np.testing.assert_array_equal(result, expected)


def test_padding_uses_generator_direction_colors_and_position():
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[3:5, 3:5] = 2
    right = oracle_sequence(grid, ("pad_right",))
    expected = grid.copy()
    expected[3:5, 5] = 6
    np.testing.assert_array_equal(right, expected)
    top = oracle_sequence(grid, ("pad_top",))
    expected = grid.copy()
    expected[2, 3:5] = 8
    np.testing.assert_array_equal(top, expected)


def test_fill_color_and_recolor_are_deterministic():
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[2:5, 2:5] = 9
    grid[3, 3] = 0
    filled = oracle_sequence(grid, ("fill_holes_different_color",))
    assert filled[3, 3] == 1
    assert filled[2, 2] == 9
    changed = oracle_sequence(grid, ("change_shape_color",))
    assert changed[2, 2] == 1
    assert changed[3, 3] == 0


def test_reject_if_either_composition_order_has_an_invalid_state():
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[1, 2] = 4
    with pytest.raises(InvalidOracle, match="out_of_bounds"):
        oracle_states(grid, "translate_up", "translate_up")


def test_spatial_chart_retains_ordered_bins():
    tokens = np.arange(16).reshape(1, 16, 1)
    np.testing.assert_array_equal(spatial_chart(tokens, (4, 4), 2),
                                  [[2.5, 4.5, 10.5, 12.5]])


def affine(weights, bias):
    value = np.eye(len(weights) + 1)
    value[:-1, :-1] = weights
    value[-1, :-1] = bias
    return value


def test_affine_fit_handles_translation_and_reports_identifiability():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(64, 3))
    matrix = affine(np.diag([1., 2., 3.]), [2., -1., 4.])
    fitted, diagnostic = fit_affine(x, apply_affine(x, matrix), 1e-8)
    np.testing.assert_allclose(fitted, matrix, atol=1e-6)
    assert not diagnostic["underidentified"]


def test_relation_retention_uses_predicted_actions_even_when_oracles_equal():
    # Identical fgx/gfx images alone cannot establish that learned actions commute.
    states = {name: np.array([[value]]) for name, value in
              (("x", 0), ("fx", 1), ("gx", 2), ("fgx", 3), ("gfx", 3))}
    embeddings = {name: np.array([1., value + 1.]) for name, value in
                  (("x", 0), ("fx", 1), ("gx", 2), ("fgx", 3), ("gfx", 3))}
    af = affine(np.array([[1., 1.], [0., 1.]]), [0., 0.])
    ag = affine(np.diag([2., 1.]), [0., 0.])
    metrics, relations = pair_measurements(states, embeddings, af, ag, 0.01)
    assert metrics["oracle_commutes"] == 1
    assert metrics["commutator_data_defect"] > 0
    assert metrics["oracle_relation_retention"] == 0
    assert len(relations) == 1

    states["gfx"] = np.array([[4]])
    metrics, _ = pair_measurements(states, embeddings, af, ag, 0.01)
    assert metrics["commutator_operator_defect"] is None
    assert metrics["commutator_data_defect"] is None


def test_additive_commuting_actions_have_perfect_transfer_and_retention():
    states = {name: np.array([[value]]) for name, value in
              (("x", 0), ("fx", 1), ("gx", 2), ("fgx", 3), ("gfx", 3))}
    embeddings = dict(x=np.array([1., 2.]), fx=np.array([2., 2.]),
                      gx=np.array([1., 4.]), fgx=np.array([2., 4.]),
                      gfx=np.array([2., 4.]))
    af, ag = affine(np.eye(2), [1., 0.]), affine(np.eye(2), [0., 2.])
    metrics, _ = pair_measurements(states, embeddings, af, ag, 0.01)
    assert metrics["action_transfer_f_relative_l2"] == 0
    assert metrics["action_transfer_g_relative_l2"] == 0
    assert metrics["commutator_operator_defect"] == 0
    assert metrics["oracle_relation_retention"] == 1
    assert metrics["fgx_prediction_defect"] == 0


def test_self_pair_tautologies_do_not_inflate_retention():
    states = {name: np.array([[value]]) for name, value in
              (("x", 0), ("fx", 1), ("gx", 1), ("fgx", 2), ("gfx", 2))}
    embeddings = {name: np.array([1., float(value)]) for name, value in
                  (("x", 0), ("fx", 1), ("gx", 1), ("fgx", 2), ("gfx", 2))}
    action = affine(np.eye(2), [0., 1.])
    metrics, relations = pair_measurements(
        states, embeddings, action, action, 0.01, same_function=True)
    assert metrics["oracle_relation_tautology_count"] == 2
    assert metrics["oracle_relation_count"] == 0
    assert metrics["oracle_relation_retention"] is None
    assert metrics["commutator_operator_defect"] is None
    assert all(row["retained"] is None for row in relations)


def test_zero_actions_and_small_correlations_are_not_misreported():
    result = delta_distance(np.zeros(2), np.zeros(2))
    assert result["relative_l2"] == 0
    assert result["cosine_distance"] is None
    assert result["both_zero"] == 1
    assert correlation([1., 2.], [3., 4.])["status"] == "fewer_than_three_pairs"
    assert correlation([1., 1., 1.], [1., 2., 3.])["pearson"] is None


def test_ood_uses_g_then_f_rows_and_excludes_all_fit_states(tmp_path):
    x = asymmetric_grid()
    states = oracle_states(x, "rot90", "translate_up")
    path = tmp_path / "exp_setting_1" / "experiment_1"
    path.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([
        dict(input=x.tolist(), output=states["fgx"].tolist(),
             transformation_suite=["translate_up", "rot90"]),
        dict(input=x.tolist(), output=states["gfx"].tolist(),
             transformation_suite=["rot90", "translate_up"]),
    ]), path / "val_ood.parquet")
    data = SplitRows(tmp_path, 1, "val_ood")
    args = SimpleNamespace(seed=42, eval_images_per_pair=3)
    selected, coverage = sample_pair(data, "val_ood", "rot90", "translate_up", args, set())
    assert len(selected) == 1
    assert selected[0]["index"] == 0
    assert coverage["available"] == 1
    selected, coverage = sample_pair(data, "val_ood", "rot90", "translate_up", args,
                                     {grid_key(states["gx"])})
    assert selected == []
    assert coverage["fit_state_overlap"] == 1


def test_audit_mismatch_saves_diagnostic_and_stops(tmp_path):
    row = dict(index=7, suite=("rot90",), input=np.array([[3]]), output=np.array([[3]]))
    with pytest.raises(ValueError, match="Oracle audit failed"):
        audit_target(row, np.array([[4]]), "test", SimpleNamespace(output_dir=tmp_path))
    with np.load(tmp_path / "oracle_audit_failure.npz", allow_pickle=False) as archive:
        assert archive["source_index"].item() == 7
        np.testing.assert_array_equal(archive["recorded_output"], [[3]])
        np.testing.assert_array_equal(archive["oracle_output"], [[4]])


def test_accuracy_pass_preserves_function_order_and_penalizes_extra_foreground():
    class Model:
        input_resolution = (2, 2)
        max_function_tokens = 2

        def __call__(self, batch):
            assert batch["task_tokens"].tolist() == [[
                TASK_TO_ID["translate_up"], TASK_TO_ID["rot90"]]]
            assert bool(batch["task_token_mask"].all())
            prediction = batch["target_grid"].clone()
            prediction[:, 1, 1] = 2  # One extra incorrect foreground cell.
            return {"predictions": prediction}

    target = np.array([[3, 0], [0, 0]])
    data = SimpleNamespace(row=lambda index: dict(index=index, input=target, output=target))
    scores = composition_accuracy(Model(), data, [4], ("translate_up", "rot90"),
                                  SimpleNamespace(batch_size=1), torch.device("cpu"))
    assert scores[4] == 0.5
