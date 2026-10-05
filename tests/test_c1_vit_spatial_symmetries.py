"""Checks for the image-only C1 ViT spatial-symmetry protocol."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from benchmarks.cogitao_vit_cross_attention.c1_vit_spatial_symmetries import (
    ViTLayerCollector, align_reflection, align_rotation, evaluate_symmetry,
    fit_operators, orbit_key, read_id_images, reflect_left_right,
    rotate_clockwise,
)
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)


def test_image_transforms_have_expected_orders():
    grid = np.arange(25).reshape(5, 5)
    rotated = grid
    for _ in range(4):
        rotated = rotate_clockwise(rotated)
    np.testing.assert_array_equal(rotated, grid)
    np.testing.assert_array_equal(
        reflect_left_right(reflect_left_right(grid)), grid)


def test_token_alignment_undoes_spatial_actions():
    tokens = np.arange(2 * 9 * 4).reshape(2, 9, 4)
    image = tokens.reshape(2, 3, 3, 4)
    rotated = np.rot90(image, k=-1, axes=(1, 2)).reshape(2, 9, 4)
    reflected = np.flip(image, axis=2).reshape(2, 9, 4)
    np.testing.assert_array_equal(align_rotation(rotated, (3, 3)), image)
    np.testing.assert_array_equal(align_reflection(reflected, (3, 3)), image)


class SyntheticCollector:
    def __init__(self, source, rotation, reflection):
        self.source = source
        self.rotation = rotation
        self.reflection = reflection

    def encode(self, grids, _device, _batch_size):
        marker = int(np.asarray(grids[0])[0, 0])
        values = self.source if marker == 0 else self.rotation if marker == 1 else self.reflection
        return {"image_input_tokens": values[:len(grids)]}


def test_fitted_actions_recover_r4_and_m2():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(3, 9, 4)).astype(np.float32)
    rotation_q = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                           [0., 0., 0., 1.], [0., 0., -1., 0.]])
    mirror_q = np.diag([1., -1., 1., -1.])
    rotation = np.rot90(source.reshape(3, 3, 3, 4), k=-1,
                        axes=(1, 2)).reshape(3, 9, 4) @ rotation_q
    reflection = np.flip(source.reshape(3, 3, 3, 4), axis=2).reshape(
        3, 9, 4) @ mirror_q
    rows = []
    for _ in range(3):
        grid = np.zeros((3, 3), dtype=np.int64)
        rows.append(dict(input=grid))
    model = SimpleNamespace(input_resolution=(3, 3), embedding_dim=4, encoder_layers=0)

    class MarkedCollector(SyntheticCollector):
        def encode(self, grids, device, batch_size):
            first = np.asarray(grids[0])
            if np.array_equal(first, rotate_clockwise(rows[0]["input"])):
                return {"image_input_tokens": self.rotation[:len(grids)]}
            if np.array_equal(first, reflect_left_right(rows[0]["input"])):
                return {"image_input_tokens": self.reflection[:len(grids)]}
            return {"image_input_tokens": self.source[:len(grids)]}

    # Use an asymmetric colored marker so rotation/reflection remain distinct.
    for row in rows:
        row["input"][0, 0] = 3
        row["input"][0, 1] = 7
    collector = MarkedCollector(source, rotation, reflection)
    layers, operators, ranks = fit_operators(
        model, collector, rows, torch.device("cpu"), 3)
    np.testing.assert_allclose(operators[("rotation", "image_input_tokens")],
                               rotation_q, atol=1e-6)
    np.testing.assert_allclose(operators[("reflection", "image_input_tokens")],
                               mirror_q, atol=1e-6)
    rotation_result, _, _ = evaluate_symmetry(
        model, collector, rows, "rotation", operators, ranks, layers,
        torch.device("cpu"), 3)
    mirror_result, _, _ = evaluate_symmetry(
        model, collector, rows, "reflection", operators, ranks, layers,
        torch.device("cpu"), 3)
    assert rotation_result["image_input_tokens"]["operator_closure_error"] < 1e-6
    assert mirror_result["image_input_tokens"]["operator_closure_error"] < 1e-6


def test_collector_never_calls_function_or_decoder_paths():
    config = dict(model=dict(config=dict(
        input_resolution=[5, 5],
        encoder=dict(hidden_dim=8, num_layers=2, num_heads=2, mlp_dim=16, dropout=0.),
        function_mlp=dict(max_tokens=2, embedding_dim=8, hidden_dim=16),
        cross_attention=dict(num_shared_latents=3, num_heads=2, mlp_dim=16, dropout=0.),
    )))
    model = FunctionConditionedViTCrossAttentionModel(config).eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A non-image path was called")

    model.function_mlp.forward = forbidden
    model.image_to_shared.forward = forbidden
    model.shared_to_output.forward = forbidden
    collector = ViTLayerCollector(model)
    try:
        features = collector.encode(
            [np.zeros((5, 5), dtype=np.int64)], torch.device("cpu"), 1)
    finally:
        collector.close()
    assert set(features) == {
        "image_input_tokens", "image_block_01", "image_block_02", "image_final_norm"
    }


def test_reader_excludes_entire_fit_dihedral_orbit(tmp_path):
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[1, 3] = 4
    reflected = reflect_left_right(grid)
    table = pa.Table.from_pylist([
        dict(input=grid.tolist(), transformation_suite=["translate_up"]),
        dict(input=reflected.tolist(), transformation_suite=["translate_up"]),
    ])
    directory = tmp_path / "exp_setting_1" / "experiment_1"
    directory.mkdir(parents=True)
    pq.write_table(table, directory / "val.parquet")
    try:
        read_id_images(tmp_path, "val", 2, 42, {orbit_key(grid)})
    except ValueError as error:
        assert "No eligible ID images" in str(error)
    else:
        raise AssertionError("A reflected fit image leaked into evaluation")
