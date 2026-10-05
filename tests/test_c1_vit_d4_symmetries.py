"""Checks for the full image-only C1 ViT D4 protocol."""

from pathlib import Path

import numpy as np

from benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries import (
    D4_ELEMENTS,
    align_d4_tokens,
    cayley_table,
    d4_transform,
    default_checkpoint,
    evaluate_group_law,
)


def test_inverse_alignment_undoes_every_d4_spatial_action():
    tokens = np.arange(2 * 25 * 3).reshape(2, 25, 3)
    images = tokens.reshape(2, 5, 5, 3)
    for element in D4_ELEMENTS:
        transformed = np.stack([
            d4_transform(image, element) for image in images
        ]).reshape(2, 25, 3)
        np.testing.assert_array_equal(
            align_d4_tokens(transformed, (5, 5), element), images)


def test_cayley_table_contains_all_d4_products_and_pure_identities():
    table = cayley_table()
    assert len(table) == 64
    assert set(table.values()) == set(D4_ELEMENTS)
    assert table[("r90", "r90")] == "r180"
    assert table[("r180", "r180")] == "e"
    assert table[("m", "m")] == "e"

    value = "e"
    for _ in range(4):
        value = table[(value, "r90")]
    assert value == "e"


def test_group_law_evaluator_accepts_an_exact_regular_representation():
    names = list(D4_ELEMENTS)
    index = {name: position for position, name in enumerate(names)}
    table = cayley_table()
    operators = {}
    for element in names:
        operator = np.zeros((len(names), len(names)))
        for source in names:
            operator[index[source], index[table[(source, element)]]] = 1.0
        operators[(element, "layer")] = operator

    results, rows, returned_table = evaluate_group_law(
        operators, ["layer"], {"layer": np.eye(len(names))})

    assert returned_table == table
    assert len(rows) == 64
    assert results["layer"]["products"] == 64
    assert results["layer"]["operator_error"]["mean"] < 1e-12
    assert results["layer"]["heldout_data_error"]["mean"] < 1e-12


def test_default_checkpoint_is_experiment_specific():
    assert default_checkpoint(4) == Path(
        "checkpoints/cogitao_vit_cross_attention/"
        "cogitao_c1_experiment_4_vit6_function_mlp_cross_attention/"
        "run_000/last.ckpt"
    )
