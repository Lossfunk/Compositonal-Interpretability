"""Scientific invariants, target isolation and real-grid feedback; no checkpoints."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments import (
    SplitRows,
    collect_atomic_fit,
    grid_key,
)
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import apply_affine
from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
    InvalidOracle,
    oracle_sequence,
)
from benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer import (
    action_relations,
    behavioral_conditions,
    fit_action_subspaces,
    sample_transfer,
    transfer_states,
)
from benchmarks.cogitao_vit_cross_attention.c1_transfer_metrics import (
    bootstrap_correlation,
    diagnostic_evidence,
    distribution_divergence,
    principal_angle_defect,
    projection_metrics,
    random_subspace,
    residual_pca,
    similarity,
)
from benchmarks.cogitao_vit_cross_attention.c1_transfer_paths import (
    ExecutionCollector,
    STAGE_ORDER,
    predict_grid,
    prediction_metrics,
    token_swap,
)
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from data.cogitao.cogitao import TASK_TO_ID


def grid():
    x = np.zeros((12, 12), dtype=np.int64)
    x[4:7, 4:6] = [[3, 3], [3, 0], [3, 3]]
    return x


def model():
    return FunctionConditionedViTCrossAttentionModel(
        dict(
            model=dict(
                config=dict(
                    input_resolution=[12, 12],
                    encoder=dict(hidden_dim=8, num_layers=1, num_heads=2, mlp_dim=16),
                    function_mlp=dict(embedding_dim=8, hidden_dim=16),
                    cross_attention=dict(num_shared_latents=3, num_heads=2, mlp_dim=16),
                )
            )
        )
    ).eval()


def dataset(tmp_path, split, rows):
    p = tmp_path / "exp_setting_1" / "experiment_1"
    p.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), p / f"{split}.parquet")
    return SplitRows(tmp_path, 1, split)


def row(x, suite):
    return dict(
        input=x.tolist(),
        output=oracle_sequence(x, suite).tolist(),
        transformation_suite=list(suite),
    )


def test_program_order_is_f_after_g():
    x = grid()
    states = transfer_states(x, "change_shape_color", "pad_right")
    np.testing.assert_array_equal(
        states["fgx"], oracle_sequence(x, ("pad_right", "change_shape_color"))
    )
    assert not np.array_equal(states["fgx"], states["gfx"])


def test_oracle_intermediate_and_direct_have_matched_target_and_real_predicted_feedback():
    x = grid()
    states = transfer_states(x, "rot90", "translate_up")

    class Spy:
        input_resolution = (12, 12)
        max_function_tokens = 2
        calls = []

        def __call__(self, batch):
            assert not batch["images"].requires_grad
            assert not bool(batch["target_grid"].any())
            self.calls.append({k: v.clone() for k, v in batch.items()})
            logits = torch.zeros((len(batch["images"]), 10, 12, 12))
            logits[:, 7] = (
                1  # Actual model predictions intentionally differ from true gx.
            )
            return dict(
                predictions=logits.argmax(1), output_log_probs=logits.log_softmax(1)
            )

    spy = Spy()
    conditions = behavioral_conditions(
        spy,
        SimpleNamespace(values={}),
        [dict(states=states)],
        "rot90",
        "translate_up",
        SimpleNamespace(batch_size=1),
        torch.device("cpu"),
    )
    assert spy.calls[4]["task_tokens"].tolist() == [
        [TASK_TO_ID["translate_up"], TASK_TO_ID["rot90"]]
    ]
    np.testing.assert_array_equal(
        spy.calls[1]["images"].argmax(1)[0].numpy(), states["gx"]
    )
    np.testing.assert_array_equal(
        spy.calls[3]["images"].argmax(1).numpy(), conditions["C_first"]["predictions"]
    )
    assert not np.array_equal(spy.calls[3]["images"].argmax(1)[0].numpy(), states["gx"])
    assert conditions["B"]["metrics"]["nll"][0] == pytest.approx(
        conditions["D"]["metrics"]["nll"][0]
    )


def test_invalid_primary_oracle_states_excluded(tmp_path):
    x = np.zeros((4, 4), dtype=np.int64)
    x[0, 1] = 3
    data = dataset(
        tmp_path,
        "val_ood",
        [
            dict(
                input=x.tolist(),
                output=x.tolist(),
                transformation_suite=["translate_up", "rot90"],
            )
        ],
    )
    selected, coverage = sample_transfer(
        data,
        "val_ood",
        "rot90",
        "translate_up",
        SimpleNamespace(seed=1, eval_images_per_pair=2, output_dir=tmp_path),
        set(),
    )
    assert selected == []
    assert coverage["invalid:out_of_bounds"] == 1


def test_reverse_invalid_does_not_exclude_primary(tmp_path):
    from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import (
        oracle_states,
    )

    x = np.zeros((10, 10), dtype=np.int64)
    x[:3, :2] = [[3, 3], [3, 0], [3, 3]]
    f, g = "pad_right", "fill_holes_different_color"
    with pytest.raises(InvalidOracle):
        oracle_states(x, f, g)
    data = dataset(tmp_path, "val_ood", [row(x, (g, f))])
    selected, coverage = sample_transfer(
        data,
        "val_ood",
        f,
        g,
        SimpleNamespace(seed=1, eval_images_per_pair=2, output_dir=tmp_path),
        set(),
    )
    assert len(selected) == 1 and "gfx" not in selected[0]["states"]
    assert coverage["reverse_valid"] == 0


def test_atomic_fit_filters_out_composition_rows(tmp_path):
    x = grid()
    y = np.roll(x, 1, axis=1)
    data = dataset(
        tmp_path,
        "train",
        [
            row(x, ("rot90",)),
            row(y, ("rot90",)),
            row(x, ("translate_up", "rot90")),
            row(y, ("translate_up", "rot90")),
        ],
    )
    selected, _, _ = collect_atomic_fit(
        data,
        ["rot90"],
        SimpleNamespace(seed=1, fit_images_per_function=4, output_dir=tmp_path),
    )
    assert all(r["suite"] == ("rot90",) for r in selected["rot90"])
    assert {r["index"] for r in selected["rot90"]} == {0, 1}
    from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_experiments import (
        fit_probes,
    )

    with pytest.raises(ValueError, match="only on the train split"):
        fit_probes(
            None,
            None,
            SimpleNamespace(path=tmp_path / "val_ood.parquet"),
            [],
            None,
            None,
            None,
        )
    with pytest.raises(ValueError, match="only be fitted"):
        fit_action_subspaces(
            None,
            None,
            SimpleNamespace(path=tmp_path / "val_ood.parquet"),
            [],
            None,
            None,
            {},
            {},
        )


@pytest.mark.parametrize("name", ["x", "fx", "gx", "fgx", "gfx"])
def test_fit_overlap_excludes_every_oracle_state(tmp_path, name):
    x = grid()
    states = transfer_states(x, "rot90", "translate_up")
    data = dataset(tmp_path, "test_ood", [row(x, ("translate_up", "rot90"))])
    selected, coverage = sample_transfer(
        data,
        "test_ood",
        "rot90",
        "translate_up",
        SimpleNamespace(seed=1, eval_images_per_pair=2, output_dir=tmp_path),
        {grid_key(states[name])},
    )
    assert selected == [] and coverage["fit_state_overlap"] == 1


def test_resized_fit_overlap_excluded(tmp_path):
    x = grid()
    m = model()
    data = dataset(tmp_path, "test_ood", [row(x, ("translate_up", "rot90"))])
    selected, coverage = sample_transfer(
        data,
        "test_ood",
        "rot90",
        "translate_up",
        SimpleNamespace(seed=1, eval_images_per_pair=2, output_dir=tmp_path),
        set(),
        m,
        {grid_key(x)},
    )
    assert not selected and coverage["resized_fit_state_overlap"] == 1


def test_pca_projector_orthogonal_and_projection_bounded():
    rng = np.random.default_rng(4)
    value = rng.normal(size=(32, 8))
    u, _, _ = residual_pca(value)
    u = u[:, :4]
    p = u @ u.T
    np.testing.assert_allclose(u.T @ u, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(p @ p, p, atol=1e-12)
    np.testing.assert_allclose(p, p.T, atol=1e-12)
    error, energy = projection_metrics(np.vstack([value, np.zeros((1, 8))]), u)
    assert np.all((0 <= error) & (error <= 1)) and np.all((0 <= energy) & (energy <= 1))
    assert error[-1] == energy[-1] == 0
    assert principal_angle_defect(u, u) == pytest.approx(0, abs=1e-12)
    assert principal_angle_defect(u, u[:, :2]) is None
    random = random_subspace(8, 4, rng)
    np.testing.assert_allclose(random.T @ random, np.eye(4), atol=1e-12)


def test_commutator_omitted_for_noncommuting_oracle():
    states = transfer_states(grid(), "change_shape_color", "pad_right")
    z = {name: np.array([float(value.sum()), 1.0]) for name, value in states.items()}
    defect, _ = action_relations(states, z, np.eye(3), np.eye(3), 0.05)
    assert defect is None


def test_c1_1_commuting_examples_have_equal_endpoints():
    states = transfer_states(grid(), "rot90", "translate_up")
    np.testing.assert_array_equal(states["fgx"], states["gfx"])


def test_hooks_target_isolation_and_token_swap_equivalence():
    m = model()
    collector = ExecutionCollector(m)
    source = torch.as_tensor(grid()[None])
    try:
        output = predict_grid(m, source, ("translate_up", "rot90"))
        assert set(STAGE_ORDER) <= set(collector.values)
        assert not bool(output["target_grid"].any())
        with token_swap(m, 0, "translate_up"):
            swapped = predict_grid(m, source, ("mirror_horizontal", "rot90"))
            captured = collector.values["function_embeddings"][:, 0]
            torch.testing.assert_close(
                captured, m.function_embedding.weight[TASK_TO_ID["translate_up"]][None]
            )
        torch.testing.assert_close(output["output_logits"], swapped["output_logits"])
        # The existing model echoes targets, but no representation or prediction reads them.
        tokens = torch.tensor([[TASK_TO_ID["translate_up"], TASK_TO_ID["rot90"]]])
        common = dict(
            images=F.one_hot(source, 10).permute(0, 3, 1, 2).float(),
            task_tokens=tokens,
            task_token_mask=tokens != 0,
        )
        with torch.inference_mode():
            changed = m(dict(common, target_grid=torch.full_like(source, 9)))
        torch.testing.assert_close(changed["output_logits"], output["output_logits"])
    finally:
        collector.close()
    assert not m.image_to_shared._forward_hooks


def test_foreground_union_and_probability_metrics():
    logits = torch.zeros((1, 10, 2, 2))
    logits[:, 0] = 2
    logits[0, 3, 0, 0] = 4
    logits[0, 2, 1, 1] = 4
    output = dict(predictions=logits.argmax(1), output_log_probs=logits.log_softmax(1))
    target = torch.tensor([[[3, 0], [0, 0]]])
    scores = prediction_metrics(output, target)
    assert scores["object_accuracy"][0] == pytest.approx(0.5)
    assert scores["nll"][0] == pytest.approx(4 * scores["cross_entropy"][0])
    assert scores["brier"][0] > 0


def test_similarity_handles_constant_and_equal_representations():
    x = np.ones((4, 2, 3))
    per, scalar = similarity(x, x)
    assert scalar["cka"] is None
    assert np.max(per["normalized_l2"]) == 0
    divergence = distribution_divergence(x, x, logits=True)
    assert np.max(divergence["js"]) == 0
    y = np.random.default_rng(4).normal(size=(4, 2, 3))
    _, scalar = similarity(y, y)
    assert scalar["cka"] == pytest.approx(1.0)


def test_taxonomy_exposes_thresholds_and_continuous_evidence():
    b = dict(
        A_object_accuracy=0.99,
        B_object_accuracy=0.95,
        C_object_accuracy=0.5,
        D_object_accuracy=0.4,
        C_first_object_accuracy=0.8,
    )
    result = diagnostic_evidence(
        b, dict(context_error=0.02, transfer_gap=0.001), 0.9, 0.7, 0.1, 0.1
    )
    assert "Type B" in result["diagnosis"] and "Type C" in result["diagnosis"]
    assert result["provisional"] and result["good_threshold"] == 0.9
    assert result["oracle_to_direct_drop"] == pytest.approx(0.55)


def test_cluster_bootstrap_cannot_invent_checkpoint_replicates():
    result = bootstrap_correlation([1, 2, 3], [3, 2, 1], 1, 20, clusters=["same"] * 3)
    assert result["pairs"] == 3 and result["checkpoint_clusters"] == 1
    assert result["pearson_ci_low"] is None


def test_affine_composition_preserves_row_vector_program_order():
    ag = np.eye(3)
    ag[:-1, :-1] = [[1.0, 1.0], [0.0, 1.0]]
    af = np.eye(3)
    af[-1, :-1] = [2.0, 3.0]
    x = np.array([[1.0, 2.0]])
    expected = apply_affine(apply_affine(x, ag), af)
    homogeneous = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    np.testing.assert_allclose(expected, (homogeneous @ ag @ af)[:, :-1])
    assert not np.allclose(expected, apply_affine(apply_affine(x, af), ag))


def test_early_trajectory_checkpoint_uses_existing_loader(tmp_path):
    from benchmarks.cogitao_vit_cross_attention.c1_vit_d4_symmetries import (
        load_c1_model,
    )

    m = model()
    config = dict(
        m.config, data=dict(train=dict(init_args=dict(setting=1, experiment=1)))
    )
    config["model"]["class_path"] = (
        "benchmarks.cogitao_vit_cross_attention.model.FunctionConditionedViTCrossAttentionModel"
    )
    path = tmp_path / "early.ckpt"
    torch.save(
        dict(
            global_step=100,
            hyper_parameters=dict(config=config),
            state_dict=m.state_dict(),
        ),
        path,
    )
    with pytest.raises(ValueError, match="only 100 training steps"):
        load_c1_model(path, 1, torch.device("cpu"))
    loaded, step = load_c1_model(path, 1, torch.device("cpu"), minimum_global_step=0)
    assert step == 100 and not loaded.training


def test_state_sufficiency_strata_keep_matched_behavior_and_count():
    from benchmarks.cogitao_vit_cross_attention.c1_compositional_transfer import (
        state_sufficiency_strata,
    )

    saved = dict(oracle_intermediate_reinference_target_match=np.array([1, 0, 1]))
    for label in ("A", "B", "C_first", "C", "D"):
        saved[label + "__object_accuracy"] = np.array([1.0, 0.0, 0.8])
        for metric in (
            "calibration_counts",
            "calibration_confidence_sum",
            "calibration_correct_sum",
        ):
            saved[label + "__" + metric] = np.ones((3, 2))
    for metric in ("context_error", "transfer_gap", "composition_error"):
        saved["image_final_norm__" + metric] = np.array([0.1, 0.9, 0.1])
    rows = state_sufficiency_strata(saved, dict(sample_count=3))
    assert [r["sample_count"] for r in rows] == [2, 1]
    assert rows[0]["B_object_accuracy"] == pytest.approx(0.9)
    assert rows[0]["context_error"] == pytest.approx(0.1)
    assert rows[1]["B_object_accuracy"] == 0


def test_atomic_weakness_is_not_classified_as_structural_transfer_failure():
    behavior = dict(
        A_object_accuracy=0.2,
        B_object_accuracy=0.2,
        C_object_accuracy=0.2,
        D_object_accuracy=0.2,
        C_first_object_accuracy=0.2,
    )
    result = diagnostic_evidence(
        behavior, dict(context_error=0.8, transfer_gap=0.3), 0.9, 0.7, 0.1, 0.1
    )
    assert result["diagnosis"] == "inconclusive"
