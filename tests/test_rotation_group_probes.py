import json
import math
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as L
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import apply, infer_objects
from benchmarks.cogitao_vit_cross_attention.degree_rotation_model import DegreeRotationViT
from benchmarks.cogitao_vit_cross_attention.object_degree_rotation_data import ObjectAnchorDegreeRotationDataset
from benchmarks.cogitao_vit_cross_attention.rotation_compositions import predict
from benchmarks.cogitao_vit_cross_attention.rotation_group_probes import (
    GROUP_ANGLES, apply_operator, fit_operator, group_laws, latent_metrics, probe_fit_metrics,
)
from benchmarks.cogitao_vit_cross_attention.run_rotation_group_tests import (
    effect_programs, evaluate, extract_features, main, wait_for_training,
)
from src.train import load_config


def exact_rotations():
    result = {}
    for angle in GROUP_ANGLES:
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        result[angle] = torch.tensor([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=torch.float64)
    return result


def test_genuine_c24_representation_satisfies_all_laws():
    source = torch.randn(32, 2, dtype=torch.float64)
    laws = group_laws(exact_rotations(), source, {(15, 30)})
    assert len(laws["composition"]) == 36
    assert len(laws["full_c24_composition"]) == 576
    for rows in laws.values():
        for row in rows:
            assert row["matrix_relative_error"] < 1e-12
            assert row["action_relative_l2"] < 1e-12
    orders = {row["angle"]: row["order"] for row in laws["cycles"]}
    assert orders == {15: 24, 30: 12, 45: 8, 60: 6, 75: 24, 90: 4}


def test_noncommuting_and_collapsed_probes_do_not_pass_controls():
    source = torch.randn(40, 2, dtype=torch.float64)
    operators = exact_rotations()
    operators[15] = torch.tensor([[1., 1., 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    operators[30] = torch.diag(torch.tensor([2., 1., 1.], dtype=torch.float64))
    laws = group_laws(operators, source, set())
    row = next(row for row in laws["commutativity"] if (row["a"], row["b"]) == (15, 30))
    assert row["matrix_relative_error"] > 0.1
    collapsed = {angle: torch.diag(torch.tensor([0., 0., 1.], dtype=torch.float64)) for angle in GROUP_ANGLES}
    collapsed[0] = torch.eye(3, dtype=torch.float64)
    controls = group_laws(collapsed, source, set())
    assert all(row["matrix_relative_error"] == 0 for row in controls["composition"])
    assert all(row["matrix_relative_error"] > 1 for row in controls["cycles"])
    fit = probe_fit_metrics(collapsed[15], source, source, source.mean(0))
    assert fit["linear_rank"] == 0
    assert fit["r2_against_training_mean"] <= 0
    assert latent_metrics(torch.zeros(3, 2), torch.zeros(3, 2))["linear_cka"] is None


def test_affine_probe_recovers_known_operator_on_disjoint_sources():
    torch.manual_seed(10)
    original = exact_rotations()[45].clone()
    original[:2, -1] = torch.tensor([0.5, -0.2])
    source = torch.randn(128, 2, dtype=torch.float64)
    fitted = fit_operator(source, apply_operator(original, source), ridge=0)
    torch.testing.assert_close(fitted, original, rtol=1e-9, atol=1e-9)
    held = torch.randn(64, 2, dtype=torch.float64)
    scores = probe_fit_metrics(fitted, held, apply_operator(original, held), apply_operator(original, source).mean(0))
    assert scores["r2_against_training_mean"] > 0.999999


@pytest.fixture
def finished_run(tmp_path):
    config = load_config(Path("config/cogitao/degree_rotation_object_anchors.yaml"))
    config["model"]["config"]["input_resolution"] = [12, 12]
    config["model"]["config"]["encoder"].update(hidden_dim=8, num_heads=2, num_layers=1, mlp_dim=16)
    folder = tmp_path / "data/exp_setting_1/experiment_1"
    folder.mkdir(parents=True)
    for split, colors in (("train", (1, 2, 3, 4)), ("val", (5, 6)), ("test", (7, 8))):
        rows = []
        for color in colors:
            grid = np.zeros((7, 7), dtype=np.int64)
            grid[1:4, 2:5] = [[color, 0, 0], [color, color, 0], [color, color, color]]
            _, target = apply(infer_objects(grid), "rot90", grid.shape)
            rows.append({"input": grid.tolist(), "output": target.tolist(), "transformation_suite": ["rot90"]})
        pq.write_table(pa.Table.from_pylist(rows), folder / f"{split}.parquet")
        config["data"][split]["init_args"].update(root=str(tmp_path / "data"), image_size=[12, 12])
    run = tmp_path / "run"
    run.mkdir()
    model = DegreeRotationViT(config).eval()
    checkpoint = run / "best.ckpt"
    torch.save(model.state_dict(), run / "initial_weights.pt")
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": {"config": config}, "pytorch-lightning_version": L.__version__}, checkpoint)
    manifest = {split: ObjectAnchorDegreeRotationDataset(config["data"][split]["init_args"]).manifest()
                for split in ("train", "val", "test")}
    (run / "config.json").write_text(json.dumps(config))
    (run / "data_manifest.json").write_text(json.dumps(manifest))
    (run / "status.json").write_text(json.dumps({"status": "complete"}))
    (run / "report.json").write_text(json.dumps({"status": "complete", "best_checkpoint": str(checkpoint)}))
    return run, model


def test_numeric_query_matches_trained_angle_encoding_and_restores_hooks(finished_run):
    _, model = finished_run
    source = torch.randint(0, 10, (2, 12, 12))
    images = F.one_hot(source, 10).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        output = predict(model, images, torch.tensor([[2, 3], [2, 3]]))
        expected = F.adaptive_avg_pool2d(output["image_embeddings"].transpose(1, 2).reshape(2, 8, 12, 12), (2, 2)).flatten(1)
    actual = extract_features(model, source, (30, 45), batch_size=2, pool_grid=2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(model.task_embedding._forward_hooks) == 0
    assert extract_features(model, source, (345,), pool_grid=2).shape == (2, 32)
    assert not torch.allclose(extract_features(model, source, (), pool_grid=2), extract_features(model, source, (0,), pool_grid=2))


def test_post_training_pipeline_fits_only_train_and_saves_group_reports(finished_run):
    output, _ = finished_run
    destination = output / "group_tests"
    destination.mkdir()
    training = wait_for_training(output, destination, wait=False)
    args = SimpleNamespace(fit_sources=4, latent_dim=4, pool_grid=2, batch_size=4, ridge=0.01, device="cpu", log_wandb=False)
    evaluate(output, destination, training, args)
    report = json.loads((destination / "report.json").read_text())
    assert report["frozen_model"]
    assert report["source_counts"] == {"train": 4, "val": 2, "test": 2}
    assert report["fit_sources"] == 4
    assert len(report["extrapolated_angles"]) == 12
    assert report["angle_kinds"]["0"] == "untrained_numeric_angle"
    assert report["angle_kinds"]["120"] == "composition_effect_trained"
    canonical = report["canonical_programs"]
    seen = {tuple(pair) for pair in json.loads((output / "data_manifest.json").read_text())["train"]["seen_pairs"]}
    assert all(tuple(program) in seen for program in canonical.values() if len(program) == 2)
    for split in ("val", "test"):
        tables = report["splits"][split]["tables"]
        assert len(tables["full_c24_composition"]) == 576
        assert len(tables["latent_regrouping"]) == 55
        assert sum(row["pair_split"] == "held_out_pairs" for row in tables["composition"]) == 12
        assert (destination / f"{split}_latent_program_consistency.csv").exists()
    assert json.loads((destination / "status.json").read_text())["status"] == "complete"


def test_unfinished_or_failed_training_never_runs_group_tests(tmp_path):
    output = tmp_path / "run"
    destination = output / "group_tests"
    destination.mkdir(parents=True)
    (output / "status.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(RuntimeError, match="completed training"):
        wait_for_training(output, destination, wait=False)
    (output / "status.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RuntimeError, match="no group tests run"):
        wait_for_training(output, destination, wait=True)


def test_initialization_control_uses_saved_weights_and_same_protocol(finished_run):
    output, model = finished_run
    destination = output / "group_tests/initialization_control"
    destination.mkdir(parents=True)
    args = SimpleNamespace(fit_sources=4, latent_dim=4, pool_grid=2, batch_size=4, ridge=0.01, device="cpu", log_wandb=False)
    training = json.loads((output / "report.json").read_text())
    report, summary = evaluate(output, destination, training, args, initialization_control=True)
    assert report["checkpoint_kind"] == "random_initialization_control"
    assert report["checkpoint"].endswith("initial_weights.pt")
    assert summary["test"]["commutativity"]["laws"] == 30
    assert summary["test"]["seen_pairs/commutativity"]["laws"] == 18
    assert summary["test"]["held_out_pairs/commutativity"]["laws"] == 12


def test_cli_completes_both_analyses_and_comparison(finished_run, monkeypatch):
    output, _ = finished_run
    monkeypatch.setattr("sys.argv", ["group-tests", "--output", str(output), "--fit-sources", "4",
                                    "--latent-dim", "4", "--pool-grid", "2", "--batch-size", "4", "--device", "cpu"])
    main()
    destination = output / "group_tests"
    comparison = json.loads((destination / "comparison.json").read_text())
    assert set(comparison) == {"trained", "initialization_control", "trained_minus_initialization"}
    # In this fixture both checkpoints have identical weights, so the matched
    # analysis must give identical scores and a zero difference.
    assert comparison["trained_minus_initialization"]["test"]["composition"]["mean/matrix_relative_error"] == 0
    assert json.loads((destination / "status.json").read_text())["status"] == "complete"
