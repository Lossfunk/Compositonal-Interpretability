from pathlib import Path

import torch

from benchmarks.cogitao_slot_attention.model import (
    TaskConditionedSlotAttentionModel,
)
from benchmarks.cogitao_slot_attention.run import build_train_arguments
from src.train import apply_override, load_config


def _config() -> dict:
    return {
        "model": {
            "class_path": "benchmarks.cogitao_slot_attention.model.TaskConditionedSlotAttentionModel",
            "config": {
                "input_resolution": [32, 32],
                "num_slots": 3,
                "slot_dim": 16,
                "num_iterations": 2,
                "slot_mlp_hidden_dim": 32,
                "image_range": "unit",
                "validation_split_names": ["val"],
                "fail_on_nonfinite": True,
                "encoder": {
                    "patch_size": 8,
                    "hidden_dim": 48,
                    "num_layers": 2,
                    "num_heads": 6,
                    "mlp_dim": 64,
                    "output_dim": 16,
                    "bottleneck_hidden_dim": 32,
                },
                "decoder": {
                    "channels": [16, 16],
                    "kernel_size": 5,
                    "output_activation": "none",
                },
                "eval_plots": {"enabled": []},
                "loss_weights": {"reconstruction_loss": 1.0},
            },
        },
        "trainer": {
            "optimizer": {
                "type": "Adam",
                "config": {"lr": 4.0e-4},
                "schedule": {
                    "warmup_steps": 10,
                    "decay_steps": 100,
                    "decay_rate": 0.5,
                },
            }
        },
    }


def test_launcher_uses_native_grid_and_paper_backbone():
    arguments = build_train_arguments(1, 1, [])
    config = load_config(Path(arguments[1]))
    for index, argument in enumerate(arguments):
        if argument == "--override":
            apply_override(config, arguments[index + 1])

    model = config["model"]["config"]
    encoder = model["encoder"]
    assert model["input_resolution"] == [20, 20]
    assert encoder["patch_size"] == 1
    assert encoder["hidden_dim"] == 128
    assert encoder["num_layers"] == 6
    assert encoder["num_heads"] == 4
    assert encoder["mlp_dim"] == 512
    assert model["decoder"]["channels"] == [64, 64]
    for split in ("train", "val", "val_ood", "test", "test_ood"):
        assert config["data"][split]["init_args"]["image_size"] == [20, 20]


def test_task_tokens_condition_slot_attention_and_predict_categorical_grid():
    torch.manual_seed(0)
    model = TaskConditionedSlotAttentionModel(_config()).eval()
    input_indices = torch.randint(0, 10, (2, 32, 32))
    images = (
        torch.nn.functional.one_hot(input_indices, num_classes=10)
        .permute(0, 3, 1, 2)
        .float()
    )
    batch = {
        "images": images,
        "target_grid": torch.randint(0, 10, (2, 32, 32)),
        "task_tokens": torch.tensor([[16, 18, 0], [11, 0, 0]]),
        "task_token_mask": torch.tensor([[True, True, False], [True, False, False]]),
    }

    output = model(batch)
    assert output["output_log_probs"].shape == (2, 10, 32, 32)
    assert output["predictions"].shape == (2, 32, 32)
    assert output["masks"].shape == (2, 3, 1, 32, 32)
    assert output["task_attention"].shape == (2, 3, 3)
    torch.testing.assert_close(
        output["task_attention"][1, :, 1:],
        torch.zeros(3, 2),
    )
    torch.testing.assert_close(
        output["output_log_probs"].exp().sum(dim=1),
        torch.ones(2, 32, 32),
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    losses = model.compute_losses(output)
    assert torch.isfinite(losses["cross_entropy_loss"])
    metrics = model.compute_metrics(output, batch)
    assert set(metrics) == {
        "grid_accuracy",
        "per_pixel_accuracy",
        "object_per_pixel_accuracy",
    }
    assert all(0.0 <= float(value) <= 100.0 for value in metrics.values())

    losses["cross_entropy_loss"].backward()


def test_paper_metrics_use_exact_grid_and_ignore_shared_background():
    model = TaskConditionedSlotAttentionModel(_config())
    target = torch.tensor(
        [
            [[0, 1], [0, 2]],
            [[0, 1], [0, 2]],
        ]
    )
    predictions = torch.tensor(
        [
            [[0, 1], [0, 2]],
            [[0, 1], [3, 4]],
        ]
    )
    metrics = model.compute_metrics(
        {"predictions": predictions, "target_grid": target}, {}
    )
    torch.testing.assert_close(metrics["grid_accuracy"], torch.tensor(50.0))
    torch.testing.assert_close(metrics["per_pixel_accuracy"], torch.tensor(75.0))
    torch.testing.assert_close(
        metrics["object_per_pixel_accuracy"], torch.tensor(200.0 / 3.0)
    )


def test_split_output_pipeline_caches_input_target_and_prediction():
    model = TaskConditionedSlotAttentionModel(_config())
    input_grid = torch.tensor([[[0, 1], [2, 3]]])
    images = (
        torch.nn.functional.one_hot(input_grid, num_classes=10)
        .permute(0, 3, 1, 2)
        .float()
    )
    target = torch.tensor([[[0, 1], [2, 4]]])
    prediction = torch.tensor([[[0, 1], [2, 3]]])
    batch = {"metadata": {"transformation_suite": ["rot90 -> translate_up"]}}
    output = {
        "images": images,
        "target_grid": target,
        "predictions": prediction,
    }

    model._cache_prediction_outputs("val", "ood", batch, output)

    sample = model._prediction_samples["val/ood"][0]
    torch.testing.assert_close(sample["input"], input_grid[0].to(torch.uint8))
    torch.testing.assert_close(sample["target"], target[0].to(torch.uint8))
    torch.testing.assert_close(sample["prediction"], prediction[0].to(torch.uint8))
    assert sample["task_suite"] == "rot90 -> translate_up"
    assert sample["grid_correct"] is False
    assert sample["per_pixel_accuracy"] == 75.0


def test_local_output_pipeline_overwrites_stable_triptychs(tmp_path):
    config = _config()
    config["logging"] = {"save_dir": str(tmp_path)}
    model = TaskConditionedSlotAttentionModel(config)
    sample = {
        "input": torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8),
        "target": torch.tensor([[0, 1], [2, 4]], dtype=torch.uint8),
        "prediction": torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8),
        "task_suite": "rot90",
        "grid_correct": False,
        "per_pixel_accuracy": 75.0,
    }
    model._prediction_samples = {"val/ood": [sample]}

    model._write_local_prediction_outputs("val", require_trainer=False)
    path = tmp_path / "qualitative_outputs" / "val" / "ood" / "sample_00.png"
    first_mtime = path.stat().st_mtime_ns
    model._write_local_prediction_outputs("val", require_trainer=False)

    assert path.is_file()
    assert path.stat().st_mtime_ns >= first_mtime
    assert [item.name for item in path.parent.iterdir()] == ["sample_00.png"]
