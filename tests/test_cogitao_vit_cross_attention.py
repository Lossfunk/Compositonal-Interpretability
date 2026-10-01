from pathlib import Path

import torch
import yaml

from benchmarks.cogitao_vit_cross_attention.model import (
    DirectPatchTransformationModel,
)
from benchmarks.cogitao_vit_cross_attention.run import build_train_arguments
from src.train import apply_override, load_config


def _small_config(patch_size: int = 1) -> dict:
    return {
        "model": {
            "class_path": (
                "benchmarks.cogitao_vit_cross_attention.model."
                "DirectPatchTransformationModel"
            ),
            "config": {
                "input_resolution": [4, 4],
                "validation_split_names": ["val", "val_ood"],
                "test_split_names": ["test", "test_ood"],
                "fail_on_nonfinite": True,
                "patch_embedding": {
                    "patch_size": patch_size,
                    "hidden_dim": 16,
                },
                "decoder": {
                    "num_layers": 2,
                    "num_heads": 4,
                    "mlp_dim": 32,
                    "dropout": 0.0,
                },
            },
        },
        "trainer": {
            "optimizer": {
                "type": "Adam",
                "config": {"lr": 4.0e-4},
            }
        },
    }


def test_c1_launcher_configures_direct_patch_decoder_without_task_tokens():
    arguments = build_train_arguments(1, [])
    config = load_config(Path(arguments[1]))
    for index, argument in enumerate(arguments):
        if argument == "--override":
            apply_override(config, arguments[index + 1])

    model = config["model"]
    model_config = model["config"]
    assert model["class_path"].endswith("DirectPatchTransformationModel")
    assert model_config["input_resolution"] == [20, 20]
    assert model_config["patch_embedding"] == {
        "patch_size": 1,
        "hidden_dim": 128,
    }
    assert model_config["decoder"]["num_layers"] == 6
    assert model_config["decoder"]["num_heads"] == 4
    assert model_config["decoder"]["mlp_dim"] == 512
    assert model_config["decoder"]["dropout"] == 0.0
    assert "function_mlp" not in model_config
    assert "cross_attention" not in model_config
    assert config["logging"]["project"] == "cogitao-compgen-slot-attention"
    assert config["data"]["train"]["init_args"]["setting"] == 1
    assert config["data"]["train"]["init_args"]["return_task_tokens"] is False
    assert "max_task_tokens" not in config["data"]["train"]["init_args"]


def test_direct_patch_decode_output_path_has_no_function_conditioning():
    torch.manual_seed(0)
    model = DirectPatchTransformationModel(_small_config()).eval()
    indices = torch.randint(0, 10, (2, 4, 4))
    images = (
        torch.nn.functional.one_hot(indices, num_classes=10)
        .permute(0, 3, 1, 2)
        .float()
    )
    batch = {
        "images": images,
        "target_grid": torch.randint(0, 10, (2, 4, 4)),
    }

    output = model(batch)
    assert output["patch_embeddings"].shape == (2, 16, 16)
    assert output["decoded_tokens"].shape == (2, 16, 16)
    assert output["output_logits"].shape == (2, 10, 4, 4)
    assert output["predictions"].shape == (2, 4, 4)
    assert not any(
        keyword in name
        for name, _ in model.named_modules()
        for keyword in ("function", "cross_attention", "slot_attention")
    )

    with_irrelevant_task_metadata = {
        **batch,
        "task_tokens": torch.tensor([[16, 18], [18, 16]]),
        "task_token_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    task_output = model(with_irrelevant_task_metadata)
    torch.testing.assert_close(output["output_logits"], task_output["output_logits"])

    loss = torch.nn.functional.nll_loss(
        output["output_log_probs"], output["target_grid"]
    )
    loss.backward()
    assert model.patch_embedding.weight.grad is not None
    assert model.decoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.output_head[-1].weight.grad is not None


def test_larger_patches_decode_back_to_full_grid():
    model = DirectPatchTransformationModel(_small_config(patch_size=2)).eval()
    indices = torch.randint(0, 10, (2, 4, 4))
    images = (
        torch.nn.functional.one_hot(indices, num_classes=10)
        .permute(0, 3, 1, 2)
        .float()
    )
    output = model(
        {
            "images": images,
            "target_grid": torch.randint(0, 10, (2, 4, 4)),
        }
    )

    assert output["patch_embeddings"].shape == (2, 4, 16)
    assert output["decoded_tokens"].shape == (2, 4, 16)
    assert output["output_logits"].shape == (2, 10, 4, 4)


def test_new_c1_sweep_covers_five_experiments():
    path = Path("benchmarks/cogitao_vit_cross_attention/sweep_setting1.yaml")
    sweep = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert sweep["name"] == "cogitao-compgen-c1-direct-patch-decoder"
    assert sweep["method"] == "grid"
    assert sweep["program"] == "benchmarks.cogitao_vit_cross_attention.run"
    assert sweep["parameters"]["experiment"]["values"] == [1, 2, 3, 4, 5]
