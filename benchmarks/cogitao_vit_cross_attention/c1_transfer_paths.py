"""Forward hooks and target-free prediction for the existing C1 architecture."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.analyze_c1_symmetries import resize_grids
from benchmarks.cogitao_vit_cross_attention.function_conditioned_model import (
    FunctionConditionedViTCrossAttentionModel,
)
from data.cogitao.cogitao import TASK_TO_ID

STAGE_ORDER = (
    "image_final_norm",
    "function_embeddings",
    "function_mlp",
    "shared_queries",
    "first_cross_attention",
    "second_cross_attention",
    "decoder_logits",
)


class ExecutionCollector:
    """Observe existing forward modules without implementing another forward path."""

    def __init__(self, model):
        self.values = {}
        self.handles = []
        for name, module in (
            ("image_final_norm", model.image_encoder),
            ("function_embeddings", model.function_embedding),
            ("function_mlp", model.function_mlp),
            ("decoder_logits", model.output_head),
        ):
            self.handles.append(module.register_forward_hook(self._output(name)))
        self.handles.append(
            model.image_to_shared.register_forward_pre_hook(self._queries)
        )
        for name, module in (
            ("first_cross_attention", model.image_to_shared),
            ("second_cross_attention", model.shared_to_output),
        ):
            self.handles.append(module.register_forward_hook(self._cross(name)))

    def _output(self, name):
        def hook(_module, _inputs, output):
            self.values[name] = output.detach()

        return hook

    def _queries(self, _module, inputs):
        self.values["shared_queries"] = inputs[0].detach()

    def _cross(self, name):
        def hook(_module, _inputs, output):
            self.values[name], self.values[name + "_attention"] = (
                v.detach() for v in output
            )

        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def token_swap(model, position, replacement):
    """Replace an embedding at one ordered slot; preserve the actual padding mask."""
    if not 0 <= position < model.max_function_tokens:
        raise ValueError("Invalid token slot")

    def hook(_module, _inputs, output):
        value = output.clone()
        value[:, position] = model.function_embedding.weight[TASK_TO_ID[replacement]]
        return value

    handle = model.function_embedding.register_forward_hook(hook, prepend=True)
    try:
        yield
    finally:
        handle.remove()


def predict_grid(model, grid_indices, suite):
    """No oracle target enters the forward path; the legacy target field is a dummy.

    Forward only echoes target_grid. It is required by the existing interface,
    so zeros are passed and discarded. Actual labels are used AFTER prediction.
    """
    if not suite or len(suite) > model.max_function_tokens:
        raise ValueError("Program exceeds token capacity or is empty")
    ids = [TASK_TO_ID[name] for name in suite]
    ids += [0] * (model.max_function_tokens - len(ids))
    tokens = torch.tensor([ids] * len(grid_indices), device=grid_indices.device)
    with torch.inference_mode():
        return model(
            dict(
                images=F.one_hot(grid_indices.long(), num_classes=10)
                .permute(0, 3, 1, 2)
                .float(),
                target_grid=torch.zeros_like(grid_indices),
                task_tokens=tokens,
                task_token_mask=tokens != 0,
            )
        )


def prediction_metrics(output, target, bins=15):
    """Use benchmark foreground-union accuracy; score probabilities after forward."""
    pred, logp = output["predictions"], output["output_log_probs"]
    if not bool(torch.isfinite(logp).all()):
        raise FloatingPointError("Non-finite prediction probabilities")
    # Reuse the canonical object-pixel calculation rather than maintaining a copy.
    object_accuracy = [
        float(
            FunctionConditionedViTCrossAttentionModel.compute_metrics(
                dict(predictions=pred[i : i + 1], target_grid=target[i : i + 1])
            )["object_per_pixel_accuracy"]
        )
        / 100
        for i in range(len(pred))
    ]
    correct = (pred == target).flatten(1)
    prob = logp.exp()
    confidence = prob.max(dim=1).values.flatten(1)
    nll = F.nll_loss(logp, target, reduction="none").flatten(1)
    entropy = -(prob * logp).sum(dim=1).flatten(1)
    truth = F.one_hot(target, num_classes=10).permute(0, 3, 1, 2)
    brier = ((prob - truth) ** 2).sum(dim=1).flatten(1)
    bin_id = (confidence * bins).long().clamp_max(bins - 1)
    ece = torch.zeros(len(pred), device=pred.device)
    bin_counts, bin_confidence, bin_correct = [], [], []
    for i in range(bins):
        mask = bin_id == i
        counts = mask.sum(dim=1)
        conf_sum = (confidence * mask).sum(dim=1)
        correct_sum = (correct * mask).sum(dim=1)
        ece += (conf_sum - correct_sum).abs() / correct.shape[1]
        bin_counts.append(counts.cpu().numpy())
        bin_confidence.append(conf_sum.cpu().numpy())
        bin_correct.append(correct_sum.cpu().numpy())
    values = dict(
        exact_accuracy=correct.all(dim=1).float(),
        cross_entropy=nll.mean(dim=1),
        nll=nll.sum(dim=1),
        confidence=confidence.mean(dim=1),
        entropy=entropy.mean(dim=1),
        brier=brier.mean(dim=1),
        example_ece=ece,
    )
    values = {name: value.cpu().numpy() for name, value in values.items()}
    values["object_accuracy"] = np.asarray(object_accuracy)
    for name, parts in (
        ("calibration_counts", bin_counts),
        ("calibration_confidence_sum", bin_confidence),
        ("calibration_correct_sum", bin_correct),
    ):
        values[name] = np.stack(parts, axis=1)
    return values


def execute(
    model, collector, grids, targets, suite, args, device, predicted_indices=None
):
    """Resize raw oracle states once; predicted grids are already at model resolution."""
    groups = defaultdict(list)
    for i, grid in enumerate(grids):
        groups[(grid.shape, targets[i].shape)].append(i)
    representations, metrics = {}, {}
    predictions = np.empty((len(grids), *model.input_resolution), dtype=np.int64)
    for indices in groups.values():
        for start in range(0, len(indices), args.batch_size):
            selection = indices[start : start + args.batch_size]
            source = (
                resize_grids(
                    [grids[i] for i in selection], model.input_resolution, device
                )
                if predicted_indices is None
                else torch.as_tensor(
                    predicted_indices[selection], dtype=torch.long, device=device
                ).detach()
            )
            target = resize_grids(
                [targets[i] for i in selection], model.input_resolution, device
            )
            collector.values.clear()
            output = predict_grid(model, source, suite)
            predictions[selection] = output["predictions"].detach().cpu().numpy()
            for name, value in prediction_metrics(output, target).items():
                if name not in metrics:
                    metrics[name] = np.empty(
                        (len(grids), *value.shape[1:]), dtype=value.dtype
                    )
                metrics[name][selection] = value
            for name, value in collector.values.items():
                value = value.float().cpu().numpy()
                if name not in representations:
                    representations[name] = np.empty(
                        (len(grids), *value.shape[1:]), dtype=np.float32
                    )
                representations[name][selection] = value
    return dict(
        metrics=metrics, representations=representations, predictions=predictions
    )


def summarize_behavior(result):
    values = result["metrics"]
    summary = {
        name: float(value.mean()) for name, value in values.items() if value.ndim == 1
    }
    counts = values["calibration_counts"].sum(axis=0)
    conf = values["calibration_confidence_sum"].sum(axis=0)
    correct = values["calibration_correct_sum"].sum(axis=0)
    summary["ece"] = float(np.abs(conf - correct).sum() / max(counts.sum(), 1))
    return summary
