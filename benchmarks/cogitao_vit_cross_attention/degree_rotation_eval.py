"""Behavioral composition tests with explicit raster-resampling controls."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import (
    ATOMIC_ANGLES,
    rotate_grids,
)
from benchmarks.cogitao_vit_cross_attention.rotation_compositions import (
    predict,
    program_name,
    score_paths,
)


def groups_for(program, dataset):
    if len(program) == 1:
        return ["atomic"]
    group = "seen_pairs" if program in dataset.seen_pairs else "held_out_pairs"
    net_kind = (
        "net_angle_atomic_trained"
        if sum(program) in ATOMIC_ANGLES
        else "net_angle_beyond_atomic_training"
    )
    effect_kind = (
        "net_effect_seen"
        if sum(program) in dataset.split_metadata["training_effects"]
        else "net_effect_unseen"
    )
    return [group, f"{group}/{net_kind}", f"{group}/{effect_kind}"]


@torch.inference_mode()
def evaluate_degrees(model, loader):
    model.eval()
    device = next(model.parameters()).device
    by_program, groups = {}, {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tokens = batch["task_tokens"]
        is_pair = tokens[:, 1].ne(0)
        direct = predict(model, batch["images"], tokens)
        summed_tokens = torch.stack(
            (batch["angle_degrees"] // 15, torch.zeros_like(tokens[:, 0])), 1
        )
        summed = predict(model, batch["images"], summed_tokens)
        first_tokens = torch.stack((tokens[:, 0], torch.zeros_like(tokens[:, 0])), 1)
        first = predict(model, batch["images"], first_tokens)
        second_ids = tokens[:, 1].clamp_min(1)
        second_tokens = torch.stack((second_ids, torch.zeros_like(second_ids)), 1)
        sequential = predict(
            model,
            F.one_hot(first["predictions"], 10).permute(0, 3, 1, 2).float(),
            second_tokens,
        )
        teacher = predict(
            model,
            F.one_hot(batch["intermediate_grid"], 10).permute(0, 3, 1, 2).float(),
            second_tokens,
        )
        for output in (sequential, teacher):
            for key in ("predictions", "output_log_probs"):
                output[key][~is_pair] = direct[key][~is_pair]
        reversed_tokens = torch.where(is_pair[:, None], tokens.flip(1), tokens)
        reversed_output = predict(model, batch["images"], reversed_tokens)
        second_angles = tokens[:, 1].float() * 15
        if "raster_reference_grid" in batch:
            raster_reference = batch["raster_reference_grid"]
            raster_valid = batch["raster_reference_valid"]
        else:
            raster_reference = rotate_grids(batch["intermediate_grid"], second_angles)
            raster_valid = torch.ones_like(is_pair)
        for i, program in enumerate(loader.dataset.programs):
            mask = batch["program_index"] == i
            count = int(mask.sum())
            if not count:
                continue
            sliced = [
                {
                    k: v[mask]
                    for k, v in out.items()
                    if k in ("predictions", "output_log_probs")
                }
                for out in (direct, summed, sequential, teacher)
            ]
            values = score_paths(model, *sliced, batch["target_grid"][mask])
            agreement = direct["predictions"][mask].eq(
                reversed_output["predictions"][mask]
            )
            values["reverse_order_grid_agreement"] = (
                100 * agreement.flatten(1).all(1).float().mean()
            )
            values["reverse_order_pixel_agreement"] = 100 * agreement.float().mean()
            valid_mask = mask & raster_valid
            valid_fraction = valid_mask.sum().float() / count
            values["raster_reference_valid_percent"] = 100 * valid_fraction
            # Invalid object states cannot be rendered without clipping/overlap.
            # Give them zero credit, and expose the validity rate explicitly.
            reference_metrics = (
                model.compute_metrics({
                    "predictions": raster_reference[valid_mask],
                    "target_grid": batch["target_grid"][valid_mask],
                }) if bool(valid_mask.any()) else {
                    "grid_accuracy": 0.0, "per_pixel_accuracy": 0.0,
                    "object_per_pixel_accuracy": 0.0,
                }
            )
            values.update({
                f"raster_reference_{k}": v * valid_fraction
                for k, v in reference_metrics.items()
            })
            if "intermediate_identity_preserved" in batch:
                values["intermediate_identity_preserved_percent"] = (
                    100 * batch["intermediate_identity_preserved"][mask].float().mean()
                )
            row = by_program.setdefault(
                program_name(program),
                {
                    "program": list(program),
                    "summed_angle": sum(program),
                    "net_angle_atomic_trained": sum(program) in ATOMIC_ANGLES,
                    "net_effect_seen_in_training": sum(program)
                    in loader.dataset.split_metadata["training_effects"],
                    "samples": 0,
                },
            )
            destinations = [row] + [
                groups.setdefault(group, {"samples": 0})
                for group in groups_for(program, loader.dataset)
            ]
            for result in destinations:
                result["samples"] += count
                for key, value in values.items():
                    result[key] = result.get(key, 0.0) + float(value) * count
            if len(program) == 2 and "intermediate_identity_preserved" in batch:
                raster_exact = raster_valid & raster_reference.eq(
                    batch["target_grid"]
                ).flatten(1).all(1)
                stable = batch["intermediate_identity_preserved"]
                controls = {
                    "raster_exact": raster_exact,
                    "raster_mismatch": ~raster_exact,
                    "identity_preserved": stable,
                    "identity_changed": ~stable,
                    "raster_exact_and_identity_preserved": raster_exact & stable,
                }
                pair_group = groups_for(program, loader.dataset)[0]
                for control, selected in controls.items():
                    selected = selected & mask
                    selected_count = int(selected.sum())
                    if not selected_count:
                        continue
                    controlled = score_paths(
                        model,
                        *[{
                            k: v[selected] for k, v in out.items()
                            if k in ("predictions", "output_log_probs")
                        } for out in (direct, summed, sequential, teacher)],
                        batch["target_grid"][selected],
                    )
                    result = groups.setdefault(f"{pair_group}/{control}", {"samples": 0})
                    result["samples"] += selected_count
                    for key, value in controlled.items():
                        result[key] = result.get(key, 0.0) + float(value) * selected_count
    for row in [*by_program.values(), *groups.values()]:
        for key in list(row):
            if key not in (
                "samples",
                "program",
                "summed_angle",
                "net_angle_atomic_trained",
                "net_effect_seen_in_training",
            ):
                row[key] /= row["samples"]
    return {"by_program": by_program, "groups": groups}
