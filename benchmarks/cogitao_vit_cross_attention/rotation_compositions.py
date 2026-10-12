"""Ordered rotation programs with a fixed seen/held-out pair split."""

from __future__ import annotations

from itertools import product

import torch
from torch.nn import functional as F

from benchmarks.cogitao_vit_cross_attention.rotation_data import RotationDataset

ANGLES = (0, 90, 180, 270)
ATOMIC_PROGRAMS = tuple((a,) for a in ANGLES)
ALL_PAIRS = tuple(product(ANGLES, repeat=2))
# Two offsets of the cyclic angle table: every first angle, second angle, and
# summed angle has equal coverage on both sides of the split.
SEEN_PAIRS = tuple((a, b) for a, b in ALL_PAIRS if ((b - a) % 360) in (0, 90))
HELD_OUT_PAIRS = tuple(pair for pair in ALL_PAIRS if pair not in SEEN_PAIRS)


class RotationProgramDataset(RotationDataset):
    def __init__(self, config):
        selection = config.get("program_selection", "seen")
        if selection not in ("seen", "all"):
            raise ValueError("program_selection must be 'seen' or 'all'.")
        if config.get("split", "train") == "train" and selection != "seen":
            raise ValueError("Held-out programs must never enter training.")
        self.programs = ATOMIC_PROGRAMS + (
            SEEN_PAIRS if selection == "seen" else ALL_PAIRS
        )
        super().__init__(config)
        if self.angles != ANGLES:
            raise ValueError(
                "Composition experiment requires the ordered four-angle vocabulary."
            )
        self.audit.update(
            atomic_examples=len(self.sources) * len(ATOMIC_PROGRAMS),
            seen_composition_examples=len(self.sources) * len(SEEN_PAIRS),
            held_out_composition_examples=(
                len(self.sources) * len(HELD_OUT_PAIRS) if selection == "all" else 0
            ),
        )

    def __len__(self):
        return len(self.sources) * len(self.programs)

    def __getitem__(self, index):
        source, program_index = divmod(index, len(self.programs))
        program = self.programs[program_index]
        total = sum(program) % 360
        ids = [self.angles.index(a) + 1 for a in program]
        tokens = ids + [0] * (2 - len(ids))
        grid = self.sources[source].long()
        return {
            "images": F.one_hot(grid, 10).permute(2, 0, 1).float(),
            "target_grid": self.targets[source, self.angles.index(total)].long(),
            "intermediate_grid": self.targets[
                source, self.angles.index(program[0])
            ].long(),
            "task_tokens": torch.tensor(tokens),
            "task_token_mask": torch.tensor(
                [True] * len(ids) + [False] * (2 - len(ids))
            ),
            "program_index": program_index,
            "angle_degrees": total,
            "source_index": self.source_indices[source],
        }

    def manifest(self):
        return {
            **super().manifest(),
            "programs": [list(p) for p in self.programs],
            "seen_pairs": [list(p) for p in SEEN_PAIRS],
            "held_out_pairs": [list(p) for p in HELD_OUT_PAIRS],
            "program_selection": self.config.get("program_selection", "seen"),
        }


def program_name(program):
    return "_then_".join(map(str, program))


def program_groups(program):
    if len(program) == 1:
        return ["atomic"]
    category = "seen_pairs" if program in SEEN_PAIRS else "held_out_pairs"
    groups = [category]
    if 0 not in program:
        groups.append(category + "_without_identity")
    return groups


def predict(model, images, tokens):
    # Targets are deliberately absent from inference; the legacy model interface
    # echoes this dummy field and actual labels are used only by the scorer.
    return model(
        {
            "images": images,
            "target_grid": torch.zeros(
                images.shape[0],
                *images.shape[-2:],
                dtype=torch.long,
                device=images.device,
            ),
            "task_tokens": tokens,
            "task_token_mask": tokens.ne(0),
        }
    )


def score_paths(model, direct, summed, sequential, teacher_forced, target):
    metrics = {"direct_cross_entropy": F.nll_loss(direct["output_log_probs"], target)}
    for prefix, output in (
        ("direct", direct),
        ("summed_angle", summed),
        ("sequential", sequential),
        ("oracle_intermediate", teacher_forced),
    ):
        metrics.update(
            {
                f"{prefix}_{k}": v
                for k, v in model.compute_metrics(
                    {"predictions": output["predictions"], "target_grid": target}
                ).items()
            }
        )
    for prefix, alternative in (
        ("direct_vs_summed", summed),
        ("direct_vs_sequential", sequential),
    ):
        agreement = direct["predictions"].eq(alternative["predictions"])
        metrics[prefix + "_grid_agreement"] = (
            100 * agreement.flatten(1).all(1).float().mean()
        )
        metrics[prefix + "_pixel_agreement"] = 100 * agreement.float().mean()
    return metrics


@torch.inference_mode()
def evaluate_programs(model, loader):
    model.eval()
    device = next(model.parameters()).device
    dataset = loader.dataset
    by_program, groups = {}, {}
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        tokens = batch["task_tokens"]
        direct = predict(model, batch["images"], tokens)
        summed_tokens = torch.stack(
            (
                batch["angle_degrees"] // 90 + 1,
                torch.zeros_like(batch["angle_degrees"]),
            ),
            1,
        )
        summed = predict(model, batch["images"], summed_tokens)
        first_tokens = torch.stack((tokens[:, 0], torch.zeros_like(tokens[:, 0])), 1)
        first = predict(model, batch["images"], first_tokens)
        # An atomic row gets a subsequent identity action for this diagnostic.
        second_ids = torch.where(
            tokens[:, 1].ne(0), tokens[:, 1], torch.ones_like(tokens[:, 1])
        )
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
        for i, program in enumerate(dataset.programs):
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
            name = program_name(program)
            destinations = [
                by_program.setdefault(
                    name,
                    {
                        "samples": 0,
                        "program": list(program),
                        "summed_angle": sum(program) % 360,
                    },
                )
            ]
            destinations.extend(
                groups.setdefault(group, {"samples": 0})
                for group in program_groups(program)
            )
            for row in destinations:
                row["samples"] += count
                for key, value in values.items():
                    row[key] = row.get(key, 0.0) + float(value) * count
    for row in [*by_program.values(), *groups.values()]:
        for key in list(row):
            if key not in ("samples", "program", "summed_angle"):
                row[key] /= row["samples"]
    return {"by_program": by_program, "groups": groups}
