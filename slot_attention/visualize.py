from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "data"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from train import build_dataset, import_from_path, load_config  # noqa: E402

from slot_attention.metrics import segmentation_ari  # noqa: E402


DEFAULT_CONFIG = Path("config/slot_attention/clevr_2d_slot_attention.yaml")
DEFAULT_CHECKPOINT = Path("checkpoints/slot_attention/clevr_2d_slot_attention/run_000")
DEFAULT_OUTPUT = Path("artifacts/clevr_2d_slot_attention")

SHAPE_MARKERS = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
    "pentagon": "p",
    "star": "*",
}


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_checkpoint(path: Path) -> Path:
    """Pick the best checkpoint in a run directory, or accept a file directly."""
    path = repo_path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    candidates = [item for item in path.glob("*.ckpt") if item.name != "last.ckpt"]
    scored: list[tuple[float, Path]] = []
    for candidate in candidates:
        matches = re.findall(
            r"ari=(-?[0-9]+(?:\.[0-9]+)?)", candidate.stem, flags=re.IGNORECASE
        )
        if matches:
            scored.append((float(matches[-1]), candidate))
    if scored:
        # Checkpoints are named by FG-ARI, where higher is better.
        return max(scored, key=lambda item: (item[0], item[1].name))[1]
    last = path / "last.ckpt"
    if last.is_file():
        return last
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint files found in: {path}")


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device):
    model_cls = import_from_path(config["model"]["class_path"])
    model = model_cls(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    return model.to(device).eval()


def split_names(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    data = config["data"]
    if requested:
        missing = [name for name in requested if name not in data]
        if missing:
            raise KeyError(f"Splits not configured in data: {', '.join(missing)}")
        return list(requested)
    configured = [str(name) for name in data.get("validation_split_names", [])]
    return configured or ["train"]


def load_split(
    config: dict[str, Any], split_name: str, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Draw a fixed sample of images, instance masks, and scene metadata."""
    dataset_config = dict(config["data"][split_name])
    init_args = dict(dataset_config.get("init_args", {}))
    init_args["return_masks"] = True
    init_args["return_metadata"] = True
    dataset_config["init_args"] = init_args
    dataset = build_dataset(dataset_config)

    rng = random.Random(seed)
    length = len(dataset)
    indices = sorted(rng.sample(range(length), k=min(count, length)))
    images, masks, metadata = [], [], []
    for index in indices:
        sample = dataset[index]
        images.append(sample["images"])
        masks.append(sample["instance_masks"])
        metadata.append(sample.get("metadata", {}))
    return torch.stack(images), torch.stack(masks), metadata


@torch.no_grad()
def run_model(
    model,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, list[torch.Tensor]] = {}
    keys = (
        "images",
        "reconstructions",
        "slot_rgb",
        "masks",
        "attention_maps",
        "slots",
    )
    for start in range(0, images.size(0), batch_size):
        # Slots are sampled from a learned Gaussian, so fix the draw to make
        # the figures reproducible across runs of this script.
        torch.manual_seed(seed + start)
        batch = images[start : start + batch_size].to(device)
        output = model({"images": batch})
        for key in keys:
            value = output[key]
            if key in {"images", "reconstructions", "slot_rgb"}:
                value = model.to_display_space(value)
            outputs.setdefault(key, []).append(value.detach().float().cpu())
    return {key: torch.cat(value) for key, value in outputs.items()}


def match_slots_to_objects(
    masks: torch.Tensor, instance_masks: torch.Tensor
) -> torch.Tensor:
    """Assign each ground-truth object the slot that covers most of it.

    Returns ``(batch, objects)`` slot indices; the assignment is greedy in
    object order so two objects never collapse onto the same slot.
    """
    hard = masks.squeeze(2).argmax(dim=1)
    batch_size, slot_count = masks.size(0), masks.size(1)
    object_count = instance_masks.size(1)
    assignment = torch.full((batch_size, object_count), -1, dtype=torch.long)
    for sample_idx in range(batch_size):
        taken: set[int] = set()
        for object_idx in range(object_count):
            target = instance_masks[sample_idx, object_idx] > 0.5
            if not bool(target.any()):
                continue
            overlaps = [
                float(((hard[sample_idx] == slot_idx) & target).sum())
                if slot_idx not in taken
                else -1.0
                for slot_idx in range(slot_count)
            ]
            best = int(np.argmax(overlaps))
            if overlaps[best] <= 0.0:
                continue
            assignment[sample_idx, object_idx] = best
            taken.add(best)
    return assignment


def plot_slot_decomposition(
    outputs: dict[str, torch.Tensor], count: int, split_name: str
) -> plt.Figure:
    """Input, composite, and each slot's RGB layer, alpha mask, and attention."""
    images = outputs["images"][:count]
    reconstructions = outputs["reconstructions"][:count]
    slot_rgb = outputs["slot_rgb"][:count]
    masks = outputs["masks"][:count]
    attention = outputs["attention_maps"][:count]
    slot_count = masks.size(1)
    rows, columns = 3 * count, 2 + slot_count
    fig, axes = plt.subplots(
        rows, columns, figsize=(1.75 * columns, 1.8 * rows), squeeze=False
    )
    for sample_idx in range(count):
        top, middle, bottom = (
            axes[3 * sample_idx],
            axes[3 * sample_idx + 1],
            axes[3 * sample_idx + 2],
        )
        attention_limit = max(float(attention[sample_idx].max()), 1e-6)
        top[0].imshow(images[sample_idx].permute(1, 2, 0).numpy())
        top[1].imshow(reconstructions[sample_idx].clamp(0, 1).permute(1, 2, 0).numpy())
        # The unsupervised segmentation: which slot owns each pixel.
        middle[0].imshow(
            masks[sample_idx].squeeze(1).argmax(dim=0).numpy(),
            cmap="tab10",
            vmin=0,
            vmax=max(9, slot_count - 1),
            interpolation="nearest",
        )
        middle[0].set_ylabel("alpha", fontsize=8)
        bottom[0].set_ylabel("attention", fontsize=8)
        for axis in (middle[1], bottom[0], bottom[1]):
            axis.axis("off")
        if sample_idx == 0:
            top[0].set_title("input", fontsize=9, fontweight="bold")
            top[1].set_title("composite", fontsize=9, fontweight="bold")
            middle[0].set_title("segmentation", fontsize=9, fontweight="bold")
        for slot_idx in range(slot_count):
            mask = masks[sample_idx, slot_idx]
            layer = slot_rgb[sample_idx, slot_idx] * mask + (1.0 - mask)
            top[2 + slot_idx].imshow(layer.clamp(0, 1).permute(1, 2, 0).numpy())
            middle[2 + slot_idx].imshow(
                mask[0].numpy(), cmap="viridis", vmin=0.0, vmax=1.0
            )
            bottom[2 + slot_idx].imshow(
                attention[sample_idx, slot_idx, 0].numpy(),
                cmap="magma",
                vmin=0.0,
                vmax=attention_limit,
            )
            if sample_idx == 0:
                top[2 + slot_idx].set_title(
                    f"slot {slot_idx}", fontsize=9, fontweight="bold"
                )
            share = float(mask.mean())
            top[2 + slot_idx].set_xlabel(f"alpha share {share:.2f}", fontsize=7)
        for row in (top, middle, bottom):
            for axis in row:
                axis.set_xticks([])
                axis.set_yticks([])
    fig.suptitle(
        f"{split_name}: slot decomposition — what each slot reconstructs and masks",
        fontsize=12,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def plot_slot_representations(
    outputs: dict[str, torch.Tensor],
    instance_masks: torch.Tensor,
    metadata: list[dict[str, Any]],
    split_name: str,
) -> tuple[plt.Figure, dict[str, Any]]:
    """PCA of slot vectors, labelled by the object each slot captured."""
    slots = outputs["slots"]
    masks = outputs["masks"]
    sample_count, slot_count, _ = slots.shape
    assignment = match_slots_to_objects(masks, instance_masks)

    labels = [[None] * slot_count for _ in range(sample_count)]
    for sample_idx in range(sample_count):
        objects = metadata[sample_idx].get("objects", [])
        for object_idx, slot_idx in enumerate(assignment[sample_idx].tolist()):
            if slot_idx < 0 or object_idx >= len(objects):
                continue
            labels[sample_idx][slot_idx] = objects[object_idx]

    flat = slots.reshape(sample_count * slot_count, -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    projected = (centered @ components[:2].T).numpy()
    eigenvalues = singular_values.square() / max(1, flat.size(0) - 1)
    explained = (eigenvalues / eigenvalues.sum().clamp_min(1e-8)).numpy()

    flat_labels = [labels[i][k] for i in range(sample_count) for k in range(slot_count)]
    slot_index = np.tile(np.arange(slot_count), sample_count)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    background = np.array([label is None for label in flat_labels])
    axes[0].scatter(
        projected[background, 0],
        projected[background, 1],
        c="lightgray",
        s=12,
        marker="x",
        label="unmatched / background",
    )
    seen_shapes: set[str] = set()
    for shape, marker in SHAPE_MARKERS.items():
        selection = np.array(
            [label is not None and label.get("shape") == shape for label in flat_labels]
        )
        if not selection.any():
            continue
        seen_shapes.add(shape)
        colors = np.array(
            [
                np.asarray(flat_labels[i]["rgb"], dtype=np.float32) / 255.0
                for i in np.nonzero(selection)[0]
            ]
        )
        axes[0].scatter(
            projected[selection, 0],
            projected[selection, 1],
            c=colors,
            marker=marker,
            s=46,
            edgecolors="black",
            linewidths=0.3,
            label=shape,
        )
    axes[0].legend(fontsize=7, loc="best", framealpha=0.85)
    axes[0].set_title(
        "slot vectors: marker = captured shape, fill = captured color", fontsize=10
    )

    scatter = axes[1].scatter(
        projected[:, 0],
        projected[:, 1],
        c=slot_index,
        cmap="tab10",
        vmin=0,
        vmax=max(9, slot_count - 1),
        s=14,
        alpha=0.8,
    )
    axes[1].set_title("same points colored by slot position", fontsize=10)
    fig.colorbar(scatter, ax=axes[1], label="slot index")

    for axis in axes[:2]:
        axis.set_xlabel(f"PC1 ({explained[0]:.1%})")
        axis.set_ylabel(f"PC2 ({explained[1]:.1%})")
        axis.grid(alpha=0.15)

    # Slot position carries no identity: a given slot index binds whatever
    # object it happened to win, so this matrix should look close to uniform.
    shape_names = sorted(seen_shapes)
    binding = np.zeros((slot_count, len(shape_names)))
    for label, index in zip(flat_labels, slot_index):
        if label is None or label.get("shape") not in seen_shapes:
            continue
        binding[index, shape_names.index(label["shape"])] += 1.0
    totals = binding.sum(axis=1, keepdims=True)
    normalized = binding / np.clip(totals, 1.0, None)
    image = axes[2].imshow(normalized, cmap="cividis", vmin=0.0, vmax=1.0)
    axes[2].set_xticks(range(len(shape_names)), shape_names, rotation=45, fontsize=8)
    axes[2].set_yticks(range(slot_count), [f"slot {i}" for i in range(slot_count)], fontsize=8)
    axes[2].set_title("P(shape | slot index)", fontsize=10)
    fig.colorbar(image, ax=axes[2])

    fig.suptitle(
        f"{split_name}: {sample_count * slot_count} slot representations "
        f"from {sample_count} scenes",
        fontsize=12,
    )
    fig.tight_layout()

    matched = int((assignment >= 0).sum())
    summary = {
        "objects_total": int(assignment.numel()),
        "objects_matched_to_a_slot": matched,
        "object_match_rate": matched / max(1, assignment.numel()),
        "distinct_slots_used": int(
            len({int(value) for value in assignment.flatten().tolist() if value >= 0})
        ),
        "shape_counts": dict(
            Counter(
                label["shape"] for label in flat_labels if label is not None
            )
        ),
    }
    return fig, summary


def evaluate_split(
    outputs: dict[str, torch.Tensor], instance_masks: torch.Tensor
) -> dict[str, float]:
    masks = outputs["masks"]
    resolution = masks.shape[-2:]
    if tuple(instance_masks.shape[-2:]) != tuple(resolution):
        instance_masks = F.interpolate(instance_masks, size=resolution, mode="nearest")
    mse = (
        (outputs["reconstructions"] - outputs["images"]).square().flatten(1).mean(1)
    )
    slot_share = masks.squeeze(2).flatten(2).mean(dim=-1)
    return {
        "reconstruction_mse": float(mse.mean()),
        "foreground_ari": float(
            torch.nanmean(segmentation_ari(instance_masks, masks, foreground_only=True))
        ),
        "ari": float(
            torch.nanmean(segmentation_ari(instance_masks, masks, foreground_only=False))
        ),
        "active_slots": float((slot_share > 0.02).float().sum(dim=1).mean()),
        "mean_max_alpha": float(masks.squeeze(2).amax(dim=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Slot Attention decompositions and slot representations."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--samples-per-split", type=int, default=256)
    parser.add_argument("--decomposition-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = load_config(repo_path(args.config))
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_model(config, checkpoint, device)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint: {checkpoint}")

    report: dict[str, Any] = {
        "config": str(args.config),
        "checkpoint": str(checkpoint),
        "splits": {},
    }
    for split_name in split_names(config, args.splits):
        images, instance_masks, metadata = load_split(
            config, split_name, args.samples_per_split, args.seed
        )
        outputs = run_model(model, images, device, args.batch_size, args.seed)
        metrics = evaluate_split(outputs, instance_masks)

        decomposition = plot_slot_decomposition(
            outputs,
            min(args.decomposition_samples, outputs["images"].size(0)),
            split_name,
        )
        decomposition_path = output_dir / f"{split_name}_slot_decomposition.png"
        decomposition.savefig(decomposition_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(decomposition)

        representation, summary = plot_slot_representations(
            outputs, instance_masks, metadata, split_name
        )
        representation_path = output_dir / f"{split_name}_slot_representations.png"
        representation.savefig(representation_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(representation)

        report["splits"][split_name] = {
            **metrics,
            **summary,
            "figures": [str(decomposition_path), str(representation_path)],
        }
        print(
            f"{split_name}: FG-ARI {metrics['foreground_ari']:.4f} | "
            f"ARI {metrics['ari']:.4f} | MSE {metrics['reconstruction_mse']:.5f} | "
            f"active slots {metrics['active_slots']:.2f}"
        )

    report_path = output_dir / "slot_attention_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
