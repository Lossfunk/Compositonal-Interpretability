from __future__ import annotations

import torch


def adjusted_rand_index(
    true_masks: torch.Tensor,
    predicted_masks: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Per-sample ARI between two soft point-to-group assignments.

    ``true_masks`` and ``predicted_masks`` are ``(batch, points, groups)``
    weights that each sum to one over the group axis. Points may be down
    weighted to zero to drop them from the comparison, which is how the
    foreground-only variant excludes background pixels.
    """
    if true_masks.dim() != 3 or predicted_masks.dim() != 3:
        raise ValueError("Masks must have shape (batch, points, groups).")
    if true_masks.shape[:2] != predicted_masks.shape[:2]:
        raise ValueError("Masks must agree on batch size and point count.")

    true_masks = true_masks.double()
    predicted_masks = predicted_masks.double()
    total_points = true_masks.sum(dim=(1, 2))
    contingency = torch.einsum("bnp,bnt->bpt", predicted_masks, true_masks)
    true_counts = contingency.sum(dim=1)
    predicted_counts = contingency.sum(dim=2)

    pair_index = (contingency * (contingency - 1.0)).sum(dim=(1, 2))
    true_index = (true_counts * (true_counts - 1.0)).sum(dim=1)
    predicted_index = (predicted_counts * (predicted_counts - 1.0)).sum(dim=1)

    denominator = (total_points * (total_points - 1.0)).clamp_min(epsilon)
    expected_index = true_index * predicted_index / denominator
    maximum_index = (true_index + predicted_index) / 2.0
    ari = (pair_index - expected_index) / (maximum_index - expected_index).clamp_min(
        epsilon
    )

    # A single occupied group on either side leaves ARI undefined; the
    # convention from the Slot Attention codebase scores that case as perfect.
    both_trivial = ((true_counts > 0).sum(dim=1) <= 1) & (
        (predicted_counts > 0).sum(dim=1) <= 1
    )
    empty = total_points <= 1
    ari = torch.where(both_trivial, torch.ones_like(ari), ari)
    return torch.where(empty, torch.full_like(ari, float("nan")), ari).float()


def segmentation_ari(
    instance_masks: torch.Tensor,
    slot_masks: torch.Tensor,
    *,
    foreground_only: bool = True,
) -> torch.Tensor:
    """ARI between ground-truth instance masks and predicted slot masks.

    ``instance_masks`` is ``(batch, objects, height, width)`` with values in
    ``[0, 1]``; overlapping objects are resolved by taking the strongest one.
    ``slot_masks`` is ``(batch, slots, 1, height, width)`` or
    ``(batch, slots, height, width)``. With ``foreground_only`` set (the
    FG-ARI reported in the paper) background pixels are dropped instead of
    being scored as one extra group.
    """
    if slot_masks.dim() == 5:
        slot_masks = slot_masks.squeeze(2)
    if slot_masks.dim() != 4:
        raise ValueError("slot_masks must be (batch, slots, [1,] height, width).")
    if instance_masks.dim() != 4:
        raise ValueError("instance_masks must be (batch, objects, height, width).")
    if instance_masks.shape[-2:] != slot_masks.shape[-2:]:
        raise ValueError("Ground-truth and slot masks must share a resolution.")

    batch_size, object_count = instance_masks.shape[:2]
    slot_count = slot_masks.size(1)
    points = instance_masks.shape[-2] * instance_masks.shape[-1]

    instance_flat = instance_masks.reshape(batch_size, object_count, points)
    occupancy = instance_flat.amax(dim=1)
    object_index = instance_flat.argmax(dim=1)
    # Background becomes group 0, so foreground objects start at 1.
    true_index = torch.where(
        occupancy > 0.5, object_index + 1, torch.zeros_like(object_index)
    )
    true_one_hot = torch.nn.functional.one_hot(
        true_index, num_classes=object_count + 1
    ).float()

    predicted_index = slot_masks.reshape(batch_size, slot_count, points).argmax(dim=1)
    predicted_one_hot = torch.nn.functional.one_hot(
        predicted_index, num_classes=slot_count
    ).float()

    if foreground_only:
        foreground = (occupancy > 0.5).float().unsqueeze(-1)
        true_one_hot = true_one_hot[..., 1:] * foreground
        predicted_one_hot = predicted_one_hot * foreground

    return adjusted_rand_index(true_one_hot, predicted_one_hot)
