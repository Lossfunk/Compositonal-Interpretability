from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

try:
    from slot_attention.attention import SoftPositionEmbed
except ImportError:  # pragma: no cover - direct-module import fallback.
    from attention import SoftPositionEmbed


class SpatialBroadcastDecoder(nn.Module):
    """Slot Attention's object-discovery decoder.

    Every slot is broadcast on its own to a small grid, given a soft position
    code, and upsampled by transposed convolutions into four channels: three
    RGB channels plus one alpha channel. The alphas are softmaxed *across
    slots*, so the slots explain disjoint parts of one image and the
    reconstruction is their alpha-weighted sum.
    """

    def __init__(
        self,
        slot_dim: int,
        output_resolution: Sequence[int],
        *,
        content_channels: int = 3,
        channels: Sequence[int] = (64, 64, 64, 64),
        kernel_size: int = 5,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.content_channels = int(content_channels)
        if self.content_channels <= 0:
            raise ValueError("content_channels must be positive.")
        self.output_resolution = tuple(int(value) for value in output_resolution)
        self.channels = tuple(int(value) for value in channels)
        if len(self.output_resolution) != 2:
            raise ValueError("output_resolution must contain height and width.")
        if not self.channels or min(self.channels) <= 0:
            raise ValueError("channels must contain positive values.")
        if output_activation not in {"none", "sigmoid"}:
            raise ValueError("output_activation must be 'none' or 'sigmoid'.")
        self.output_activation = output_activation

        upsample_factor = 2 ** len(self.channels)
        if any(size % upsample_factor != 0 for size in self.output_resolution):
            raise ValueError(
                "Each output dimension must be divisible by 2 ** len(channels); "
                f"got resolution={self.output_resolution}, channels={self.channels}."
            )
        self.broadcast_resolution = tuple(
            size // upsample_factor for size in self.output_resolution
        )
        if min(self.broadcast_resolution) <= 0:
            raise ValueError("The broadcast resolution must be positive.")

        self.position_embedding = SoftPositionEmbed(
            self.slot_dim, self.broadcast_resolution
        )
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding stays symmetric.")
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        in_channels = self.slot_dim
        for out_channels in self.channels:
            blocks.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=1,
                )
            )
            blocks.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Conv2d(
            in_channels, self.content_channels + 1, kernel_size=3, stride=1, padding=1
        )

    def decode_slots(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode each slot independently into RGB layers and alpha logits."""
        if slots.dim() != 3:
            raise ValueError("slots must have shape (batch, slots, slot_dim).")
        batch_size, slot_count, slot_dim = slots.shape
        if slot_dim != self.slot_dim:
            raise ValueError(f"Expected slot_dim={self.slot_dim}, got {slot_dim}.")

        height, width = self.broadcast_resolution
        broadcast = slots.reshape(batch_size * slot_count, slot_dim, 1, 1)
        broadcast = broadcast.expand(-1, -1, height, width)
        features = self.position_embedding(broadcast)
        decoded = self.output(self.blocks(features))
        decoded = decoded.reshape(
            batch_size, slot_count, self.content_channels + 1, *self.output_resolution
        )
        slot_rgb = decoded[:, :, : self.content_channels]
        if self.output_activation == "sigmoid":
            slot_rgb = torch.sigmoid(slot_rgb)
        return slot_rgb, decoded[:, :, self.content_channels :]

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        slot_rgb, alpha_logits = self.decode_slots(slots)
        masks = F.softmax(alpha_logits, dim=1)
        reconstruction = (slot_rgb * masks).sum(dim=1)
        return {
            "reconstruction": reconstruction,
            "slot_rgb": slot_rgb,
            "alpha_logits": alpha_logits,
            "masks": masks,
        }
