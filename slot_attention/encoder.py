from __future__ import annotations

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer


class VisionTransformerSlotEncoder(nn.Module):
    """Three-layer ViT trunk that emits per-patch features for Slot Attention.

    Slot Attention's original encoder is a stride-1 CNN with a soft position
    embedding. Here the trunk is the same tiny ViT the other CLEVR-2D
    experiments use, so the patch tokens (position embedding included) already
    carry spatial identity. The paper's ``LayerNorm -> MLP`` bottleneck is kept
    between the trunk and the slot module.
    """

    def __init__(
        self,
        input_resolution: tuple[int, int],
        *,
        input_channels: int = 3,
        patch_size: int = 8,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_dim: int = 512,
        output_dim: int | None = None,
        bottleneck_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        height, width = (int(input_resolution[0]), int(input_resolution[1]))
        if height != width:
            raise ValueError("The ViT trunk requires square input images.")
        if height % int(patch_size) != 0:
            raise ValueError("input_resolution must be divisible by patch_size.")

        self.input_resolution = (height, width)
        self.input_channels = int(input_channels)
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.feature_resolution = (height // self.patch_size, width // self.patch_size)
        self.output_dim = int(output_dim) if output_dim is not None else self.hidden_dim
        bottleneck_hidden_dim = int(
            bottleneck_hidden_dim
            if bottleneck_hidden_dim is not None
            else self.hidden_dim
        )

        self.trunk = VisionTransformer(
            image_size=height,
            patch_size=self.patch_size,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            hidden_dim=self.hidden_dim,
            mlp_dim=int(mlp_dim),
            dropout=float(dropout),
            attention_dropout=float(attention_dropout),
            num_classes=self.hidden_dim,
        )
        self.trunk.heads = nn.Identity()

        if self.input_channels != 3:
            self.trunk.conv_proj = nn.Conv2d(
                self.input_channels,
                self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            fan_in = self.input_channels * self.patch_size * self.patch_size
            nn.init.trunc_normal_(self.trunk.conv_proj.weight, std=fan_in**-0.5)
            if self.trunk.conv_proj.bias is not None:
                nn.init.zeros_(self.trunk.conv_proj.bias)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, bottleneck_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_hidden_dim, self.output_dim),
        )

    @property
    def num_tokens(self) -> int:
        return self.feature_resolution[0] * self.feature_resolution[1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch features shaped ``(batch, tokens, output_dim)``."""
        patch_tokens = self.trunk._process_input(images)
        class_token = self.trunk.class_token.expand(patch_tokens.size(0), -1, -1)
        tokens = self.trunk.encoder(torch.cat([class_token, patch_tokens], dim=1))
        return self.bottleneck(tokens[:, 1:])
