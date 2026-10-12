"""Fresh joint-token ViT with shared numeric degree features."""

from __future__ import annotations

import math

import torch
from torch import nn

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import NET_ANGLES
from benchmarks.cogitao_vit_cross_attention.joint_token_model import (
    JointTokenViTTransformationModel,
)


class AngleFeatureEmbedding(nn.Module):
    """Encode angles through sin/cos and a shared projection, including net angles.

    The model still jointly processes two ordered tokens for a composition.
    No addition or algebraic constraint is imposed inside its forward pass.
    """

    def __init__(self, angles, hidden_dim):
        super().__init__()
        radians = torch.tensor(angles, dtype=torch.float32) * (math.pi / 180)
        features = torch.stack((radians.sin(), radians.cos()), -1)
        self.register_buffer("angle_features", torch.cat((torch.zeros(1, 2), features)))
        self.projection = nn.Linear(2, hidden_dim)

    def forward(self, ids):
        return self.projection(self.angle_features[ids])


class DegreeRotationViT(JointTokenViTTransformationModel):
    def __init__(self, config):
        super().__init__(config)
        self.angles = tuple(config["model"]["config"]["angles"])
        if self.angles != NET_ANGLES:
            raise ValueError("Expected net-angle vocabulary 15, 30, ..., 180 degrees.")
        self.task_embedding = AngleFeatureEmbedding(self.angles, self.hidden_dim)

    def training_step(self, batch, batch_index):
        loss = super().training_step(batch, batch_index)
        self.log(
            "train/step_cross_entropy_loss",
            loss,
            on_step=True,
            on_epoch=False,
            batch_size=batch["images"].shape[0],
        )
        return loss
