"""Reuse the joint-token ViT with a dedicated rotation-angle vocabulary."""

from torch import nn

from benchmarks.cogitao_vit_cross_attention.joint_token_model import (
    JointTokenViTTransformationModel,
)
from benchmarks.cogitao_vit_cross_attention.rotation_data import validate_angles


class RotationTokenViT(JointTokenViTTransformationModel):
    def __init__(self, config):
        super().__init__(config)
        self.angles = validate_angles(
            config["model"]["config"].get("angles", (0, 90, 180, 270))
        )
        self.task_embedding = nn.Embedding(
            len(self.angles) + 1, self.hidden_dim, padding_idx=0
        )
