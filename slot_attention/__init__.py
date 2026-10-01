"""Slot Attention object discovery on CLEVR-2D with a three-layer ViT trunk."""

from slot_attention.attention import SlotAttention, SoftPositionEmbed
from slot_attention.decoder import SpatialBroadcastDecoder
from slot_attention.encoder import VisionTransformerSlotEncoder
from slot_attention.metrics import adjusted_rand_index, segmentation_ari
from slot_attention.model import SlotAttentionObjectDiscoveryModel

__all__ = [
    "SlotAttention",
    "SoftPositionEmbed",
    "SpatialBroadcastDecoder",
    "VisionTransformerSlotEncoder",
    "SlotAttentionObjectDiscoveryModel",
    "adjusted_rand_index",
    "segmentation_ari",
]
