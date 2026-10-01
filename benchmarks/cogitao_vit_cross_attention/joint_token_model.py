"""COGITAO C1 ViT with joint image and ordered function-token encoding."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    try:
        import pytorch_lightning as L
    except ImportError:
        L = None

from data.cogitao.cogitao import NUM_GRID_CLASSES, TASK_VOCABULARY


_BaseModule = L.LightningModule if L is not None else nn.Module


class JointTokenViTEncoder(nn.Module):
    """Fresh ViT encoder for image patches and ordered function tokens."""

    def __init__(
        self,
        input_resolution: tuple[int, int],
        *,
        patch_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        height, width = input_resolution
        if patch_size <= 0 or height % patch_size or width % patch_size:
            raise ValueError("The patch size must divide both grid dimensions.")
        if hidden_dim % num_heads:
            raise ValueError("The hidden dimension must be divisible by num_heads.")
        self.patch_grid = (height // patch_size, width // patch_size)
        self.num_image_tokens = self.patch_grid[0] * self.patch_grid[1]
        self.patch_embedding = nn.Conv2d(
            NUM_GRID_CLASSES, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.image_position_embedding = nn.Parameter(
            torch.empty(1, self.num_image_tokens, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        nn.init.trunc_normal_(self.image_position_embedding, std=0.02)

    def forward(
        self,
        images: torch.Tensor,
        function_tokens: torch.Tensor,
        function_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_tokens = self.patch_embedding(images).flatten(2).transpose(1, 2)
        image_tokens = image_tokens + self.image_position_embedding
        joint_tokens = torch.cat((image_tokens, function_tokens), dim=1)
        image_padding = torch.zeros(
            images.size(0), self.num_image_tokens,
            dtype=torch.bool, device=images.device,
        )
        padding_mask = torch.cat((image_padding, ~function_mask), dim=1)
        return self.transformer(
            joint_tokens, src_key_padding_mask=padding_mask
        )


class JointTokenViTTransformationModel(_BaseModule):
    """Encode image and ordered function tokens jointly, then decode image tokens."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        model_config = config["model"]["config"]
        self.input_resolution = tuple(
            int(value) for value in model_config.get("input_resolution", (20, 20))
        )
        encoder_config = dict(model_config.get("encoder", {}))
        self.patch_size = int(encoder_config.get("patch_size", 1))
        self.hidden_dim = int(encoder_config.get("hidden_dim", 128))
        self.encoder = JointTokenViTEncoder(
            self.input_resolution,
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=int(encoder_config.get("num_layers", 6)),
            num_heads=int(encoder_config.get("num_heads", 4)),
            mlp_dim=int(encoder_config.get("mlp_dim", 512)),
            dropout=float(encoder_config.get("dropout", 0.0)),
        )
        self.num_image_tokens = self.encoder.num_image_tokens
        self.patch_grid = self.encoder.patch_grid
        self.max_task_tokens = int(model_config.get("max_task_tokens", 2))
        if self.max_task_tokens != 2:
            raise ValueError("C1 expects two ordered, padded function tokens.")
        self.task_embedding = nn.Embedding(
            len(TASK_VOCABULARY) + 1, self.hidden_dim, padding_idx=0
        )
        self.task_position_embedding = nn.Parameter(
            torch.empty(1, self.max_task_tokens, self.hidden_dim)
        )
        self.task_modality_embedding = nn.Parameter(torch.empty(1, 1, self.hidden_dim))
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(
                self.hidden_dim,
                NUM_GRID_CLASSES * self.patch_size * self.patch_size,
            ),
        )
        self.validation_split_names = list(
            model_config.get("validation_split_names", ["val", "val_ood"])
        )
        self.test_split_names = list(
            model_config.get("test_split_names", ["test", "test_ood"])
        )
        self.fail_on_nonfinite = bool(model_config.get("fail_on_nonfinite", True))
        nn.init.trunc_normal_(self.task_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.task_modality_embedding, std=0.02)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters({"config": config})

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["images"].float()
        if images.ndim != 4 or images.size(1) != NUM_GRID_CLASSES:
            raise ValueError("COGITAO images must be [batch, 10, height, width].")
        if tuple(images.shape[-2:]) != self.input_resolution:
            raise ValueError(f"Expected input resolution {self.input_resolution}.")
        task_tokens = batch["task_tokens"].long()
        task_mask = batch["task_token_mask"].bool()
        expected_shape = (images.size(0), self.max_task_tokens)
        if task_tokens.shape != expected_shape or task_mask.shape != expected_shape:
            raise ValueError(f"Expected task tokens and mask shaped {expected_shape}.")
        task_features = (
            self.task_embedding(task_tokens)
            + self.task_position_embedding
            + self.task_modality_embedding
        )
        task_features = task_features * task_mask.unsqueeze(-1).to(task_features.dtype)
        encoded = self.encoder(images, task_features, task_mask)
        image_embeddings = encoded[:, : self.num_image_tokens]
        function_embeddings = encoded[:, self.num_image_tokens:]
        patch_logits = self.output_head(image_embeddings)
        patch_rows, patch_cols = self.patch_grid
        height, width = self.input_resolution
        output_logits = patch_logits.reshape(
            images.size(0), patch_rows, patch_cols,
            NUM_GRID_CLASSES, self.patch_size, self.patch_size,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            images.size(0), NUM_GRID_CLASSES, height, width
        )
        output_log_probs = F.log_softmax(output_logits, dim=1)
        return {
            "images": images,
            "target_grid": batch["target_grid"].long(),
            "image_embeddings": image_embeddings,
            "function_embeddings": function_embeddings,
            "output_logits": output_logits,
            "output_log_probs": output_log_probs,
            "predictions": output_log_probs.argmax(dim=1),
        }

    @staticmethod
    def compute_metrics(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predictions = output["predictions"]
        target = output["target_grid"]
        correct = predictions.eq(target)
        object_mask = predictions.ne(0) | target.ne(0)
        object_pixels = object_mask.flatten(1).sum(dim=1)
        correct_object = (correct & object_mask).flatten(1).sum(dim=1)
        object_accuracy = torch.where(
            object_pixels > 0,
            correct_object.float() / object_pixels.clamp_min(1),
            correct.flatten(1).all(dim=1).float(),
        )
        return {
            "grid_accuracy": 100.0 * correct.flatten(1).all(dim=1).float().mean(),
            "per_pixel_accuracy": 100.0 * correct.float().mean(),
            "object_per_pixel_accuracy": 100.0 * object_accuracy.mean(),
        }

    @staticmethod
    def _domain_name(split_name: str) -> str:
        return "ood" if split_name.endswith("_ood") else "id"

    def _shared_step(
        self, batch: dict[str, torch.Tensor], stage: str, split_name: str | None = None
    ) -> torch.Tensor:
        output = self(batch)
        loss = F.nll_loss(output["output_log_probs"], output["target_grid"])
        if self.fail_on_nonfinite and not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"Non-finite {stage} cross-entropy loss.")
        values = {
            "cross_entropy_loss": loss,
            "total_loss": loss,
            **self.compute_metrics(output),
        }
        prefix = stage if split_name is None else f"{stage}/{self._domain_name(split_name)}"
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            add_dataloader_idx=False,
            batch_size=output["images"].size(0),
        )
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: dict[str, torch.Tensor], _batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        split_name = self.validation_split_names[dataloader_idx]
        return self._shared_step(batch, "val", split_name)

    def test_step(
        self, batch: dict[str, torch.Tensor], _batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        split_name = self.test_split_names[dataloader_idx]
        return self._shared_step(batch, "test", split_name)

    def _log_generalization_gap(self, stage: str) -> None:
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        metrics = trainer.callback_metrics
        id_key = f"{stage}/id/grid_accuracy"
        ood_key = f"{stage}/ood/grid_accuracy"
        if id_key in metrics and ood_key in metrics:
            self.log(
                f"{stage}/grid_accuracy_id_minus_ood",
                metrics[id_key] - metrics[ood_key],
                on_step=False, on_epoch=True, sync_dist=False,
            )

    def on_validation_epoch_end(self) -> None:
        self._log_generalization_gap("val")

    def on_test_epoch_end(self) -> None:
        self._log_generalization_gap("test")

    def configure_optimizers(self):
        optimizer_config = self.config["trainer"]["optimizer"]
        if optimizer_config.get("type") != "AdamW":
            raise ValueError("Joint-token C1 training expects AdamW.")
        optimizer = torch.optim.AdamW(
            self.parameters(), **dict(optimizer_config.get("config", {}))
        )
        schedule = dict(optimizer_config.get("schedule", {}))
        warmup_steps = max(0, int(schedule.get("warmup_steps", 200)))
        total_steps = int(schedule.get("total_steps", self.trainer.estimated_stepping_batches))
        min_lr_ratio = float(schedule.get("min_lr_ratio", 0.01))
        if total_steps <= 0 or not 0 <= min_lr_ratio <= 1:
            raise ValueError("Invalid total_steps or min_lr_ratio in optimizer schedule.")

        def scale(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
