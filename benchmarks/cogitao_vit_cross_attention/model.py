from __future__ import annotations

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

from data.cogitao.cogitao import NUM_GRID_CLASSES


_BaseModule = L.LightningModule if L is not None else nn.Module


class DirectPatchTransformationModel(_BaseModule):
    """Transform a C1 input grid directly through patch-token decoder blocks.

    This model deliberately has no task/function input. Each one-hot input patch
    is embedded, combined with a learned position embedding, decoded with
    self-attention blocks, and projected back to categorical output patches.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]
        self.input_resolution = tuple(
            int(value)
            for value in self.model_config.get("input_resolution", (20, 20))
        )
        height, width = self.input_resolution

        patch_config = dict(self.model_config.get("patch_embedding", {}))
        self.patch_size = int(patch_config.get("patch_size", 1))
        self.embedding_dim = int(patch_config.get("hidden_dim", 128))
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("input_resolution must be divisible by patch_size.")
        self.patch_grid = (height // self.patch_size, width // self.patch_size)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        decoder_config = dict(self.model_config.get("decoder", {}))
        self.decoder_layers = int(decoder_config.get("num_layers", 6))
        self.decoder_heads = int(decoder_config.get("num_heads", 4))
        self.decoder_mlp_dim = int(decoder_config.get("mlp_dim", 512))
        decoder_dropout = float(decoder_config.get("dropout", 0.0))
        if self.embedding_dim % self.decoder_heads:
            raise ValueError(
                "patch hidden_dim must be divisible by decoder num_heads."
            )

        self.patch_embedding = nn.Conv2d(
            NUM_GRID_CLASSES,
            self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_patches, self.embedding_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.decoder_heads,
            dim_feedforward=self.decoder_mlp_dim,
            dropout=decoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.decoder_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(
                self.embedding_dim,
                NUM_GRID_CLASSES * self.patch_size * self.patch_size,
            ),
        )

        self.validation_split_names = [
            str(name)
            for name in self.model_config.get(
                "validation_split_names", ["val", "val_ood"]
            )
        ]
        self.test_split_names = [
            str(name)
            for name in self.model_config.get("test_split_names", ["test", "test_ood"])
        ]
        self.fail_on_nonfinite = bool(
            self.model_config.get("fail_on_nonfinite", True)
        )

        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters({"config": config})

    def embed_patches(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.size(1) != NUM_GRID_CLASSES:
            raise ValueError("COGITAO images must be [B, 10, H, W] one-hot grids.")
        if tuple(images.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"Expected image resolution {self.input_resolution}, "
                f"got {tuple(images.shape[-2:])}."
            )
        return self.patch_embedding(images).flatten(2).transpose(1, 2)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["images"].float()
        patch_embeddings = self.embed_patches(images)
        decoded_tokens = self.decoder(patch_embeddings + self.position_embedding)
        output_patches = self.output_head(decoded_tokens)
        height, width = self.input_resolution
        patch_rows, patch_columns = self.patch_grid
        output_logits = output_patches.reshape(
            images.size(0),
            patch_rows,
            patch_columns,
            NUM_GRID_CLASSES,
            self.patch_size,
            self.patch_size,
        ).permute(0, 3, 1, 4, 2, 5).reshape(
            images.size(0), NUM_GRID_CLASSES, height, width
        )
        output_log_probs = F.log_softmax(output_logits, dim=1)
        return {
            "images": images,
            "target_grid": batch["target_grid"].long(),
            "patch_embeddings": patch_embeddings,
            "decoded_tokens": decoded_tokens,
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
        correct_object_pixels = (correct & object_mask).flatten(1).sum(dim=1)
        object_accuracy = torch.where(
            object_pixels > 0,
            correct_object_pixels.float() / object_pixels.clamp_min(1),
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
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        split_name: str | None = None,
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
        prefix = stage
        if split_name is not None:
            prefix = f"{stage}/{self._domain_name(split_name)}"
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

    def training_step(
        self, batch: dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        _batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        split_name = (
            self.validation_split_names[dataloader_idx]
            if dataloader_idx < len(self.validation_split_names)
            else f"loader_{dataloader_idx}"
        )
        return self._shared_step(batch, "val", split_name)

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        _batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        split_name = (
            self.test_split_names[dataloader_idx]
            if dataloader_idx < len(self.test_split_names)
            else f"loader_{dataloader_idx}"
        )
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
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def on_validation_epoch_end(self) -> None:
        self._log_generalization_gap("val")

    def on_test_epoch_end(self) -> None:
        self._log_generalization_gap("test")

    def configure_optimizers(self):
        optimizer_config = self.config["trainer"]["optimizer"]
        optimizer_type = optimizer_config.get("type", "Adam")
        kwargs = dict(optimizer_config.get("config", {}))
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(parameters, **kwargs)
        elif optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(parameters, **kwargs)
        else:
            raise ValueError(f"Optimizer type {optimizer_type} not implemented.")

        schedule_config = optimizer_config.get("schedule")
        if not schedule_config:
            return optimizer
        warmup_steps = max(0, int(schedule_config.get("warmup_steps", 0)))
        decay_steps = max(1, int(schedule_config.get("decay_steps", 100_000)))
        decay_rate = float(schedule_config.get("decay_rate", 0.5))

        def scale(step: int) -> float:
            warmup = 1.0 if warmup_steps == 0 else min(1.0, (step + 1) / warmup_steps)
            return warmup * decay_rate ** (step / decay_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
