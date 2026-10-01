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

from data.cogitao.cogitao import NUM_GRID_CLASSES, TASK_VOCABULARY


_BaseModule = L.LightningModule if L is not None else nn.Module


class CrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention followed by a residual feed-forward block."""

    def __init__(
        self,
        dimension: int,
        num_heads: int,
        mlp_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dimension)
        self.memory_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, mlp_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_memory = self.memory_norm(memory)
        attended, weights = self.attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=True,
            average_attn_weights=False,
        )
        shared = queries + attended
        shared = shared + self.mlp(self.mlp_norm(shared))
        return shared, weights


class FunctionConditionedViTCrossAttentionModel(_BaseModule):
    """Fuse paper-style ViT image latents with an MLP function latent.

    The image path receives only one-hot grid cells. Ordered C1 function
    embeddings are concatenated and encoded by a separate MLP. The resulting
    function latent conditions learned queries which cross-attend to the image
    latents, producing shared latents. Learned spatial output queries then
    cross-attend to those shared latents to predict the output grid.
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
        self.num_image_tokens = height * width

        encoder_config = dict(self.model_config.get("encoder", {}))
        self.embedding_dim = int(encoder_config.get("hidden_dim", 128))
        self.encoder_layers = int(encoder_config.get("num_layers", 6))
        self.encoder_heads = int(encoder_config.get("num_heads", 4))
        self.encoder_mlp_dim = int(encoder_config.get("mlp_dim", 512))
        encoder_dropout = float(encoder_config.get("dropout", 0.0))
        if self.embedding_dim % self.encoder_heads:
            raise ValueError("encoder hidden_dim must be divisible by num_heads.")

        # Paper-style pixel/cell tokenization: a 1x1 stride-1 projection of the
        # ten-channel one-hot grid, followed by learned 1-D position embeddings.
        self.image_tokenizer = nn.Conv2d(
            NUM_GRID_CLASSES,
            self.embedding_dim,
            kernel_size=1,
            stride=1,
        )
        self.image_position_embedding = nn.Parameter(
            torch.empty(1, self.num_image_tokens, self.embedding_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.encoder_heads,
            dim_feedforward=self.encoder_mlp_dim,
            dropout=encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.image_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.encoder_layers,
            norm=nn.LayerNorm(self.embedding_dim),
        )

        function_config = dict(self.model_config.get("function_mlp", {}))
        self.max_function_tokens = int(function_config.get("max_tokens", 2))
        function_embedding_dim = int(
            function_config.get("embedding_dim", self.embedding_dim)
        )
        function_hidden_dim = int(function_config.get("hidden_dim", 512))
        if self.max_function_tokens != 2:
            raise ValueError("The C1 function MLP expects exactly two padded tokens.")
        self.function_embedding = nn.Embedding(
            len(TASK_VOCABULARY) + 1,
            function_embedding_dim,
            padding_idx=0,
        )
        self.function_mlp = nn.Sequential(
            nn.Linear(self.max_function_tokens * function_embedding_dim, function_hidden_dim),
            nn.GELU(),
            nn.Linear(function_hidden_dim, self.embedding_dim),
        )

        cross_config = dict(self.model_config.get("cross_attention", {}))
        num_shared_latents = int(cross_config.get("num_shared_latents", 16))
        cross_heads = int(cross_config.get("num_heads", 4))
        cross_mlp_dim = int(cross_config.get("mlp_dim", 512))
        cross_dropout = float(cross_config.get("dropout", 0.0))
        self.shared_queries = nn.Parameter(
            torch.empty(1, num_shared_latents, self.embedding_dim)
        )
        self.output_queries = nn.Parameter(
            torch.empty(1, self.num_image_tokens, self.embedding_dim)
        )
        self.image_to_shared = CrossAttentionBlock(
            self.embedding_dim, cross_heads, cross_mlp_dim, cross_dropout
        )
        self.shared_to_output = CrossAttentionBlock(
            self.embedding_dim, cross_heads, cross_mlp_dim, cross_dropout
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, NUM_GRID_CLASSES),
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

        nn.init.trunc_normal_(self.image_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.shared_queries, std=0.02)
        nn.init.trunc_normal_(self.output_queries, std=0.02)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters({"config": config})

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.size(1) != NUM_GRID_CLASSES:
            raise ValueError("COGITAO images must be [B, 10, H, W] one-hot grids.")
        if tuple(images.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"Expected image resolution {self.input_resolution}, "
                f"got {tuple(images.shape[-2:])}."
            )
        tokens = self.image_tokenizer(images).flatten(2).transpose(1, 2)
        return self.image_encoder(tokens + self.image_position_embedding)

    def encode_functions(
        self,
        task_tokens: torch.Tensor,
        task_mask: torch.Tensor,
    ) -> torch.Tensor:
        if task_tokens.shape != task_mask.shape:
            raise ValueError("task_tokens and task_token_mask must have equal shape.")
        if task_tokens.size(1) != self.max_function_tokens:
            raise ValueError(
                f"Expected {self.max_function_tokens} padded function tokens, "
                f"got {task_tokens.size(1)}."
            )
        vectors = self.function_embedding(task_tokens)
        vectors = vectors * task_mask.unsqueeze(-1).to(vectors.dtype)
        # Fixed halves preserve order: [embedding(function_1) || embedding(function_2)].
        return self.function_mlp(vectors.flatten(start_dim=1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["images"].float()
        image_latents = self.encode_images(images)
        function_latent = self.encode_functions(
            batch["task_tokens"].long(),
            batch["task_token_mask"].bool(),
        )

        shared_queries = self.shared_queries.expand(images.size(0), -1, -1)
        shared_queries = shared_queries + function_latent.unsqueeze(1)
        shared_latents, image_cross_attention = self.image_to_shared(
            shared_queries, image_latents
        )

        output_queries = self.output_queries.expand(images.size(0), -1, -1)
        output_latents, output_cross_attention = self.shared_to_output(
            output_queries, shared_latents
        )
        output_logits = self.output_head(output_latents)
        height, width = self.input_resolution
        output_logits = output_logits.transpose(1, 2).reshape(
            images.size(0), NUM_GRID_CLASSES, height, width
        )
        output_log_probs = F.log_softmax(output_logits, dim=1)
        return {
            "images": images,
            "target_grid": batch["target_grid"].long(),
            "image_latents": image_latents,
            "function_latent": function_latent,
            "shared_latents": shared_latents,
            "image_cross_attention": image_cross_attention,
            "output_cross_attention": output_cross_attention,
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
