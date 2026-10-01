from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from torch import nn
from torch.nn import functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - W&B is optional for local model tests.
    wandb = None

from data.cogitao.cogitao import (
    ARC_PALETTE,
    MAX_COMPOSITION_DEPTH,
    NUM_GRID_CLASSES,
    TASK_VOCABULARY,
)
from slot_attention.decoder import SpatialBroadcastDecoder
from slot_attention.encoder import VisionTransformerSlotEncoder
from slot_attention.model import SlotAttentionObjectDiscoveryModel


class TaskConditionedSlotAttentionModel(SlotAttentionObjectDiscoveryModel):
    """Predict a transformed COGITAO grid from image patches and task tokens.

    Task tokens are appended to the encoded image-token sequence before Slot
    Attention. Slots are decoded as a mixture of categorical cell distributions.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.loss_weights = {
            "cross_entropy_loss": self.loss_weights["reconstruction_loss"]
        }
        self.test_split_names = [
            str(name)
            for name in self.model_config.get("test_split_names", ["test", "test_ood"])
        ]
        output_logging = dict(self.model_config.get("output_logging", {}))
        self.log_output_tables = bool(output_logging.get("enabled", True))
        self.max_logged_outputs = max(1, int(output_logging.get("max_samples", 8)))
        self.save_local_outputs = bool(output_logging.get("save_local", True))
        self.max_local_outputs = max(1, int(output_logging.get("max_local_samples", 4)))
        self.local_output_root = (
            Path(config.get("logging", {}).get("save_dir", "runs"))
            / "qualitative_outputs"
        )
        self.output_log_every_n_epochs = max(
            1,
            int(
                output_logging.get(
                    "every_n_epochs", self.validation_plot_every_n_epochs
                )
            ),
        )
        self._prediction_samples: dict[str, list[dict[str, Any]]] = {}
        encoder_config = dict(self.model_config.get("encoder", {}))
        self.encoder = VisionTransformerSlotEncoder(
            self.input_resolution,
            input_channels=NUM_GRID_CLASSES,
            patch_size=int(encoder_config.get("patch_size", 8)),
            hidden_dim=int(encoder_config.get("hidden_dim", 192)),
            num_layers=int(encoder_config.get("num_layers", 3)),
            num_heads=int(encoder_config.get("num_heads", 6)),
            mlp_dim=int(encoder_config.get("mlp_dim", 512)),
            output_dim=int(encoder_config.get("output_dim", self.slot_dim)),
            bottleneck_hidden_dim=encoder_config.get("bottleneck_hidden_dim"),
            dropout=float(encoder_config.get("dropout", 0.0)),
            attention_dropout=float(encoder_config.get("attention_dropout", 0.0)),
        )

        decoder_config = dict(self.model_config.get("decoder", {}))
        self.decoder = SpatialBroadcastDecoder(
            self.slot_dim,
            self.input_resolution,
            content_channels=NUM_GRID_CLASSES,
            channels=decoder_config.get("channels", (64, 64, 64, 64)),
            kernel_size=int(decoder_config.get("kernel_size", 5)),
            output_activation="none",
        )

        feature_dim = self.encoder.output_dim
        self.max_task_tokens = MAX_COMPOSITION_DEPTH
        self.task_embedding = nn.Embedding(
            len(TASK_VOCABULARY) + 1,
            feature_dim,
            padding_idx=0,
        )
        self.task_position_embedding = nn.Parameter(
            torch.empty(1, self.max_task_tokens, feature_dim)
        )
        self.modality_embedding = nn.Embedding(2, feature_dim)
        nn.init.normal_(self.task_position_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding.weight, std=0.02)

    def _task_features(self, task_tokens: torch.Tensor) -> torch.Tensor:
        if task_tokens.size(1) > self.max_task_tokens:
            raise ValueError(
                f"Expected at most {self.max_task_tokens} task tokens, "
                f"got {task_tokens.size(1)}."
            )
        positions = self.task_position_embedding[:, : task_tokens.size(1)]
        task_features = self.task_embedding(task_tokens) + positions
        return task_features + self.modality_embedding.weight[1].view(1, 1, -1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["images"].float()
        if images.size(1) != NUM_GRID_CLASSES:
            raise ValueError("COGITAO images must be ten-channel one-hot grids.")
        image_features = self.encoder(images)
        image_features = image_features + self.modality_embedding.weight[0].view(
            1, 1, -1
        )
        task_tokens = batch["task_tokens"].long()
        task_mask = batch["task_token_mask"].bool()
        if task_tokens.shape != task_mask.shape:
            raise ValueError("task_tokens and task_token_mask must have equal shape.")
        task_features = self._task_features(task_tokens)
        features = torch.cat((image_features, task_features), dim=1)
        image_mask = torch.ones(
            image_features.shape[:2],
            dtype=torch.bool,
            device=images.device,
        )
        input_mask = torch.cat((image_mask, task_mask), dim=1)
        slot_output = self.slot_attention(features, input_mask=input_mask)
        slot_logits, alpha_logits = self.decoder.decode_slots(slot_output["slots"])
        masks = F.softmax(alpha_logits, dim=1)
        slot_log_probs = F.log_softmax(slot_logits, dim=2)
        output_log_probs = torch.logsumexp(
            slot_log_probs + masks.clamp_min(1.0e-8).log(), dim=1
        )
        predictions = output_log_probs.argmax(dim=1)

        image_token_count = image_features.size(1)
        height, width = self.encoder.feature_resolution
        attention = slot_output["attention"][:, :, :image_token_count]
        attention_maps = F.interpolate(
            attention.reshape(-1, 1, height, width),
            size=self.input_resolution,
            mode="bilinear",
            align_corners=False,
        ).reshape(images.size(0), self.num_slots, 1, *self.input_resolution)
        return {
            "images": images,
            "target_grid": batch["target_grid"].long(),
            "slots": slot_output["slots"],
            "attention": slot_output["attention"],
            "image_attention": attention,
            "task_attention": slot_output["attention"][:, :, image_token_count:],
            "attention_maps": attention_maps,
            "output_log_probs": output_log_probs,
            "predictions": predictions,
            "slot_logits": slot_logits,
            "masks": masks,
            "alpha_logits": alpha_logits,
            "reconstructions": output_log_probs.exp(),
            "slot_rgb": slot_log_probs.exp(),
        }

    def compute_losses(
        self, output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        target = output["target_grid"]
        return {"cross_entropy_loss": F.nll_loss(output["output_log_probs"], target)}

    def compute_metrics(
        self,
        output: dict[str, torch.Tensor],
        _batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        correct = output["predictions"].eq(output["target_grid"])
        object_mask = output["predictions"].ne(0) | output["target_grid"].ne(0)
        object_pixels = object_mask.flatten(1).sum(dim=1)
        correct_object_pixels = (correct & object_mask).flatten(1).sum(dim=1)
        object_accuracy = torch.where(
            object_pixels > 0,
            correct_object_pixels.float() / object_pixels.clamp_min(1),
            correct.flatten(1).all(dim=1).float(),
        )
        return {
            # COGITAO reports all evaluation accuracies as percentages.
            "grid_accuracy": 100.0 * correct.flatten(1).all(dim=1).float().mean(),
            "per_pixel_accuracy": 100.0 * correct.float().mean(),
            "object_per_pixel_accuracy": 100.0 * object_accuracy.mean(),
        }

    @staticmethod
    def _domain_name(split_name: str) -> str:
        return "ood" if split_name.endswith("_ood") else "id"

    def _log_paper_values(
        self,
        stage: str,
        domain: str | None,
        values: dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        prefix = stage if domain is None else f"{stage}/{domain}"
        self.log_dict(
            {f"{prefix}/{name}": value for name, value in values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
            add_dataloader_idx=False,
            batch_size=batch_size,
        )

    def _shared_paper_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
        split_name: str | None = None,
    ) -> torch.Tensor:
        output = self(batch)
        losses = self.compute_losses(output)
        total = self._weighted_total(losses)
        domain = None if split_name is None else self._domain_name(split_name)
        location = stage if domain is None else f"{stage}/{domain}"
        self._raise_on_nonfinite(losses, total, location)
        if stage == "val" and split_name is not None:
            self._cache_validation_output(output, split_name)
        self._cache_prediction_outputs(stage, domain, batch, output)
        values = {**losses, "total_loss": total, **self.compute_metrics(output, batch)}
        self._log_paper_values(stage, domain, values, output["images"].size(0))
        return total

    @staticmethod
    def _cache_key(stage: str, domain: str | None = None) -> str:
        return stage if domain is None else f"{stage}/{domain}"

    def _cache_prediction_outputs(
        self,
        stage: str,
        domain: str | None,
        batch: dict[str, Any],
        output: dict[str, torch.Tensor],
    ) -> None:
        if not self.log_output_tables:
            return
        key = self._cache_key(stage, domain)
        samples = self._prediction_samples.setdefault(key, [])
        remaining = self.max_logged_outputs - len(samples)
        if remaining <= 0:
            return

        input_grids = output["images"].argmax(dim=1).detach().cpu()
        target_grids = output["target_grid"].detach().cpu()
        predictions = output["predictions"].detach().cpu()
        metadata = batch.get("metadata", {})
        task_suites = metadata.get("transformation_suite", [])
        for index in range(min(remaining, input_grids.size(0))):
            correct = predictions[index].eq(target_grids[index])
            task_suite = (
                str(task_suites[index]) if index < len(task_suites) else "unavailable"
            )
            samples.append(
                {
                    "input": input_grids[index].to(torch.uint8),
                    "target": target_grids[index].to(torch.uint8),
                    "prediction": predictions[index].to(torch.uint8),
                    "task_suite": task_suite,
                    "grid_correct": bool(correct.all()),
                    "per_pixel_accuracy": 100.0 * float(correct.float().mean()),
                }
            )

    @staticmethod
    def _wandb_grid(grid: torch.Tensor, caption: str):
        rgb = ARC_PALETTE[grid.long()].numpy()
        return wandb.Image(rgb, caption=caption)

    @staticmethod
    def _save_grid_triptych(path: Path, sample: dict[str, Any]) -> None:
        panels = [
            (255.0 * ARC_PALETTE[sample[name].long()]).round().to(torch.uint8)
            for name in ("input", "target", "prediction")
        ]
        separator = torch.full((panels[0].size(0), 4, 3), 255, dtype=torch.uint8)
        triptych = torch.cat(
            (panels[0], separator, panels[1], separator, panels[2]), dim=1
        )
        PILImage.fromarray(triptych.numpy()).save(path)

    def _write_local_prediction_outputs(
        self, stage: str, *, require_trainer: bool = True
    ) -> None:
        if not self.save_local_outputs:
            return
        trainer = getattr(self, "_trainer", None)
        if require_trainer and (
            trainer is None
            or bool(getattr(trainer, "sanity_checking", False))
            or not bool(getattr(trainer, "is_global_zero", True))
        ):
            return

        prefix = f"{stage}/"
        for key, samples in self._prediction_samples.items():
            if key != stage and not key.startswith(prefix):
                continue
            split_dir = self.local_output_root.joinpath(*key.split("/"))
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, sample in enumerate(samples[: self.max_local_outputs]):
                self._save_grid_triptych(split_dir / f"sample_{index:02d}.png", sample)

    def _log_prediction_outputs(self, stage: str) -> None:
        if not self.log_output_tables or wandb is None:
            return
        trainer = getattr(self, "_trainer", None)
        if (
            trainer is None
            or bool(getattr(trainer, "sanity_checking", False))
            or not bool(getattr(trainer, "is_global_zero", True))
        ):
            return
        if stage != "test" and self.current_epoch % self.output_log_every_n_epochs:
            return
        experiment = getattr(getattr(self, "logger", None), "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return

        payload: dict[str, Any] = {"epoch": int(self.current_epoch)}
        prefix = f"{stage}/"
        for key, samples in self._prediction_samples.items():
            if key != stage and not key.startswith(prefix):
                continue
            table = wandb.Table(
                columns=[
                    "sample",
                    "task_suite",
                    "input_grid",
                    "target_output",
                    "predicted_output",
                    "perfect_grid_match",
                    "per_pixel_accuracy",
                ]
            )
            for index, sample in enumerate(samples):
                table.add_data(
                    index,
                    sample["task_suite"],
                    self._wandb_grid(sample["input"], "input"),
                    self._wandb_grid(sample["target"], "ground truth"),
                    self._wandb_grid(sample["prediction"], "prediction"),
                    sample["grid_correct"],
                    sample["per_pixel_accuracy"],
                )
            if samples:
                payload[f"outputs/{key}"] = table
        if len(payload) > 1:
            experiment.log(payload)

    def _reset_prediction_outputs(self, stage: str) -> None:
        prefix = f"{stage}/"
        self._prediction_samples = {
            key: value
            for key, value in self._prediction_samples.items()
            if key != stage and not key.startswith(prefix)
        }

    def on_train_epoch_start(self) -> None:
        self._reset_prediction_outputs("train")

    def on_train_epoch_end(self) -> None:
        self._write_local_prediction_outputs("train")
        self._log_prediction_outputs("train")

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._reset_prediction_outputs("val")

    def on_test_epoch_start(self) -> None:
        self._reset_prediction_outputs("test")

    def training_step(
        self, batch: dict[str, torch.Tensor], _batch_idx: int
    ) -> torch.Tensor:
        return self._shared_paper_step(batch, "train")

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
        return self._shared_paper_step(batch, "val", split_name)

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
        return self._shared_paper_step(batch, "test", split_name)

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
                prog_bar=False,
                sync_dist=False,
            )

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        self._log_generalization_gap("val")
        self._write_local_prediction_outputs("val")
        self._log_prediction_outputs("val")

    def on_test_epoch_end(self) -> None:
        self._log_generalization_gap("test")
        self._write_local_prediction_outputs("test")
        self._log_prediction_outputs("test")
