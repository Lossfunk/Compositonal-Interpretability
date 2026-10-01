from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - logging is optional in tests.
    wandb = None

try:
    import lightning.pytorch as L
except ImportError:  # pragma: no cover
    try:
        import pytorch_lightning as L
    except ImportError:
        L = None

try:
    from slot_attention.attention import SlotAttention
    from slot_attention.decoder import SpatialBroadcastDecoder
    from slot_attention.encoder import VisionTransformerSlotEncoder
    from slot_attention.metrics import segmentation_ari
except ImportError:  # pragma: no cover - direct-module import fallback.
    from attention import SlotAttention
    from decoder import SpatialBroadcastDecoder
    from encoder import VisionTransformerSlotEncoder
    from metrics import segmentation_ari


_BaseModule = L.LightningModule if L is not None else nn.Module


class SlotAttentionObjectDiscoveryModel(_BaseModule):
    """Slot Attention object discovery on CLEVR-2D with a tiny ViT trunk.

    The image is encoded into patch features, Slot Attention iterates a GRU to
    turn those features into a set of slots, and the spatial broadcast decoder
    turns each slot back into an RGB layer plus an alpha mask. Training signal
    is the reconstruction error of the alpha-composited image alone; the
    segmentation is never supervised.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_config = config["model"]["config"]
        self.input_resolution = tuple(
            int(value) for value in self.model_config.get("input_resolution", (96, 96))
        )
        self.num_slots = int(self.model_config.get("num_slots", 4))
        self.slot_dim = int(self.model_config.get("slot_dim", 64))
        self.image_range = str(self.model_config.get("image_range", "signed"))
        if self.image_range not in {"signed", "unit"}:
            raise ValueError("image_range must be 'signed' or 'unit'.")

        self.loss_weights = dict(
            self.model_config.get("loss_weights", {"reconstruction_loss": 1.0})
        )
        unknown_losses = set(self.loss_weights) - {"reconstruction_loss"}
        if unknown_losses:
            raise ValueError(
                "Unsupported Slot Attention losses: " + ", ".join(sorted(unknown_losses))
            )

        encoder_config = dict(self.model_config.get("encoder", {}))
        self.encoder = VisionTransformerSlotEncoder(
            self.input_resolution,
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
        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            input_dim=self.encoder.output_dim,
            slot_dim=self.slot_dim,
            mlp_hidden_dim=int(self.model_config.get("slot_mlp_hidden_dim", 128)),
            num_iterations=int(self.model_config.get("num_iterations", 3)),
            implicit_differentiation=bool(
                self.model_config.get("implicit_differentiation", False)
            ),
        )
        decoder_config = dict(self.model_config.get("decoder", {}))
        self.decoder = SpatialBroadcastDecoder(
            self.slot_dim,
            self.input_resolution,
            channels=decoder_config.get("channels", (64, 64, 64, 64)),
            kernel_size=int(decoder_config.get("kernel_size", 5)),
            output_activation=str(
                decoder_config.get(
                    "output_activation",
                    "none" if self.image_range == "signed" else "sigmoid",
                )
            ),
        )

        self.validation_split_names = [
            str(name)
            for name in self.model_config.get("validation_split_names", ["validation"])
        ]
        self.eval_plot_config = dict(self.model_config.get("eval_plots", {}))
        self.enabled_eval_plots = {
            str(name) for name in self.eval_plot_config.get("enabled", [])
        }
        self.max_reconstruction_samples = max(
            1, int(self.eval_plot_config.get("max_reconstruction_samples", 8))
        )
        self.max_decomposition_samples = max(
            1, int(self.eval_plot_config.get("max_decomposition_samples", 6))
        )
        self.max_slot_pca_samples = max(
            2, int(self.eval_plot_config.get("max_slot_pca_samples", 512))
        )
        self.validation_plot_every_n_epochs = max(
            1, int(self.eval_plot_config.get("every_n_epochs", 1))
        )
        self._reset_validation_plot_cache()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters({"config": config})

    # ------------------------------------------------------------------
    # Image space helpers
    # ------------------------------------------------------------------
    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to the model resolution and map ``[0, 1]`` into model space."""
        if tuple(images.shape[-2:]) != self.input_resolution:
            images = F.interpolate(
                images,
                size=self.input_resolution,
                mode="bilinear",
                align_corners=False,
            )
        if self.image_range == "signed":
            return images * 2.0 - 1.0
        return images

    def to_display_space(self, images: torch.Tensor) -> torch.Tensor:
        if self.image_range == "signed":
            images = (images + 1.0) / 2.0
        return images.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward and losses
    # ------------------------------------------------------------------
    def forward(
        self, batch: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        raw_images = batch["images"] if isinstance(batch, dict) else batch
        images = self._prepare_images(raw_images)
        features = self.encoder(images)
        slot_output = self.slot_attention(features)
        decoded = self.decoder(slot_output["slots"])

        height, width = self.encoder.feature_resolution
        attention = slot_output["attention"].reshape(
            images.size(0), -1, 1, height, width
        )
        attention_maps = F.interpolate(
            attention.flatten(0, 1),
            size=self.input_resolution,
            mode="bilinear",
            align_corners=False,
        ).reshape(images.size(0), -1, 1, *self.input_resolution)
        return {
            "images": images,
            "slots": slot_output["slots"],
            "attention": slot_output["attention"],
            "attention_maps": attention_maps,
            "reconstructions": decoded["reconstruction"],
            "slot_rgb": decoded["slot_rgb"],
            "masks": decoded["masks"],
            "alpha_logits": decoded["alpha_logits"],
        }

    def compute_losses(
        self, output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "reconstruction_loss": F.mse_loss(
                output["reconstructions"], output["images"]
            )
        }

    def compute_metrics(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Segmentation quality, when the dataset supplies instance masks."""
        metrics: dict[str, torch.Tensor] = {}
        masks = output["masks"]
        occupancy = masks.squeeze(2).amax(dim=1)
        metrics["mean_max_mask"] = occupancy.mean()
        # How many slots actually claim pixels, averaged over the batch.
        slot_share = masks.squeeze(2).flatten(2).mean(dim=-1)
        metrics["active_slots"] = (slot_share > 0.02).float().sum(dim=1).mean()

        instance_masks = batch.get("instance_masks") if isinstance(batch, dict) else None
        if instance_masks is None:
            return metrics
        instance_masks = instance_masks.to(masks.device)
        if tuple(instance_masks.shape[-2:]) != self.input_resolution:
            instance_masks = F.interpolate(
                instance_masks,
                size=self.input_resolution,
                mode="nearest",
            )
        foreground_ari = segmentation_ari(
            instance_masks, masks.detach(), foreground_only=True
        )
        full_ari = segmentation_ari(
            instance_masks, masks.detach(), foreground_only=False
        )
        metrics["foreground_ari"] = torch.nanmean(foreground_ari)
        metrics["ari"] = torch.nanmean(full_ari)
        return metrics

    def _weighted_total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.loss_weights[name] * losses[name] for name in self.loss_weights)

    def _raise_on_nonfinite(
        self, losses: dict[str, torch.Tensor], total: torch.Tensor, stage: str
    ) -> None:
        if not self.model_config.get("fail_on_nonfinite", True):
            return
        bad = [
            name
            for name, value in {**losses, "total_loss": total}.items()
            if not bool(torch.isfinite(value.detach()).all())
        ]
        if bad:
            raise FloatingPointError(
                f"Non-finite {stage} losses: {', '.join(sorted(bad))}"
            )

    def _log_losses(
        self,
        values: dict[str, torch.Tensor],
        prefix: str = "",
        prog_bar: bool = False,
    ) -> None:
        if not hasattr(self, "log_dict"):
            return
        self.log_dict(
            {f"{prefix}{name}": value for name, value in values.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=prog_bar,
            sync_dist=False,
            add_dataloader_idx=False,
            batch_size=getattr(self, "_current_batch_size", None),
        )

    # ------------------------------------------------------------------
    # Validation caching
    # ------------------------------------------------------------------
    def _eval_plot_enabled(self, name: str) -> bool:
        return name in self.enabled_eval_plots

    @staticmethod
    def _empty_cache() -> dict[str, list]:
        return {
            "images": [],
            "reconstructions": [],
            "slot_rgb": [],
            "masks": [],
            "attention_maps": [],
            "slots": [],
            "slot_colors": [],
            "slot_areas": [],
        }

    def _reset_validation_plot_cache(self) -> None:
        self._validation_plot_cache = {
            split_name: self._empty_cache()
            for split_name in self.validation_split_names
        }

    def on_validation_epoch_start(self) -> None:
        self._reset_validation_plot_cache()

    def _cache_validation_output(
        self, output: dict[str, torch.Tensor], split_name: str
    ) -> None:
        if not self.enabled_eval_plots:
            return
        cache = self._validation_plot_cache.setdefault(split_name, self._empty_cache())
        display_images = self.to_display_space(output["images"]).detach().float().cpu()

        needs_reconstructions = self._eval_plot_enabled("validation_reconstructions")
        needs_decomposition = self._eval_plot_enabled("slot_decomposition")
        if needs_reconstructions or needs_decomposition:
            limit = max(
                self.max_reconstruction_samples if needs_reconstructions else 0,
                self.max_decomposition_samples if needs_decomposition else 0,
            )
            cached = sum(item.size(0) for item in cache["images"])
            remaining = limit - cached
            if remaining > 0:
                cache["images"].append(display_images[:remaining])
                cache["reconstructions"].append(
                    self.to_display_space(output["reconstructions"][:remaining])
                    .detach()
                    .float()
                    .cpu()
                )
                cache["slot_rgb"].append(
                    self.to_display_space(output["slot_rgb"][:remaining])
                    .detach()
                    .float()
                    .cpu()
                )
                cache["masks"].append(
                    output["masks"][:remaining].detach().float().cpu()
                )
                cache["attention_maps"].append(
                    output["attention_maps"][:remaining].detach().float().cpu()
                )

        if self._eval_plot_enabled("slot_representation_pca"):
            cached = sum(item.size(0) for item in cache["slots"])
            remaining = self.max_slot_pca_samples - cached
            if remaining > 0:
                masks = output["masks"][:remaining].detach().float().cpu()
                images = display_images[:remaining].unsqueeze(1)
                weight = masks.flatten(3).sum(dim=-1).clamp_min(1.0e-6)
                # Average image color under each slot's alpha mask: a slot that
                # owns a red square lands on a red point in the scatter.
                slot_colors = (images * masks).flatten(3).sum(dim=-1) / weight
                cache["slots"].append(
                    output["slots"][:remaining].detach().float().cpu()
                )
                cache["slot_colors"].append(slot_colors)
                cache["slot_areas"].append(masks.flatten(2).mean(dim=-1))

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    @staticmethod
    def _display_image(image: torch.Tensor) -> np.ndarray:
        return image.detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

    def _cached(self, split_name: str, key: str) -> torch.Tensor | None:
        cache = self._validation_plot_cache.get(split_name, {})
        items = cache.get(key)
        if not items:
            return None
        return torch.cat(items)

    def _plot_validation_reconstructions(self, split_name: str) -> plt.Figure | None:
        images = self._cached(split_name, "images")
        if images is None:
            return None
        images = images[: self.max_reconstruction_samples]
        reconstructions = self._cached(split_name, "reconstructions")[
            : self.max_reconstruction_samples
        ]
        errors = (reconstructions - images).abs().mean(dim=1)
        sample_mse = (reconstructions - images).square().flatten(1).mean(1)
        error_limit = max(float(torch.quantile(errors.flatten(), 0.995)), 0.05)
        fig, axes = plt.subplots(
            images.size(0), 3, figsize=(8.4, 2.65 * images.size(0)), squeeze=False
        )
        for sample_idx in range(images.size(0)):
            axes[sample_idx, 0].imshow(self._display_image(images[sample_idx]))
            axes[sample_idx, 1].imshow(self._display_image(reconstructions[sample_idx]))
            axes[sample_idx, 2].imshow(
                errors[sample_idx].numpy(), cmap="magma", vmin=0.0, vmax=error_limit
            )
            axes[sample_idx, 1].text(
                0.5,
                -0.06,
                f"MSE {float(sample_mse[sample_idx]):.6f}",
                ha="center",
                va="top",
                transform=axes[sample_idx, 1].transAxes,
                fontsize=8,
            )
            for axis in axes[sample_idx]:
                axis.axis("off")
        for axis, title in zip(axes[0], ("input", "slot composite", "absolute error")):
            axis.set_title(title, fontsize=10, fontweight="bold")
        fig.suptitle(
            f"{split_name}: Slot Attention reconstructions | "
            f"mean MSE {float(sample_mse.mean()):.6f}"
        )
        fig.tight_layout()
        return fig

    def _plot_slot_decomposition(self, split_name: str) -> plt.Figure | None:
        images = self._cached(split_name, "images")
        if images is None:
            return None
        count = min(self.max_decomposition_samples, images.size(0))
        images = images[:count]
        reconstructions = self._cached(split_name, "reconstructions")[:count]
        slot_rgb = self._cached(split_name, "slot_rgb")[:count]
        masks = self._cached(split_name, "masks")[:count]
        attention_maps = self._cached(split_name, "attention_maps")[:count]
        slot_count = masks.size(1)

        rows = 3 * count
        columns = 2 + slot_count
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(1.7 * columns, 1.75 * rows),
            squeeze=False,
        )
        for sample_idx in range(count):
            top, middle, bottom = (
                axes[3 * sample_idx],
                axes[3 * sample_idx + 1],
                axes[3 * sample_idx + 2],
            )
            attention_limit = max(float(attention_maps[sample_idx].max()), 1e-6)
            top[0].imshow(self._display_image(images[sample_idx]))
            top[0].set_ylabel(f"sample {sample_idx}", fontsize=8)
            top[1].imshow(self._display_image(reconstructions[sample_idx]))
            # The unsupervised segmentation: which slot owns each pixel.
            middle[0].imshow(
                masks[sample_idx].squeeze(1).argmax(dim=0).numpy(),
                cmap="tab10",
                vmin=0,
                vmax=max(9, slot_count - 1),
                interpolation="nearest",
            )
            middle[1].axis("off")
            bottom[0].axis("off")
            bottom[1].axis("off")
            if sample_idx == 0:
                top[0].set_title("input", fontsize=9, fontweight="bold")
                top[1].set_title("composite", fontsize=9, fontweight="bold")
                middle[0].set_title("segmentation", fontsize=9, fontweight="bold")
            for slot_idx in range(slot_count):
                mask = masks[sample_idx, slot_idx]
                # What the slot draws where it claims the image; elsewhere the
                # layer fades to white so the slot's own extent is visible.
                masked = slot_rgb[sample_idx, slot_idx] * mask + (1.0 - mask)
                top[2 + slot_idx].imshow(self._display_image(masked))
                middle[2 + slot_idx].imshow(
                    mask[0].numpy(), cmap="viridis", vmin=0.0, vmax=1.0
                )
                bottom[2 + slot_idx].imshow(
                    attention_maps[sample_idx, slot_idx, 0].numpy(),
                    cmap="magma",
                    vmin=0.0,
                    vmax=attention_limit,
                )
                if sample_idx == 0:
                    top[2 + slot_idx].set_title(
                        f"slot {slot_idx}", fontsize=9, fontweight="bold"
                    )
                middle[2 + slot_idx].set_xlabel("alpha", fontsize=7)
                bottom[2 + slot_idx].set_xlabel("attention", fontsize=7)
            for row in (top, middle, bottom):
                for axis in row:
                    axis.set_xticks([])
                    axis.set_yticks([])
        fig.suptitle(
            f"{split_name}: per-slot reconstruction, alpha mask, and "
            "slot-attention map"
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        return fig

    def _plot_slot_representation_pca(self, split_name: str) -> plt.Figure | None:
        slots = self._cached(split_name, "slots")
        if slots is None or slots.size(0) < 2:
            return None
        slot_colors = self._cached(split_name, "slot_colors")
        slot_areas = self._cached(split_name, "slot_areas")
        sample_count, slot_count, _ = slots.shape
        flat = slots.reshape(sample_count * slot_count, -1)
        centered = flat - flat.mean(dim=0, keepdim=True)
        _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
        if components.size(0) < 2:
            return None
        projected = centered @ components[:2].T
        eigenvalues = singular_values.square() / max(1, flat.size(0) - 1)
        explained = eigenvalues / eigenvalues.sum().clamp_min(1.0e-8)

        colors = slot_colors.reshape(-1, 3).clamp(0.0, 1.0).numpy()
        areas = slot_areas.reshape(-1).numpy()
        slot_index = (
            torch.arange(slot_count).repeat(sample_count).numpy()
        )

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
        axes[0].scatter(
            projected[:, 0].numpy(),
            projected[:, 1].numpy(),
            c=colors,
            s=8.0 + 120.0 * areas,
            edgecolors="black",
            linewidths=0.2,
            alpha=0.85,
        )
        axes[0].set_title(
            "slot vectors colored by the mean image color under their mask",
            fontsize=10,
        )
        scatter = axes[1].scatter(
            projected[:, 0].numpy(),
            projected[:, 1].numpy(),
            c=slot_index,
            cmap="tab10",
            vmin=0,
            vmax=max(9, slot_count - 1),
            s=10.0,
            alpha=0.8,
        )
        axes[1].set_title("same points colored by slot position", fontsize=10)
        fig.colorbar(scatter, ax=axes[1], label="slot index")
        for axis in axes:
            axis.set_xlabel(f"PC1 ({float(explained[0]):.1%})")
            axis.set_ylabel(f"PC2 ({float(explained[1]):.1%})")
            axis.grid(alpha=0.15)
        fig.suptitle(
            f"{split_name}: PCA of {sample_count * slot_count} slot representations"
        )
        fig.tight_layout()
        return fig

    def on_validation_epoch_end(self) -> None:
        trainer = getattr(self, "_trainer", None)
        epoch = int(getattr(self, "current_epoch", 0))
        should_log = epoch % self.validation_plot_every_n_epochs == 0
        if (
            not should_log
            or not self.enabled_eval_plots
            or bool(getattr(trainer, "sanity_checking", False))
            or (
                trainer is not None
                and not bool(getattr(trainer, "is_global_zero", True))
            )
        ):
            self._reset_validation_plot_cache()
            return
        experiment = getattr(getattr(self, "logger", None), "experiment", None)
        if experiment is None or not hasattr(experiment, "log") or wandb is None:
            self._reset_validation_plot_cache()
            return
        payload: dict[str, Any] = {"epoch": epoch}
        plotters = {
            "validation_reconstructions": (
                "reconstructions",
                self._plot_validation_reconstructions,
            ),
            "slot_decomposition": ("slot_decomposition", self._plot_slot_decomposition),
            "slot_representation_pca": (
                "slot_representation_pca",
                self._plot_slot_representation_pca,
            ),
        }
        for split_name in self.validation_split_names:
            for plot_name, (suffix, plotter) in plotters.items():
                if not self._eval_plot_enabled(plot_name):
                    continue
                figure = plotter(split_name)
                if figure is not None:
                    payload[f"val/{split_name}_{suffix}"] = wandb.Image(figure)
                    plt.close(figure)
        if len(payload) > 1:
            experiment.log(payload)
        self._reset_validation_plot_cache()

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    def training_step(self, batch: dict[str, torch.Tensor], _batch_idx: int):
        output = self(batch)
        losses = self.compute_losses(output)
        total = self._weighted_total(losses)
        self._raise_on_nonfinite(losses, total, "training")
        self._current_batch_size = output["images"].size(0)
        self._log_losses({**losses, "total_loss": total}, prog_bar=True)
        return total

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        _batch_idx: int,
        dataloader_idx: int = 0,
    ):
        output = self(batch)
        losses = self.compute_losses(output)
        total = self._weighted_total(losses)
        split_name = (
            self.validation_split_names[dataloader_idx]
            if dataloader_idx < len(self.validation_split_names)
            else f"loader_{dataloader_idx}"
        )
        self._raise_on_nonfinite(losses, total, f"validation/{split_name}")
        self._cache_validation_output(output, split_name)
        self._current_batch_size = output["images"].size(0)
        metrics = self.compute_metrics(output, batch)
        self._log_losses(
            {**losses, "total_loss": total, **metrics}, prefix=f"val_{split_name}_"
        )
        return total

    def configure_optimizers(self):
        optimizer_config = self.config["trainer"]["optimizer"]
        optimizer_type = optimizer_config.get("type", "Adam")
        kwargs = dict(optimizer_config.get("config", {}))
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
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
            """Linear warmup, then the paper's exponential decay."""
            warmup = 1.0 if warmup_steps == 0 else min(1.0, (step + 1) / warmup_steps)
            return warmup * decay_rate ** (step / decay_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
