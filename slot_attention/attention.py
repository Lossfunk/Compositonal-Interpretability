from __future__ import annotations

import torch
from torch import nn


class SlotAttention(nn.Module):
    """Slot Attention from Locatello et al. 2020, iterated with a GRU.

    Slots are sampled from a shared Gaussian, then refined for
    ``num_iterations`` rounds. Each round scores slots against the input
    features, normalizes the scores over the *slot* axis so slots compete for
    the same feature, aggregates a weighted mean per slot, and pushes that
    update through a ``GRUCell`` followed by a residual MLP.
    """

    def __init__(
        self,
        num_slots: int,
        input_dim: int,
        slot_dim: int,
        mlp_hidden_dim: int,
        *,
        num_iterations: int = 3,
        epsilon: float = 1.0e-8,
        implicit_differentiation: bool = False,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_iterations = int(num_iterations)
        self.epsilon = float(epsilon)
        self.implicit_differentiation = bool(implicit_differentiation)
        if self.num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive.")

        self.scale = self.slot_dim**-0.5

        self.slots_mu = nn.Parameter(torch.empty(1, 1, self.slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.empty(1, 1, self.slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.norm_inputs = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.to_k = nn.Linear(self.input_dim, self.slot_dim, bias=False)
        self.to_v = nn.Linear(self.input_dim, self.slot_dim, bias=False)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, int(mlp_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(mlp_hidden_dim), self.slot_dim),
        )

    def sample_slots(
        self,
        batch_size: int,
        num_slots: int | None = None,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        slot_count = self.num_slots if num_slots is None else int(num_slots)
        device = self.slots_mu.device if device is None else device
        noise = torch.randn(
            batch_size,
            slot_count,
            self.slot_dim,
            device=device,
            dtype=self.slots_mu.dtype,
            generator=generator,
        )
        return self.slots_mu + self.slots_log_sigma.exp() * noise

    def _step(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        input_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, slot_count, slot_dim = slots.shape
        previous_slots = slots
        queries = self.to_q(self.norm_slots(slots)) * self.scale
        logits = torch.einsum("bsd,bnd->bsn", queries, keys)
        # Softmax over slots: every input feature is distributed across the
        # slots, which is what makes the slots compete instead of duplicate.
        attention = logits.softmax(dim=1)
        if input_mask is not None:
            attention = attention * input_mask[:, None, :].to(attention.dtype)
        # Weighted mean over inputs, so a slot that wins few features still
        # receives an update of comparable scale.
        weights = attention + self.epsilon
        if input_mask is not None:
            weights = weights * input_mask[:, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        updates = torch.einsum("bsn,bnd->bsd", weights, values)
        slots = self.gru(
            updates.reshape(-1, slot_dim),
            previous_slots.reshape(-1, slot_dim),
        ).reshape(batch_size, slot_count, slot_dim)
        slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attention

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        num_slots: int | None = None,
        slots: torch.Tensor | None = None,
        input_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape (batch, tokens, input_dim).")
        if inputs.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {inputs.size(-1)}."
            )
        if input_mask is not None:
            if input_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    "input_mask must have shape (batch, tokens), got "
                    f"{tuple(input_mask.shape)} for inputs {tuple(inputs.shape)}."
                )
            input_mask = input_mask.to(device=inputs.device, dtype=torch.bool)
        inputs = self.norm_inputs(inputs)
        keys = self.to_k(inputs)
        values = self.to_v(inputs)
        if slots is None:
            slots = self.sample_slots(
                inputs.size(0),
                num_slots,
                device=inputs.device,
                generator=generator,
            )

        iterations = self.num_iterations
        attention = None
        if self.implicit_differentiation:
            # Run the fixed-point search without tracking gradients, then take
            # a single differentiable step from the converged slots.
            with torch.no_grad():
                for _ in range(iterations - 1):
                    slots, attention = self._step(slots, keys, values, input_mask)
            slots = slots.detach()
            iterations = 1
        for _ in range(iterations):
            slots, attention = self._step(slots, keys, values, input_mask)
        return {"slots": slots, "attention": attention}


class SoftPositionEmbed(nn.Module):
    """Additive position code built from a linear projection of a pixel grid.

    The grid holds ``(x, 1 - x, y, 1 - y)`` so a single linear layer can express
    any affine ramp over the feature map.
    """

    def __init__(self, hidden_dim: int, resolution: tuple[int, int]) -> None:
        super().__init__()
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.projection = nn.Linear(4, int(hidden_dim))
        self.register_buffer(
            "grid", self._build_grid(self.resolution), persistent=False
        )

    @staticmethod
    def _build_grid(resolution: tuple[int, int]) -> torch.Tensor:
        height, width = resolution
        rows = torch.linspace(0.0, 1.0, height)
        columns = torch.linspace(0.0, 1.0, width)
        y_grid, x_grid = torch.meshgrid(rows, columns, indexing="ij")
        grid = torch.stack([x_grid, 1.0 - x_grid, y_grid, 1.0 - y_grid], dim=-1)
        return grid.unsqueeze(0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Add the position code to ``(batch, channels, height, width)``."""
        code = self.projection(self.grid.to(features.dtype))
        return features + code.permute(0, 3, 1, 2)
