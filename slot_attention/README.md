# Slot Attention

A self-contained replication of *Object-Centric Learning with Slot Attention*
(Locatello et al., 2020) on CLEVR-2D. Nothing here imports from `src/`, and
nothing in `src/` imports from here — the folder is a standalone reference
implementation of the module and its object-discovery objective.

| File | Contents |
| --- | --- |
| `attention.py` | `SlotAttention` (iterative attention + `GRUCell` + residual MLP) and `SoftPositionEmbed` |
| `encoder.py` | `VisionTransformerSlotEncoder`: the repo's three-layer ViT trunk, emitting one token per patch |
| `decoder.py` | `SpatialBroadcastDecoder`: per-slot RGB layers and alpha masks composed by a softmax over slots |
| `metrics.py` | ARI and the foreground-only FG-ARI reported in the paper |
| `model.py` | `SlotAttentionObjectDiscoveryModel`, the Lightning module wired into `src/train.py` |
| `visualize.py` | Offline figures: per-slot reconstructions/masks and slot-representation PCA |

## The module

One iteration, given input features `x` and current slots `s`:

1. `q = W_q LN(s)`, `k = W_k LN(x)`, `v = W_v LN(x)`.
2. `attn = softmax(q k^T / sqrt(D), dim=slots)` — the softmax runs over the
   **slot** axis, so slots compete for each input feature instead of all
   drifting onto the same one. This is the only structural difference from
   ordinary cross-attention, and it is what makes the slots specialize.
3. `updates = weighted_mean(attn, v)` — normalizing over inputs keeps the
   update scale comparable for a slot that wins few features.
4. `s = GRUCell(updates, s)`, then `s = s + MLP(LN(s))`.

Slots are initialized by sampling from a learned Gaussian
(`slots_mu`, `exp(slots_log_sigma)`), which is why they are exchangeable: the
module is permutation-equivariant in its slots and slot index carries no
identity. `implicit_differentiation: true` runs the first `T-1` iterations
under `no_grad` and backpropagates through a single step from the fixed point.

## Object discovery

Each slot is broadcast to a 6×6 grid, given a soft position code, and upsampled
by four transposed convolutions to 96×96 with four output channels: RGB plus
one alpha. The alphas are softmaxed across slots and the reconstruction is the
alpha-weighted sum of the slot layers. The loss is the MSE of that composite
against the input — **the segmentation is never supervised**. Instance masks
appear only in the validation splits, only to score FG-ARI.

See `docs/clevr_2d_slot_attention.md` for how to run it and what the figures
show.
