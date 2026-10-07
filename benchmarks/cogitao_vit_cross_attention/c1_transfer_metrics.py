"""Geometry and diagnosis for compositional transfer (no model or oracle logic)."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_metrics import (
    EPS,
    correlation,
)


def residual_pca(residuals):
    """Centered PCA directions; project *uncentered* action residuals onto them.

    The training mean is reported separately. Zero-variance directions are never
    used to pad a requested rank. OOD PCA is descriptive, never a fitted ID basis.
    """
    value = np.asarray(residuals, dtype=np.float64)
    centered = value - value.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        max(centered.shape)
        * np.finfo(float).eps
        * (singular[0] if len(singular) else 0)
    )
    rank = int(np.sum(singular > tolerance))
    energy = singular[:rank] ** 2
    fraction = energy / energy.sum() if energy.sum() > EPS else np.zeros(rank)
    return (
        vt[:rank].T,
        fraction,
        dict(
            samples=len(value),
            dimension=value.shape[1],
            effective_rank=rank,
            centered=True,
            residual_mean_norm=float(np.linalg.norm(value.mean(axis=0))),
        ),
    )


def rank_choices(basis, variance, fixed, thresholds):
    choices = [(f"k{rank}", rank) for rank in fixed if rank <= basis.shape[1]]
    for threshold in thresholds:
        if len(variance):
            rank = min(
                len(variance), int(np.searchsorted(np.cumsum(variance), threshold)) + 1
            )
            choices.append((f"variance_{threshold:g}", rank))
    return choices


def projection_metrics(residuals, basis):
    value = np.asarray(residuals, dtype=np.float64)
    projected = (value @ basis) @ basis.T
    norms = np.linalg.norm(value, axis=1)
    error = np.linalg.norm(value - projected, axis=1) / (norms + EPS)
    energy = 1 - np.sum((value @ basis) ** 2, axis=1) / (norms**2 + EPS)
    # A zero action has no residual energy and provides no directional evidence.
    energy = np.where(norms <= EPS, 0.0, energy)
    return np.clip(error, 0, 1), np.clip(energy, 0, 1)


def principal_angle_defect(left, right):
    if left.shape[1] == 0 or left.shape[1] != right.shape[1]:
        return None
    singular = np.linalg.svd(left.T @ right, compute_uv=False)
    return float(np.mean(1 - np.clip(singular, 0, 1) ** 2))


def random_subspace(dimension, rank, rng):
    return np.linalg.qr(rng.normal(size=(dimension, rank)), mode="reduced")[0]


def similarity(left, right):
    """Matched rows; CKA is undefined for constant program-only representations."""
    x = np.asarray(left, dtype=np.float64).reshape(len(left), -1)
    y = np.asarray(right, dtype=np.float64).reshape(len(right), -1)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    cosine = 1 - np.clip(np.sum(x * y, axis=1) / np.maximum(nx * ny, EPS), -1, 1)
    cosine = np.where((nx <= EPS) & (ny <= EPS), 0, cosine)
    l2 = np.linalg.norm(x - y, axis=1) / ((nx + ny) / 2 + EPS)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    kx, ky = xc @ xc.T, yc @ yc.T
    scale = np.linalg.norm(kx) * np.linalg.norm(ky)
    cka = float(np.sum(kx * ky) / scale) if len(x) >= 3 and scale > EPS else None

    # Sample-space SVD avoids constructing a huge feature covariance matrix.
    def basis(value, gram):
        eigenvalues, vectors = np.linalg.eigh(gram)
        if eigenvalues[-1] <= EPS:
            return np.empty((value.shape[1], 0))
        valid = np.flatnonzero(
            eigenvalues > max(value.shape) * np.finfo(float).eps * eigenvalues[-1]
        )[-32:][::-1]
        return value.T @ (vectors[:, valid] / np.sqrt(eigenvalues[valid]))

    ux, uy = basis(xc, kx), basis(yc, ky)
    rank = min(ux.shape[1], uy.shape[1])
    angle = principal_angle_defect(ux[:, :rank], uy[:, :rank])
    return dict(cosine_distance=cosine, normalized_l2=l2), dict(
        cka=cka,
        principal_angle_defect=angle,
        similarity_rank=rank,
        cka_status="ok" if cka is not None else "too_few_or_constant_samples",
    )


def distribution_divergence(left, right, logits=False):
    """Symmetric KL and JS, averaged across all query/pixel distributions."""
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if logits:
        x = np.exp(x - logsumexp(x, axis=-1, keepdims=True))
        y = np.exp(y - logsumexp(y, axis=-1, keepdims=True))
    x, y = np.maximum(x, EPS), np.maximum(y, EPS)
    x, y = x / x.sum(axis=-1, keepdims=True), y / y.sum(axis=-1, keepdims=True)
    m = (x + y) / 2
    kl = 0.5 * (np.sum(x * np.log(x / y), axis=-1) + np.sum(y * np.log(y / x), axis=-1))
    js = 0.5 * (np.sum(x * np.log(x / m), axis=-1) + np.sum(y * np.log(y / m), axis=-1))
    return dict(
        symmetric_kl=kl.reshape(len(x), -1).mean(axis=1),
        js=js.reshape(len(x), -1).mean(axis=1),
    )


def diagnostic_evidence(behavior, latent, good, bad, gap_threshold, low_error):
    """Multi-label provisional diagnoses with all values and thresholds exposed."""
    a, b, c, d, first = (
        behavior.get(key)
        for key in (
            "A_object_accuracy",
            "B_object_accuracy",
            "C_object_accuracy",
            "D_object_accuracy",
            "C_first_object_accuracy",
        )
    )
    error, gap = latent.get("context_error"), latent.get("transfer_gap")
    complete = all(value is not None for value in (a, b, c, d, first, error, gap))
    labels = []
    if complete:
        if a >= good and b <= bad and gap >= gap_threshold and error > low_error:
            labels.append("Type A: state-geometry transfer failure")
        if b >= good and d <= bad and error <= low_error:
            labels.append("Type B: program/binding failure")
        if b >= good and c <= bad and first < good:
            labels.append("Type C: model-intermediate compounding failure")
        if (
            a >= good
            and b <= bad
            and d <= bad
            and error > low_error
            and gap >= gap_threshold
        ):
            labels.append("Type D: mixed")
    return dict(
        diagnosis="; ".join(labels) if labels else "inconclusive",
        provisional=True,
        A_object_accuracy=a,
        B_object_accuracy=b,
        C_object_accuracy=c,
        D_object_accuracy=d,
        C_first_object_accuracy=first,
        context_error=error,
        transfer_gap=gap,
        atomic_to_context_drop=None if a is None or b is None else a - b,
        oracle_to_direct_drop=None if b is None or d is None else b - d,
        oracle_to_sequential_drop=None if b is None or c is None else b - c,
        good_threshold=good,
        bad_threshold=bad,
        transfer_gap_threshold=gap_threshold,
        low_latent_error_threshold=low_error,
    )


def bootstrap_correlation(x, y, seed, repeats=200, clusters=None):
    """Cluster resampling when checkpoint ids are supplied; never invent replicates."""
    result = correlation(x, y)
    result.update(
        pearson_ci_low=None,
        pearson_ci_high=None,
        bootstrap_unit="checkpoint" if clusters is not None else "example",
        independent_replicates=False,
    )
    if result["status"] != "ok" or not repeats:
        return result
    valid = np.array(
        [
            a is not None and b is not None and np.isfinite(a) and np.isfinite(b)
            for a, b in zip(x, y)
        ]
    )
    xv, yv = np.asarray(x, dtype=float)[valid], np.asarray(y, dtype=float)[valid]
    rng = np.random.default_rng(seed)
    groups = None
    if clusters is not None:
        cv = np.asarray(clusters)[valid]
        names = np.unique(cv)
        result["checkpoint_clusters"] = len(names)
        if len(names) < 3:
            return {
                **result,
                "bootstrap_status": "fewer_than_three_checkpoint_clusters",
            }
        groups = [np.flatnonzero(cv == name) for name in names]
    scores = []
    for _ in range(repeats):
        indices = (
            rng.integers(0, len(xv), len(xv))
            if groups is None
            else np.concatenate(
                [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            )
        )
        if np.ptp(xv[indices]) > EPS and np.ptp(yv[indices]) > EPS:
            scores.append(float(np.corrcoef(xv[indices], yv[indices])[0, 1]))
    if scores:
        result["pearson_ci_low"], result["pearson_ci_high"] = map(
            float, np.quantile(scores, [0.025, 0.975])
        )
    return result
