"""Post-hoc affine representation probes for the discrete rotation group C24.

These are fitted probes of frozen hidden states, not operators extracted from
network weights. With column homogeneous coordinates, [a,b] acts as rho(b)rho(a).
"""

from itertools import product
from math import gcd

import torch

from benchmarks.cogitao_vit_cross_attention.degree_rotation_data import ATOMIC_ANGLES

GROUP_ANGLES = tuple(range(0, 360, 15))


def finite_scalar(value):
    value = torch.as_tensor(value)
    return float(value) if bool(torch.isfinite(value)) else None


def relative_error(left, right):
    denominator = (left.norm() + right.norm()) / 2
    if not bool(torch.isfinite(denominator)):
        return None
    return finite_scalar((left - right).norm() / denominator.clamp_min(1e-12))


def latent_metrics(left, right):
    left, right = left.double(), right.double()
    norms = left.norm(dim=1) * right.norm(dim=1)
    valid = norms > 1e-12
    centered_left = left - left.mean(0)
    centered_right = right - right.mean(0)
    cross = centered_left.T @ centered_right
    cka_denominator = (centered_left.T @ centered_left).norm() * (centered_right.T @ centered_right).norm()
    return {
        "relative_l2": relative_error(left, right),
        "rmse": finite_scalar((left - right).square().mean().sqrt()),
        "cosine_mean": finite_scalar(((left * right).sum(1)[valid] / norms[valid]).mean()) if bool(valid.any()) else None,
        "linear_cka": finite_scalar(cross.square().sum() / cka_denominator) if cka_denominator > 1e-12 else None,
        "left_variance": finite_scalar(centered_left.square().mean()),
        "right_variance": finite_scalar(centered_right.square().mean()),
    }


def fit_projector(features, dimensions=64, seed=42):
    """PCA and whitening fitted exclusively on training-source representations."""
    features = features.float()
    dimensions = min(int(dimensions), features.shape[0] - 1, features.shape[1])
    if dimensions < 1:
        raise ValueError("At least two training representations are required.")
    mean = features.mean(0)
    centered = features - mean
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, basis = torch.pca_lowrank(
            centered, q=min(dimensions + 8, *centered.shape), center=False, niter=2
        )
    basis = basis[:, :dimensions]
    scales = (centered @ basis).std(0, correction=1)
    if float(scales.max()) <= 1e-10:
        raise ValueError("All training representations are constant; a group probe is unidentifiable.")
    retained = scales > max(float(scales.max()) * 1e-5, 1e-8)
    basis, scales = basis[:, retained], scales[retained]
    return {"mean": mean, "basis": basis, "scales": scales}


def project(features, projector):
    return ((features.float() - projector["mean"]) @ projector["basis"] / projector["scales"]).double()


def homogeneous(features):
    return torch.cat((features.double(), torch.ones(len(features), 1, dtype=torch.float64)), 1)


def fit_operator(source, target, ridge=0.01):
    if ridge < 0 or source.shape != target.shape:
        raise ValueError("Expected nonnegative ridge and equally shaped source/target latents.")
    source, target = source.double(), target.double()
    design = homogeneous(source)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * (len(source) * ridge)
    penalty[-1, -1] = 0  # The affine intercept is not penalized.
    if ridge:
        coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    else:
        coefficients = torch.linalg.lstsq(design, target).solution
    operator = torch.eye(design.shape[1], dtype=torch.float64)
    operator[:-1] = coefficients.T
    return operator


def apply_operator(operator, features):
    return (homogeneous(features) @ operator.T)[:, :-1]


def operator_comparison(left, right, source):
    return {
        "matrix_relative_error": relative_error(left[:-1], right[:-1]),
        **{f"action_{key}": value for key, value in latent_metrics(
            apply_operator(left, source), apply_operator(right, source)
        ).items()},
    }


def probe_fit_metrics(operator, source, target, training_target_mean):
    predicted = apply_operator(operator, source)
    error = (predicted - target).square().mean()
    baseline = (target - training_target_mean).square().mean()
    singular = torch.linalg.svdvals(operator[:-1, :-1])
    return {
        **latent_metrics(predicted, target),
        "r2_against_training_mean": finite_scalar(1 - error / baseline) if baseline > 1e-12 else None,
        "mean_baseline_mse": finite_scalar(baseline),
        "linear_rank": int(torch.linalg.matrix_rank(operator[:-1, :-1])),
        "largest_singular_value": finite_scalar(singular.max()),
        "smallest_singular_value": finite_scalar(singular.min()),
    }


def group_laws(operators, source, seen_pairs, extrapolated_angles=()):
    """Nontrivial laws of independently fitted rho; no laws are enforced in fitting."""
    seen_pairs, extrapolated = set(seen_pairs), set(extrapolated_angles)
    identity = torch.eye(next(iter(operators.values())).shape[0], dtype=torch.float64)
    result = {name: [] for name in (
        "composition", "commutativity", "identity", "inverse", "cycles",
        "factorization_associativity", "full_c24_composition",
    )}
    for a, b in product(ATOMIC_ANGLES, repeat=2):
        metadata = {"a": a, "b": b, "total": (a + b) % 360,
                    "pair_split": "seen_pairs" if (a, b) in seen_pairs else "held_out_pairs"}
        result["composition"].append({**metadata, **operator_comparison(
            operators[b] @ operators[a], operators[(a + b) % 360], source
        )})
        result["commutativity"].append({**metadata, "tautological": a == b, **operator_comparison(
            operators[b] @ operators[a], operators[a] @ operators[b], source
        )})
    for a in GROUP_ANGLES:
        metadata = {"angle": a, "extrapolation": a in extrapolated or 0 in extrapolated}
        result["identity"].append({**metadata, "side": "left", **operator_comparison(
            operators[0] @ operators[a], operators[a], source
        )})
        result["identity"].append({**metadata, "side": "right", **operator_comparison(
            operators[a] @ operators[0], operators[a], source
        )})
        inverse = (-a) % 360
        metadata = {"angle": a, "inverse_angle": inverse,
                    "extrapolation": a in extrapolated or inverse in extrapolated}
        for side, composed in (
            ("left", operators[inverse] @ operators[a]),
            ("right", operators[a] @ operators[inverse]),
        ):
            result["inverse"].append({**metadata, "side": side, **operator_comparison(composed, identity, source)})
    result["identity"].append({"angle": 0, "side": "rho0_vs_identity", "extrapolation": 0 in extrapolated,
                               **operator_comparison(operators[0], identity, source)})
    for a in ATOMIC_ANGLES:
        order = 360 // gcd(a, 360)
        result["cycles"].append({"angle": a, "order": order, "extrapolation": False,
                                 **operator_comparison(torch.linalg.matrix_power(operators[a], order), identity, source)})
    for a, b, c in product(ATOMIC_ANGLES, repeat=3):
        if a + b + c > 180:
            continue
        # Ordinary matrix multiplication is associative by definition. Instead,
        # compare separately fitted summed-angle maps in the two factorizations.
        left = operators[c] @ operators[a + b]
        right = operators[b + c] @ operators[a]
        total = operators[a + b + c]
        result["factorization_associativity"].append({
            "a": a, "b": b, "c": c, "total": a + b + c,
            **operator_comparison(left, right, source),
            "left_vs_total_matrix_error": relative_error(left[:-1], total[:-1]),
            "right_vs_total_matrix_error": relative_error(right[:-1], total[:-1]),
        })
    for a, b in product(GROUP_ANGLES, repeat=2):
        total = (a + b) % 360
        result["full_c24_composition"].append({
            "a": a, "b": b, "total": total,
            "extrapolation": bool({a, b, total} & extrapolated),
            **operator_comparison(operators[b] @ operators[a], operators[total], source),
        })
    return result


def numeric_summary(rows):
    """Macro summaries of laws; source counts and each per-law row remain saved."""
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    summary = {"laws": len(rows)}
    for key in keys:
        if key in ("a", "b", "c", "angle", "inverse_angle", "total", "order", "extrapolation", "samples"):
            continue
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
        if values:
            summary[f"mean/{key}"] = sum(values) / len(values)
            summary[f"max/{key}"] = max(values)
    return summary
