"""Numerical measurements for oracle atomic-pair representation experiments."""

from itertools import combinations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from benchmarks.cogitao_vit_cross_attention.c1_atomic_pair_oracle import STATE_NAMES

EPS = 1e-12


def spatial_chart(tokens, resolution, pool_grid):
    """Ordered spatial-bin means, flattened with channels retained in each bin."""
    height, width = resolution
    if not 1 <= pool_grid <= min(height, width):
        raise ValueError("pool_grid must fit the image token grid")
    maps = np.asarray(tokens, dtype=np.float64).reshape(len(tokens), height, width, -1)
    rows = np.array_split(np.arange(height), pool_grid)
    cols = np.array_split(np.arange(width), pool_grid)
    bins = [maps[:, r[:, None], c[None, :], :].mean(axis=(1, 2))
            for r in rows for c in cols]
    return np.concatenate(bins, axis=1)


def fit_affine(x, y, ridge):
    """Fit a row-vector homogeneous affine action; only weights are regularized."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.shape != y.shape or ridge <= 0:
        raise ValueError("At least two matched pairs and positive ridge are required")
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x - mean_x, y - mean_y
    gram = centered_x.T @ centered_x
    penalty = ridge * max(float(np.trace(gram)) / x.shape[1], EPS)
    weights = np.linalg.solve(gram + penalty * np.eye(x.shape[1]),
                              centered_x.T @ centered_y)
    matrix = np.eye(x.shape[1] + 1)
    matrix[:-1, :-1] = weights
    matrix[-1, :-1] = mean_y - mean_x @ weights
    rank = int(np.linalg.matrix_rank(centered_x))
    prediction = apply_affine(x, matrix)
    return matrix, dict(samples=len(x), dimension=x.shape[1], rank=rank,
                        underidentified=rank < x.shape[1], ridge_penalty=penalty,
                        training_relative_error=float(relative_error(prediction, y).mean()))


def apply_affine(z, matrix):
    return z @ matrix[:-1, :-1] + matrix[-1, :-1]


def relative_error(predicted, target):
    return np.linalg.norm(predicted - target, axis=-1) / np.maximum(
        np.linalg.norm(target, axis=-1), EPS)


def delta_distance(a, b):
    """Symmetric relative L2 and cosine distance; both-zero is reported explicitly."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = None if min(na, nb) <= EPS else float(
        1 - np.clip(np.dot(a, b) / (na * nb), -1, 1))
    return dict(relative_l2=float(np.linalg.norm(a - b) /
                                  max((na + nb) / 2, EPS)),
                cosine_distance=cosine, source_delta_norm=na, context_delta_norm=nb,
                both_zero=int(max(na, nb) <= EPS))


def operator_commutator_metrics(af, ag):
    """Pair/layer constants; compute matrix products once, outside image loops."""
    first, second = af @ ag, ag @ af
    norm = float(np.linalg.norm(first - second))
    return dict(absolute=norm, relative=norm / max(
        (np.linalg.norm(first) + np.linalg.norm(second)) / 2, EPS))


def predict_states(embeddings, af, ag):
    """Accept a vector or batch, retaining the mathematical composition order."""
    x = embeddings["x"]
    fx, gx = apply_affine(x, af), apply_affine(x, ag)
    return dict(x=x, fx=fx, gx=gx, fgx=apply_affine(gx, af),
                gfx=apply_affine(fx, ag),
                f_context=apply_affine(embeddings["gx"], af),
                g_context=apply_affine(embeddings["fx"], ag))


def pair_measurements(states, embeddings, af, ag, tolerance,
                      operator_defects=None, predicted=None, same_function=False):
    """Relations are tested on frozen predicted actions, not identical encodings."""
    x, fx, gx, fgx, gfx = (embeddings[name] for name in STATE_NAMES)
    if predicted is None:
        predicted = predict_states(embeddings, af, ag)
    result = {}
    for name, before, after in (("f", fx - x, fgx - gx),
                                ("g", gx - x, gfx - fx)):
        result.update({f"action_transfer_{name}_{key}": value
                       for key, value in delta_distance(before, after).items()})
    for name, base, changed, context, changed_context in (
        ("f", "x", "fx", "gx", "fgx"), ("g", "x", "gx", "fx", "gfx")
    ):
        result[f"{name}_source_oracle_no_op"] = int(np.array_equal(states[base], states[changed]))
        result[f"{name}_context_oracle_no_op"] = int(
            np.array_equal(states[context], states[changed_context]))
    result["oracle_commutes"] = int(np.array_equal(states["fgx"], states["gfx"]))
    applicable = bool(result["oracle_commutes"] and not same_function)
    result["commutator_applicable"] = int(applicable)
    if applicable and operator_defects is None:
        operator_defects = operator_commutator_metrics(af, ag)
    result["commutator_operator_defect"] = (
        operator_defects["absolute"] if applicable else None)
    result["commutator_operator_relative_defect"] = (
        operator_defects["relative"] if applicable else None)
    scale = max(float(np.linalg.norm(x)), EPS)
    result["commutator_data_defect"] = (float(np.linalg.norm(
        predicted["fgx"] - predicted["gfx"]) / scale)
        if applicable else None)
    for name in STATE_NAMES[1:]:
        result[f"{name}_prediction_defect"] = float(
            np.linalg.norm(predicted[name] - embeddings[name]) / scale)
    result["f_context_prediction_defect"] = float(
        np.linalg.norm(predicted["f_context"] - fgx) / scale)
    result["g_context_prediction_defect"] = float(
        np.linalg.norm(predicted["g_context"] - gfx) / scale)
    relations = []
    for left, right in combinations(STATE_NAMES, 2):
        if np.array_equal(states[left], states[right]):
            defect = float(np.linalg.norm(predicted[left] - predicted[right]) / scale)
            tautology = bool(same_function and (left, right) in (
                ("fx", "gx"), ("fgx", "gfx")))
            relations.append(dict(left=left, right=right, defect=defect,
                                  structural_tautology=int(tautology),
                                  retained=None if tautology else int(defect <= tolerance)))
    scored_relations = [relation for relation in relations if not relation["structural_tautology"]]
    result["oracle_relation_observed_count"] = len(relations)
    result["oracle_relation_tautology_count"] = len(relations) - len(scored_relations)
    result["oracle_relation_count"] = len(scored_relations)
    result["oracle_relation_retention"] = (float(np.mean([
        relation["retained"] for relation in scored_relations])) if scored_relations else None)
    result["oracle_relation_defect"] = (float(np.mean([
        relation["defect"] for relation in scored_relations])) if scored_relations else None)
    result["oracle_relation_max_defect"] = (max(
        relation["defect"] for relation in scored_relations) if scored_relations else None)
    return result, relations


def aggregate(records):
    if not records:
        return {}
    result = {"samples": len(records)}
    for key in records[0]:
        values = [row[key] for row in records if row.get(key) is not None
                  and isinstance(row[key], (int, float, np.number))]
        if values:
            result[key + "_mean"] = float(np.mean(values))
            result[key + "_count"] = len(values)
    for function in ("f", "g"):
        active = [row for row in records if not row[f"{function}_source_oracle_no_op"]
                  and not row[f"{function}_context_oracle_no_op"]]
        result[f"action_transfer_{function}_active_samples"] = len(active)
        for metric in ("relative_l2", "cosine_distance"):
            key = f"action_transfer_{function}_{metric}"
            values = [row[key] for row in active if row[key] is not None]
            if values:
                result[key + "_active_mean"] = float(np.mean(values))
    return result


def correlation(values, accuracy):
    pairs = [(float(x), float(y)) for x, y in zip(values, accuracy)
             if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
    result = dict(pairs=len(pairs), pearson=None, pearson_pvalue=None,
                  spearman=None, spearman_pvalue=None)
    if len(pairs) < 3:
        return {**result, "status": "fewer_than_three_pairs"}
    x, y = np.asarray(pairs).T
    if np.ptp(x) <= EPS or np.ptp(y) <= EPS:
        return {**result, "status": "constant_metric_or_accuracy"}
    p, s = pearsonr(x, y), spearmanr(x, y)
    return dict(result, status="ok", pearson=float(p.statistic),
                pearson_pvalue=float(p.pvalue), spearman=float(s.statistic),
                spearman_pvalue=float(s.pvalue))
