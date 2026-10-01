"""Exact, sample-specific C1 rot90 and translate_up operators on raw grids.

COGITAO rotates each object's bounding-box patch counterclockwise in place.
Translation moves its top-left anchor up one row. The returned sparse matrix
acts on flattened raw color indices; background columns are zero.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, sparse

ROT = "rot90"
UP = "translate_up"

# Homogeneous column-vector convention: [anchor_row, anchor_col, height, width, 1].
ATTRIBUTE_ROT = np.array([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1],
], dtype=np.int64)
ATTRIBUTE_UP = np.array([
    [1, 0, 0, 0, -1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
], dtype=np.int64)


def step(grid: np.ndarray, operation: str) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Apply one object transformation and return its exact pixel operator."""
    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 2 or operation not in (ROT, UP):
        raise ValueError("Expected a 2-D grid and rot90 or translate_up")
    height, width = grid.shape
    labels, n_objects = ndimage.label(grid != 0, structure=np.ones((3, 3)))
    result = np.zeros_like(grid)
    source, destination = [], []
    for object_id in range(1, n_objects + 1):
        rows, cols = np.where(labels == object_id)
        top, left = int(rows.min()), int(cols.min())
        bottom, right = int(rows.max()), int(cols.max())
        patch_height, patch_width = bottom - top + 1, right - left + 1
        if operation == ROT:
            dest_rows = top + (patch_width - 1 - (cols - left))
            dest_cols = left + (rows - top)
        else:
            dest_rows = rows - 1
            dest_cols = cols
        if (dest_rows.min() < 0 or dest_rows.max() >= height or
                dest_cols.min() < 0 or dest_cols.max() >= width):
            raise ValueError(f"{operation} moves an object out of the grid")
        if np.any(result[dest_rows, dest_cols] != 0):
            raise ValueError(f"{operation} causes object overlap")
        result[dest_rows, dest_cols] = grid[rows, cols]
        source.extend((rows * width + cols).tolist())
        destination.extend((dest_rows * width + dest_cols).tolist())
    matrix = sparse.coo_matrix(
        (np.ones(len(source), dtype=np.int8), (destination, source)),
        shape=(height * width, height * width),
    ).tocsr()
    if not np.array_equal(matrix @ grid.ravel(), result.ravel()):
        raise AssertionError("Pixel matrix does not reconstruct its output")
    return result, matrix


def compose(
    grid: np.ndarray, operations: tuple[str, ...]
) -> tuple[np.ndarray, sparse.csr_matrix, list[sparse.csr_matrix]]:
    """Apply operations left to right; matrices multiply right to left."""
    current = np.asarray(grid, dtype=np.int64)
    total = sparse.eye(current.size, format="csr", dtype=np.int8)
    steps = []
    for operation in operations:
        current, matrix = step(current, operation)
        total = matrix @ total
        steps.append(matrix)
    return current, total, steps


def foreground_correspondence(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Source and destination indices for every retained foreground pixel."""
    dest, source = matrix.nonzero()
    order = np.argsort(source)
    return source[order], dest[order]
