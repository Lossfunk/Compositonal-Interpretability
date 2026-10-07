"""Object-preserving COGITAO oracle for atomic-pair experiments.

Semantics reference: yassinetb/COGITAO, arcworld/transformations/
shape_transformations.py. Objects are inferred once as eight-connected
foreground components; their identities (including explicit zero-valued crop
points) persist through composition. Recorded targets must audit this inference.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from data.cogitao.cogitao import TASK_VOCABULARY

ORACLE_SOURCE = (
    "https://github.com/yassinetb/COGITAO/blob/main/"
    "arcworld/transformations/shape_transformations.py"
)
STATE_NAMES = ("x", "fx", "gx", "fgx", "gfx")


class InvalidOracle(ValueError):
    """An oracle state cannot be represented without clipping or ambiguity."""


@dataclass(frozen=True)
class ObjectState:
    points: dict[tuple[int, int], int]

    @property
    def anchor(self):
        if not self.points:
            raise InvalidOracle("empty_object")
        return tuple(min(key[axis] for key in self.points) for axis in (0, 1))

    def patch(self):
        top, left = self.anchor
        bottom = max(row for row, _ in self.points)
        right = max(col for _, col in self.points)
        value = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int64)
        for (row, col), color in self.points.items():
            value[row - top, col - left] = color
        return value


def from_patch(patch, anchor, *, move_to_anchor=False):
    coordinates = np.argwhere(patch != 0)
    if not len(coordinates):
        raise InvalidOracle("empty_object")
    shift = coordinates.min(axis=0) if move_to_anchor else np.zeros(2, dtype=int)
    return ObjectState({
        (int(row - shift[0] + anchor[0]), int(col - shift[1] + anchor[1])):
        int(patch[row, col]) for row, col in coordinates
    })


def infer_objects(grid):
    value = np.asarray(grid, dtype=np.int64)
    if value.ndim != 2 or value.size == 0 or np.any((value < 0) | (value > 9)):
        raise InvalidOracle("invalid_grid")
    labels, count = ndimage.label(value != 0, structure=np.ones((3, 3)))
    if count == 0:
        raise InvalidOracle("empty_input")
    return tuple(ObjectState({(int(row), int(col)): int(value[row, col])
                              for row, col in np.argwhere(labels == label)})
                 for label in range(1, count + 1))


def transform_object(obj, function):
    """Reconstruct generator actions, including its crop point-cloud behavior."""
    if function not in TASK_VOCABULARY:
        raise InvalidOracle("unsupported_function:" + function)
    top, left = obj.anchor
    patch = obj.patch()
    height, width = patch.shape
    if function.startswith("translate_"):
        dr, dc = {"translate_up": (-1, 0), "translate_right": (0, 1)}[function]
        return ObjectState({(r + dr, c + dc): color
                            for (r, c), color in obj.points.items()})
    if function in ("rot90", "mirror_horizontal", "mirror_vertical"):
        changed = (np.rot90(patch) if function == "rot90" else
                   np.flipud(patch) if function == "mirror_horizontal" else
                   np.fliplr(patch))
        return from_patch(changed, (top, left), move_to_anchor=True)
    if function == "change_shape_color":
        return ObjectState({point: color % 9 + 1
                            for point, color in obj.points.items() if color})
    if function.startswith("fill_holes_"):
        filled = ndimage.binary_fill_holes(patch != 0)
        if np.array_equal(filled, patch != 0):
            return obj  # The upstream action is a valid no-op on non-hollow shapes.
        colors = list(obj.points.values())
        most_frequent = max(set(colors), key=colors.count)
        if function == "fill_holes_same_color":
            changed = filled.astype(np.int64) * most_frequent
        else:
            baseline = filled.astype(np.int64) * colors[0]
            changed = patch + (baseline != patch) * (most_frequent % 9 + 1)
        if np.any(changed > 9):
            raise InvalidOracle("invalid_fill_color")
        return from_patch(changed, (top, left))
    if function == "empty_inside_pixels":
        interior = ndimage.binary_erosion(
            patch != 0, structure=ndimage.generate_binary_structure(2, 1))
        return from_patch(np.where(interior, 0, patch), (top, left), move_to_anchor=True)
    if function in ("crop_bottom_side", "crop_top_side"):
        start, stop = ((0, (height + 1) // 2) if function == "crop_bottom_side"
                       else (height // 2, height))
        # The generator constructs a point-cloud dict containing zeros too.
        return ObjectState({(top + r, left + c): int(patch[r, c])
                            for r in range(start, stop) for c in range(width)})
    if function == "crop_contours":
        if min(height, width) < 3:
            raise InvalidOracle("crop_contours_too_small")
        return from_patch(patch[1:-1, 1:-1], (top + 1, left + 1),
                          move_to_anchor=True)
    if function == "extend_contours_same_color":
        changed = np.pad(patch, 1)
        changed[0, 1:-1], changed[-1, 1:-1] = patch[0], patch[-1]
        changed[1:-1, 0], changed[1:-1, -1] = patch[:, 0], patch[:, -1]
        return from_patch(changed, (top - 1, left - 1), move_to_anchor=True)
    if function.startswith("pad_"):
        padding, color, anchor = {
            "pad_top": (((1, 0), (0, 0)), 8, (top - 1, left)),
            "pad_left": (((0, 0), (1, 0)), 7, (top, left - 1)),
            "pad_right": (((0, 0), (0, 1)), 6, (top, left)),
        }[function]
        changed = np.pad(patch, padding)
        if function == "pad_top":
            changed[0, :] = color
        elif function == "pad_left":
            changed[:, 0] = color
        else:
            changed[:, -1] = color
        return from_patch(changed, anchor, move_to_anchor=True)
    if function in ("double_right", "double_down"):
        axis = 1 if function == "double_right" else 0
        return from_patch(np.concatenate((patch, patch), axis=axis), (top, left),
                          move_to_anchor=True)
    raise InvalidOracle("unsupported_function:" + function)


def render(objects, shape):
    grid = np.zeros(shape, dtype=np.int64)
    occupied = set()
    for obj in objects:
        if not obj.points or not any(obj.points.values()):
            raise InvalidOracle("empty_object")
        for (row, col), color in obj.points.items():
            if not (0 <= row < shape[0] and 0 <= col < shape[1]):
                raise InvalidOracle("out_of_bounds")
            if (row, col) in occupied:
                raise InvalidOracle("object_overlap")
            occupied.add((row, col))
            grid[row, col] = color
    return grid


def apply(objects, function, shape):
    transformed = tuple(transform_object(obj, function) for obj in objects)
    return transformed, render(transformed, shape)


def oracle_sequence(grid, suite):
    objects = infer_objects(grid)
    value = np.asarray(grid, dtype=np.int64)
    for function in suite:
        objects, value = apply(objects, function, grid.shape)
    return value


def oracle_states(grid, f, g, *, require_reverse=True):
    """fgx=f(g(x)), task order [g,f]. Reverse validity is required by default.

    Transfer comparisons can set require_reverse=False: x/fx/gx/fgx remain
    required, while unsupported gfx is omitted rather than dropping the example.
    Object identities persist through both ordered programs.
    """
    x = np.asarray(grid, dtype=np.int64)
    objects = infer_objects(x)
    f_objects, fx = apply(objects, f, x.shape)
    g_objects, gx = apply(objects, g, x.shape)
    _, fgx = apply(g_objects, f, x.shape)
    states = dict(x=x, fx=fx, gx=gx, fgx=fgx)
    try:
        _, states["gfx"] = apply(f_objects, g, x.shape)
    except InvalidOracle:
        if require_reverse:
            raise
    return states
