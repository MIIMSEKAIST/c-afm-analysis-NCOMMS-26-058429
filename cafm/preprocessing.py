"""First-order scan-line flattening followed by one median-filter pass."""

import numpy as np
from scipy.ndimage import median_filter


def flatten_lines_order1(image, line_axis=1):
    values = np.asarray(image, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Line flattening requires a finite 2D current map")
    if line_axis not in (0, 1) or values.shape[line_axis] < 2:
        raise ValueError("The scan axis must be 0 or 1 with at least two pixels")
    lines = values if line_axis == 1 else values.T
    x = np.arange(lines.shape[1], dtype=float) - (lines.shape[1] - 1) / 2
    centered = lines - lines.mean(axis=1, keepdims=True)
    slopes = np.sum(centered * x, axis=1, keepdims=True) / np.sum(x * x)
    flattened = centered - slopes * x
    return flattened if line_axis == 1 else flattened.T


def prepare_map(raw, line_axis=1):
    return median_filter(flatten_lines_order1(raw, line_axis), size=3, mode="reflect")
