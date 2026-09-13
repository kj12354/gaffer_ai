"""Physical-millimetre morphology helpers (spacing-aware)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, distance_transform_edt


def ellipsoid(radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    """Binary structuring element that is a ball of ``radius_mm`` in physical space."""
    radii = [max(1, int(np.ceil(radius_mm / float(s)))) for s in spacing_zyx]
    zz, yy, xx = np.ogrid[
        -radii[0] : radii[0] + 1,
        -radii[1] : radii[1] + 1,
        -radii[2] : radii[2] + 1,
    ]
    selem = (
        (zz * spacing_zyx[0]) ** 2
        + (yy * spacing_zyx[1]) ** 2
        + (xx * spacing_zyx[2]) ** 2
    ) <= (radius_mm**2 + 1e-6)
    return selem


def dilate_mm(mask: np.ndarray, radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    return binary_dilation(mask, structure=ellipsoid(radius_mm, spacing_zyx))


def erode_mm(mask: np.ndarray, radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    return binary_erosion(mask, structure=ellipsoid(radius_mm, spacing_zyx))


def open_mm(mask: np.ndarray, radius_mm: float, spacing_zyx: np.ndarray) -> np.ndarray:
    return binary_opening(mask, structure=ellipsoid(radius_mm, spacing_zyx))


def distance_mm(mask: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """EDT in millimetres. ``mask`` is True for the *foreground* to measure from."""
    return distance_transform_edt(~mask, sampling=tuple(spacing_zyx)).astype(np.float32)


def neighbor_offsets(connectivity: int = 26) -> np.ndarray:
    """3D neighborhood offsets, excluding the origin."""
    offs = []
    r = (-1, 0, 1)
    for dz in r:
        for dy in r:
            for dx in r:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                if connectivity == 6 and abs(dz) + abs(dy) + abs(dx) != 1:
                    continue
                offs.append((dz, dy, dx))
    return np.asarray(offs, dtype=np.int8)
