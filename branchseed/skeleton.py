"""Skeleton / path extraction along a daughter lumen."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize

from .morphology import neighbor_offsets


NEIGH26 = neighbor_offsets(26)


@dataclass
class TracedPath:
    """A proximal centreline from ostium outward."""

    points_zyx: np.ndarray  # (N, 3) float, numpy order
    length_mm: float
    seed_zyx: np.ndarray
    seed_ok: bool
    direction_zyx: np.ndarray  # unit vector in numpy (z,y,x) index-delta space
    hit_bifurcation: bool


def _in_bounds(z: int, y: int, x: int, shape: tuple[int, ...]) -> bool:
    return 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]


def ridge_walk(
    start_zyx: np.ndarray,
    lumen: np.ndarray,
    dist_from_aorta: np.ndarray,
    lumen_radius: np.ndarray,
    spacing_zyx: np.ndarray,
    max_mm: float,
    min_mm: float,
) -> TracedPath:
    """Walk a centreline ridge away from the aorta.

    At each step pick the 26-neighbour that most increases aorta-distance
    while staying inside the lumen and near its distance-transform ridge.
    This is more stable than skeletonizing 1–2 voxel tubes.
    """
    shape = lumen.shape
    z0, y0, x0 = [int(round(c)) for c in start_zyx]
    z0 = int(np.clip(z0, 0, shape[0] - 1))
    y0 = int(np.clip(y0, 0, shape[1] - 1))
    x0 = int(np.clip(x0, 0, shape[2] - 1))

    # Snap start onto the lumen if needed.
    if not lumen[z0, y0, x0]:
        found = False
        for dz, dy, dx in NEIGH26:
            nz, ny, nx = z0 + int(dz), y0 + int(dy), x0 + int(dx)
            if _in_bounds(nz, ny, nx, shape) and lumen[nz, ny, nx]:
                z0, y0, x0 = nz, ny, nx
                found = True
                break
        if not found:
            pts = np.array([[z0, y0, x0]], dtype=np.float64)
            return TracedPath(pts, 0.0, pts[0], False, np.array([1.0, 0.0, 0.0]), False)

    visited = set()
    path = [(float(z0), float(y0), float(x0))]
    length = 0.0
    seed = np.array(path[0], dtype=np.float64)
    seed_ok = False
    hit_bif = False
    z, y, x = z0, y0, x0

    sp = np.asarray(spacing_zyx, dtype=np.float64)
    max_steps = int(max_mm / max(float(np.min(sp)), 0.3)) + 8

    for _ in range(max_steps):
        visited.add((z, y, x))
        d0 = float(dist_from_aorta[z, y, x])
        candidates: list[tuple[float, int, int, int]] = []
        for dz, dy, dx in NEIGH26:
            nz, ny, nx = z + int(dz), y + int(dy), x + int(dx)
            if not _in_bounds(nz, ny, nx, shape):
                continue
            if (nz, ny, nx) in visited or not lumen[nz, ny, nx]:
                continue
            d1 = float(dist_from_aorta[nz, ny, nx])
            if d1 < d0 - 0.35:
                continue
            step = np.array([dz, dy, dx], dtype=np.float64) * sp
            step_mm = float(np.linalg.norm(step))
            score = (
                1.6 * (d1 - d0)
                + 0.5 * float(lumen_radius[nz, ny, nx])
                - 0.05 * step_mm
            )
            candidates.append((score, nz, ny, nx))

        if not candidates:
            break
        candidates.sort(reverse=True)
        # A clear second outgoing neighbour at similar aorta-distance is a fork.
        if (
            len(candidates) >= 2
            and length >= min_mm * 0.6
            and candidates[0][0] - candidates[1][0] < 0.25
        ):
            d_a = float(dist_from_aorta[candidates[0][1], candidates[0][2], candidates[0][3]])
            d_b = float(dist_from_aorta[candidates[1][1], candidates[1][2], candidates[1][3]])
            if abs(d_a - d_b) < 1.2:
                hit_bif = True
                break

        _score, nz, ny, nx = candidates[0]
        step = (np.array([nz - z, ny - y, nx - x], dtype=np.float64) * sp)
        length += float(np.linalg.norm(step))
        z, y, x = nz, ny, nx
        path.append((float(z), float(y), float(x)))
        if (not seed_ok) and length >= min_mm:
            seed = np.array(path[-1], dtype=np.float64)
            seed_ok = True
        if length >= max_mm:
            break

    pts = np.asarray(path, dtype=np.float64)
    direction = _direction_from_path(pts, spacing_zyx)
    if not seed_ok and length > 0:
        seed = _interpolate_at_length(pts, spacing_zyx, min(min_mm, length))
        seed_ok = length >= min_mm * 0.85
    return TracedPath(pts, length, seed, seed_ok, direction, hit_bif)


def _interpolate_at_length(
    points_zyx: np.ndarray, spacing_zyx: np.ndarray, target_mm: float
) -> np.ndarray:
    if len(points_zyx) == 1:
        return points_zyx[0].copy()
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    acc = 0.0
    for i in range(1, len(points_zyx)):
        seg = (points_zyx[i] - points_zyx[i - 1]) * sp
        slen = float(np.linalg.norm(seg))
        if acc + slen >= target_mm and slen > 1e-6:
            t = (target_mm - acc) / slen
            return points_zyx[i - 1] + t * (points_zyx[i] - points_zyx[i - 1])
        acc += slen
    return points_zyx[-1].copy()


def _direction_from_path(points_zyx: np.ndarray, spacing_zyx: np.ndarray) -> np.ndarray:
    """Unit direction in *physical* (z,y,x) millimetres, later remapped to xyz."""
    if len(points_zyx) < 2:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    # Use up to the first ~5 mm of the path (PCA if enough points).
    acc = 0.0
    use = [0]
    for i in range(1, len(points_zyx)):
        acc += float(np.linalg.norm((points_zyx[i] - points_zyx[i - 1]) * sp))
        use.append(i)
        if acc >= 5.0:
            break
    pts = points_zyx[use] * sp
    if len(pts) >= 4:
        centered = pts - pts.mean(axis=0)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            direction = vt[0]
        except np.linalg.LinAlgError:
            direction = pts[-1] - pts[0]
    else:
        direction = pts[-1] - pts[0]
    n = float(np.linalg.norm(direction))
    if n < 1e-8:
        direction = (points_zyx[-1] - points_zyx[0]) * sp
        n = float(np.linalg.norm(direction))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    direction = direction / n
    # Point outward (increasing aorta-distance along the path).
    if np.dot(direction, pts[-1] - pts[0]) < 0:
        direction = -direction
    return direction


def skeletonize_component(lumen: np.ndarray) -> np.ndarray:
    """3D skeleton of a binary lumen (used for debugging / fallback)."""
    if not np.any(lumen):
        return lumen
    return skeletonize(lumen.astype(bool))
