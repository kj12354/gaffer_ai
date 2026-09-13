"""Cross-sectional radius / circularity along a short ostium→seed path.

Diagnostic geometry only — not used as a keep/reject gate unless a later
calibration shows a clean true-vs-extra split.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def resample_path_zyx(
    points_zyx: np.ndarray,
    spacing_zyx: np.ndarray,
    step_mm: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced samples along a (N,3) voxel path. Returns points + unit dirs."""
    pts = np.atleast_2d(np.asarray(points_zyx, dtype=np.float64))
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    if len(pts) == 1:
        return pts.copy(), np.array([[1.0, 0.0, 0.0]])
    phys = pts * sp.reshape(1, 3)
    seg = np.linalg.norm(np.diff(phys, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        d = np.array([1.0, 0.0, 0.0])
        return pts[:1].copy(), d.reshape(1, 3)
    n = max(3, int(math.floor(total / step_mm)) + 1)
    targets = np.linspace(0.0, total, n)
    out_pts = np.empty((n, 3), dtype=np.float64)
    out_dir = np.empty((n, 3), dtype=np.float64)
    for i, t in enumerate(targets):
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = int(np.clip(k, 0, len(seg) - 1))
        slen = float(seg[k]) if seg[k] > 1e-9 else 1.0
        alpha = (t - cum[k]) / slen
        out_pts[i] = pts[k] + alpha * (pts[k + 1] - pts[k])
        step = (pts[k + 1] - pts[k]) * sp
        nrm = float(np.linalg.norm(step))
        out_dir[i] = step / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    return out_pts, out_dir


def _plane_axes(direction_zyx_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction_zyx_mm, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-8:
        d = np.array([1.0, 0.0, 0.0])
    else:
        d = d / nrm
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a1 = np.cross(d, helper)
    a1 = a1 / float(np.linalg.norm(a1))
    a2 = np.cross(d, a1)
    return a1, a2


def _section_mask(
    center_zyx: np.ndarray,
    direction_zyx_mm: np.ndarray,
    lumen: np.ndarray,
    spacing_zyx: np.ndarray,
    half_mm: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """In-plane (u,v) mm coords of lumen hits + area-equivalent radius."""
    a1, a2 = _plane_axes(direction_zyx_mm)
    sp = np.asarray(spacing_zyx, dtype=np.float64)
    step = 0.5 * float(np.min(sp))
    n = int(np.ceil(half_mm / step))
    us: list[float] = []
    vs: list[float] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            u = i * step
            v = j * step
            if u * u + v * v > half_mm * half_mm:
                continue
            pos = center_zyx + (u * a1 + v * a2) / sp
            zi, yi, xi = [int(round(c)) for c in pos]
            if (
                0 <= zi < lumen.shape[0]
                and 0 <= yi < lumen.shape[1]
                and 0 <= xi < lumen.shape[2]
                and lumen[zi, yi, xi]
            ):
                us.append(u)
                vs.append(v)
    if not us:
        return np.zeros((0, 2)), 0.0
    pts = np.column_stack([us, vs])
    area = float(len(us)) * (step**2)
    r_area = float(math.sqrt(area / math.pi)) if area > 0 else 0.0
    return pts, r_area


def _circularity(pts_uv: np.ndarray) -> float:
    """1 = circular, ~0 = elongated. Axis ratio of the in-plane point cloud."""
    if len(pts_uv) < 4:
        return float("nan")
    centered = pts_uv - pts_uv.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts_uv) - 1, 1)
    try:
        evals = np.sort(np.linalg.eigvalsh(cov))
    except np.linalg.LinAlgError:
        return float("nan")
    ev_min = max(float(evals[0]), 1e-8)
    ev_max = max(float(evals[-1]), 1e-8)
    return float(math.sqrt(ev_min / ev_max))


def path_consistency(
    points_zyx: np.ndarray,
    lumen: np.ndarray,
    spacing_zyx: np.ndarray,
    step_mm: float = 1.5,
) -> tuple[float, float]:
    """Return (radius_cv, mean_circularity) along the traced path."""
    pts = np.atleast_2d(np.asarray(points_zyx, dtype=np.float64))
    if pts.size == 0 or not np.any(lumen):
        return float("nan"), float("nan")
    samples, dirs = resample_path_zyx(pts, spacing_zyx, step_mm=step_mm)
    dt = distance_transform_edt(lumen, sampling=tuple(spacing_zyx))
    # Sample DT with linear interpolation at fractional voxel centres.
    coords = np.array(
        [
            np.clip(samples[:, 0], 0, lumen.shape[0] - 1),
            np.clip(samples[:, 1], 0, lumen.shape[1] - 1),
            np.clip(samples[:, 2], 0, lumen.shape[2] - 1),
        ]
    )
    r_dt = map_coordinates(dt, coords, order=1, mode="nearest")
    radii: list[float] = []
    circs: list[float] = []
    for i in range(len(samples)):
        uv, r_area = _section_mask(samples[i], dirs[i], lumen, spacing_zyx)
        r = float(r_dt[i]) if r_dt[i] > 0.15 else r_area
        if r > 0:
            radii.append(r)
        c = _circularity(uv)
        if math.isfinite(c):
            circs.append(c)
    if len(radii) < 2:
        cv = float("nan")
    else:
        mu = float(np.mean(radii))
        cv = float("nan") if mu < 1e-6 else float(np.std(radii, ddof=1) / mu)
    mean_c = float(np.mean(circs)) if circs else float("nan")
    return cv, mean_c
