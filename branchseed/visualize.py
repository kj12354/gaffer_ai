"""Static 3D visual check: aorta surface, ostia, and direction arrows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-branchseed")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from .io import VolumePair, mm_to_index_xyz, np_zyx_to_mm


def _aorta_surface_mm(vol: VolumePair, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Marching-cubes aorta surface vertices in LPS millimetres."""
    aorta = vol.aorta
    if step > 1:
        aorta = aorta[::step, ::step, ::step]
        spacing = vol.spacing_zyx * step
        origin = np.zeros(3)
    else:
        spacing = vol.spacing_zyx
        origin = np.zeros(3)
    try:
        verts_zyx, faces, *_ = marching_cubes(
            aorta.astype(np.float32), level=0.5, spacing=tuple(spacing)
        )
    except (ValueError, RuntimeError):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    # verts are in (z,y,x) mm relative to the (possibly strided) array origin.
    verts_idx = verts_zyx / spacing + origin
    if step > 1:
        verts_idx = verts_idx * step
    pts = np.array([np_zyx_to_mm(vol.image, z, y, x) for z, y, x in verts_idx])
    return pts, faces


def render_case(
    vol: VolumePair,
    prediction: dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Write a PNG showing the aorta, predicted ostia, and direction arrows."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Downsample very large aortas so the static figure stays cheap.
    n = int(vol.aorta.sum())
    step = 2 if n > 25000 else 1
    verts, faces = _aorta_surface_mm(vol, step=step)

    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    if len(verts) and len(faces):
        mesh = Poly3DCollection(verts[faces], alpha=0.18, linewidths=0.05)
        mesh.set_facecolor((0.25, 0.45, 0.75, 0.18))
        mesh.set_edgecolor((0.2, 0.3, 0.45, 0.08))
        ax.add_collection3d(mesh)

    daughters = prediction.get("daughters", [])
    # tab10 wraps: linspace(0, 1, 11) maps the first and last branch to the
    # same color. Use tab20 (20 hues) plus distinct markers so ≥15 instances
    # stay visually separable.
    palette = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
    markers = ("o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">", "8", "H", "d")
    for i, d in enumerate(daughters):
        color = palette[i % len(palette)]
        marker = markers[i % len(markers)]
        o = np.asarray(d["ostium_xyz_mm"], dtype=float)
        s = np.asarray(d["seed_xyz_mm"], dtype=float)
        direc = np.asarray(d["direction_xyz"], dtype=float)
        ax.scatter(
            *o, color=color, marker=marker, s=64, depthshade=False,
            label=d["instance_id"], zorder=5, edgecolors="k", linewidths=0.4,
        )
        ax.plot([o[0], s[0]], [o[1], s[1]], [o[2], s[2]], color=color, lw=2.0, zorder=4)
        ax.quiver(
            o[0], o[1], o[2],
            direc[0], direc[1], direc[2],
            length=12.0, normalize=True, color=color, arrow_length_ratio=0.28,
            linewidth=1.4,
        )

    ax.set_xlabel("X mm (LPS)")
    ax.set_ylabel("Y mm (LPS)")
    ax.set_zlabel("Z mm (LPS)")
    ax.view_init(elev=22, azim=-70)
    ax.set_title(title or f"{prediction.get('case_id', vol.case_id)} - ostia + directions")
    if daughters:
        ax.legend(loc="upper left", fontsize=8)
    if len(verts):
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        pad = 12.0
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        ax.set_zlim(mins[2] - pad, maxs[2] + pad)
    try:
        ax.set_box_aspect(
            (
                max(np.ptp(verts[:, 0]) if len(verts) else 1, 1),
                max(np.ptp(verts[:, 1]) if len(verts) else 1, 1),
                max(np.ptp(verts[:, 2]) if len(verts) else 1, 1),
            )
        )
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _window_ct(ct: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((ct - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def _mark_ostia_axial(
    ax,
    ostia_xyz: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    image,
    crop_x: slice,
    crop_y: slice,
) -> None:
    for ost, lab, col in zip(ostia_xyz, labels, colors):
        idx = mm_to_index_xyz(image, ost)
        x, y = idx[0] - crop_x.start, idx[1] - crop_y.start
        ax.scatter([x], [y], c=col, s=40, marker="+", linewidths=1.4)
        ax.annotate(lab, (x + 0.8, y + 0.8), color=col, fontsize=7)


def render_ostium_pair_check(
    vol: VolumePair,
    ostia_xyz_mm: list[list[float]],
    labels: list[str],
    output_path: str | Path,
    title: str,
    crop_mm: float = 28.0,
    mip_half_mm: float = 3.0,
) -> Path:
    """Cropped axial / thin-MIP / sagittal-coronal check around ostia."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ostia = [np.asarray(p, dtype=float) for p in ostia_xyz_mm]
    center = np.mean(ostia, axis=0)
    cidx = mm_to_index_xyz(vol.image, center)
    cx, cy, cz = [int(round(v)) for v in cidx]
    sp = vol.spacing_xyz
    hx = max(6, int(round(crop_mm / float(sp[0]))))
    hy = max(6, int(round(crop_mm / float(sp[1]))))
    hz = max(2, int(round(mip_half_mm / float(sp[2]))))
    shape = vol.ct.shape  # z, y, x
    x0, x1 = max(0, cx - hx), min(shape[2], cx + hx + 1)
    y0, y1 = max(0, cy - hy), min(shape[1], cy + hy + 1)
    z0, z1 = max(0, cz - hz), min(shape[0], cz + hz + 1)
    # Window from aorta voxels so contrast blood stays bright.
    aorta_hu = vol.ct[vol.aorta]
    lo = float(np.percentile(aorta_hu, 5)) - 80.0
    hi = float(np.percentile(aorta_hu, 95)) + 80.0
    colors = ["#d62728", "#1f77b4"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 7.2))

    def _axial_at(z_mm: float, ax, panel: str) -> None:
        z_idx = int(round(mm_to_index_xyz(vol.image, [center[0], center[1], z_mm])[2]))
        z_idx = int(np.clip(z_idx, 0, shape[0] - 1))
        sl = vol.ct[z_idx, y0:y1, x0:x1]
        msk = vol.aorta[z_idx, y0:y1, x0:x1]
        ax.imshow(_window_ct(sl, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1)
        if np.any(msk):
            ax.contour(msk.astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
        _mark_ostia_axial(ax, ostia, labels, colors, vol.image, slice(x0, x1), slice(y0, y1))
        ax.set_title(f"{panel}  z={z_mm:.1f} mm", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    _axial_at(ostia[0][2], axes[0, 0], labels[0])
    _axial_at(ostia[1][2], axes[0, 1], labels[1])
    # Thin MIP about the pair midpoint.
    mip = vol.ct[z0:z1, y0:y1, x0:x1].max(axis=0)
    axes[0, 2].imshow(_window_ct(mip, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1)
    mip_aorta = vol.aorta[z0:z1, y0:y1, x0:x1].max(axis=0)
    if np.any(mip_aorta):
        axes[0, 2].contour(mip_aorta.astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    _mark_ostia_axial(axes[0, 2], ostia, labels, colors, vol.image, slice(x0, x1), slice(y0, y1))
    axes[0, 2].set_title(f"axial thin-MIP ±{mip_half_mm:.0f} mm", fontsize=8)
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])

    # Coronal (x–z at center y) and sagittal (y–z at center x).
    cor = vol.ct[z0:z1, cy, x0:x1]
    sag = vol.ct[z0:z1, y0:y1, cx]
    axes[1, 0].imshow(_window_ct(cor, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].contour(vol.aorta[z0:z1, cy, x0:x1].astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    axes[1, 0].set_title("coronal crop (x–z)", fontsize=8)
    axes[1, 1].imshow(_window_ct(sag, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].contour(vol.aorta[z0:z1, y0:y1, cx].astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    axes[1, 1].set_title("sagittal crop (y–z)", fontsize=8)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_xticks([])
        ax.set_yticks([])
        for ost, lab, col in zip(ostia, labels, colors):
            idx = mm_to_index_xyz(vol.image, ost)
            if ax is axes[1, 0]:
                ax.scatter([idx[0] - x0], [idx[2] - z0], c=col, s=36, marker="+")
            else:
                ax.scatter([idx[1] - y0], [idx[2] - z0], c=col, s=36, marker="+")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        0.85,
        title + "\n\nCyan = aorta mask edge.\n"
        "Crosses = predicted ostia.\n"
        "Look for a ~2 mm bright lumen\nleaving the posterior wall,\n"
        "not just vesselness on fat/bone.",
        transform=axes[1, 2].transAxes,
        fontsize=8,
        va="top",
        family="sans-serif",
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_ostium_review_card(
    vol: VolumePair,
    ostium_xyz_mm: list[float],
    label: str,
    output_path: str | Path,
    subtitle: str = "",
    crop_mm: float = 24.0,
    mip_half_mm: float = 3.0,
) -> Path:
    """Single-ostium axial / MIP / sag / cor crop for manual review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ost = np.asarray(ostium_xyz_mm, dtype=float)
    cidx = mm_to_index_xyz(vol.image, ost)
    cx, cy, cz = [int(round(v)) for v in cidx]
    sp = vol.spacing_xyz
    hx = max(6, int(round(crop_mm / float(sp[0]))))
    hy = max(6, int(round(crop_mm / float(sp[1]))))
    hz = max(2, int(round(mip_half_mm / float(sp[2]))))
    shape = vol.ct.shape
    x0, x1 = max(0, cx - hx), min(shape[2], cx + hx + 1)
    y0, y1 = max(0, cy - hy), min(shape[1], cy + hy + 1)
    z0, z1 = max(0, cz - hz), min(shape[0], cz + hz + 1)
    aorta_hu = vol.ct[vol.aorta]
    lo = float(np.percentile(aorta_hu, 5)) - 80.0
    hi = float(np.percentile(aorta_hu, 95)) + 80.0

    def _mark(ax, x, y) -> None:
        ax.scatter([x], [y], c="#ff4d4d", s=42, marker="+", linewidths=1.5)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    z_idx = int(np.clip(cz, 0, shape[0] - 1))
    sl = vol.ct[z_idx, y0:y1, x0:x1]
    msk = vol.aorta[z_idx, y0:y1, x0:x1]
    axes[0, 0].imshow(_window_ct(sl, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1)
    if np.any(msk):
        axes[0, 0].contour(msk.astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    _mark(axes[0, 0], cx - x0, cy - y0)
    axes[0, 0].set_title(f"axial  z={ost[2]:.1f} mm", fontsize=8)

    mip = vol.ct[z0:z1, y0:y1, x0:x1].max(axis=0)
    axes[0, 1].imshow(_window_ct(mip, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1)
    mip_aorta = vol.aorta[z0:z1, y0:y1, x0:x1].max(axis=0)
    if np.any(mip_aorta):
        axes[0, 1].contour(mip_aorta.astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    _mark(axes[0, 1], cx - x0, cy - y0)
    axes[0, 1].set_title(f"axial thin-MIP ±{mip_half_mm:.0f} mm", fontsize=8)

    cor = vol.ct[z0:z1, cy, x0:x1]
    axes[1, 0].imshow(_window_ct(cor, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].contour(vol.aorta[z0:z1, cy, x0:x1].astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    axes[1, 0].scatter([cx - x0], [cz - z0], c="#ff4d4d", s=36, marker="+")
    axes[1, 0].set_title("coronal (x–z)", fontsize=8)

    sag = vol.ct[z0:z1, y0:y1, cx]
    axes[1, 1].imshow(_window_ct(sag, lo, hi), cmap="gray", origin="lower", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].contour(vol.aorta[z0:z1, y0:y1, cx].astype(float), levels=[0.5], colors="#39d0ff", linewidths=0.7)
    axes[1, 1].scatter([cy - y0], [cz - z0], c="#ff4d4d", s=36, marker="+")
    axes[1, 1].set_title("sagittal (y–z)", fontsize=8)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{label}  {subtitle}".strip(), fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)
    return output_path
