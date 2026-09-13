"""Hessian / Frangi vesselness and the search shell around the aorta."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.filters import frangi

from .intensity import IntensityRange, blood_mask
from .morphology import dilate_mm, distance_mm
from .params import PipelineParams


def compute_vesselness(
    ct: np.ndarray,
    spacing_zyx: np.ndarray,
    params: PipelineParams,
) -> np.ndarray:
    """3D Frangi vesselness for bright tubes, sigmas set in millimetres.

    ``skimage.filters.frangi`` takes sigmas in *voxels*. Physical scales are
    converted with the mean spacing so ~1–2 voxel daughters stay visible on
    both 0.78 mm and 1.5 mm grids.
    """
    mean_sp = float(np.mean(spacing_zyx))
    sigmas = tuple(max(0.4, s / mean_sp) for s in params.vessel_sigmas_mm)
    resp = frangi(
        ct,
        sigmas=sigmas,
        alpha=params.frangi_alpha,
        beta=params.frangi_beta,
        gamma=params.frangi_gamma,
        black_ridges=False,
        mode="reflect",
    )
    return resp.astype(np.float32, copy=False)


def adaptive_vesselness_threshold(
    vesselness: np.ndarray,
    support: np.ndarray,
    params: PipelineParams,
) -> float:
    """Threshold from the vesselness distribution *inside this case's shell*."""
    vals = vesselness[support]
    if vals.size == 0:
        return params.vesselness_floor
    p = float(np.percentile(vals, params.vesselness_percentile))
    return max(params.vesselness_floor, params.vesselness_far_scale * p)


def build_search_masks(
    ct: np.ndarray,
    aorta: np.ndarray,
    spacing_zyx: np.ndarray,
    rng: IntensityRange,
    params: PipelineParams,
) -> dict[str, np.ndarray]:
    """Build blood, vesselness, and ostium-seed maps.

    Seeds are near-wall blood voxels that touch *outward* blood (beyond
    ``near_wall_mm``). A thin partial-volume sheet hugging the wall does not
    reach that far, so it does not become a seed. Growth then uses **all**
    contrast-like voxels in the shell so curved 5 mm paths are not cut short.
    """
    dist = distance_mm(aorta, spacing_zyx)
    blood = blood_mask(ct, rng) & ~aorta
    shell = (dist > 0) & (dist <= params.growth_limit_mm)

    vesselness = compute_vesselness(ct, spacing_zyx, params)
    v_thresh = adaptive_vesselness_threshold(vesselness, blood & shell, params)

    ring = blood & (dist > 0) & (dist <= params.near_wall_mm)
    outward = blood & (dist > params.near_wall_mm) & (dist <= params.growth_limit_mm)
    # A real ostium has contrast continuing away from the wall.
    seeds = ring & binary_dilation(outward, structure=np.ones((3, 3, 3)))
    # Thin / dim daughters: vesselness peaks on the contact ring.
    seeds = seeds | (ring & (vesselness >= v_thresh))

    contact = dilate_mm(aorta, float(np.min(spacing_zyx)) * 1.1, spacing_zyx) & ~aorta
    seeds = seeds | (contact & outward)

    # Two-tier lumen: bright contrast is always allowed; dimmer voxels
    # (partial-volume tubes) only if vesselness supports them.
    bright = (ct >= rng.p10) & (ct <= rng.blood_high) & ~aorta & shell
    dim_vessel = blood & shell & (vesselness >= v_thresh * 0.6)
    allowed = bright | dim_vessel
    return {
        "dist": dist,
        "shell": shell,
        "blood": blood,
        "vesselness": vesselness,
        "v_thresh": np.array(v_thresh, dtype=np.float32),
        "sheet": ring & ~seeds,
        "protrusions": seeds,
        "allowed": allowed,
        "seeds": seeds,
        "contact_ring": contact,
    }
