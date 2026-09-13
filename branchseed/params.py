"""Physical-unit parameters shared by every case.

Values are millimetres or unitless ratios. Nothing here is a per-case HU
threshold or a voxel-index constant — those are calibrated from the supplied
aorta mask at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineParams:
    """Tunable detector settings, all in physical millimetres unless noted."""

    # Search envelope around the parent lumen boundary.
    shell_outer_mm: float = 12.0
    crop_margin_mm: float = 10.0

    # Frangi / vesselness scales (physical tube radii to enhance).
    vessel_sigmas_mm: tuple[float, ...] = (0.7, 1.1, 1.8)
    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5
    frangi_gamma: float = 15.0

    # Outward growth.
    max_trace_mm: float = 10.0
    # Challenge brief: contrast-filled lumen followable ≥5 mm beyond the wall.
    min_path_mm: float = 5.0
    growth_limit_mm: float = 14.0
    dist_slack_mm: float = 0.9
    sheet_open_mm: float = 2.4
    near_wall_mm: float = 2.5

    # Ostium clustering / de-duplication.
    ostium_merge_mm: float = 3.5
    # Unconditional near-duplicate floor (separate from path-divergence).
    # ~3 voxels on 1.5 mm data; well below opposite-wall / lumbar-pair spacing.
    near_dup_ostium_mm: float = 5.0
    # Nearby seeds of one origin: merge only if 3D ostia are close AND
    # traced paths do not diverge (protects genuine separate ostia).
    path_merge_ostium_mm: float = 8.0
    path_merge_min_dir_cosine: float = 0.40
    path_diverge_dir_cosine: float = 0.25
    # Contact-patch split: only a long wall-kiss with two opposing ridge
    # directions. True-match contacts on 19–23 are typically 0.8–6.5 mm;
    # alongside kisses reach ~12–14 mm. The fused case-20 pair is 23.7 mm.
    # Off until the full-dataset ≥18 mm contacts are visually reviewed.
    enable_contact_split: bool = False
    split_min_contact_spread_mm: float = 18.0
    split_max_dir_cosine: float = 0.15
    split_min_walks: int = 6
    split_min_cluster_walks: int = 3

    # Eligibility: origin diameter (partial-volume aware; 2 mm is the policy).
    min_origin_diameter_mm: float = 1.6
    min_seed_radius_mm: float = 0.45

    # Crop-cap and iliac exclusion.
    cap_slice_margin: int = 2
    cap_dir_z_abs: float = 0.75
    iliac_inferior_window_mm: float = 12.0
    iliac_dir_z: float = -0.40
    iliac_terminus_mm: float = 10.0

    # Intensity calibration (percentiles / ratios of THIS case's aorta HU).
    blood_low_percentile: float = 8.0
    blood_low_relax: float = 0.20
    blood_high_percentile: float = 92.0
    calcif_margin_hu: float = 80.0
    calcif_abs_cap_hu: float = 850.0

    # Vesselness gate (adaptive floor as a fraction of in-shell response).
    vesselness_percentile: float = 70.0
    vesselness_floor: float = 0.015
    vesselness_far_scale: float = 0.35

    # Post-merge quality filter (confirmed-false patterns from crop review).
    # Logistic LOOCV had no zero-FN threshold; these rules drop 0 confirmed_real.
    quality_filter: bool = True
    context_radius_mm: float = 5.0
    tube_clear_radius_mm: float = 2.6
    bone_frac_drop: float = 0.22
    adj_lumen_drop: float = 0.28
    blob_vess_max: float = 0.08
    blob_radius_mm: float = 3.0
    short_path_drop_mm: float = 6.0
    short_circ_max: float = 0.38
    # Near-zero vesselness + unstable radius: wall smear, not a coherent tube.
    # case_23's true match has v≈0 but radius_cv≈0.31, so the CV floor is required.
    smear_vess_max: float = 0.02
    smear_radius_cv: float = 0.40
