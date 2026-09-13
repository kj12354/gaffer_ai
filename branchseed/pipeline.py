"""End-to-end Branchseed detection pipeline."""

from __future__ import annotations

import os
from typing import Any

# Keep native numeric libs on the organiser's 4-core CPU budget.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

from .branches import DetectionResult, extract_daughters
from .intensity import calibrate_blood_range
from .io import VolumePair, crop_pair, load_case
from .output import build_prediction
from .params import PipelineParams
from .vesselness import build_search_masks

__all__ = ["detect_daughters", "run_on_volume", "PipelineParams"]


def detect_daughters(
    image_path: str,
    mask_path: str,
    case_id: str | None = None,
    params: PipelineParams | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the detector on one CT / aorta-mask pair and return prediction JSON."""
    params = params or PipelineParams()
    vol = load_case(image_path, mask_path, case_id=case_id)
    payload, _ = run_on_volume_detailed(vol, params=params, verbose=verbose)
    return payload


def run_on_volume(
    vol: VolumePair,
    params: PipelineParams | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    payload, _ = run_on_volume_detailed(vol, params=params, verbose=verbose)
    return payload


def run_on_volume_detailed(
    vol: VolumePair,
    params: PipelineParams | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], DetectionResult]:
    params = params or PipelineParams()
    log = print if verbose else (lambda *a, **k: None)

    rng = calibrate_blood_range(vol.ct, vol.aorta, params)
    log(
        f"[{vol.case_id}] aorta n={rng.n_voxels}  HU p10/p50/p90="
        f"{rng.p10:.0f}/{rng.p50:.0f}/{rng.p90:.0f}  "
        f"blood=[{rng.blood_low:.0f}, {rng.blood_high:.0f}]  "
        f"spacing={tuple(round(float(s), 3) for s in vol.spacing_xyz)}"
    )

    ct_c, aorta_c, _sl, origin_zyx = crop_pair(vol, params.crop_margin_mm)
    log(
        f"[{vol.case_id}] cropped ROI {ct_c.shape} "
        f"(from full {vol.ct.shape}) origin_zyx={tuple(int(x) for x in origin_zyx)}"
    )

    search = build_search_masks(ct_c, aorta_c, vol.spacing_zyx, rng, params)
    log(
        f"[{vol.case_id}] seeds={int(search['seeds'].sum())}  "
        f"allowed={int(search['allowed'].sum())}  "
        f"v_thresh={float(search['v_thresh']):.4f}"
    )

    detected = extract_daughters(vol, ct_c, aorta_c, search, rng, origin_zyx, params)
    if detected.rejected and verbose:
        shown = detected.rejected[:12]
        extra = "" if len(detected.rejected) <= 12 else f" (+{len(detected.rejected)-12} more)"
        log(f"[{vol.case_id}] rejected: {shown}{extra}")
    if detected.merge_log and verbose:
        for line in detected.merge_log:
            log(f"[{vol.case_id}] merge {line}")
    log(f"[{vol.case_id}] kept {len(detected.daughters)} daughter(s)")

    return build_prediction(vol, detected.daughters, origin_zyx), detected
