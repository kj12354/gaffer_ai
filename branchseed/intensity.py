"""Per-case adaptive HU calibration from the supplied aorta mask."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import PipelineParams


@dataclass
class IntensityRange:
    """Contrast-enhanced blood window derived from this case's aorta."""

    blood_low: float
    blood_high: float
    p05: float
    p10: float
    p50: float
    p90: float
    p95: float
    n_voxels: int

    @property
    def strict_low(self) -> float:
        """A tighter floor used far from the wall, where partial volume is less."""
        return float(0.55 * self.p50 + 0.45 * self.blood_low)


def calibrate_blood_range(
    ct: np.ndarray,
    aorta: np.ndarray,
    params: PipelineParams | None = None,
) -> IntensityRange:
    """Sample HU inside the parent mask and set a blood intensity window.

    Daughter lumina are often slightly dimmer than the aortic core because of
    partial volume, so the lower bound sits below the aorta's 10th percentile
    rather than at the median.
    """
    params = params or PipelineParams()
    vals = np.asarray(ct[aorta], dtype=np.float64)
    if vals.size < 10:
        raise ValueError("Aorta mask has too few voxels to calibrate intensity")

    p05, p10, p50, p90, p95 = np.percentile(vals, [5.0, 10.0, 50.0, 90.0, 95.0])
    p_low = float(np.percentile(vals, params.blood_low_percentile))
    p_high = float(np.percentile(vals, params.blood_high_percentile))

    span = max(p50 - p10, 20.0)
    blood_low = p_low - params.blood_low_relax * span
    # Never drop into typical unenhanced soft tissue if the aorta is bright,
    # but do allow dim partial-volume daughters on weaker-contrast scans.
    floor = min(p05, 0.35 * p50)
    blood_low = max(blood_low, floor)

    blood_high = min(p_high + params.calcif_margin_hu, params.calcif_abs_cap_hu)
    if blood_high <= blood_low + 20:
        blood_high = min(float(vals.max()), params.calcif_abs_cap_hu)

    return IntensityRange(
        blood_low=float(blood_low),
        blood_high=float(blood_high),
        p05=float(p05),
        p10=float(p10),
        p50=float(p50),
        p90=float(p90),
        p95=float(p95),
        n_voxels=int(vals.size),
    )


def blood_mask(ct: np.ndarray, rng: IntensityRange) -> np.ndarray:
    """Boolean mask of contrast-like voxels (includes parent lumen)."""
    return (ct >= rng.blood_low) & (ct <= rng.blood_high)
