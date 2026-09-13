"""JSON prediction formatting (challenge schema)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .branches import Daughter
from .io import VolumePair, np_zyx_to_mm


def _round_vec(vec: np.ndarray, ndigits: int = 6) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(vec, dtype=np.float64)]


def daughter_to_dict(
    daughter: Daughter,
    instance_id: str,
    vol: VolumePair,
    origin_zyx: np.ndarray,
) -> dict[str, Any]:
    ostium_full = daughter.ostium_zyx + origin_zyx.astype(np.float64)
    seed_full = daughter.seed_zyx + origin_zyx.astype(np.float64)
    ostium_mm = np_zyx_to_mm(vol.image, *ostium_full)
    seed_mm = np_zyx_to_mm(vol.image, *seed_full)
    # Challenge definition: unit vector from ostium into the daughter.
    # Compute it in SimpleITK LPS from the two landmarks so axis order cannot flip.
    direction = seed_mm - ostium_mm
    n = float(np.linalg.norm(direction))
    if n < 1e-8:
        direction = np.asarray(daughter.direction_xyz, dtype=np.float64)
        n = float(np.linalg.norm(direction))
    if n < 1e-8:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / n
    return {
        "instance_id": instance_id,
        "parent_instance_id": "aorta",
        "ostium_xyz_mm": _round_vec(ostium_mm),
        "seed_xyz_mm": _round_vec(seed_mm),
        "radius_mm": round(float(daughter.radius_mm), 4),
        "direction_xyz": _round_vec(direction),
    }


def build_prediction(
    vol: VolumePair,
    daughters: list[Daughter],
    origin_zyx: np.ndarray,
) -> dict[str, Any]:
    items = []
    for i, d in enumerate(daughters, start=1):
        items.append(daughter_to_dict(d, f"branch_{i:03d}", vol, origin_zyx))
    return {
        "case_id": vol.case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": items,
    }


def write_prediction(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
