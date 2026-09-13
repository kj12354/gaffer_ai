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


_DAUGHTER_KEYS = (
    "instance_id",
    "parent_instance_id",
    "ostium_xyz_mm",
    "seed_xyz_mm",
    "radius_mm",
    "direction_xyz",
)


def _is_xyz(value: object) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return False
    return all(isinstance(x, (int, float)) and np.isfinite(float(x)) for x in value)


def validate_prediction(payload: dict[str, Any]) -> None:
    """Raise ValueError if ``payload`` does not match the challenge schema."""
    if not isinstance(payload, dict):
        raise ValueError("prediction must be a JSON object")
    if not payload.get("case_id"):
        raise ValueError("prediction missing case_id")
    parent = payload.get("parent")
    if not isinstance(parent, dict) or parent.get("instance_id") != "aorta":
        raise ValueError("prediction parent.instance_id must be 'aorta'")
    daughters = payload.get("daughters")
    if not isinstance(daughters, list):
        raise ValueError("prediction daughters must be a list (empty if none found)")
    seen: set[str] = set()
    for i, item in enumerate(daughters):
        if not isinstance(item, dict):
            raise ValueError(f"daughters[{i}] must be an object")
        missing = [k for k in _DAUGHTER_KEYS if k not in item]
        if missing:
            raise ValueError(f"daughters[{i}] missing fields: {', '.join(missing)}")
        rid = str(item["instance_id"])
        if rid in seen:
            raise ValueError(f"duplicate instance_id: {rid}")
        seen.add(rid)
        if item.get("parent_instance_id") != "aorta":
            raise ValueError(f"{rid}: parent_instance_id must be 'aorta'")
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            if not _is_xyz(item[key]):
                raise ValueError(f"{rid}: {key} must be three finite millimetre / unit numbers")
        try:
            radius = float(item["radius_mm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{rid}: radius_mm must be a number") from exc
        if not np.isfinite(radius) or radius < 0:
            raise ValueError(f"{rid}: radius_mm must be a non-negative finite number")
        direction = np.asarray(item["direction_xyz"], dtype=np.float64)
        nrm = float(np.linalg.norm(direction))
        if nrm < 0.5 or nrm > 1.5:
            raise ValueError(f"{rid}: direction_xyz must be a unit vector (got norm {nrm:.4f})")


def write_prediction(payload: dict[str, Any], path: str | Path) -> None:
    validate_prediction(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
