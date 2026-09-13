"""Compare predictions to EVAL_SET annotations.json landmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


MATCH_MM = 10.0
DIR_COS_OK = 0.35
RADIUS_ABS_OK = (0.35, 8.0)


@dataclass
class CaseMetrics:
    case_id: str
    n_gt: int
    n_pred: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    mean_ostium_err_mm: float | None
    mean_seed_err_mm: float | None
    mean_dir_cosine: float | None
    flags: list[str] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)


def load_gt_landmarks(annotations_path: Path) -> list[dict[str, Any]]:
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    out = []
    for d in data.get("daughters", []):
        out.append(
            {
                "instance_id": d["instance_id"],
                "ostium_xyz_mm": np.asarray(d["ostium_xyz_mm"], dtype=np.float64),
                "seed_xyz_mm": np.asarray(d["seed_xyz_mm"], dtype=np.float64),
                "direction_xyz": np.asarray(d["direction_xyz"], dtype=np.float64),
                "radius_mm": d.get("radius_mm"),
            }
        )
    return out


def load_pred_landmarks(prediction_path: Path) -> list[dict[str, Any]]:
    data = json.loads(prediction_path.read_text(encoding="utf-8"))
    out = []
    for d in data.get("daughters", []):
        out.append(
            {
                "instance_id": d["instance_id"],
                "ostium_xyz_mm": np.asarray(d["ostium_xyz_mm"], dtype=np.float64),
                "seed_xyz_mm": np.asarray(d["seed_xyz_mm"], dtype=np.float64),
                "direction_xyz": np.asarray(d["direction_xyz"], dtype=np.float64),
                "radius_mm": d.get("radius_mm"),
            }
        )
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def match_case(gt: list[dict], pred: list[dict], match_mm: float = MATCH_MM) -> CaseMetrics:
    n_gt, n_pred = len(gt), len(pred)
    flags: list[str] = []
    if n_gt == 0 and n_pred == 0:
        return CaseMetrics("?", 0, 0, 0, 1.0, 1.0, 1.0, None, None, None, [], [])
    if n_pred == 0:
        return CaseMetrics("?", n_gt, 0, 0, 0.0, 0.0, 0.0, None, None, None, ["no_predictions"], [])
    if n_gt == 0:
        return CaseMetrics("?", 0, n_pred, 0, 0.0, 1.0, 0.0, None, None, None, ["gt_empty"], [])

    cost = np.full((n_gt, n_pred), 1e6)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            cost[i, j] = float(np.linalg.norm(g["ostium_xyz_mm"] - p["ostium_xyz_mm"]))
    ri, cj = linear_sum_assignment(cost)

    matches = []
    ostium_errs, seed_errs, cosines = [], [], []
    used_pred = set()
    for i, j in zip(ri, cj):
        dist = float(cost[i, j])
        if dist > match_mm:
            continue
        used_pred.add(j)
        g, p = gt[i], pred[j]
        seed_err = float(np.linalg.norm(g["seed_xyz_mm"] - p["seed_xyz_mm"]))
        cg = _unit(g["direction_xyz"])
        cp = _unit(p["direction_xyz"])
        cos = float(np.clip(np.dot(cg, cp), -1.0, 1.0))
        ostium_errs.append(dist)
        seed_errs.append(seed_err)
        cosines.append(cos)
        rec = {
            "gt": g["instance_id"],
            "pred": p["instance_id"],
            "ostium_err_mm": round(dist, 3),
            "seed_err_mm": round(seed_err, 3),
            "direction_cosine": round(cos, 3),
            "pred_radius_mm": p["radius_mm"],
            "gt_radius_mm": g["radius_mm"],
        }
        if cos < DIR_COS_OK:
            rec["flag"] = "direction_mismatch"
            flags.append(f"{p['instance_id']} direction cosine {cos:.2f} vs {g['instance_id']}")
        if p["radius_mm"] is not None and not (RADIUS_ABS_OK[0] <= float(p["radius_mm"]) <= RADIUS_ABS_OK[1]):
            flags.append(f"{p['instance_id']} implausible radius {p['radius_mm']}")
        if g["radius_mm"] is not None and p["radius_mm"] is not None:
            ratio = float(p["radius_mm"]) / max(float(g["radius_mm"]), 1e-3)
            if ratio < 0.35 or ratio > 2.8:
                flags.append(
                    f"{p['instance_id']} radius {p['radius_mm']} vs GT {g['radius_mm']} (ratio {ratio:.2f})"
                )
        if abs(float(np.linalg.norm(cp)) - 1.0) > 0.05:
            flags.append(f"{p['instance_id']} direction is not unit length")
        matches.append(rec)

    n_matched = len(matches)
    prec = n_matched / n_pred if n_pred else 0.0
    reca = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * prec * reca / (prec + reca) if (prec + reca) else 0.0
    matched_gt_ids = {m["gt"] for m in matches}
    matched_pred_ids = {m["pred"] for m in matches}
    for g in gt:
        if g["instance_id"] not in matched_gt_ids:
            flags.append(f"FN {g['instance_id']}")
    for p in pred:
        if p["instance_id"] not in matched_pred_ids:
            flags.append(f"FP {p['instance_id']}")

    return CaseMetrics(
        case_id="?",
        n_gt=n_gt,
        n_pred=n_pred,
        n_matched=n_matched,
        precision=prec,
        recall=reca,
        f1=f1,
        mean_ostium_err_mm=float(np.mean(ostium_errs)) if ostium_errs else None,
        mean_seed_err_mm=float(np.mean(seed_errs)) if seed_errs else None,
        mean_dir_cosine=float(np.mean(cosines)) if cosines else None,
        flags=flags,
        matches=matches,
    )


def evaluate_eval_set(
    pred_dir: str | Path,
    eval_dir: str | Path,
    match_mm: float = MATCH_MM,
) -> dict[str, Any]:
    """Score all case_19..23 predictions against EVAL_SET annotations."""
    pred_dir = Path(pred_dir)
    eval_dir = Path(eval_dir)
    cases = []
    for case_dir in sorted(eval_dir.glob("case_*")):
        ann = case_dir / "annotations.json"
        if not ann.exists():
            continue
        num = case_dir.name.split("_")[1]
        pred_path = None
        for cand in (
            pred_dir / f"subject{int(num):03d}.json",
            pred_dir / f"case_{num}.json",
            pred_dir / f"{case_dir.name}.json",
            pred_dir / case_dir.name / "prediction.json",
        ):
            if cand.exists():
                pred_path = cand
                break
        if pred_path is None:
            cases.append(
                CaseMetrics(case_dir.name, 0, 0, 0, 0.0, 0.0, 0.0, None, None, None, ["missing_prediction"], [])
            )
            continue
        metrics = match_case(load_gt_landmarks(ann), load_pred_landmarks(pred_path), match_mm)
        metrics.case_id = case_dir.name
        cases.append(metrics)

    n_gt = sum(c.n_gt for c in cases)
    n_pred = sum(c.n_pred for c in cases)
    n_matched = sum(c.n_matched for c in cases)
    prec = n_matched / n_pred if n_pred else 0.0
    reca = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * prec * reca / (prec + reca) if (prec + reca) else 0.0
    ostium_errs = [c.mean_ostium_err_mm for c in cases if c.mean_ostium_err_mm is not None]
    return {
        "match_threshold_mm": match_mm,
        "overall": {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "n_matched": n_matched,
            "precision": round(prec, 4),
            "recall": round(reca, 4),
            "f1": round(f1, 4),
            "mean_ostium_err_mm": round(float(np.mean(ostium_errs)), 3) if ostium_errs else None,
        },
        "cases": [_case_to_dict(c) for c in cases],
    }


def _case_to_dict(c: CaseMetrics) -> dict[str, Any]:
    return {
        "case_id": c.case_id,
        "n_gt": c.n_gt,
        "n_pred": c.n_pred,
        "n_matched": c.n_matched,
        "precision": round(c.precision, 4),
        "recall": round(c.recall, 4),
        "f1": round(c.f1, 4),
        "mean_ostium_err_mm": None if c.mean_ostium_err_mm is None else round(c.mean_ostium_err_mm, 3),
        "mean_seed_err_mm": None if c.mean_seed_err_mm is None else round(c.mean_seed_err_mm, 3),
        "mean_dir_cosine": None if c.mean_dir_cosine is None else round(c.mean_dir_cosine, 3),
        "flags": c.flags,
        "matches": c.matches,
    }
