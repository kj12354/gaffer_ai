#!/usr/bin/env python3
"""Leave-one-case-out keep/drop model on crop-adjudicated detections.

Uses only confirmed_real / confirmed_false. Uncertain rows are held out.
Fits a small L2 logistic model in numpy (no sklearn) and reports the
effect of the rule-based quality filter already in PipelineParams.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.params import PipelineParams
from branchseed.pipeline import run_on_volume_detailed
from branchseed.branches import _likely_artifact
from branchseed.validate import load_gt_landmarks, match_case

ROOT = Path(__file__).resolve().parent
VERDICTS = ROOT / "predictions" / "review_verdicts.csv"
FEAT_CSV = ROOT / "predictions" / "filter_features.csv"
EVAL = ROOT / "EVAL_SET"
PRED = ROOT / "predictions"

FEATURE_KEYS = (
    "vesselness",
    "radius_cv",
    "circularity",
    "path_mm",
    "radius_mm",
    "bone_frac",
    "adj_lumen_frac",
)


def _eval_pair(num: int) -> tuple[Path, Path]:
    return EVAL / f"case_{num}" / f"orig{num}.nii.gz", EVAL / f"case_{num}" / f"aorta{num}.nii.gz"


def _subject_pair(num: int) -> tuple[Path, Path]:
    folder = ROOT / f"subject{num:03d}"
    images = sorted(folder.glob("orig*"))
    masks = [m for m in sorted(folder.glob("mask*")) if "daughter" not in m.name.lower()]
    return images[0], masks[0]


def _load_verdicts() -> dict[tuple[str, str], str]:
    out = {}
    with VERDICTS.open() as f:
        for r in csv.DictReader(f):
            out[(r["case"], r["instance_id"])] = (r.get("verdict") or "").strip()
    return out


def _collect(params: PipelineParams) -> list[dict]:
    jobs = [(f"case_{n}", *_eval_pair(n)) for n in range(19, 24)]
    jobs.append(("subject025", *_subject_pair(25)))
    rows: list[dict] = []
    for case_id, image, mask in jobs:
        print(f"--- {case_id} ---")
        vol = load_case(image, mask, case_id=case_id)
        payload, det = run_on_volume_detailed(vol, params=params, verbose=True)
        pred_lm = [
            {
                "instance_id": d["instance_id"],
                "ostium_xyz_mm": np.asarray(d["ostium_xyz_mm"], dtype=np.float64),
                "seed_xyz_mm": np.asarray(d["seed_xyz_mm"], dtype=np.float64),
                "direction_xyz": np.asarray(d["direction_xyz"], dtype=np.float64),
                "radius_mm": d.get("radius_mm"),
            }
            for d in payload["daughters"]
        ]
        gt_path = EVAL / case_id / "annotations.json"
        gt = load_gt_landmarks(gt_path) if gt_path.exists() else []
        mets = match_case(gt, pred_lm) if gt else None
        matched_pred = set()
        if mets is not None:
            for m in mets.matches:
                matched_pred.add(m["pred"])
        for i, (dtr, d) in enumerate(zip(payload["daughters"], det.daughters), start=1):
            iid = dtr["instance_id"]
            rows.append(
                {
                    "case": case_id,
                    "instance_id": iid,
                    "status": "draft_match" if iid in matched_pred else "extra",
                    "ostium_x": dtr["ostium_xyz_mm"][0],
                    "ostium_y": dtr["ostium_xyz_mm"][1],
                    "ostium_z": dtr["ostium_xyz_mm"][2],
                    "radius_mm": float(d.radius_mm),
                    "path_mm": float(d.path_length_mm),
                    "vesselness": float(d.med_vesselness),
                    "mean_hu": float(d.mean_hu),
                    "radius_cv": float(d.radius_cv),
                    "circularity": float(d.mean_circularity),
                    "bone_frac": float(d.bone_frac),
                    "adj_lumen_frac": float(d.adj_lumen_frac),
                    "rule_drop": _likely_artifact(d, params) or "",
                }
            )
    return rows


def _match_verdicts(rows: list[dict], verdicts: dict[tuple[str, str], str]) -> None:
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r)
    used: set[tuple[str, str]] = set()
    for (case, iid), verd in verdicts.items():
        cands = by_case.get(case, [])
        # Prefer exact instance_id; else nearest ostium from the old CSV.
        hit = next((r for r in cands if r["instance_id"] == iid and (case, iid) not in used), None)
        if hit is None:
            continue
        hit["verdict"] = verd
        used.add((case, iid))


def _xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    kept = []
    for r in rows:
        v = r.get("verdict", "")
        if v not in {"confirmed_real", "confirmed_false"}:
            continue
        if any(not np.isfinite(float(r[k])) for k in FEATURE_KEYS):
            continue
        kept.append(r)
    X = np.array([[float(r[k]) for k in FEATURE_KEYS] for r in kept], dtype=np.float64)
    y = np.array([1.0 if r["verdict"] == "confirmed_real" else 0.0 for r in kept])
    return X, y, kept


def _standardize(X: np.ndarray, mu: np.ndarray | None = None, sd: np.ndarray | None = None):
    if mu is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    Z = np.concatenate([np.ones((Z.shape[0], 1)), Z], axis=1)
    return Z, mu, sd


def _fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    def nll(w: np.ndarray) -> float:
        z = X @ w
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        return loss + 0.5 * l2 * float(np.sum(w[1:] ** 2)) / max(len(y), 1)

    res = minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B")
    return res.x


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _loocv(rows: list[dict]) -> None:
    X, y, kept = _xy(rows)
    print(f"\nLabeled for model: {len(kept)}  real={int(y.sum())}  false={int((1-y).sum())}")
    if len(kept) < 8:
        print("Too few labeled rows for LOOCV.")
        return
    cases = sorted({r["case"] for r in kept})
    preds = np.zeros(len(kept))
    for hold in cases:
        tr = [i for i, r in enumerate(kept) if r["case"] != hold]
        te = [i for i, r in enumerate(kept) if r["case"] == hold]
        if not tr or not te:
            continue
        Ztr, mu, sd = _standardize(X[tr])
        w = _fit_logreg(Ztr, y[tr])
        Zte, _, _ = _standardize(X[te], mu, sd)
        preds[te] = _sigmoid(Zte @ w)
        names = ("bias",) + FEATURE_KEYS
        coef = ", ".join(f"{n}={c:+.2f}" for n, c in zip(names, w))
        print(f"  hold {hold}: n_te={len(te)}  {coef}")

    # Threshold sweep that never drops a true on LOOCV if possible.
    print("\nLOOCV threshold sweep (drop if p_real < t):")
    best = None
    for t in np.linspace(0.15, 0.75, 13):
        pred_pos = preds >= t
        tp = int(((y == 1) & pred_pos).sum())
        fn = int(((y == 1) & ~pred_pos).sum())
        tn = int(((y == 0) & ~pred_pos).sum())
        fp = int(((y == 0) & pred_pos).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        print(f"  t={t:.2f}  TP={tp} FN={fn} TN={tn} FP={fp}  P={prec:.2f} R={rec:.2f}")
        if fn == 0:
            best = (t, tn)
    if best:
        print(f"Safest no-FN threshold: t={best[0]:.2f}  (drops {best[1]} confirmed_false)")
    else:
        print("No threshold has zero FN on LOOCV — do not ship the logistic scores as a gate.")

    Z, mu, sd = _standardize(X)
    w = _fit_logreg(Z, y)
    names = ("bias",) + FEATURE_KEYS
    print("\nFull-fit coefficients (standardized):")
    for n, c in zip(names, w):
        print(f"  {n:16s} {c:+.3f}")


def _rule_report(rows: list[dict]) -> None:
    print("\nRule-based quality filter on this run (params as compiled):")
    for split in ("confirmed_real", "confirmed_false", "uncertain", ""):
        sub = [r for r in rows if r.get("verdict", "") == split]
        if split == "":
            sub = [r for r in rows if not r.get("verdict")]
            label = "unlabeled"
        else:
            label = split
        dropped = [r for r in sub if r.get("rule_drop")]
        print(f"  {label:16s} n={len(sub):2d}  dropped={len(dropped):2d}")
        for r in dropped:
            print(f"    {r['case']} {r['instance_id']}  {r['rule_drop']}  "
                  f"v={r['vesselness']:.3f} bone={r['bone_frac']:.2f} adj={r['adj_lumen_frac']:.2f}")


def _score_draft(rows: list[dict]) -> None:
    """What the rule would do to draft-matched 19–23 detections."""
    scored = [r for r in rows if r["case"].startswith("case_")]
    matched = [r for r in scored if r["status"] == "draft_match"]
    extras = [r for r in scored if r["status"] == "extra"]
    drop_m = [r for r in matched if r.get("rule_drop")]
    drop_e = [r for r in extras if r.get("rule_drop")]
    print("\nDraft impact if this rule is applied (cases 19–23):")
    print(f"  keep matches {len(matched) - len(drop_m)}/{len(matched)}  "
          f"(dropped matches: {[r['case']+':'+r['instance_id'] for r in drop_m] or 'none'})")
    print(f"  drop extras  {len(drop_e)}/{len(extras)}")
    n_pred = len(scored) - len(drop_m) - len(drop_e)
    n_match = len(matched) - len(drop_m)
    n_gt = 19
    prec = n_match / max(n_pred, 1)
    rec = n_match / n_gt
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  implied P={prec:.3f} R={rec:.3f} F1={f1:.3f}  (from {n_match} / {n_pred} vs {n_gt} GT)")


def main() -> None:
    params = PipelineParams()
    params.quality_filter = False
    rows = _collect(params)
    _match_verdicts(rows, _load_verdicts())
    # Recompute rule_drop with current thresholds (filter was off during detect).
    # _likely_artifact already stored from the unfiltered daughters.

    fieldnames = [
        "case", "instance_id", "status", "verdict", "ostium_x", "ostium_y", "ostium_z",
        "radius_mm", "path_mm", "vesselness", "mean_hu", "radius_cv", "circularity",
        "bone_frac", "adj_lumen_frac", "rule_drop",
    ]
    with FEAT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {FEAT_CSV}  n={len(rows)}")

    print("\nFeature medians by verdict:")
    for verd in ("confirmed_real", "confirmed_false", "uncertain"):
        sub = [r for r in rows if r.get("verdict") == verd]
        if not sub:
            continue
        print(f"  {verd} n={len(sub)}")
        for k in FEATURE_KEYS:
            vals = np.array([float(r[k]) for r in sub], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                print(f"    {k:16s} med={np.median(vals):.3f}  p10={np.percentile(vals,10):.3f}  "
                      f"p90={np.percentile(vals,90):.3f}")

    _loocv(rows)
    _rule_report(rows)
    _score_draft(rows)


if __name__ == "__main__":
    main()
