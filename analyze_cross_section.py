#!/usr/bin/env python3
"""Cross-sectional radius CV / circularity on cases 19–23 (diagnostic, no gate)."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.pipeline import run_on_volume_detailed

MATCH_MM = 10.0
CSV_PATH = Path("predictions/lumbar_pattern_review.csv")


def _stats(vals: np.ndarray) -> dict[str, float]:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {}
    qs = (0, 10, 25, 50, 75, 90, 95, 100)
    out = {
        "n": float(vals.size),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
    }
    for q, v in zip(qs, np.percentile(vals, qs)):
        out[f"p{q:02d}"] = float(v)
    return out


def _print_stats(name: str, s: dict[str, float]) -> None:
    if not s:
        print(f"{name}: no finite values")
        return
    print(
        f"{name}: n={int(s['n'])}  min={s['min']:.4f}  p10={s['p10']:.4f}  "
        f"p25={s['p25']:.4f}  median={s['median']:.4f}  mean={s['mean']:.4f}  "
        f"p75={s['p75']:.4f}  p90={s['p90']:.4f}  max={s['max']:.4f}"
    )


def _separation(true: np.ndarray, extra: np.ndarray, higher_is_signal: bool) -> None:
    true = true[np.isfinite(true)]
    extra = extra[np.isfinite(extra)]
    if true.size == 0 or extra.size == 0:
        print("  (insufficient finite values)")
        return
    if higher_is_signal:
        t_all = float(true.min())
        extras_below = int((extra < t_all).sum())
        print(
            f"  Lowest true = {t_all:.4f}. Floor just below keeps all {true.size} "
            f"trues and drops {extras_below}/{extra.size} extras."
        )
        cands = np.unique(np.concatenate([true, extra]))
        best = None
        for t in cands:
            tp = int((true >= t).sum())
            fn = int((true < t).sum())
            fp = int((extra >= t).sum())
            tn = int((extra < t).sum())
            youden = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
            rec = (t, youden, tp, fn, fp, tn)
            if best is None or rec[1] > best[1]:
                best = rec
    else:
        t_all = float(true.max())
        extras_above = int((extra > t_all).sum())
        print(
            f"  Highest true = {t_all:.4f}. Ceiling just above keeps all {true.size} "
            f"trues and drops {extras_above}/{extra.size} extras."
        )
        cands = np.unique(np.concatenate([true, extra]))
        best = None
        for t in cands:
            tp = int((true <= t).sum())
            fn = int((true > t).sum())
            fp = int((extra <= t).sum())
            tn = int((extra > t).sum())
            youden = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
            rec = (t, youden, tp, fn, fp, tn)
            if best is None or rec[1] > best[1]:
                best = rec
    t, youden, tp, fn, fp, tn = best
    print(
        f"  Best Youden cut = {t:.4f} (Youden={youden:.3f}): "
        f"keep {tp}/{true.size} true, drop {tn}/{extra.size} extras, "
        f"lose {fn} true, leave {fp} extras."
    )
    lo = float(max(true.min(), extra.min()))
    hi = float(min(true.max(), extra.max()))
    n_t = int(((true >= lo) & (true <= hi)).sum())
    n_e = int(((extra >= lo) & (extra <= hi)).sum())
    print(
        f"  Overlap band [{lo:.4f}, {hi:.4f}] contains "
        f"{n_t}/{true.size} trues and {n_e}/{extra.size} extras."
    )


def main() -> None:
    existing: dict[tuple[str, str], dict] = {}
    if CSV_PATH.exists():
        with CSV_PATH.open() as f:
            for row in csv.DictReader(f):
                existing[(row["case"], row["instance_id"])] = row

    rows = []
    for num in (19, 20, 21, 22, 23):
        vol = load_case(
            f"EVAL_SET/case_{num}/orig{num}.nii.gz",
            f"EVAL_SET/case_{num}/aorta{num}.nii.gz",
        )
        payload, detected = run_on_volume_detailed(vol, verbose=False)
        gt = json.loads(Path(f"EVAL_SET/case_{num}/annotations.json").read_text())
        G, P = gt["daughters"], payload["daughters"]
        matched: dict[int, tuple[str, float]] = {}
        if G and P:
            cost = np.array(
                [
                    [
                        np.linalg.norm(
                            np.array(g["ostium_xyz_mm"]) - np.array(p["ostium_xyz_mm"])
                        )
                        for p in P
                    ]
                    for g in G
                ]
            )
            ri, cj = linear_sum_assignment(cost)
            for i, j in zip(ri, cj):
                if cost[i, j] <= MATCH_MM:
                    matched[j] = (G[i]["instance_id"], float(cost[i, j]))
        for j, (p, d) in enumerate(zip(P, detected.daughters)):
            status = "draft_match" if j in matched else "extra"
            gt_id, err = matched.get(j, ("", None))
            ost = np.array(p["ostium_xyz_mm"])
            prev = existing.get((f"case_{num}", p["instance_id"]), {})
            rows.append(
                {
                    "case": f"case_{num}",
                    "instance_id": p["instance_id"],
                    "status": status,
                    "gt_id": gt_id,
                    "ostium_err_mm": "" if err is None else round(err, 2),
                    "ostium_x": round(float(ost[0]), 2),
                    "ostium_y": round(float(ost[1]), 2),
                    "ostium_z": round(float(ost[2]), 2),
                    "radius_mm": round(float(p["radius_mm"]), 4),
                    "path_length_mm": round(float(d.path_length_mm), 2),
                    "extent_mm": round(float(d.extent_mm), 2),
                    "vesselness": round(float(d.med_vesselness), 4),
                    "mean_hu": round(float(d.mean_hu), 1),
                    "radius_cv": "" if not math.isfinite(d.radius_cv) else round(float(d.radius_cv), 4),
                    "mean_circularity": (
                        ""
                        if not math.isfinite(d.mean_circularity)
                        else round(float(d.mean_circularity), 4)
                    ),
                    "review_flag": prev.get("review_flag", ""),
                }
            )

    true_cv = np.array([float(r["radius_cv"]) for r in rows if r["status"] == "draft_match" and r["radius_cv"] != ""])
    extra_cv = np.array([float(r["radius_cv"]) for r in rows if r["status"] == "extra" and r["radius_cv"] != ""])
    true_c = np.array([float(r["mean_circularity"]) for r in rows if r["status"] == "draft_match" and r["mean_circularity"] != ""])
    extra_c = np.array([float(r["mean_circularity"]) for r in rows if r["status"] == "extra" and r["mean_circularity"] != ""])

    print("=== Radius CV (std/mean along ostium→seed path) ===")
    _print_stats("draft-matched (true)", _stats(true_cv))
    _print_stats("unmatched extras     ", _stats(extra_cv))
    print("Separation (lower CV = more tube-like):")
    _separation(true_cv, extra_cv, higher_is_signal=False)

    print("\n=== Mean circularity (in-plane axis ratio, 1 = circle) ===")
    _print_stats("draft-matched (true)", _stats(true_c))
    _print_stats("unmatched extras     ", _stats(extra_c))
    print("Separation (higher circularity = more tube-like):")
    _separation(true_c, extra_c, higher_is_signal=True)

    print("\nPer-detection:")
    print(f"{'case':<9} {'id':<12} {'status':<13} {'vess':>7} {'rad_cv':>8} {'circ':>7}")
    for r in rows:
        print(
            f"{r['case']:<9} {r['instance_id']:<12} {r['status']:<13} "
            f"{r['vesselness']:>7} {str(r['radius_cv']):>8} {str(r['mean_circularity']):>7}"
        )

    Path("visuals").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    if true_cv.size or extra_cv.size:
        lo = 0.0
        hi = max(
            0.8,
            float(np.nanmax(np.concatenate([true_cv, extra_cv]))) + 0.05 if (true_cv.size + extra_cv.size) else 0.8,
        )
        bins = np.linspace(lo, hi, 16)
        axes[0].hist(true_cv, bins=bins, alpha=0.65, color="#1f77b4", label=f"draft-matched (n={true_cv.size})")
        axes[0].hist(extra_cv, bins=bins, alpha=0.55, color="#d62728", label=f"extras (n={extra_cv.size})")
    axes[0].set_xlabel("radius CV (std/mean)")
    axes[0].set_ylabel("detections (cases 19–23)")
    axes[0].set_title("Radius variation along path")
    axes[0].legend(fontsize=8)
    if true_c.size or extra_c.size:
        bins_c = np.linspace(0.0, 1.0, 16)
        axes[1].hist(true_c, bins=bins_c, alpha=0.65, color="#1f77b4", label=f"draft-matched (n={true_c.size})")
        axes[1].hist(extra_c, bins=bins_c, alpha=0.55, color="#d62728", label=f"extras (n={extra_c.size})")
    axes[1].set_xlabel("mean circularity (minor/major)")
    axes[1].set_title("Cross-section circularity")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("visuals/cross_section_consistency.png", dpi=140)
    plt.close(fig)
    print("Wrote visuals/cross_section_consistency.png")

    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
