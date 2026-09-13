#!/usr/bin/env python3
"""Vesselness matched-vs-extra distributions + path/extent audit (no new gates)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.pipeline import run_on_volume_detailed

MATCH_MM = 10.0


def _stats(vals: np.ndarray) -> dict[str, float]:
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
    print(
        f"{name}: n={int(s['n'])}  min={s['min']:.4f}  p10={s['p10']:.4f}  "
        f"p25={s['p25']:.4f}  median={s['median']:.4f}  mean={s['mean']:.4f}  "
        f"p75={s['p75']:.4f}  p90={s['p90']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}"
    )


def main() -> None:
    rows: list[dict] = []
    for num in (19, 20, 21, 22, 23):
        vol = load_case(
            f"EVAL_SET/case_{num}/orig{num}.nii.gz",
            f"EVAL_SET/case_{num}/aorta{num}.nii.gz",
        )
        payload, detected = run_on_volume_detailed(vol, verbose=False)
        gt = json.loads(Path(f"EVAL_SET/case_{num}/annotations.json").read_text())
        G, P = gt["daughters"], payload["daughters"]
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
        matched: dict[int, tuple[str, float]] = {}
        if G and P:
            ri, cj = linear_sum_assignment(cost)
            for i, j in zip(ri, cj):
                if cost[i, j] <= MATCH_MM:
                    matched[j] = (G[i]["instance_id"], float(cost[i, j]))
        for j, (p, d) in enumerate(zip(P, detected.daughters)):
            status = "draft_match" if j in matched else "extra"
            gt_id, err = matched.get(j, ("", None))
            rows.append(
                {
                    "case": f"case_{num}",
                    "instance_id": p["instance_id"],
                    "status": status,
                    "gt_id": gt_id,
                    "ostium_err_mm": err,
                    "vesselness": float(d.med_vesselness),
                    "path_length_mm": float(d.path_length_mm),
                    "extent_mm": float(d.extent_mm),
                    "followable_mm": float(max(d.path_length_mm, d.extent_mm)),
                }
            )

    matched_v = np.array([r["vesselness"] for r in rows if r["status"] == "draft_match"])
    extra_v = np.array([r["vesselness"] for r in rows if r["status"] == "extra"])
    print("=== Vesselness by draft-match status (cases 19–23) ===")
    _print_stats("draft-matched (true)", _stats(matched_v))
    _print_stats("unmatched extras     ", _stats(extra_v))
    print("\nPer-detection vesselness / path / extent:")
    print(
        f"{'case':<9} {'id':<12} {'status':<13} {'gt':<12} "
        f"{'vess':>7} {'path':>7} {'extent':>7} {'follow':>7}"
    )
    for r in rows:
        print(
            f"{r['case']:<9} {r['instance_id']:<12} {r['status']:<13} "
            f"{(r['gt_id'] or '-'):<12} {r['vesselness']:7.4f} "
            f"{r['path_length_mm']:7.2f} {r['extent_mm']:7.2f} "
            f"{r['followable_mm']:7.2f}"
        )

    print("\n=== Path-length floor audit (reported path < 5.0 mm) ===")
    short = [r for r in rows if r["path_length_mm"] < 5.0]
    if not short:
        print("None.")
    else:
        for r in short:
            print(
                f"  {r['case']} {r['instance_id']} {r['status']}: "
                f"path={r['path_length_mm']:.2f}  extent={r['extent_mm']:.2f}  "
                f"followable={r['followable_mm']:.2f}  vesselness={r['vesselness']:.4f}"
            )

    # Separation scan: keep all / most matches, drop extras.
    print("\n=== Separation scan (do not apply) ===")
    if matched_v.size and extra_v.size:
        t_all_true = float(matched_v.min())
        extras_below_all_true = int((extra_v < t_all_true).sum())
        extras_at_or_above = int((extra_v >= t_all_true).sum())
        print(
            f"Lowest true vesselness = {t_all_true:.4f}. "
            f"A floor just below that keeps all {matched_v.size} matches and "
            f"drops {extras_below_all_true}/{extra_v.size} extras "
            f"(overlap: {extras_at_or_above} extras still survive)."
        )
        # Youden-like sweep on unique midpoints.
        cands = np.unique(np.concatenate([matched_v, extra_v]))
        best = None
        for t in cands:
            tp = int((matched_v >= t).sum())
            fn = int((matched_v < t).sum())
            fp = int((extra_v >= t).sum())
            tn = int((extra_v < t).sum())
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            youden = tpr - fpr
            rec = (t, youden, tp, fn, fp, tn, tpr, fpr)
            if best is None or rec[1] > best[1]:
                best = rec
        t, youden, tp, fn, fp, tn, tpr, fpr = best
        print(
            f"Best Youden floor = {t:.4f}  (Youden={youden:.3f}): "
            f"keep {tp}/{matched_v.size} true, drop {tn}/{extra_v.size} extras, "
            f"lose {fn} true, leave {fp} extras."
        )
        overlap_lo = float(max(matched_v.min(), extra_v.min()))
        overlap_hi = float(min(matched_v.max(), extra_v.max()))
        n_true_in = int(((matched_v >= overlap_lo) & (matched_v <= overlap_hi)).sum())
        n_extra_in = int(((extra_v >= overlap_lo) & (extra_v <= overlap_hi)).sum())
        print(
            f"Overlap band [{overlap_lo:.4f}, {overlap_hi:.4f}] contains "
            f"{n_true_in}/{matched_v.size} trues and {n_extra_in}/{extra_v.size} extras."
        )

    Path("visuals").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bins = np.linspace(0.0, max(0.7, float(max(matched_v.max(), extra_v.max())) + 0.02), 18)
    ax.hist(matched_v, bins=bins, alpha=0.65, color="#1f77b4", label=f"draft-matched (n={matched_v.size})")
    ax.hist(extra_v, bins=bins, alpha=0.55, color="#d62728", label=f"unmatched extras (n={extra_v.size})")
    if matched_v.size:
        ax.axvline(float(matched_v.min()), color="#1f77b4", ls="--", lw=1.0, label="min true")
    if extra_v.size:
        ax.axvline(float(np.median(extra_v)), color="#d62728", ls=":", lw=1.0, label="median extra")
    ax.set_xlabel("median Frangi vesselness on the grown instance")
    ax.set_ylabel("detections (cases 19–23)")
    ax.set_title("Vesselness: draft-matched vs unmatched extras")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(left=0)
    fig.tight_layout()
    out = Path("visuals/vesselness_distribution.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
