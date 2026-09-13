#!/usr/bin/env python3
"""Re-detect 19–23 after the 5 mm followable-path fix; refresh review CSV + validate."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branchseed.io import load_case
from branchseed.output import write_prediction
from branchseed.pipeline import run_on_volume_detailed
from branchseed.validate import evaluate_eval_set

MATCH_MM = 10.0
BEFORE = {
    19: 4,
    20: 3,
    21: 10,
    22: 11,
    23: 2,
}
BEFORE_OVERALL = {"n_pred": 30, "n_matched": 17, "precision": 0.5667, "recall": 0.8947, "f1": 0.6939}


def _flag(num: int, instance_id: str, status: str) -> str:
    if num == 21 and instance_id == "branch_006":
        return "likely_genuine_fp_near_zero_vesselness"
    if num == 21 and instance_id in {"branch_007", "branch_008", "branch_009"}:
        return "ambiguous_manual_review_do_not_auto_remove"
    if num == 22 and instance_id == "branch_008":
        return "independent_extra_not_oversplit_of_007"
    if num == 22 and instance_id == "branch_007" and status == "draft_match":
        return "draft_match_independent_of_008"
    if num == 22 and instance_id in {"branch_001", "branch_011"}:
        return "singleton_unpaired_review_do_not_auto_remove"
    return ""


def main() -> None:
    rows = []
    counts = {}
    short_reported = []
    for num in (19, 20, 21, 22, 23):
        vol = load_case(
            f"EVAL_SET/case_{num}/orig{num}.nii.gz",
            f"EVAL_SET/case_{num}/aorta{num}.nii.gz",
        )
        payload, detected = run_on_volume_detailed(vol, verbose=False)
        write_prediction(payload, Path("predictions") / f"subject{num:03d}.json")
        counts[num] = len(payload["daughters"])
        gt = json.loads(Path(f"EVAL_SET/case_{num}/annotations.json").read_text())
        G, P = gt["daughters"], payload["daughters"]
        matched = {}
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
            if d.path_length_mm < 5.0:
                short_reported.append((num, p["instance_id"], d.path_length_mm, d.extent_mm))
            ost = np.array(p["ostium_xyz_mm"])
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
                    "review_flag": _flag(num, p["instance_id"], status),
                }
            )

    print("=== Candidate counts (path-length fix only; no vesselness floor) ===")
    for num in (19, 20, 21, 22, 23):
        delta = counts[num] - BEFORE[num]
        print(f"  case_{num}: {BEFORE[num]} -> {counts[num]}  (Δ {delta:+d})")
    print(f"  total: {sum(BEFORE.values())} -> {sum(counts.values())}")
    if short_reported:
        print("WARNING: still reporting path < 5 mm:")
        for item in short_reported:
            print(f"  {item}")
    else:
        print("All surviving detections report followable path_length_mm ≥ 5.0")

    csv_path = Path("predictions/lumbar_pattern_review.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path}")

    report = evaluate_eval_set("predictions", "EVAL_SET")
    Path("predictions/validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    ov = report["overall"]
    print("\n=== validate.py ===")
    print(
        f"Vesselness floor: NOT applied (no clean separation). Δ from vesselness = 0."
    )
    print(
        f"BEFORE path fix: P={BEFORE_OVERALL['precision']:.3f} R={BEFORE_OVERALL['recall']:.3f} "
        f"F1={BEFORE_OVERALL['f1']:.3f} pred={BEFORE_OVERALL['n_pred']} matched={BEFORE_OVERALL['n_matched']}"
    )
    print(
        f"AFTER path fix:  P={ov['precision']:.3f} R={ov['recall']:.3f} "
        f"F1={ov['f1']:.3f} pred={ov['n_pred']} matched={ov['n_matched']}"
    )
    for case in report["cases"]:
        loc = "n/a" if case["mean_ostium_err_mm"] is None else f"{case['mean_ostium_err_mm']:.2f} mm"
        print(
            f"  {case['case_id']}: pred {case['n_pred']}/{case['n_gt']}  "
            f"matched={case['n_matched']}  F1={case['f1']:.2f}  ostium={loc}"
        )
        for flag in case["flags"]:
            print(f"    ! {flag}")
    print(f"Wrote predictions/validation_metrics.json")


if __name__ == "__main__":
    main()
