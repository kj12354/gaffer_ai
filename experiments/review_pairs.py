#!/usr/bin/env python3
"""Case 21 pair-check images + case 22 path-merge verification + review CSV."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branchseed.io import load_case, np_zyx_to_mm
from branchseed.output import write_prediction
from branchseed.pipeline import run_on_volume_detailed
from branchseed.validate import evaluate_eval_set
from branchseed.visualize import render_ostium_pair_check

MATCH_MM = 10.0
BEFORE_OVERALL = {"n_pred": 30, "n_matched": 17, "precision": 0.5667, "recall": 0.8947, "f1": 0.6939}
BEFORE_C22 = {"n_pred": 11, "n_matched": 6, "precision": 6 / 11, "recall": 1.0, "f1": 0.7059}

# Pre-fix landmarks for the suspected case-22 oversplit.
C22_DRAFT002_OST = np.array([22.026584, -140.195638, 308.544445])
C22_EXTRA008_OST = np.array([16.76716, -150.679269, 303.028572])


def _nearest(pred_daughters: list[dict], ost: np.ndarray) -> dict:
    dists = [float(np.linalg.norm(np.array(d["ostium_xyz_mm"]) - ost)) for d in pred_daughters]
    i = int(np.argmin(dists))
    return {**pred_daughters[i], "_dist_to_query": dists[i]}


def main() -> None:
    # --- 1. Case 21 pair-check images ---
    vol21 = load_case("EVAL_SET/case_21/orig21.nii.gz", "EVAL_SET/case_21/aorta21.nii.gz")
    pred21 = json.loads(Path("predictions/subject021.json").read_text())
    by_id = {d["instance_id"]: d for d in pred21["daughters"]}
    p1 = render_ostium_pair_check(
        vol21,
        [by_id["branch_006"]["ostium_xyz_mm"], by_id["branch_007"]["ostium_xyz_mm"]],
        ["006", "007"],
        "visuals/case21_pair1_check.png",
        "Case 21 pair 1  —  branch_006 / 007  (Z ~ 355–358 mm)",
    )
    p2 = render_ostium_pair_check(
        vol21,
        [by_id["branch_008"]["ostium_xyz_mm"], by_id["branch_009"]["ostium_xyz_mm"]],
        ["008", "009"],
        "visuals/case21_pair2_check.png",
        "Case 21 pair 2  —  branch_008 / 009  (Z ~ 336–340 mm)",
    )
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")

    # --- 2. Re-run detectors (merge fix on) ---
    results = {}
    for num in (19, 20, 21, 22, 23):
        vol = load_case(f"EVAL_SET/case_{num}/orig{num}.nii.gz", f"EVAL_SET/case_{num}/aorta{num}.nii.gz")
        payload, detected = run_on_volume_detailed(vol, verbose=True)
        write_prediction(payload, Path("predictions") / f"subject{num:03d}.json")
        results[num] = (payload, detected, vol)

    payload22, detected22, vol22 = results[22]
    print("\n=== Case 22 path-divergence merge vs the 007/008 pair ===")
    d_ost = float(np.linalg.norm(C22_DRAFT002_OST - C22_EXTRA008_OST))
    print(f"Pre-fix 3D ostium distance (draft-matched 007 vs extra 008): {d_ost:.2f} mm")
    print(f"Pre-fix Δz only: {abs(C22_DRAFT002_OST[2] - C22_EXTRA008_OST[2]):.2f} mm")
    xy = np.linalg.norm((C22_DRAFT002_OST - C22_EXTRA008_OST)[:2])
    print(f"Pre-fix in-plane (XY) wall separation: {xy:.2f} mm")

    after_a = _nearest(payload22["daughters"], C22_DRAFT002_OST)
    after_b = _nearest(payload22["daughters"], C22_EXTRA008_OST)
    same = after_a["instance_id"] == after_b["instance_id"]
    print(f"After merge: nearest to old 007 -> {after_a['instance_id']} (Δ {after_a['_dist_to_query']:.2f} mm)")
    print(f"After merge: nearest to old 008 -> {after_b['instance_id']} (Δ {after_b['_dist_to_query']:.2f} mm)")
    print(f"Merged into one instance? {same}")
    print("Merge log (case 22):")
    for line in detected22.merge_log:
        print(f"  {line}")

    # Direction / seed geometry of the two surviving nearest detections
    if not same:
        oa = np.array(after_a["ostium_xyz_mm"])
        ob = np.array(after_b["ostium_xyz_mm"])
        sa = np.array(after_a["seed_xyz_mm"])
        sb = np.array(after_b["seed_xyz_mm"])
        cos = float(np.dot(after_a["direction_xyz"], after_b["direction_xyz"]))
        print(
            f"Surviving pair: 3D ostium={np.linalg.norm(oa-ob):.2f} mm  "
            f"seed_sep={np.linalg.norm(sa-sb):.2f} mm  dir_cos={cos:.2f}"
        )

    # --- 3. Review CSV with confidence fields; flag 22 singletons ---
    rows = []
    for num in (21, 22):
        payload, detected, vol = results[num]
        gt = json.loads(Path(f"EVAL_SET/case_{num}/annotations.json").read_text())
        G, P = gt["daughters"], payload["daughters"]
        cost = np.array(
            [
                [np.linalg.norm(np.array(g["ostium_xyz_mm"]) - np.array(p["ostium_xyz_mm"])) for p in P]
                for g in G
            ]
        )
        ri, cj = linear_sum_assignment(cost)
        matched = {}
        for i, j in zip(ri, cj):
            if cost[i, j] <= MATCH_MM:
                matched[j] = (G[i]["instance_id"], float(cost[i, j]))
        # Map JSON daughters back to internal Daughter objects (same order).
        internals = detected.daughters
        for j, p in enumerate(P):
            d = internals[j]
            ost = np.array(p["ostium_xyz_mm"])
            flag = ""
            if num == 22 and p["instance_id"] in {"branch_001", "branch_011"}:
                flag = "singleton_unpaired_review_do_not_auto_remove"
            status = "draft_match" if j in matched else "extra"
            gt_id, err = matched.get(j, ("", None))
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
                    "vesselness": round(float(d.med_vesselness), 4),
                    "mean_hu": round(float(d.mean_hu), 1),
                    "review_flag": flag,
                }
            )

    csv_path = Path("predictions/lumbar_pattern_review.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")
    print("Case 22 singletons (flagged, not removed):")
    for r in rows:
        if r["review_flag"]:
            print(
                f"  {r['instance_id']}: vesselness={r['vesselness']}  "
                f"path_length_mm={r['path_length_mm']}  radius_mm={r['radius_mm']}  "
                f"z={r['ostium_z']}"
            )

    report = evaluate_eval_set("predictions", "EVAL_SET")
    Path("predictions/validation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    ov = report["overall"]
    c22 = next(c for c in report["cases"] if c["case_id"] == "case_22")
    print("\n=== validate.py after path-divergence merge ===")
    print(
        f"OVERALL before: P={BEFORE_OVERALL['precision']:.3f} R={BEFORE_OVERALL['recall']:.3f} "
        f"F1={BEFORE_OVERALL['f1']:.3f} pred={BEFORE_OVERALL['n_pred']} matched={BEFORE_OVERALL['n_matched']}"
    )
    print(
        f"OVERALL after:  P={ov['precision']:.3f} R={ov['recall']:.3f} "
        f"F1={ov['f1']:.3f} pred={ov['n_pred']} matched={ov['n_matched']}"
    )
    print(
        f"CASE 22 before: pred={BEFORE_C22['n_pred']} matched={BEFORE_C22['n_matched']} "
        f"P={BEFORE_C22['precision']:.3f} R={BEFORE_C22['recall']:.3f} F1={BEFORE_C22['f1']:.3f}"
    )
    print(
        f"CASE 22 after:  pred={c22['n_pred']} matched={c22['n_matched']} "
        f"P={c22['precision']:.3f} R={c22['recall']:.3f} F1={c22['f1']:.3f}"
    )
    print(f"Case 22 flags: {c22['flags']}")


if __name__ == "__main__":
    main()
