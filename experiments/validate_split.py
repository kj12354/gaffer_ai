#!/usr/bin/env python3
"""Compare gated splitter vs baseline match sets. Writes 19–23 predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branchseed.io import load_case
from branchseed.output import write_prediction
from branchseed.pipeline import run_on_volume_detailed
from branchseed.validate import load_gt_landmarks, match_case, MATCH_MM

BASELINE_COUNTS = {19: 4, 20: 3, 21: 9, 22: 11, 23: 2}
# Baseline GT matches from the last stable validate.py run.
BASELINE_MATCHED_GT = {
    19: {"branch_001", "branch_002", "branch_003"},
    20: {"branch_001", "branch_002", "branch_004"},
    21: {"branch_001", "branch_002", "branch_003"},
    22: {"branch_001", "branch_002", "branch_003", "branch_004", "branch_005", "branch_006"},
    23: {"branch_001", "branch_003"},
}


def _eval_pair(n: int) -> tuple[Path, Path]:
    return Path(f"EVAL_SET/case_{n}/orig{n}.nii.gz"), Path(f"EVAL_SET/case_{n}/aorta{n}.nii.gz")


def _subject025() -> tuple[Path, Path]:
    folder = Path("subject025")
    img = sorted(folder.glob("orig*"))[0]
    msk = [m for m in sorted(folder.glob("mask*")) if "daughter" not in m.name.lower()][0]
    return img, msk


def main() -> None:
    print("=== gated contact-patch splitter validation ===")
    overall_gt = overall_pred = overall_match = 0
    for n in (19, 20, 21, 22, 23):
        img, msk = _eval_pair(n)
        vol = load_case(img, msk, case_id=f"case_{n}")
        payload, det = run_on_volume_detailed(vol, verbose=False)
        write_prediction(payload, Path(f"predictions/subject{n:03d}.json"))
        gt = load_gt_landmarks(Path(f"EVAL_SET/case_{n}/annotations.json"))
        pred = [
            {
                "instance_id": d["instance_id"],
                "ostium_xyz_mm": np.asarray(d["ostium_xyz_mm"], dtype=np.float64),
                "seed_xyz_mm": np.asarray(d["seed_xyz_mm"], dtype=np.float64),
                "direction_xyz": np.asarray(d["direction_xyz"], dtype=np.float64),
                "radius_mm": d.get("radius_mm"),
            }
            for d in payload["daughters"]
        ]
        m = match_case(gt, pred, MATCH_MM)
        overall_gt += m.n_gt
        overall_pred += m.n_pred
        overall_match += m.n_matched
        matched_gt = {rec["gt"] for rec in m.matches}
        lost = BASELINE_MATCHED_GT[n] - matched_gt
        gained = matched_gt - BASELINE_MATCHED_GT[n]
        n_split = sum(1 for d in det.daughters if d.split_derived)
        print(
            f"case_{n}: pred {m.n_pred} (was {BASELINE_COUNTS[n]})  "
            f"matched {m.n_matched}/{m.n_gt}  "
            f"P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}  "
            f"split_derived={n_split}"
        )
        for line in det.merge_log:
            if line.startswith("split") or "split" in line:
                print(f"    merge {line}")
        if lost:
            print(f"    FRAGMENT / LOST baseline GT: {sorted(lost)}")
        if gained:
            print(f"    newly matched GT: {sorted(gained)}")
        for flag in m.flags:
            print(f"    ! {flag}")

    p = overall_match / overall_pred if overall_pred else 0.0
    r = overall_match / overall_gt if overall_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print(
        f"OVERALL  GT={overall_gt} pred={overall_pred} matched={overall_match}  "
        f"P={p:.3f} R={r:.3f} F1={f1:.3f}   baseline P=0.586 R=0.895 F1=0.708"
    )

    print("\n=== subject025 (fragmentation / count check) ===")
    img, msk = _subject025()
    vol = load_case(img, msk, case_id="subject025")
    payload, det = run_on_volume_detailed(vol, verbose=False)
    n_split = sum(1 for d in det.daughters if d.split_derived)
    print(f"subject025 kept={len(det.daughters)} split_derived={n_split}")
    for line in det.merge_log:
        if "split" in line:
            print(f"    merge {line}")


if __name__ == "__main__":
    main()
