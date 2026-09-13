#!/usr/bin/env python3
"""Why draft GT ostia on cases 20 and 23 are missed, and direction of the case-20 match."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case, np_zyx_to_mm
from branchseed.params import PipelineParams
from branchseed.pipeline import run_on_volume_detailed
from branchseed.validate import load_gt_landmarks


def _mm(vol, d, origin):
    return np_zyx_to_mm(vol.image, *(d.ostium_zyx + origin))


def report(case_num: int) -> None:
    image = Path(f"EVAL_SET/case_{case_num}/orig{case_num}.nii.gz")
    mask = Path(f"EVAL_SET/case_{case_num}/aorta{case_num}.nii.gz")
    gt = load_gt_landmarks(Path(f"EVAL_SET/case_{case_num}/annotations.json"))
    params = PipelineParams()
    params.quality_filter = False
    vol = load_case(image, mask, case_id=f"case_{case_num}")
    payload, det = run_on_volume_detailed(vol, params=params, verbose=True)
    print(f"\n=== case_{case_num}  kept={len(det.daughters)}  rejected={len(det.rejected)} ===")
    print("rejected:", det.rejected[:20], ("..." if len(det.rejected) > 20 else ""))

    pred_mm = [np.asarray(d["ostium_xyz_mm"], dtype=np.float64) for d in payload["daughters"]]
    # Recover origin from first daughter if present via payload vs det alignment.
    # origin is baked into payload ostia already.
    print("\nGT vs nearest kept detection:")
    for g in gt:
        go = g["ostium_xyz_mm"]
        gs = g["seed_xyz_mm"]
        gd = g["direction_xyz"]
        gd = gd / (np.linalg.norm(gd) + 1e-9)
        if pred_mm:
            dists = [float(np.linalg.norm(p - go)) for p in pred_mm]
            j = int(np.argmin(dists))
            p = payload["daughters"][j]
            po = np.asarray(p["ostium_xyz_mm"])
            ps = np.asarray(p["seed_xyz_mm"])
            pd = np.asarray(p["direction_xyz"])
            pd = pd / (np.linalg.norm(pd) + 1e-9)
            print(
                f"  {g['instance_id']}  ost={np.round(go,2)}  "
                f"nearest {p['instance_id']} Δ={dists[j]:.2f} mm  "
                f"dir_cos={float(np.dot(gd, pd)):.3f}  "
                f"seedΔ={float(np.linalg.norm(ps-gs)):.2f}"
            )
            print(f"           gt_dir={np.round(gd,3)}  pred_dir={np.round(pd,3)}")
        else:
            print(f"  {g['instance_id']}  no predictions")

    # Features of kept extras / matches
    print("\nKept features:")
    for djson, d in zip(payload["daughters"], det.daughters):
        print(
            f"  {djson['instance_id']}  v={d.med_vesselness:.3f}  path={d.path_length_mm:.1f}  "
            f"r={d.radius_mm:.2f}  cv={d.radius_cv:.2f}  circ={d.mean_circularity:.2f}  "
            f"bone={d.bone_frac:.2f}  adj={d.adj_lumen_frac:.2f}  HU={d.mean_hu:.0f}"
        )


def main() -> None:
    for n in (20, 23):
        report(n)


if __name__ == "__main__":
    main()
