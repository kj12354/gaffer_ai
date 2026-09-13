#!/usr/bin/env python3
"""CLI verdict entry for review/ crops. Keys: r real / f false / u uncertain / n next / q quit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CSV_PATH = Path("predictions/review_verdicts.csv")
VALID = {"r": "confirmed_real", "f": "confirmed_false", "u": "uncertain", "": ""}


def main() -> None:
    p = argparse.ArgumentParser(description="Record manual verdicts for review crops")
    p.add_argument("--case", default="", help="Filter, e.g. case_21 or subject025")
    p.add_argument("--priority-only", action="store_true",
                   help="Only 21/007-009 and subject025")
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    args = p.parse_args()
    rows = list(csv.DictReader(args.csv.open()))
    if args.case:
        rows = [r for r in rows if r["case"] == args.case]
    if args.priority_only:
        keep = []
        for r in rows:
            if r["case"] == "case_21" and r["instance_id"] in {"branch_007", "branch_008", "branch_009"}:
                keep.append(r)
            elif r["case"] == "subject025":
                keep.append(r)
        rows = keep
    if not rows:
        raise SystemExit("No rows. Run generate_review.py first.")
    print(f"{len(rows)} detections. r=real  f=false  u=uncertain  s=skip  q=quit")
    by_key = {(r["case"], r["instance_id"]): r for r in csv.DictReader(args.csv.open())}
    for r in rows:
        print("-" * 60)
        print(
            f"{r['case']} {r['instance_id']}  {r['status']}  gt={r.get('gt_id') or '-'}  "
            f"z={r['ostium_z']}  vess={r['vesselness']}  cv={r['radius_cv']}  "
            f"circ={r['mean_circularity']}  path={r['path_length_mm']}  HU={r['mean_hu']}"
        )
        print(f"  crop: {r.get('crop_path', '')}")
        print(f"  current verdict: {r.get('verdict') or '(empty)'}")
        ans = input("  verdict [r/f/u/s/q]: ").strip().lower()
        if ans == "q":
            break
        if ans in VALID and ans != "":
            by_key[(r["case"], r["instance_id"])]["verdict"] = VALID[ans]
            print(f"  -> {VALID[ans]}")
    all_rows = list(by_key.values())
    fieldnames = list(all_rows[0].keys())
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
