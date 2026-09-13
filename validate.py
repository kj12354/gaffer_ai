#!/usr/bin/env python3
"""Score predictions against EVAL_SET annotations for cases 19–23."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.validate import evaluate_eval_set


def main() -> None:
    p = argparse.ArgumentParser(description="Validate Branchseed predictions vs annotations.json")
    p.add_argument("--pred-dir", type=Path, default=Path("predictions"))
    p.add_argument("--eval-dir", type=Path, default=Path("EVAL_SET"))
    p.add_argument("--output", type=Path, default=Path("predictions/validation_metrics.json"))
    p.add_argument("--match-mm", type=float, default=10.0)
    args = p.parse_args()

    report = evaluate_eval_set(args.pred_dir, args.eval_dir, match_mm=args.match_mm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    ov = report["overall"]
    print("=== Branchseed validation (cases 19–23) ===")
    print(
        f"GT={ov['n_gt']}  pred={ov['n_pred']}  matched={ov['n_matched']}  "
        f"P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}"
    )
    if ov["mean_ostium_err_mm"] is not None:
        print(f"mean ostium error: {ov['mean_ostium_err_mm']:.2f} mm  (match ≤ {args.match_mm:.1f} mm)")
    for case in report["cases"]:
        loc = "n/a" if case["mean_ostium_err_mm"] is None else f"{case['mean_ostium_err_mm']:.2f} mm"
        print(
            f"  {case['case_id']}: pred {case['n_pred']}/{case['n_gt']}  "
            f"F1={case['f1']:.2f}  ostium={loc}  dir_cos={case['mean_dir_cosine']}"
        )
        for flag in case["flags"]:
            print(f"    ! {flag}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
