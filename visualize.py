#!/usr/bin/env python3
"""Simple visual check: aorta surface, ostia, and direction arrows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.pipeline import detect_daughters
from branchseed.visualize import render_case


def _pair_for_eval_case(eval_dir: Path, num: int) -> tuple[Path, Path]:
    folder = eval_dir / f"case_{num}"
    return folder / f"orig{num}.nii.gz", folder / f"aorta{num}.nii.gz"


def main() -> None:
    p = argparse.ArgumentParser(description="Render Branchseed visual checks")
    p.add_argument("--image", type=Path)
    p.add_argument("--aorta-mask", type=Path)
    p.add_argument("--prediction", type=Path, help="Existing prediction JSON (else run detector)")
    p.add_argument("--output", type=Path, default=Path("visuals/preview.png"))
    p.add_argument("--eval-dir", type=Path, default=Path("EVAL_SET"))
    p.add_argument("--pred-dir", type=Path, default=Path("predictions"))
    p.add_argument("--cases", default="19,21,22", help="EVAL_SET case numbers for the required 3-case check")
    p.add_argument("--output-dir", type=Path, default=Path("visuals"))
    args = p.parse_args()

    if args.image and args.aorta_mask:
        vol = load_case(args.image, args.aorta_mask)
        if args.prediction and args.prediction.exists():
            pred = json.loads(args.prediction.read_text(encoding="utf-8"))
        else:
            pred = detect_daughters(str(args.image), str(args.aorta_mask), verbose=True)
        path = render_case(vol, pred, args.output)
        print(f"Wrote {path}")
        return

    nums = [int(x.strip()) for x in args.cases.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for num in nums:
        image, mask = _pair_for_eval_case(args.eval_dir, num)
        if not image.exists() or not mask.exists():
            print(f"skip case_{num}: missing {image} or {mask}")
            continue
        vol = load_case(image, mask)
        pred_path = args.pred_dir / f"subject{num:03d}.json"
        if pred_path.exists():
            pred = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            pred = detect_daughters(str(image), str(mask), verbose=True)
        out = args.output_dir / f"case_{num}_ostia.png"
        render_case(vol, pred, out, title=f"case_{num} — aorta, ostia, directions")
        print(f"Wrote {out}  ({len(pred.get('daughters', []))} daughters)")


if __name__ == "__main__":
    main()
