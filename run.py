#!/usr/bin/env python3
"""Branchseed CLI.

Required:
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

Optional batch:
    python run.py --batch-dir EVAL_SET --output-dir predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python run.py` from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import infer_case_id
from branchseed.output import write_prediction
from branchseed.pipeline import detect_daughters
from branchseed.resources import Timer


def _find_pair(case_dir: Path) -> tuple[Path, Path] | None:
    images = sorted(
        list(case_dir.glob("orig*.nii"))
        + list(case_dir.glob("orig*.nii.gz"))
        + list(case_dir.glob("image*.nii"))
        + list(case_dir.glob("image*.nii.gz"))
    )
    masks = sorted(
        list(case_dir.glob("aorta*.nii"))
        + list(case_dir.glob("aorta*.nii.gz"))
        + list(case_dir.glob("mask*.nii"))
        + list(case_dir.glob("mask*.nii.gz"))
    )
    # Prefer dedicated aorta* over combined viewing masks.
    masks = [m for m in masks if "daughter" not in m.name.lower() and "and_" not in m.name.lower()]
    if not images or not masks:
        return None
    return images[0], masks[0]


def _run_one(image: Path, mask: Path, output: Path, case_id: str | None, verbose: bool) -> None:
    with Timer() as timer:
        payload = detect_daughters(str(image), str(mask), case_id=case_id, verbose=verbose)
    write_prediction(payload, output)
    n = len(payload["daughters"])
    print(f"Wrote {output}  ({n} daughter(s))")
    print(timer.stats.format_line())


def main() -> None:
    p = argparse.ArgumentParser(description="Detect direct aortic daughter arteries.")
    p.add_argument("--image", type=Path, help="CT volume (NIfTI)")
    p.add_argument("--aorta-mask", type=Path, help="Binary parent-aorta mask (NIfTI)")
    p.add_argument("--output", type=Path, help="Output prediction JSON")
    p.add_argument("--case-id", default=None, help="Override case_id in the JSON")
    p.add_argument("--batch-dir", type=Path, help="Folder of subject*/case_* directories")
    p.add_argument("--output-dir", type=Path, default=Path("predictions"))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.batch_dir:
        roots = []
        batch = args.batch_dir
        kids = sorted([d for d in batch.iterdir() if d.is_dir() and (d.name.startswith("subject") or d.name.startswith("case_"))])
        if kids:
            roots = kids
        elif _find_pair(batch):
            roots = [batch]
        if not roots:
            p.error(f"No subject*/case_* folders with orig/mask pairs under {batch}")
        for folder in roots:
            pair = _find_pair(folder)
            if pair is None:
                print(f"skip {folder}: no image/mask pair")
                continue
            image, mask = pair
            cid = args.case_id or infer_case_id(image)
            out = args.output_dir / f"{cid}.json"
            print(f"=== {cid} ===")
            _run_one(image, mask, out, cid, verbose=not args.quiet)
        return

    if not (args.image and args.aorta_mask and args.output):
        p.error("Provide --image, --aorta-mask and --output (or --batch-dir)")
    _run_one(args.image, args.aorta_mask, args.output, args.case_id, verbose=not args.quiet)


if __name__ == "__main__":
    main()
