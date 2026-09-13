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


def _fail(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _as_user_error(exc: BaseException) -> str:
    """Turn I/O / schema failures into one line. No traceback."""
    if isinstance(exc, FileNotFoundError):
        name = exc.filename or str(exc)
        return f"missing file: {name}"
    msg = str(exc).strip() or type(exc).__name__
    low = msg.lower()
    if "unable to determine imageio" in low or "file too small" in low:
        return "could not read NIfTI (unrecognized or corrupted file)"
    if "different sizes" in low or "not co-registered" in low:
        return msg.splitlines()[0]
    if "could not read" in low or "unsupported nifti" in low:
        return msg.splitlines()[0]
    if isinstance(exc, (OSError, ValueError, RuntimeError)):
        return msg.splitlines()[0]
    return f"{type(exc).__name__}: {msg.splitlines()[0]}"


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
    image = Path(image)
    mask = Path(mask)
    output = Path(output)
    if not image.exists():
        _fail(f"missing file: {image}")
    if not mask.exists():
        _fail(f"missing file: {mask}")
    try:
        with Timer() as timer:
            payload = detect_daughters(str(image), str(mask), case_id=case_id, verbose=verbose)
        write_prediction(payload, output)
    except SystemExit:
        raise
    except Exception as exc:
        _fail(_as_user_error(exc))
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
        batch = Path(args.batch_dir)
        if not batch.exists():
            _fail(f"missing file: {batch}")
        if not batch.is_dir():
            _fail(f"--batch-dir is not a directory: {batch}")
        kids = sorted(
            [
                d
                for d in batch.iterdir()
                if d.is_dir() and (d.name.startswith("subject") or d.name.startswith("case_"))
            ]
        )
        roots = kids if kids else ([batch] if _find_pair(batch) else [])
        if not roots:
            _fail(f"No subject*/case_* folders with image/mask pairs under {batch}")
        n_ok = 0
        n_fail = 0
        for folder in roots:
            pair = _find_pair(folder)
            if pair is None:
                print(f"skip {folder}: no image/mask pair")
                continue
            image, mask = pair
            cid = args.case_id or infer_case_id(image)
            out = Path(args.output_dir) / f"{cid}.json"
            print(f"=== {cid} ===")
            try:
                _run_one(image, mask, out, cid, verbose=not args.quiet)
                n_ok += 1
            except SystemExit as exc:
                # Keep the rest of the batch running; still a non-zero process
                # status if anything failed.
                n_fail += 1
                if exc.code not in (0, None):
                    print(f"error: {folder.name} failed", file=sys.stderr)
        if n_fail:
            _fail(f"batch finished with {n_fail} failure(s), {n_ok} ok")
        return

    if not (args.image and args.aorta_mask and args.output):
        p.error("Provide --image, --aorta-mask and --output (or --batch-dir)")
    _run_one(args.image, args.aorta_mask, args.output, args.case_id, verbose=not args.quiet)


if __name__ == "__main__":
    main()
