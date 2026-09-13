#!/usr/bin/env python3
"""Diagnostic full-dataset batch: same run.py pipeline, no logic changes.

Runs every subject* folder via `run.py --image ...` in an isolated subprocess
so a crash cannot abort the rest of the set, and peak RSS is per case.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk

from branchseed.io import read_image_robust

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = Path(sys.executable)
RUN_PY = ROOT / "run.py"
OUT_DIR = ROOT / "predictions"
SUMMARY = OUT_DIR / "full_dataset_run_summary.csv"
HIGH_COUNT = 15


def find_pair(case_dir: Path) -> tuple[Path, Path] | None:
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
    masks = [m for m in masks if "daughter" not in m.name.lower() and "and_" not in m.name.lower()]
    if not images or not masks:
        return None
    return images[0], masks[0]


def _finite(x: object) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def sanity_flags(pred_path: Path, image_path: Path) -> list[str]:
    flags: list[str] = []
    try:
        data = json.loads(pred_path.read_text())
    except Exception as exc:
        return [f"bad_json:{exc}"]
    daughters = data.get("daughters") or []
    if len(daughters) == 0:
        flags.append("zero_detections")
    if len(daughters) > HIGH_COUNT:
        flags.append(f"high_count:{len(daughters)}")
    image, _src = read_image_robust(image_path, "sanity")
    size = image.GetSize()
    for d in daughters:
        rid = d.get("instance_id", "?")
        radius = d.get("radius_mm")
        if not _finite(radius):
            flags.append(f"nan_inf:{rid}.radius")
        elif float(radius) < 0:
            flags.append(f"negative_radius:{rid}")
        for key in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            vec = d.get(key) or []
            if len(vec) != 3 or not all(_finite(c) for c in vec):
                flags.append(f"nan_inf:{rid}.{key}")
                continue
            if key == "direction_xyz":
                continue
            idx = image.TransformPhysicalPointToContinuousIndex(tuple(float(c) for c in vec))
            for axis, (c, n) in enumerate(zip(idx, size)):
                if c < -1.0 or c > float(n) + 1.0:
                    flags.append(f"oob:{rid}.{key}[{axis}]={c:.2f}")
                    break
    return flags


def parse_run_output(text: str) -> tuple[int | None, float | None, float | None]:
    n_det: int | None = None
    runtime: float | None = None
    peak_mb: float | None = None
    for line in text.splitlines():
        if "daughter(s))" in line and "Wrote" in line:
            # "Wrote predictions/subject001.json  (3 daughter(s))"
            try:
                n_det = int(line.rsplit("(", 1)[1].split()[0])
            except (IndexError, ValueError):
                pass
        if line.startswith("[compute]"):
            parts = {p.split("=")[0]: p.split("=")[1] for p in line.split() if "=" in p}
            runtime = float(parts.get("wall_clock_s", "nan"))
            peak_mb = float(parts.get("peak_rss_mb", "nan"))
    return n_det, runtime, peak_mb


def main() -> None:
    subjects = sorted(
        d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("subject")
    )
    if not subjects:
        sys.exit("No subject* folders in the dataset root")
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    inspect: list[str] = []
    print(f"Full-dataset diagnostic: {len(subjects)} subject folders")
    for folder in subjects:
        pair = find_pair(folder)
        cid = folder.name
        if pair is None:
            row = {
                "case_id": cid,
                "n_detections": "",
                "runtime_s": "",
                "peak_mem_gb": "",
                "flag": "no_image_mask_pair",
            }
            rows.append(row)
            inspect.append(f"{cid}: no_image_mask_pair")
            print(f"=== {cid} === SKIP no image/mask pair")
            continue
        image, mask = pair
        out = OUT_DIR / f"{cid}.json"
        print(f"=== {cid} ===")
        proc = subprocess.run(
            [
                str(PYTHON),
                str(RUN_PY),
                "--image",
                str(image),
                "--aorta-mask",
                str(mask),
                "--output",
                str(out),
                "--case-id",
                cid,
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        print(proc.stdout, end="" if (proc.stdout or "").endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        n_det, runtime, peak_mb = parse_run_output(proc.stdout or "")
        flags: list[str] = []
        if proc.returncode != 0:
            flags.append(f"crash:exit_{proc.returncode}")
            err_tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            if err_tail:
                flags.append("err:" + err_tail[-1][:120])
        warn_hits = [
            ln.strip()
            for ln in combined.splitlines()
            if any(k in ln for k in ("Traceback", "Error", "Warning", "Exception"))
            and "UserWarning" not in ln
        ]
        if warn_hits and proc.returncode == 0:
            flags.append("warning")
        if out.exists() and proc.returncode == 0:
            flags.extend(sanity_flags(out, image))
            if n_det is None:
                try:
                    n_det = len(json.loads(out.read_text()).get("daughters") or [])
                except Exception:
                    flags.append("bad_json")
        elif proc.returncode == 0:
            flags.append("missing_output")
        flag = ";".join(flags)
        peak_gb = "" if peak_mb is None else f"{peak_mb / 1024.0:.4f}"
        row = {
            "case_id": cid,
            "n_detections": "" if n_det is None else n_det,
            "runtime_s": "" if runtime is None else f"{runtime:.2f}",
            "peak_mem_gb": peak_gb,
            "flag": flag,
        }
        rows.append(row)
        print(
            f"[summary] {cid}  n={row['n_detections']}  "
            f"t={row['runtime_s']}s  mem={peak_gb}GB  flag={flag or 'ok'}"
        )
        if flag:
            inspect.append(f"{cid}: {flag}")

    with SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["case_id", "n_detections", "runtime_s", "peak_mem_gb", "flag"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {SUMMARY}")
    print("=== Cases needing manual inspection ===")
    if inspect:
        for line in inspect:
            print(f"  {line}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
