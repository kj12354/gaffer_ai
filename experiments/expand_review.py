#!/usr/bin/env python3
"""Additive review pages for subject001-018. Does not rewrite 19-23 or 025."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.output import write_prediction
from branchseed.pipeline import run_on_volume_detailed
from branchseed.visualize import render_ostium_review_card
from generate_review import _html_page, _subject_pair

REVIEW_DIR = Path("review")
CROPS = REVIEW_DIR / "crops"
PRED_DIR = Path("predictions")
SUMMARY = PRED_DIR / "all_cases_detection_summary.csv"
EXISTING_VERDICTS = PRED_DIR / "review_verdicts.csv"
PROTECTED = {"case_19", "case_20", "case_21", "case_22", "case_23", "subject025"}


def _write_index(extra_ids: list[str]) -> None:
    existing = ["case_19", "case_20", "case_21", "case_22", "case_23", "subject025"]
    ids = extra_ids + existing
    links = "".join(f'<li><a href="{r}.html">{r}</a></li>' for r in ids)
    (REVIEW_DIR / "index.html").write_text(
        "<!doctype html><html><body style='font-family:sans-serif'>"
        "<h1>Branchseed adjudication</h1>"
        "<p>Open a case, then r/f/u + n/p. Cases 19–23 and subject025 are unchanged.</p>"
        f"<ul>{links}</ul></body></html>",
        encoding="utf-8",
    )


def _process_case(case_id: str, image: Path, mask: Path, write_pred: Path | None) -> list[dict]:
    vol = load_case(image, mask, case_id=case_id)
    payload, detected = run_on_volume_detailed(vol, verbose=True)
    if write_pred is not None:
        write_prediction(payload, write_pred)
    P = payload["daughters"]
    rows = []
    for p, d in zip(P, detected.daughters):
        ost = np.array(p["ostium_xyz_mm"])
        crop_name = f"{case_id}_{p['instance_id']}.png"
        crop_abs = CROPS / crop_name
        subtitle = (
            f"extra  v={d.med_vesselness:.3f}  cv={d.radius_cv:.2f}  "
            f"circ={d.mean_circularity:.2f}  path={d.path_length_mm:.1f}  "
            f"HU={d.mean_hu:.0f}"
        )
        render_ostium_review_card(
            vol, p["ostium_xyz_mm"], p["instance_id"], crop_abs, subtitle=subtitle
        )
        rows.append(
            {
                "case": case_id,
                "instance_id": p["instance_id"],
                "status": "extra",
                "gt_id": "",
                "ostium_err_mm": "",
                "ostium_x": round(float(ost[0]), 2),
                "ostium_y": round(float(ost[1]), 2),
                "ostium_z": round(float(ost[2]), 2),
                "radius_mm": round(float(p["radius_mm"]), 4),
                "path_length_mm": round(float(d.path_length_mm), 2),
                "vesselness": round(float(d.med_vesselness), 4),
                "mean_hu": round(float(d.mean_hu), 1),
                "radius_cv": "" if not np.isfinite(d.radius_cv) else round(float(d.radius_cv), 4),
                "mean_circularity": (
                    "" if not np.isfinite(d.mean_circularity) else round(float(d.mean_circularity), 4)
                ),
                "crop_path": str(crop_abs),
                "crop_rel": f"crops/{crop_name}",
                "verdict": "",
            }
        )
    html = REVIEW_DIR / f"{case_id}.html"
    html.write_text(_html_page(case_id, rows), encoding="utf-8")
    print(f"Wrote {html}  ({len(rows)} detections)  pred={write_pred}")
    return rows


def _rows_from_existing_verdicts() -> list[dict]:
    if not EXISTING_VERDICTS.exists():
        return []
    rows = []
    for r in csv.DictReader(EXISTING_VERDICTS.open()):
        row = dict(r)
        row["verdict"] = ""  # worksheet is blank; do not copy old marks
        rows.append(row)
    return rows


def _process_024_csv_only() -> list[dict]:
    image, mask = _subject_pair(24)
    vol = load_case(image, mask, case_id="subject024")
    payload, detected = run_on_volume_detailed(vol, verbose=False)
    rows = []
    for p, d in zip(payload["daughters"], detected.daughters):
        ost = np.array(p["ostium_xyz_mm"])
        rows.append(
            {
                "case": "subject024",
                "instance_id": p["instance_id"],
                "status": "extra",
                "gt_id": "",
                "ostium_err_mm": "",
                "ostium_x": round(float(ost[0]), 2),
                "ostium_y": round(float(ost[1]), 2),
                "ostium_z": round(float(ost[2]), 2),
                "radius_mm": round(float(p["radius_mm"]), 4),
                "path_length_mm": round(float(d.path_length_mm), 2),
                "vesselness": round(float(d.med_vesselness), 4),
                "mean_hu": round(float(d.mean_hu), 1),
                "radius_cv": "" if not np.isfinite(d.radius_cv) else round(float(d.radius_cv), 4),
                "mean_circularity": (
                    "" if not np.isfinite(d.mean_circularity) else round(float(d.mean_circularity), 4)
                ),
                "crop_path": "",
                "verdict": "",
            }
        )
    return rows


def main() -> None:
    REVIEW_DIR.mkdir(exist_ok=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    extra_ids: list[str] = []
    new_rows: list[dict] = []
    for num in range(1, 19):
        case_id = f"case_{num:03d}"
        extra_ids.append(case_id)
        image, mask = _subject_pair(num)
        pred_path = PRED_DIR / f"subject{num:03d}.json"
        new_rows.extend(_process_case(case_id, image, mask, pred_path))

    _write_index(extra_ids)

    summary = new_rows + _rows_from_existing_verdicts() + _process_024_csv_only()
    # stable column set
    fields = [
        "case",
        "instance_id",
        "status",
        "gt_id",
        "ostium_err_mm",
        "ostium_x",
        "ostium_y",
        "ostium_z",
        "radius_mm",
        "path_length_mm",
        "vesselness",
        "mean_hu",
        "radius_cv",
        "mean_circularity",
        "crop_path",
        "verdict",
    ]
    with SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {SUMMARY}  ({len(summary)} rows, verdict blank)")
    print("Did not write review_verdicts.csv; 19-23/025 HTML and crops untouched.")
    # sanity: protected html still present
    for name in PROTECTED:
        p = REVIEW_DIR / f"{name}.html"
        print(f"  protected {p} exists={p.exists()}")


if __name__ == "__main__":
    main()
