#!/usr/bin/env python3
"""Contact-spread census on unsplit grown instances, subject001–025.

Measures the instance the splitter would see (parent contact, before any
split). Does not change detection logic.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.branches import (
    _contact_ridge_clusters,
    _contact_spread_mm,
    _contact_voxels,
    _drop_cap_seeds,
    _ostium_zyx,
    grow_instances,
)
from branchseed.intensity import calibrate_blood_range
from branchseed.io import crop_pair, load_case, np_zyx_to_mm
from branchseed.params import PipelineParams
from branchseed.pipeline import run_on_volume_detailed
from branchseed.visualize import render_ostium_review_card
from full_dataset_run import find_pair
from generate_review import _eval_pair, _html_page

OUT = Path("predictions/contact_spread_full_dataset.csv")
REVIEW = Path("review")
CROPS = REVIEW / "crops"
NEAR_MM = 15.0  # visual-review band around the 18 mm floor
FLAG_HTML = REVIEW / "spread_flags.html"


def _paths(n: int) -> tuple[str, Path, Path]:
    if 19 <= n <= 23:
        img, msk = _eval_pair(n)
        return f"case_{n}", img, msk
    pair = find_pair(Path(f"subject{n:03d}"))
    if pair is None:
        raise FileNotFoundError(f"subject{n:03d}")
    return f"subject{n:03d}", pair[0], pair[1]


def _census_case(n: int, params: PipelineParams) -> list[dict]:
    case_id, img, msk = _paths(n)
    vol = load_case(img, msk, case_id=case_id)
    rng = calibrate_blood_range(vol.ct, vol.aorta, params)
    ct_c, aorta_c, _sl, origin = crop_pair(vol, params.crop_margin_mm)
    from branchseed.vesselness import build_search_masks
    from scipy.ndimage import distance_transform_edt

    search = build_search_masks(ct_c, aorta_c, vol.spacing_zyx, rng, params)
    seeds = _drop_cap_seeds(search["seeds"], aorta_c, params.cap_slice_margin)
    instances = grow_instances(
        seeds, search["allowed"], search["dist"], vol.spacing_zyx, params.ostium_merge_mm
    )
    lumen_radius = distance_transform_edt(
        search["blood"] | aorta_c, sampling=tuple(vol.spacing_zyx)
    ).astype(np.float32)
    payload, _det = run_on_volume_detailed(vol, params=params, verbose=False)
    kept = payload["daughters"]

    rows: list[dict] = []
    for inst_id in range(1, int(instances.max()) + 1):
        inst = instances == inst_id
        contact = _contact_voxels(inst, aorta_c)
        if not np.any(contact):
            continue
        spread = _contact_spread_mm(contact, vol.spacing_zyx)
        ost = _ostium_zyx(contact, aorta_c)
        ost_mm = np_zyx_to_mm(vol.image, *(ost + origin))
        clustered = _contact_ridge_clusters(
            inst, contact, search["dist"], lumen_radius, vol.spacing_zyx, params
        )
        cluster_cos = "" if clustered is None else round(float(clustered[0]), 3)
        over_18 = spread >= params.split_min_contact_spread_mm
        would_full = clustered is not None and over_18
        branch = ""
        if kept:
            dists = [
                float(np.linalg.norm(np.asarray(p["ostium_xyz_mm"]) - ost_mm))
                for p in kept
            ]
            j = int(np.argmin(dists))
            if dists[j] <= 12.0:
                branch = kept[j]["instance_id"]
        instance_id = branch or f"id{inst_id}"
        rows.append(
            {
                "case_id": case_id,
                "instance_id": instance_id,
                "internal_id": f"id{inst_id}",
                "contact_spread_mm": round(float(spread), 3),
                "over_18mm_threshold": "yes" if over_18 else "no",
                "would_full_split": "yes" if would_full else "no",
                "cluster_cos": cluster_cos,
                "n_contact": int(contact.sum()),
                "n_instance": int(inst.sum()),
                "ostium_x": round(float(ost_mm[0]), 2),
                "ostium_y": round(float(ost_mm[1]), 2),
                "ostium_z": round(float(ost_mm[2]), 2),
                "nearest_kept": branch,
                "kept_ostium_err_mm": "" if not branch else round(dists[j], 2),
            }
        )
    print(
        f"{case_id}: {len(rows)} contacted instances  "
        f"over18={sum(1 for r in rows if r['over_18mm_threshold']=='yes')}  "
        f"full_split={sum(1 for r in rows if r['would_full_split']=='yes')}",
        flush=True,
    )
    return rows


def _write_flag_review(flag_rows: list[dict]) -> None:
    REVIEW.mkdir(exist_ok=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    cards: list[dict] = []
    by_case: dict[str, list[dict]] = {}
    for r in flag_rows:
        by_case.setdefault(r["case_id"], []).append(r)
    for case_id, rows in by_case.items():
        n = int("".join(ch for ch in case_id if ch.isdigit()))
        _cid, img, msk = _paths(n)
        vol = load_case(img, msk, case_id=case_id)
        for r in rows:
            crop_name = f"spread_{case_id}_{r['internal_id']}.png"
            crop_abs = CROPS / crop_name
            ost = [r["ostium_x"], r["ostium_y"], r["ostium_z"]]
            render_ostium_review_card(
                vol,
                ost,
                r["instance_id"],
                crop_abs,
                subtitle=(
                    f"spread={r['contact_spread_mm']:.1f}  "
                    f"over18={r['over_18mm_threshold']}  "
                    f"full_split={r['would_full_split']}  "
                    f"cos={r['cluster_cos'] or '—'}"
                ),
            )
            cards.append(
                {
                    "case": case_id,
                    "instance_id": f"{r['instance_id']} ({r['internal_id']})",
                    "status": "over_18" if r["over_18mm_threshold"] == "yes" else "near_18",
                    "gt_id": r["would_full_split"],
                    "ostium_err_mm": r["contact_spread_mm"],
                    "ostium_z": r["ostium_z"],
                    "radius_mm": r["n_contact"],
                    "vesselness": r["cluster_cos"] or "",
                    "radius_cv": "",
                    "mean_circularity": "",
                    "path_length_mm": r["n_instance"],
                    "mean_hu": "",
                    "crop_rel": f"crops/{crop_name}",
                }
            )
    FLAG_HTML.write_text(_html_page("spread ≥15 mm flags", cards), encoding="utf-8")
    print(f"Wrote {FLAG_HTML} ({len(cards)} crops)")


def main() -> None:
    params = PipelineParams()
    all_rows: list[dict] = []
    for n in range(1, 26):
        all_rows.extend(_census_case(n, params))

    fields = [
        "case_id",
        "instance_id",
        "contact_spread_mm",
        "over_18mm_threshold",
        "would_full_split",
        "cluster_cos",
        "internal_id",
        "n_contact",
        "n_instance",
        "ostium_x",
        "ostium_y",
        "ostium_z",
        "nearest_kept",
        "kept_ostium_err_mm",
    ]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    spreads = np.array([r["contact_spread_mm"] for r in all_rows], dtype=float)
    over = [r for r in all_rows if r["over_18mm_threshold"] == "yes"]
    near = [r for r in all_rows if r["contact_spread_mm"] >= NEAR_MM]
    full = [r for r in all_rows if r["would_full_split"] == "yes"]
    print(f"\nWrote {OUT}  n={len(all_rows)}")
    print(
        f"spread mm  min={spreads.min():.1f}  p50={np.median(spreads):.1f}  "
        f"p90={np.percentile(spreads, 90):.1f}  p95={np.percentile(spreads, 95):.1f}  "
        f"max={spreads.max():.1f}"
    )
    print(f"near ≥{NEAR_MM:.0f} mm: {len(near)}   over 18 mm: {len(over)}   full split gate: {len(full)}")
    print("FLAGGED (≥15 mm):")
    for r in sorted(near, key=lambda x: -x["contact_spread_mm"]):
        print(
            f"  {r['case_id']:12s} {r['instance_id']:12s}  "
            f"spread={r['contact_spread_mm']:5.1f}  over18={r['over_18mm_threshold']}  "
            f"full_split={r['would_full_split']}  cos={r['cluster_cos']}"
        )
    _write_flag_review(near)


if __name__ == "__main__":
    main()
