#!/usr/bin/env python3
"""Ridge curvature / pinch diagnostics for flagged long-contact instances.

Does not change detection. Splitter stays off.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.branches import (
    _contact_spread_mm,
    _contact_voxels,
    _drop_cap_seeds,
    grow_instances,
)
from branchseed.intensity import calibrate_blood_range
from branchseed.io import crop_pair, load_case, np_zyx_to_mm
from branchseed.params import PipelineParams
from branchseed.vesselness import build_search_masks
from branchseed.visualize import render_ostium_review_card
from full_dataset_run import find_pair
from generate_review import _eval_pair

OUT = Path("visuals/split_diag")
CSV = Path("predictions/spread_curvature_diag.csv")

TARGETS = [
    # near-orthogonal (plausible fuse)
    ("subject025", 25, 44),
    ("subject018", 18, 26),
    # known fuse
    ("case_20", 20, None),  # pick largest-spread contacted instance
    # extreme opposing cosine / long sheet
    ("subject018", 18, 1),
    ("subject008", 8, 18),
    ("subject001", 1, 1),
    ("subject007", 7, 18),
    ("subject003", 3, None),
    ("subject024", 24, None),
    ("subject016", 16, None),  # largest over-18 on 016 if id not set
]


def _paths(case_id: str, n: int):
    if case_id.startswith("case_"):
        return _eval_pair(n)
    return find_pair(Path(f"subject{n:03d}"))


def _order_contact(contact: np.ndarray, spacing: np.ndarray):
    coords = np.argwhere(contact).astype(np.float64)
    if len(coords) < 4:
        return coords, None, None
    phys = coords * spacing.reshape(1, 3)
    centered = phys - phys.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    order = np.argsort(proj)
    return coords[order], phys[order], proj[order] - proj.min()


def _smooth(y: np.ndarray, win: int = 5) -> np.ndarray:
    if len(y) < win:
        return y
    k = np.ones(win) / win
    return np.convolve(y, k, mode="same")


def _profile(inst, contact, lumen_r, spacing):
    coords, phys, arc = _order_contact(contact, spacing)
    if phys is None or len(phys) < 6:
        return None
    # downsample to ~1 mm steps along the strip
    arc = np.asarray(arc, dtype=np.float64)
    keep = [0]
    for i in range(1, len(arc)):
        if arc[i] - arc[keep[-1]] >= 1.0:
            keep.append(i)
    if keep[-1] != len(arc) - 1:
        keep.append(len(arc) - 1)
    idx = np.array(keep)
    phys = phys[idx]
    coords = coords[idx]
    arc = arc[idx]
    tangents = np.zeros_like(phys)
    for i in range(len(phys)):
        if i == 0:
            t = phys[1] - phys[0]
        elif i == len(phys) - 1:
            t = phys[-1] - phys[-2]
        else:
            t = phys[i + 1] - phys[i - 1]
        n = float(np.linalg.norm(t))
        tangents[i] = t / n if n > 1e-8 else 0.0
    turn = np.zeros(len(phys))
    for i in range(1, len(phys)):
        turn[i] = float(np.clip(np.dot(tangents[i], tangents[i - 1]), -1, 1))
    turn_deg = np.degrees(np.arccos(np.clip(turn, -1, 1)))
    turn_deg[0] = 0.0
    rad = np.array(
        [float(lumen_r[int(c[0]), int(c[1]), int(c[2])]) for c in np.round(coords)]
    )
    # endpoint cosine of first vs last third mean tangents
    n = len(tangents)
    a = tangents[: max(1, n // 3)].mean(0)
    b = tangents[-max(1, n // 3) :].mean(0)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    end_cos = float(np.dot(a, b))
    # curvature concentration: max local turn / mean turn
    td = turn_deg[1:]
    conc = float(td.max() / (td.mean() + 1e-6)) if len(td) else 0.0
    # radius waist: min in interior vs mean of both end thirds
    if len(rad) >= 7:
        i0, i1 = max(1, n // 5), min(n - 1, 4 * n // 5)
        waist = float(rad[i0:i1].min()) if i1 > i0 else float(rad.min())
        ends = float(np.mean(np.r_[rad[: n // 3], rad[-n // 3 :]]))
        waist_ratio = waist / (ends + 1e-6)
        waist_s = float(arc[i0 + int(np.argmin(rad[i0:i1]))]) if i1 > i0 else float(arc[int(np.argmin(rad))])
    else:
        waist, ends, waist_ratio, waist_s = float("nan"), float("nan"), float("nan"), float("nan")
    # distributed vs spiked turning: Gini-like share of top 15% samples
    if len(td):
        s = np.sort(td)[::-1]
        k = max(1, int(round(0.15 * len(s))))
        spike_share = float(s[:k].sum() / (s.sum() + 1e-6))
    else:
        spike_share = 0.0
    return {
        "arc": arc,
        "rad": rad,
        "turn_deg": turn_deg,
        "coords": coords,
        "phys": phys,
        "end_cos": end_cos,
        "conc": conc,
        "waist": waist,
        "ends": ends,
        "waist_ratio": waist_ratio,
        "waist_s": waist_s,
        "spike_share": spike_share,
        "mean_turn": float(td.mean()) if len(td) else 0.0,
        "max_turn": float(td.max()) if len(td) else 0.0,
    }


def _pick_instance(instances, aorta, spacing, want_id, prefer_largest=False):
    best = None
    for inst_id in range(1, int(instances.max()) + 1):
        if want_id is not None and inst_id != want_id:
            continue
        inst = instances == inst_id
        contact = _contact_voxels(inst, aorta)
        if not np.any(contact):
            continue
        spread = _contact_spread_mm(contact, spacing)
        if want_id is not None:
            return inst_id, inst, contact, spread
        if best is None or spread > best[0]:
            best = (spread, inst_id, inst, contact)
    if best is None:
        return None
    return best[1], best[2], best[3], best[0]


def analyze(case_id: str, n: int, want_id: int | None, params: PipelineParams) -> dict | None:
    img, msk = _paths(case_id, n)
    vol = load_case(img, msk, case_id=case_id)
    rng = calibrate_blood_range(vol.ct, vol.aorta, params)
    ct_c, aorta_c, _sl, origin = crop_pair(vol, params.crop_margin_mm)
    search = build_search_masks(ct_c, aorta_c, vol.spacing_zyx, rng, params)
    seeds = _drop_cap_seeds(search["seeds"], aorta_c, params.cap_slice_margin)
    instances = grow_instances(
        seeds, search["allowed"], search["dist"], vol.spacing_zyx, params.ostium_merge_mm
    )
    lumen_r = distance_transform_edt(
        search["blood"] | aorta_c, sampling=tuple(vol.spacing_zyx)
    ).astype(np.float32)
    picked = _pick_instance(
        instances, aorta_c, vol.spacing_zyx, want_id, prefer_largest=want_id is None
    )
    if picked is None:
        print(f"{case_id}: instance not found")
        return None
    inst_id, inst, contact, spread = picked
    prof = _profile(inst, contact, lumen_r, vol.spacing_zyx)
    tag = f"{case_id}_id{inst_id}"
    print(
        f"{tag}: spread={spread:.1f} nC={int(contact.sum())} nI={int(inst.sum())} "
        f"end_cos={None if not prof else round(prof['end_cos'], 3)} "
        f"waist_ratio={None if not prof else round(prof['waist_ratio'], 2)} "
        f"max_turn={None if not prof else round(prof['max_turn'], 1)} "
        f"mean_turn={None if not prof else round(prof['mean_turn'], 1)} "
        f"conc={None if not prof else round(prof['conc'], 2)} "
        f"spike_share={None if not prof else round(prof['spike_share'], 2)}",
        flush=True,
    )
    dest = OUT / tag
    dest.mkdir(parents=True, exist_ok=True)

    def _mm(zyx):
        return np_zyx_to_mm(vol.image, *(np.asarray(zyx, dtype=np.float64) + origin))

    points = []
    if prof is not None:
        points.append(("end0", prof["coords"][0], 0.0))
        points.append(("end1", prof["coords"][-1], float(prof["arc"][-1])))
        mid = int(np.argmin(np.abs(prof["arc"] - 0.5 * prof["arc"][-1])))
        points.append(("mid", prof["coords"][mid], float(prof["arc"][mid])))
        if np.isfinite(prof["waist_s"]):
            w = int(np.argmin(np.abs(prof["arc"] - prof["waist_s"])))
            points.append(("waist", prof["coords"][w], float(prof["arc"][w])))
        fig, ax = plt.subplots(2, 1, figsize=(8.2, 5.2), sharex=True)
        ax[0].plot(prof["arc"], prof["rad"], color="#1f77b4")
        ax[0].set_ylabel("lumen radius (mm)")
        ax[0].set_title(
            f"{tag}  spread={spread:.1f}  end_cos={prof['end_cos']:.2f}  "
            f"waist/ends={prof['waist_ratio']:.2f}"
        )
        if np.isfinite(prof["waist_s"]):
            ax[0].axvline(prof["waist_s"], color="#d62728", ls="--", lw=0.8)
        ax[1].plot(prof["arc"], prof["turn_deg"], color="#ff7f0e")
        ax[1].set_ylabel("local turn (deg)")
        ax[1].set_xlabel("arc along contact (mm)")
        fig.tight_layout()
        fig.savefig(dest / "profile.png", dpi=140)
        plt.close(fig)

    for name, zyx, s in points:
        ost = _mm(zyx)
        render_ostium_review_card(
            vol,
            ost.tolist(),
            f"{tag} {name}",
            dest / f"{name}.png",
            subtitle=f"s={s:.1f}mm  spread={spread:.1f}",
            crop_mm=30.0,
        )
    row = {
        "case_id": case_id,
        "internal_id": f"id{inst_id}",
        "contact_spread_mm": round(float(spread), 2),
        "end_cos": "" if not prof else round(prof["end_cos"], 3),
        "waist_mm": "" if not prof else round(prof["waist"], 2),
        "end_radius_mm": "" if not prof else round(prof["ends"], 2),
        "waist_ratio": "" if not prof else round(prof["waist_ratio"], 3),
        "mean_turn_deg": "" if not prof else round(prof["mean_turn"], 2),
        "max_turn_deg": "" if not prof else round(prof["max_turn"], 2),
        "turn_concentration": "" if not prof else round(prof["conc"], 2),
        "top15_turn_share": "" if not prof else round(prof["spike_share"], 3),
        "verdict": "",
    }
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = PipelineParams()
    # disable split even if someone flipped the flag
    params.enable_contact_split = False
    rows = []
    seen = set()
    for case_id, n, iid in TARGETS:
        key = (case_id, iid)
        if key in seen:
            continue
        seen.add(key)
        row = analyze(case_id, n, iid, params)
        if row:
            rows.append(row)
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {CSV} and {OUT}/")


if __name__ == "__main__":
    main()
