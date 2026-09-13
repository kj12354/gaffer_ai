#!/usr/bin/env python3
"""Read-only diagnosis of case 20 fused FN and case 23 branch_002.

Does not change detection logic. Re-runs the live pipeline and inspects
seed patches / instances / rejects / merges at the draft GT ostia.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branchseed.branches import _drop_cap_seeds, _label_and_merge, extract_daughters
from branchseed.intensity import calibrate_blood_range
from branchseed.io import crop_pair, load_case, mm_to_index_xyz, np_zyx_to_mm
from branchseed.output import build_prediction
from branchseed.params import PipelineParams
from branchseed.vesselness import build_search_masks


def _gt_ostia(num: int) -> list[dict]:
    data = json.loads(Path(f"EVAL_SET/case_{num}/annotations.json").read_text())
    out = []
    for d in data["daughters"]:
        out.append(
            {
                "id": d["instance_id"],
                "ost": np.asarray(d["ostium_xyz_mm"], dtype=np.float64),
                "seed": np.asarray(d["seed_xyz_mm"], dtype=np.float64),
                "dir": np.asarray(d["direction_xyz"], dtype=np.float64),
                "notes": d.get("notes") or d.get("guide", {}).get("notes", ""),
                "conf": d.get("confidence") or d.get("guide", {}).get("confidence", ""),
                "diam": d.get("origin_diameter_estimate_mm")
                or d.get("guide", {}).get("origin_diameter_estimate_mm"),
            }
        )
    return out


def _mm_to_crop_zyx(vol, origin_zyx, xyz_mm: np.ndarray) -> np.ndarray:
    idx = mm_to_index_xyz(vol.image, xyz_mm)  # x,y,z
    full = np.array([idx[2], idx[1], idx[0]], dtype=np.float64)
    return full - origin_zyx.astype(np.float64)


def _sample(arr, zyx, default=float("nan")):
    z, y, x = [int(round(c)) for c in zyx]
    if not (0 <= z < arr.shape[0] and 0 <= y < arr.shape[1] and 0 <= x < arr.shape[2]):
        return default
    return arr[z, y, x]


def _ball_stats(arr, zyx, spacing, radius_mm=3.0):
    z, y, x = [int(round(c)) for c in zyx]
    rz, ry, rx = [max(1, int(np.ceil(radius_mm / s))) for s in spacing]
    z0, z1 = max(0, z - rz), min(arr.shape[0], z + rz + 1)
    y0, y1 = max(0, y - ry), min(arr.shape[1], y + ry + 1)
    x0, x1 = max(0, x - rx), min(arr.shape[2], x + rx + 1)
    patch = arr[z0:z1, y0:y1, x0:x1]
    if patch.size == 0:
        return {}
    if patch.dtype == bool or patch.dtype == np.uint8:
        return {"sum": int(patch.sum()), "frac": float(patch.mean())}
    vals = np.asarray(patch, dtype=np.float64)
    return {
        "min": float(np.min(vals)),
        "p50": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def _patch_table(mask, spacing, origin, vol, title: str) -> list[dict]:
    labeled, n = label(mask, structure=np.ones((3, 3, 3)))
    rows = []
    for i in range(1, n + 1):
        com = np.asarray(center_of_mass(mask, labeled, i), dtype=np.float64)
        mm = np_zyx_to_mm(vol.image, *(com + origin))
        rows.append(
            {
                "id": i,
                "n": int((labeled == i).sum()),
                "com_crop": com,
                "com_mm": mm,
            }
        )
    print(f"  {title}: {n} 26-connected patch(es)")
    for r in rows:
        print(
            f"    patch {r['id']:3d}  nvox={r['n']:4d}  "
            f"com_mm=[{r['com_mm'][0]:7.2f}, {r['com_mm'][1]:8.2f}, {r['com_mm'][2]:7.2f}]"
        )
    return rows


def diagnose(num: int, focus: list[str]) -> None:
    params = PipelineParams()
    image = Path(f"EVAL_SET/case_{num}/orig{num}.nii.gz")
    mask = Path(f"EVAL_SET/case_{num}/aorta{num}.nii.gz")
    vol = load_case(image, mask, case_id=f"case_{num}")
    rng = calibrate_blood_range(vol.ct, vol.aorta, params)
    ct_c, aorta_c, _sl, origin = crop_pair(vol, params.crop_margin_mm)
    search = build_search_masks(ct_c, aorta_c, vol.spacing_zyx, rng, params)
    seeds_raw = search["seeds"]
    seeds = _drop_cap_seeds(seeds_raw, aorta_c, params.cap_slice_margin)
    raw_labels, n_raw = label(seeds, structure=np.ones((3, 3, 3)))
    merged_markers = _label_and_merge(seeds, vol.spacing_zyx, params.ostium_merge_mm)

    from branchseed.branches import grow_instances

    instances = grow_instances(
        seeds, search["allowed"], search["dist"], vol.spacing_zyx, params.ostium_merge_mm
    )

    detected = extract_daughters(vol, ct_c, aorta_c, search, rng, origin, params)
    payload = build_prediction(vol, detected.daughters, origin)

    gt = _gt_ostia(num)
    print("=" * 72)
    print(f"CASE {num}  spacing={tuple(round(float(s), 3) for s in vol.spacing_xyz)}")
    print(
        f"  blood=[{rng.blood_low:.0f},{rng.blood_high:.0f}]  "
        f"v_thresh={float(search['v_thresh']):.4f}  "
        f"seeds={int(seeds.sum())} (raw {int(seeds_raw.sum())})  "
        f"raw_patches={n_raw}  merged_markers={int(merged_markers.max())}  "
        f"instances={int(instances.max())}  "
        f"kept={len(detected.daughters)}"
    )
    print(f"  merge floors: near_dup={params.near_dup_ostium_mm}  "
          f"ostium_merge={params.ostium_merge_mm}  "
          f"path_merge={params.path_merge_ostium_mm}")
    print("  rejected:")
    for r in detected.rejected:
        print(f"    {r}")
    if not detected.rejected:
        print("    (none)")
    print("  merge_log:")
    for line in detected.merge_log:
        print(f"    {line}")
    if not detected.merge_log:
        print("    (none)")

    print("\n  GT ostium pair distances:")
    for i, a in enumerate(gt):
        for b in gt[i + 1 :]:
            d = float(np.linalg.norm(a["ost"] - b["ost"]))
            print(f"    {a['id']}–{b['id']}: {d:.2f} mm")

    print("\n  Kept predictions:")
    pred_mm = []
    for p, d in zip(payload["daughters"], detected.daughters):
        po = np.asarray(p["ostium_xyz_mm"])
        pred_mm.append(po)
        print(
            f"    {p['instance_id']}  ost={np.round(po, 2).tolist()}  "
            f"r={d.radius_mm:.2f}  path={d.path_length_mm:.1f}  "
            f"v={d.med_vesselness:.3f}  HU={d.mean_hu:.0f}  "
            f"cv={d.radius_cv:.2f}  circ={d.mean_circularity:.2f}"
        )

    print("\n  Raw seed patches (after cap drop, before centroid merge):")
    raw_rows = _patch_table(seeds, vol.spacing_zyx, origin, vol, "raw seed patches")
    print("  After _label_and_merge (ostium_merge_mm):")
    _patch_table(merged_markers > 0, vol.spacing_zyx, origin, vol, "merged seed markers")

    for g in gt:
        if focus and g["id"] not in focus:
            # still print a short nearest-pred line for context
            pass
        crop = _mm_to_crop_zyx(vol, origin, g["ost"])
        print("\n" + "-" * 72)
        print(f"  GT {g['id']}  conf={g['conf']}  origin_diam={g['diam']}")
        print(f"    ostium_mm={np.round(g['ost'], 3).tolist()}")
        print(f"    seed_mm  ={np.round(g['seed'], 3).tolist()}")
        print(f"    dir      ={np.round(g['dir'] / (np.linalg.norm(g['dir']) + 1e-9), 3).tolist()}")
        print(f"    notes: {g['notes'][:200]}")
        print(f"    crop_zyx={np.round(crop, 2).tolist()}  in_crop={all(0 <= crop[i] < ct_c.shape[i] for i in range(3))}")

        hu = float(_sample(ct_c, crop))
        vess = float(_sample(search["vesselness"], crop))
        dist = float(_sample(search["dist"], crop))
        seed_here = bool(_sample(seeds, crop, False))
        seed_raw = bool(_sample(seeds_raw, crop, False))
        allowed = bool(_sample(search["allowed"], crop, False))
        blood = bool(_sample(search["blood"], crop, False))
        aorta = bool(_sample(aorta_c, crop, False))
        inst_id = int(_sample(instances, crop, 0))
        raw_id = int(_sample(raw_labels, crop, 0))
        mer_id = int(_sample(merged_markers, crop, 0))
        print(
            f"    at ostium voxel: HU={hu:.1f}  vess={vess:.4f}  dist_wall={dist:.2f}  "
            f"seed={seed_here} (raw={seed_raw}) allowed={allowed} blood={blood} aorta={aorta}"
        )
        print(f"    maps: raw_patch={raw_id}  merged_marker={mer_id}  grown_instance={inst_id}")
        print(f"    ball 3mm seeds={_ball_stats(seeds, crop, vol.spacing_zyx)}")
        print(f"    ball 3mm vesselness={_ball_stats(search['vesselness'], crop, vol.spacing_zyx)}")
        print(f"    ball 3mm HU={_ball_stats(ct_c, crop, vol.spacing_zyx)}")
        print(f"    ball 3mm allowed={_ball_stats(search['allowed'], crop, vol.spacing_zyx)}")

        # nearest raw seed patches
        if raw_rows:
            dists = [float(np.linalg.norm(r["com_mm"] - g["ost"])) for r in raw_rows]
            order = np.argsort(dists)
            print("    nearest raw seed patches:")
            for k in order[:5]:
                r = raw_rows[k]
                print(
                    f"      patch {r['id']}  Δ={dists[k]:.2f} mm  nvox={r['n']}  "
                    f"com={np.round(r['com_mm'], 2).tolist()}"
                )

        if pred_mm:
            dists = [float(np.linalg.norm(p - g["ost"])) for p in pred_mm]
            j = int(np.argmin(dists))
            p = payload["daughters"][j]
            d = detected.daughters[j]
            gd = g["dir"] / (np.linalg.norm(g["dir"]) + 1e-9)
            pd = np.asarray(p["direction_xyz"])
            pd = pd / (np.linalg.norm(pd) + 1e-9)
            print(
                f"    nearest kept {p['instance_id']}  Δost={dists[j]:.2f} mm  "
                f"dir_cos={float(np.dot(gd, pd)):.3f}  "
                f"seedΔ={float(np.linalg.norm(np.asarray(p['seed_xyz_mm']) - g['seed'])):.2f}"
            )

        # Diagnose mechanism
        if inst_id > 0:
            print(f"    MECHANISM hint: a grown instance (id{inst_id}) covers this ostium voxel.")
        elif mer_id > 0:
            print(f"    MECHANISM hint: merged seed marker {mer_id} is here but instance did not grow onto it.")
        elif any(float(np.linalg.norm(r["com_mm"] - g["ost"])) < 6.0 for r in raw_rows):
            print("    MECHANISM hint: a raw seed patch is within 6 mm; check merge into a neighbour.")
        else:
            print("    MECHANISM hint: no seed / instance at this ostium — detection never triggered here.")


def main() -> None:
    print("Live params: near_dup=5.0  ostium_merge=3.5  path_merge=8.0  quality_filter=ON")
    print("Case 20 focus: GT branch_003 (unmatched) vs branch_004 (kept, fused heading)")
    diagnose(20, focus=["branch_003", "branch_004"])
    print("\n")
    print("Case 23 focus: GT branch_002 (unmatched) vs branch_003 (kept posterior)")
    diagnose(23, focus=["branch_002", "branch_003"])


if __name__ == "__main__":
    main()
