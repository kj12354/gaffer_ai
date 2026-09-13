#!/usr/bin/env python3
"""Diagnostic lumbar-pattern review for subject025 (no filtering)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from branchseed.io import load_case, np_zyx_to_mm
from branchseed.visualize import render_case

# LPS +X is patient left. Side is ostium X minus aorta centroid X.
LEFT_MM = 0.0


def aorta_centroid_xyz(vol) -> np.ndarray:
    coords = np.argwhere(vol.aorta)
    pts = np.array([np_zyx_to_mm(vol.image, *c) for c in coords[:: max(1, len(coords) // 4000)]])
    return pts.mean(axis=0)


def main() -> None:
    vol = load_case("subject025/orig25.nii.gz", "subject025/mask25.nii.gz", case_id="subject025")
    pred = json.loads(Path("predictions/subject025.json").read_text())
    out_png = Path("visuals/subject025_ostia.png")
    render_case(vol, pred, out_png, title="subject025 — aorta, ostia, directions")
    print(f"Wrote {out_png}  ({len(pred['daughters'])} daughters)")

    centroid = aorta_centroid_xyz(vol)
    print(f"Aorta centroid LPS mm: {centroid.round(2).tolist()}  (+X = patient left)")

    rows = []
    for d in pred["daughters"]:
        ost = np.asarray(d["ostium_xyz_mm"], dtype=float)
        dx = float(ost[0] - centroid[0])
        side = "L" if dx >= LEFT_MM else "R"
        rows.append(
            {
                "instance_id": d["instance_id"],
                "ostium_x": round(float(ost[0]), 2),
                "ostium_y": round(float(ost[1]), 2),
                "ostium_z": round(float(ost[2]), 2),
                "side": side,
                "dx_from_centroid_mm": round(dx, 2),
                "radius_mm": d.get("radius_mm"),
            }
        )
    rows.sort(key=lambda r: -r["ostium_z"])

    print("\nSuperior → inferior sequence:")
    print(f"{'id':<12} {'side':<4} {'Z':>8} {'dX':>7} {'radius':>8}")
    seq = []
    for r in rows:
        seq.append(r["side"])
        print(
            f"{r['instance_id']:<12} {r['side']:<4} {r['ostium_z']:8.1f} "
            f"{r['dx_from_centroid_mm']:7.2f} {r['radius_mm']}"
        )
    print(f"Side sequence: {''.join(seq)}   L={seq.count('L')}  R={seq.count('R')}")

    print("\nSame-side ΔZ (consecutive on that side, superior → inferior):")
    for side in ("L", "R"):
        zs = [r["ostium_z"] for r in rows if r["side"] == side]
        deltas = [round(zs[i] - zs[i + 1], 1) for i in range(len(zs) - 1)]
        print(f"  {side}: n={len(zs)}  ΔZ={deltas}  (lumbar-like ~25–40 mm)")

    # Opposite-side pairs at nearly the same Z (possible lumbar pair).
    print("\nNear-Z L/R pairs (|ΔZ| < 8 mm):")
    left = [r for r in rows if r["side"] == "L"]
    right = [r for r in rows if r["side"] == "R"]
    paired = 0
    for a in left:
        for b in right:
            dz = abs(a["ostium_z"] - b["ostium_z"])
            if dz < 8.0:
                paired += 1
                print(
                    f"  {a['instance_id']}(L z={a['ostium_z']:.1f})  "
                    f"{b['instance_id']}(R z={b['ostium_z']:.1f})  ΔZ={dz:.1f}"
                )
    if paired == 0:
        print("  none")

    csv_path = Path("predictions/subject025_lumbar_pattern.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
