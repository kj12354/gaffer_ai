#!/usr/bin/env python3
"""Build one-page-per-case review HTML + crops. Does not write verdicts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from branchseed.io import load_case
from branchseed.pipeline import run_on_volume_detailed
from branchseed.visualize import render_ostium_review_card

MATCH_MM = 10.0
REVIEW_DIR = Path("review")
CROPS = REVIEW_DIR / "crops"
VERDICTS = Path("predictions/review_verdicts.csv")


def _eval_pair(num: int) -> tuple[Path, Path]:
    return Path(f"EVAL_SET/case_{num}/orig{num}.nii.gz"), Path(f"EVAL_SET/case_{num}/aorta{num}.nii.gz")


def _subject_pair(num: int) -> tuple[Path, Path]:
    folder = Path(f"subject{num:03d}")
    images = sorted(folder.glob("orig*"))
    masks = [m for m in sorted(folder.glob("mask*") ) if "daughter" not in m.name.lower()]
    return images[0], masks[0]


def _match_gt(pred: list[dict], gt: list[dict]) -> dict[int, tuple[str, float]]:
    if not pred or not gt:
        return {}
    cost = np.array(
        [
            [np.linalg.norm(np.array(g["ostium_xyz_mm"]) - np.array(p["ostium_xyz_mm"])) for p in pred]
            for g in gt
        ]
    )
    ri, cj = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(ri, cj):
        if cost[i, j] <= MATCH_MM:
            out[j] = (gt[i]["instance_id"], float(cost[i, j]))
    return out


def _html_page(case_id: str, rows: list[dict]) -> str:
    cards = []
    for i, r in enumerate(rows):
        rel = Path(r["crop_rel"])
        cards.append(
            f"""
<article class="card" data-idx="{i}" data-case="{r['case']}" data-id="{r['instance_id']}">
  <div class="meta">
    <h2>{r['instance_id']}  <span class="status {r['status']}">{r['status']}</span></h2>
    <p>gt={r['gt_id'] or '—'}  err={r['ostium_err_mm'] or '—'}
       z={r['ostium_z']}  r={r['radius_mm']}</p>
    <ul>
      <li>vesselness <b>{r['vesselness']}</b></li>
      <li>radius_cv <b>{r['radius_cv']}</b></li>
      <li>circularity <b>{r['mean_circularity']}</b></li>
      <li>path_mm <b>{r['path_length_mm']}</b></li>
      <li>mean_HU <b>{r['mean_hu']}</b></li>
    </ul>
    <div class="verdict" data-key="{r['case']}|{r['instance_id']}">
      <button data-v="confirmed_real">r real</button>
      <button data-v="confirmed_false">f false</button>
      <button data-v="uncertain">u uncertain</button>
      <span class="chosen"></span>
    </div>
  </div>
  <img src="{rel.as_posix()}" alt="{r['instance_id']}"/>
</article>"""
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Review {case_id}</title>
<style>
body {{ font-family: sans-serif; background:#111; color:#eee; margin:0; }}
header {{ position:sticky; top:0; background:#1b1b1b; padding:10px 16px; border-bottom:1px solid #333; z-index:2; }}
.card {{ display:flex; gap:16px; padding:16px; border-bottom:1px solid #333; align-items:flex-start; }}
.card.active {{ outline:2px solid #6cf; }}
.meta {{ min-width:240px; }}
img {{ max-width:640px; background:#000; }}
.status.draft_match {{ color:#7d7; }}
.status.extra {{ color:#f8a; }}
button {{ margin-right:6px; padding:4px 8px; cursor:pointer; }}
button.on {{ background:#2a6; color:#fff; }}
kbd {{ background:#333; padding:1px 5px; border-radius:3px; }}
</style></head><body>
<header>
  <strong>{case_id}</strong> — {len(rows)} detections.
  Keys: <kbd>r</kbd> real <kbd>f</kbd> false <kbd>u</kbd> uncertain
  <kbd>n</kbd>/<kbd>j</kbd> next <kbd>p</kbd>/<kbd>k</kbd> prev.
  <button id="export">download verdicts.csv</button>
  <a href="index.html" style="color:#9cf;margin-left:12px">all cases</a>
</header>
{''.join(cards)}
<script>
const KEY='branchseed_verdicts_v1';
const store=JSON.parse(localStorage.getItem(KEY)||'{{}}');
const cards=[...document.querySelectorAll('.card')];
let cur=0;
function paint(){{
  cards.forEach((c,i)=>c.classList.toggle('active', i===cur));
  document.querySelectorAll('.verdict').forEach(v=>{{
    const k=v.dataset.key; const val=store[k]||'';
    v.querySelector('.chosen').textContent=val;
    v.querySelectorAll('button').forEach(b=>b.classList.toggle('on', b.dataset.v===val));
  }});
}}
function setV(v){{
  const c=cards[cur]; const k=c.dataset.case+'|'+c.dataset.id;
  store[k]=v; localStorage.setItem(KEY, JSON.stringify(store)); paint();
}}
document.querySelectorAll('.verdict button').forEach(b=>b.onclick=()=>{{
  cur=cards.indexOf(b.closest('.card')); setV(b.dataset.v);
}});
document.addEventListener('keydown', e=>{{
  if(e.key==='r') setV('confirmed_real');
  if(e.key==='f') setV('confirmed_false');
  if(e.key==='u') setV('uncertain');
  if(e.key==='n'||e.key==='j'){{ cur=Math.min(cards.length-1, cur+1); paint(); cards[cur].scrollIntoView({{block:'center'}}); }}
  if(e.key==='p'||e.key==='k'){{ cur=Math.max(0, cur-1); paint(); cards[cur].scrollIntoView({{block:'center'}}); }}
}});
document.getElementById('export').onclick=()=>{{
  let csv='case,instance_id,verdict\\n';
  Object.entries(store).forEach(([k,v])=>{{ const [c,id]=k.split('|'); csv+=c+','+id+','+v+'\\n'; }});
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv]));
  a.download='review_verdicts.csv'; a.click();
}};
paint();
</script></body></html>
"""


def _process_case(case_id: str, image: Path, mask: Path, gt_path: Path | None) -> list[dict]:
    vol = load_case(image, mask, case_id=case_id)
    payload, detected = run_on_volume_detailed(vol, verbose=False)
    P = payload["daughters"]
    gt = []
    if gt_path and gt_path.exists():
        gt = json.loads(gt_path.read_text()).get("daughters", [])
    matched = _match_gt(P, gt)
    rows = []
    for j, (p, d) in enumerate(zip(P, detected.daughters)):
        status = "draft_match" if j in matched else "extra"
        gt_id, err = matched.get(j, ("", None))
        ost = np.array(p["ostium_xyz_mm"])
        crop_name = f"{case_id}_{p['instance_id']}.png"
        crop_abs = CROPS / crop_name
        render_ostium_review_card(
            vol, p["ostium_xyz_mm"], p["instance_id"], crop_abs,
            subtitle=f"{status}  v={d.med_vesselness:.3f}  path={d.path_length_mm:.1f}",
        )
        rows.append(
            {
                "case": case_id,
                "instance_id": p["instance_id"],
                "status": status,
                "gt_id": gt_id,
                "ostium_err_mm": "" if err is None else round(err, 2),
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
    print(f"Wrote {html}  ({len(rows)} detections)")
    return rows


def main() -> None:
    REVIEW_DIR.mkdir(exist_ok=True)
    CROPS.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for num in (19, 20, 21, 22, 23):
        img, msk = _eval_pair(num)
        all_rows.extend(
            _process_case(f"case_{num}", img, msk, Path(f"EVAL_SET/case_{num}/annotations.json"))
        )
    all_rows.extend(_process_case("subject025", *_subject_pair(25), None))

    links = "".join(
        f'<li><a href="{r}.html">{r}</a></li>'
        for r in ("case_19", "case_20", "case_21", "case_22", "case_23", "subject025")
    )
    (REVIEW_DIR / "index.html").write_text(
        f"<!doctype html><html><body style='font-family:sans-serif'>"
        f"<h1>Branchseed adjudication</h1><p>Open a case, then r/f/u + n/p.</p>"
        f"<ul>{links}</ul></body></html>",
        encoding="utf-8",
    )
    fields = [k for k in all_rows[0].keys() if k != "crop_rel"]
    with VERDICTS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {VERDICTS}  (verdict column empty — do not auto-adjudicate)")
    print(f"Open review/index.html")


if __name__ == "__main__":
    main()
