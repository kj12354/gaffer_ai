# Internal false-positive read (cases 21–22)

Draft GT is not exhaustive. These notes override the earlier “possible unlabeled lumbar / oversplit” story. Nothing here is an auto-filter.

## Case 21

- **branch_006 — likely genuine false positive.** Median vesselness 0.0094 with blood-range HU. Posterior-wall pair crops do not show a ~2 mm contrast lumen leaving the wall. Do not treat as an unlabeled lumbar candidate.
- **branch_007, branch_008, branch_009 — still ambiguous.** Vesselness in the same band as many draft matches (0.29–0.43). Axial / thin-MIP crops do not confirm a clear outward lumen, but vesselness alone cannot reject them. Flag for manual review; do not auto-remove.

## Case 22

- **branch_007 and branch_008 are not one oversplit vessel.** 3D ostium distance 13.0 mm, in-plane wall separation 11.7 mm, direction cosine −0.36, seed separation 22 mm. The path-divergence merge (8 mm window) correctly keeps both. Treat them as independent: `branch_007` is a draft match to GT `branch_002`; `branch_008` is a separate extra / likely FP.
- **branch_001 and branch_011** remain isolated unpaired extras. Flag for individual review; do not drop just because they are unpaired.

## Vesselness

Across cases 19–23, draft-matched and unmatched extras overlap almost completely (true min 0.00, extra min 0.009; overlap band contains 10/17 trues and 13/13 extras). No clean separation point. No additional vesselness floor was applied.

## Crop adjudication (2026-09-13)

Reviewer: AI visual pass on the 24 mm 4-panel crops only (axial + thin MIP ±3 mm + short cor/sag). Not a 3D scroll. 1.5 mm voxels make a 2 mm lumen ~1 pixel, so many extras stay **uncertain** on purpose.

Verdicts are in `predictions/review_verdicts.csv`. Counts: **18 confirmed_real / 16 confirmed_false / 17 uncertain**.

All 16 draft matches were kept **confirmed_real** (crops match GT, including the more obvious case-22 viscerals).

### Priority extras

- **case_21 / 007, 008, 009 — uncertain.** Same as before: no clear outward lumen on the crop, but not obviously bone/smear either. Do not auto-drop.
- **subject025 left cluster (009–013):** 009 uncertain; 010 / 012 / 013 **false** (contact between two lumens or marker inside a blob, vesselness ~0).
- **subject025 right-only tail (016–022):** **016 real** (clear superior leaving lumen). 017–019 uncertain. **020–022 false** (inferior wall sitting on vertebra).

### Other extras that looked clearly false

Bone / vertebra contact: 21/002, 21/006, 21/010, 22/001, 22/002, 025/006, 025/015.
Adjacent-lumen contact (not a leaving branch): 025/002, 025/007, 025/008, 025/012.

### Extras that looked real

- **025/004** — connecting contrast tube between two lumens.
- **025/016** — superior branch leaving a round aorta, same look as the draft-matched viscerals.

Do not train a drop-filter on the uncertain set. The 16 confirmed_false extras are the only safe negative labels from this pass.

## Quality filter (2026-09-13)

Leave-one-case-out logistic regression on the 34 confirmed labels had **no threshold with zero FN**. It is not in the live pipeline.

Shipped rules (0 confirmed_real dropped):

- short + flat ostium (`path < 6 mm` and circularity `< 0.38`) → case 21 `branch_002`
- near-zero vesselness + unstable radius (`v < 0.02` and radius_cv `≥ 0.40`) → case 21 `branch_006`, subject025 `012`
- low vesselness + leftover lumen beside a fat blob → subject025 `007`, `013`
- high bone fraction in the outward ostium ball → subject025 `006`

Draft score after the filter: **P 0.55 → 0.59, R 0.84 (unchanged), F1 0.67 → 0.70**. subject025 22 → 18. Bone/vertebra extras on cases 21–22 still remain; cancellous bone sits inside the blood HU window on these CTs, so a HU bone gate cannot take them.

## Alongside-span eligibility (2026-09-13)

Case 23 `branch_001` (anterior, runs inferiorly beside the aorta) was rejected as `short_path` because radial wall-distance was 4.7 mm. Followable length now also uses Euclidean span from the ostium. Span-only survivors must be elongated and have some vesselness, so wall blobs do not all pass.

Draft score after that plus the quality filter: **P=0.586  R=0.895  F1=0.708** (17/19 matches). Remaining FNs: case 20 `branch_003` (superior pair / crop-cap) and case 23 `branch_002` (low-confidence posterior partner of the kept match). Case 20 `branch_004` still has a near-orthogonal direction. All 25 subject JSONs were regenerated with this pipeline.

## Span-only strip landmarks (2026-09-13)

Alongside daughters kiss the aorta as a long contact strip. The contact centroid then sits mid-vessel, so `seed − ostium` pointed back toward the true origin (case 23 match cosine **−0.77**).

Output-only fix, after keep/reject: if the instance is span-only (ridge/extent < 5 mm) and the contact is an elongated strip (aniso ≥ 1.80, span ≥ 5 mm), snap the ostium to the origin end of the kiss and place the seed ~5 mm along the strip. Detection counts do not change.

Do **not** apply this to radially leaving instances. An ungated version moved the case 20 fused 003/004 ostium onto 003 (lost the 004 match) and pulled case 22 viscerals off-axis.

After the gated snap: case 23 `branch_001` cosine **−0.77 → 0.983**. Draft score unchanged (**P=0.586 R=0.895 F1=0.708**, 17/19, pred 29). Case 20/21/22 flags unchanged.

Tried and reverted (do not re-enable as-is):

- Vesselness-weighted / multi-start ridge as the live walk: fixes case 20 `004` cosine but explodes case 21 extras.
- Landmark transplant from those walks after keep/reject: flattened the case 20 alongside seed (cosine 0.995 → 0.30) and did not fix the fused `003`/`004` heading.
- Ostium split of two-patch contacts: did not recover case 20 `003` (one connected patch); added case 22 extras.
- Crop-cap diameter gate: the only crop-cap reject on the eval set is case 20 `id1` at **20.9 mm** (aorta continuing through the crop), not the 2.5 mm FN.

Remaining FNs are still not safe to force: case 20 `003` is fused with `004` (the other superior candidate is the 21 mm crop stump); case 23 `002` is the low-confidence posterior partner.

## Seed clamp at 5 mm (2026-09-13)

Challenge definition: seed is 5 mm outward from the ostium. After keep/reject, every kept seed is placed on that heading at exactly 5 mm. Detection set and directions do not change.

Case 20 alongside match seed error **26.0 → 7.7 mm** (residual is the 7.4 mm ostium offset). Mean seed error on that case **12.2 → 5.6 mm**. Draft P/R/F1 unchanged.

An in-loop snap onto instance voxels changed quality features and added a case-21 extra (F1 0.694). Reverted to output-only. A far-end anisotropy bone rule never fired on 19–23; not shipped.

## 45% discovery attempts (2026-09-13)

Tried and reverted:

- Opposite ≥5 mm walks from one ostium: never fired on the fused case-20 pair (one connected contact; walks did not meet opposite-cosine + seed-sep).
- Bright-blob past the seed (`dump_blob`): killed draft matches (case 19 `003`, 21 `001`, 22 viscerals, 23 `003`). F1 0.708 → 0.585. Real tubes sit next to other bright tissue at 8 mm on 1.5 mm data.

Do not re-enable. Hidden-test 45% is still best served by the recall-first detector, not another drop-filter.
