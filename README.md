# Branchseed

Detect every eligible artery that branches directly off a supplied abdominal-aorta mask in a contrast CT, and write one JSON file of ostium / seed / radius / direction landmarks per case.

Coordinates are **SimpleITK LPS millimetres** (`TransformIndexToPhysicalPoint` / `TransformContinuousIndexToPhysicalPoint`). NiBabel affines are not used.

The detector is classical CV only (adaptive HU calibration, 3D Frangi vesselness, outward growth, centreline walk). No GPU, no network, no pretrained weights.

## Setup

Python 3.11 or 3.12, CPU only. Versions in `requirements.txt` are pinned.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Example on a development case:

```bash
.venv/bin/python run.py --image EVAL_SET/case_19/orig19.nii.gz --aorta-mask EVAL_SET/case_19/aorta19.nii.gz --output predictions/subject019.json
```

Batch a folder of `subject*` / `case_*` pairs:

```bash
.venv/bin/python run.py --batch-dir EVAL_SET --output-dir predictions
```

Each run prints wall-clock time and peak RSS so compute efficiency can be reported honestly.

## Validate (cases 19–23)

```bash
.venv/bin/python validate.py --pred-dir predictions --eval-dir EVAL_SET --output predictions/validation_metrics.json
```

Reports precision / recall / F1 on matched daughters, mean ostium localisation error (mm), and flags implausible directions or radii.

On the draft annotations for cases 19–23 (19 reference daughters, 10 mm match radius) extras are scored as false positives, but the draft is not an exhaustive expert reference. Vesselness on draft-matched vs unmatched extras overlaps heavily (true matches include near-zero Frangi scores), so there is no blanket vesselness cutoff. A leave-one-case-out logistic model on crop-adjudicated labels also had no safe threshold (it always dropped a true match). The live quality filter is therefore three conservative geometric rules only: short+flat contact, low-vesselness smear with unstable radius, and low-vesselness leftover lumen next to a fat blob. Daughters that run alongside the aorta are eligible from ostium-span (not only radial wall-distance), which recovered case 23 `branch_001`. Draft score: P=0.586  R=0.895  F1=0.708.

Path length in internal review is the followable outward extent (max of ridge walk and wall-distance), gated at the stated ≥5 mm rule. Typical runtime is 0.5–4 s per 1.5 mm case and ~7 s / ~0.8 GB on a 512³-class 0.78 mm volume.

## Visual check

The challenge asks for a simple visual check on at least three cases (aorta, ostia, direction arrows):

```bash
.venv/bin/python visualize.py --eval-dir EVAL_SET --pred-dir predictions --cases 19,21,22 --output-dir visuals
```

## Method (short)

1. Confirm the CT and aorta mask share size, spacing, origin, and direction.
2. Calibrate a blood HU window from voxels **inside that case's aorta mask**.
3. Crop to the aorta plus a ~10 mm margin and dilate a ~12 mm search shell.
4. Run 3D Frangi vesselness at physical scales that cover ~1–2 voxel tubes.
5. Suppress the circumferential partial-volume sheet on the wall; keep protrusions as ostium seeds.
6. Watershed-grow outward and reject paths shorter than 5 mm of followable lumen (ridge walk, radial extent, or ostium-span for alongside courses).
7. Drop crop-cap artefacts and the terminal iliac split; merge ostia closer than 3.5 mm, then collapse near-duplicates closer than 5 mm.
8. Reject remaining artefacts that match crop-review false patterns (short+flat ostium, near-zero vesselness with unstable radius, or a fat leftover lumen beside a low-vesselness blob).
9. Ostium = parent-surface centroid (strip-end snap for span-only alongside kisses); seed = 5 mm along that heading; direction = unit vector from ostium to seed; radius = local lumen estimate at the seed.

If nothing eligible is found, `daughters` is an empty list.

## Known limitations

- Case 20: two superior ostia ~6 mm apart grow as one instance (contact spread 23.7 mm). Draft FN `branch_003`.
- Case 23: posterior partner of the kept match is not seeded (ostium voxel inside the aorta mask).
- Long wall-kisses / iliac sheets can produce one elongated contact; a directional contact splitter was tested and **not shipped** (smooth curvature is not two ostia).
- Bone/vertebra extras on some cases sit inside the blood HU window; they are not all dropped.
- Manual crop review of subjects 001–018 (128 detections): **39 confirmed real / 5 confirmed false / 84 uncertain**. The five false calls are flush vertebra contact. Uncertain 1 px lumbars and low-contrast 016–018 were not used to train a drop-filter.
- Rejected experiments (logistic quality gate, ungated dump-blob, live contact splitter) live under `experiments/` and are not on the run path.

## Layout

```
run.py / validate.py / visualize.py
branchseed/          live detector
predictions/         development-set JSON + draft metrics
visuals/             required 3-case aorta / ostia / direction checks
review/              crop adjudication pages (not required to run)
experiments/         rejected approaches and diagnostic scripts
```
