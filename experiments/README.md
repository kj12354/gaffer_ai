# Experiments (not on the live path)

Rejected or diagnostic work referenced from `predictions/REVIEW_NOTES.md`.
`python run.py` does not import anything here. `enable_contact_split` in
`branchseed/params.py` stays `False`.

| Script | What it tested |
|---|---|
| `train_filter.py` | Leave-one-case-out logistic quality gate (no zero-FN threshold) |
| `validate_split.py` / `contact_spread_report.py` / `diagnose_spread_curvature.py` | Gated contact-patch splitter and curvature census |
| `rerun_gates.py` / `analyze_vesselness.py` / `analyze_cross_section.py` | Extra drop-filters and vesselness / lumen checks |
| `diagnose_fn.py` | Case 20 / 23 fused-ostium diagnosis |
| `generate_review.py` / `expand_review.py` / `adjudicate.py` | Crop review pages |
| `full_dataset_run.py` | Isolated subprocess batch (same `run.py` detector) |

`artifacts/` holds CSVs and extra visuals from those runs. The shipped
development predictions are only `../predictions/subject*.json`.
