# Hito 2 · Capstone NextLap (IIT414W)

Team: [confirm team name]  
Members: Álvaro Torres and Sebastián Ramorino  
Repository: https://github.com/Seba-RC/Capstone_NextLap2026  
Commit: `846f23f3fcdd9f22df88b18fe3b6bf74af0d0a3e` (Hito 2 submission)

## Evidence map

| Requirement | Evidence |
|---|---|
| Hito 1 carry-forward evidence | `hito1_evidence/HITO1.md`; `hito1_evidence/DECISION_LOG.md`; `hito1_evidence/Hito1_Notebook.ipynb`; `hito1_evidence/outputs/` |
| Executed comparison and ablation | `HITO2.md` §1; `outputs/comparison_metrics.csv`; `outputs/validation_predictions.csv` |
| Errors by race and context | `HITO2.md` §2; `outputs/race_error_comparison.csv`; `outputs/context_error_slices.csv` |
| Inspected validation case | `outputs/inspected_validation_case.csv`; `outputs/inspected_case_timeline.csv` |
| Local and global SHAP | `HITO2.md` §3; `outputs/shap_local_case.csv`; `outputs/shap_local_case_metadata.csv`; `outputs/shap_global_summary.csv`; `outputs/shap_global_summary.png` |
| Reproducibility and audit | `run_hito2.py`; `outputs/run_audit.json`; `requirements.txt`; `base_NextLap/` |
| Decisions, AI record and individual explanations | `DECISION_LOG.md`; `HITO2.md` §4 |

## Data

The `base_NextLap/` directory contains the released clean training/validation files, original timing snapshots, `source_manifest_v2.json` and the course reference helper. The helper verifies snapshot SHA-256 hashes before the run. Training rounds are 1–7; validation rounds are 8–9. Reserved final-evaluation data are not included or used.

## Environment and run

Recorded environment: Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, SHAP 0.50.0.

Create an environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_hito2.py
```

The script writes the comparison, race/context diagnostics, row-level predictions, inspected case, SHAP tables/plot and `run_audit.json` into `outputs/`. It also verifies that the refitted reference and candidate predictions match the saved Hito 1 predictions.

Before submission, replace the team/repository/commit placeholders and ensure both individual explanations and contribution statements are confirmed by the actual members.
