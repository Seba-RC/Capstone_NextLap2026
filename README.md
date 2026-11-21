# Capstone NextLap · Final report (IIT414W)

Team 2: Álvaro Torres and Sebastián Ramorino
Repository: https://github.com/Seba-RC/Capstone_NextLap2026
Submitted commit: the SHA on the first page of the report PDF. The Hito 2 submission was `846f23f3fcdd9f22df88b18fe3b6bf74af0d0a3e`.

**Question.** Does recent pace variability (`recent_std_s`, the sample standard deviation of up to three valid recent laps in the same driver, race and tyre stint) improve the prediction of `next_lap_s` at the end of lap t?

**Final decision.** Retain OLS + `recent_std_s`. Validation MAE is 0.365223 s against 0.386707 s for the six-input OLS reference / matched control on the same 1,808 IDs. The difference is −0.021484 s (95% driver-race CI [−0.031503, −0.012860]), and MAE is lower in both validation races. The short-history flag suggested in feedback was tested and rejected (`DECISION_LOG.md`, "Response to feedback").

## Evidence map

| Report component | Evidence |
|---|---|
| C1 Split, fitting, input availability, common IDs | `run_hito2.py` (assertions), `outputs/run_audit.json`, `hito1_evidence/Hito1_Notebook.ipynb` (leakage checks L1–L4) |
| C2 OLS reference, candidate, matched control | `outputs/comparison_metrics.csv`, `outputs/validation_predictions.csv`, `hito1_evidence/outputs/trial_log.csv` |
| C3 Race comparison, slices, inspected case | `outputs/race_error_comparison.csv`, `outputs/context_error_slices.csv`, `outputs/inspected_validation_case.csv`, `outputs/inspected_case_timeline.csv` |
| C4 Local and global SHAP | `outputs/shap_local_case.csv`, `outputs/shap_local_case_metadata.csv`, `outputs/shap_global_summary.csv`, `outputs/shap_global_summary.png` |
| C5 Decision record, AI use, contributions | `DECISION_LOG.md` (all trials, including unsuccessful ones) |
| C6 Feedback response (short-history flag) | `experiments/run_short_history_flag.py`, `outputs/exp1_short_history_flag/` |
| Exploratory trials (median loss, traffic gap, trees) | `experiments/README.md`, `outputs/exp2_median_traffic_gap/`, `outputs/exp3_tree_models/` |
| Hito 1 and Hito 2 write-ups | `hito1_evidence/HITO1.md`, `HITO2.md` |

## Data

`base_NextLap/` holds the released clean training and validation files (`train_clean_v1.csv`, `validation_clean_v1.csv`), the nine original race timing snapshots in `data/`, `source_manifest_v2.json` and the course helper `nextlap_reference_v1.py`. Every run checks the snapshot SHA-256 hashes against the manifest before using them.

- Training: rounds 1–7, 5,772 clean pairs.
- Validation (development): rounds 8–9, 1,808 clean pairs (R8 Austria 1,066; R9 Britain 742).
- Reserved final-evaluation data are not included and were not used.

## Environment

Recorded environment: Python 3.13 (3.13.5 for Hito 2; re-run on 3.13.18), with the exact versions in `requirements.txt` (pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, SHAP 0.50.0, matplotlib 3.10.8). Random seed: 414.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python run_hito2.py                              # main comparison, slices, inspected case, SHAP  -> outputs/
python experiments/run_short_history_flag.py     # feedback response: flag + training-only check -> outputs/exp1_short_history_flag/
python experiments/run_median_traffic_gap.py     # exploratory: median loss, traffic gap          -> outputs/exp2_median_traffic_gap/
python experiments/run_tree_models.py            # exploratory: gradient boosting, random forest  -> outputs/exp3_tree_models/
```

Each script takes a few seconds. `run_hito2.py` asserts that the refitted reference and candidate match the saved Hito 1 predictions. Each experiment asserts that its refitted candidate matches the saved Hito 2 predictions before comparing anything. Re-running everything on 21 November reproduced every Hito 2 CSV byte-for-byte.
