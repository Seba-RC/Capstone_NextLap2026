# Hito 1 · Capstone NextLap (IIT414W)

Team: [team name] · Members: Álvaro Torres (Member 1), Sebastián Ramorino (Member 2)
Repository: [URL] · Commit: [exact commit hash]

| Part | File |
|---|---|
| Framing and experiment plan | [HITO1.md](HITO1.md) |
| Executable evidence | [Hito1_Notebook.ipynb](Hito1_Notebook.ipynb) |
| Generated metric tables | [outputs/](outputs/) |
| Decisions, AI record, contributions | [DECISION_LOG.md](DECISION_LOG.md) |

## Run

Data location: extract `IIT414W_Capstone_NextLap_StudentData_v1.zip` (course release v1, 30 October 2026). The dataset is not duplicated here; `data/source_manifest_v2.json` inside the release carries the SHA-256 of each race snapshot, and the notebook verifies them.

```
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `Hito1_Notebook.ipynb`, set `DATA_DIR` in the first code cell to the extracted data folder, and run all cells. Recorded run: Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1.

## Outputs

| File | Content |
|---|---|
| `outputs/reference_metrics.csv` | MAE in seconds, population and row count for median, persistence and OLS. |
| `outputs/reference_predictions.csv` | One row per validation `pair_id` with the three predictions. |
| `outputs/extension_metrics.csv`, `outputs/extension_predictions.csv` | Reference OLS vs OLS + `recent_std_s` on the same rows. |
| `outputs/trial_log.csv` | One row per comparison run, including unsuccessful ones. |

The course helper `nextlap_reference_v1.py` is reused with attribution for `make_model`, `load_snapshots` and the feature constants.
