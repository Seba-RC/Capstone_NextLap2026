# Capstone NextLap · Hito 2

IIT414W · Executed experiments, error analysis and interpretation

## 1. Executed comparison and ablation

### Experimental setup

We carried forward the Hito 1 hypothesis: recent pace variability contains additional predictive information about `next_lap_s`. The candidate feature is `recent_std_s`, the sample standard deviation (`ddof=1`) of up to three valid observed lap times ending at lap *t*, within the same season, race, driver and tyre stint. The history is built from the original timing snapshots before retrospective clean-pair selection. A history with fewer than two valid observations is missing and is imputed using a median learned from training rows only.

The target, training/validation split, OLS model family, preprocessing pipeline, metric and validation IDs were held fixed. Training used rounds 1–7 (5,772 rows); validation used rounds 8–9 (1,808 rows: R8 = 1,066; R9 = 742). The same validation `pair_id` values were used for all predictions. No final-evaluation data were used.

### Comparison

| Version | Model and inputs | Validation n | MAE (s) | Output |
|---|---|---:|---:|---|
| OLS reference | OLS, six published inputs | 1,808 | 0.386707 | `outputs/comparison_metrics.csv` |
| Candidate | OLS + six published inputs + `recent_std_s` | 1,808 | 0.365223 | `outputs/comparison_metrics.csv` |
| Matched control | Candidate setup with `recent_std_s` removed; same run as OLS reference | 1,808 | 0.386707 | `outputs/comparison_metrics.csv` |

The matched control is the OLS reference itself: removing `recent_std_s` leaves exactly the six reference inputs and the same OLS pipeline. The Hito 2 script refits both versions and verifies that their predictions match the saved Hito 1 predictions (`outputs/run_audit.json`). The control-minus-candidate MAE difference is **+0.021484 seconds**. The candidate MAE is approximately **5.56% lower** than the control MAE.

This supports the hypothesis for this single feature definition, OLS setup and fixed clean-pair validation population. It does not establish a causal effect, performance on every race lap, or performance on unseen future races. The result is conditional on the retrospective clean-pair population.

### Predeclared check and observed result

The Hito 1 decision rule required: (1) lower candidate MAE on all 1,808 validation IDs; (2) lower candidate error in both validation rounds; and (3) a 95% driver-race cluster-bootstrap interval for the paired difference entirely below zero. Hito 1 recorded the bootstrap interval as **[−0.031503, −0.012860] seconds** for candidate-minus-control error, over 43 driver-race groups. Both race-level differences were also negative under that sign convention. The Hito 2 refit reproduces the Hito 1 reference and candidate predictions to numerical tolerance.

The matched comparison therefore supports retaining `recent_std_s`, with an important short-history qualification discussed below.

## 2. Errors by context

### By validation race

The signed difference below is control MAE minus candidate MAE; a positive value favours the candidate.

| Race | n | Control MAE (s) | Candidate MAE (s) | Control − candidate (s) |
|---|---:|---:|---:|---:|
| R8 · Austrian Grand Prix | 1,066 | 0.373571 | 0.357604 | +0.015967 |
| R9 · British Grand Prix | 742 | 0.405579 | 0.376169 | +0.029410 |
| Pooled, row-weighted | 1,808 | 0.386707 | 0.365223 | +0.021484 |

The candidate improves in both races, so the pooled result is not driven by only one of the two validation rounds. The pooled MAE is calculated over all validation rows (equivalently, by weighting each race MAE by its row count).

### Context slices

| Context | Group | n | Control MAE (s) | Candidate MAE (s) | Control − candidate (s) |
|---|---|---:|---:|---:|---:|
| `recent_std_s` status | Observed | 1,660 | 0.385889 | 0.360673 | +0.025216 |
| `recent_std_s` status | Imputed (short history) | 148 | 0.395886 | 0.416261 | −0.020375 |
| Current compound | HARD | 814 | 0.424496 | 0.397919 | +0.026576 |
| Current compound | MEDIUM | 861 | 0.327214 | 0.317517 | +0.009697 |
| Current compound | SOFT | 133 | 0.540563 | 0.473944 | +0.066619 |

The candidate improves on rows where the feature is observed, but is worse on the 148 rows where the feature is imputed. This is a material limitation: the pooled improvement does not apply uniformly to every row. All three compound slices show lower candidate MAE, but the SOFT slice is relatively small (133 rows), so that difference should be interpreted cautiously. These are descriptive slices, not causal comparisons.

### Inspected validation case

We inspected the validation row with the largest absolute error under the candidate; ties are resolved by `pair_id` order.

| Field | Value |
|---|---|
| `pair_id` | `2026-R9-23-L6` |
| Race / driver | R9 British Grand Prix / driver 23 |
| Current lap | 6 |
| Current compound | HARD |
| `recent_count` / `recent_std_s` | 3 / 0.979325 s |
| Observed next-lap time | 105.128 s |
| Candidate prediction | 96.825754 s |
| Candidate absolute error | 8.302246 s |
| Control prediction / absolute error | 97.302929 s / 7.825071 s |

The candidate underpredicts this next lap by more than eight seconds and is approximately 0.477175 seconds worse in absolute error than the control on this individual row. The original timing snapshot shows the three laps available to the feature were 96.460 s (lap 4), 96.299 s (lap 5) and 98.070 s (lap 6); lap 7 then took 105.128 s, followed by 97.882 s on lap 8. The inspected rows are marked accurate, green-flag (`track_status = 1`), and without pit-in, pit-out or deleted flags. Thus the next-lap time is a sudden one-lap slowdown not signalled by the three-lap variability feature. The evidence does not identify its physical cause. This case demonstrates that a lower average MAE does not prevent large individual errors. A sensible next check is to inspect other large residuals and any additional context available by the end of lap *t*, while avoiding features that become available only later. The exact timing rows are saved in `outputs/inspected_case_timeline.csv`.

## 3. SHAP interpretation

### Model and method

We explain the fitted **OLS + `recent_std_s`** candidate because it is the model whose added information was tested by the ablation. The pipeline's fitted preprocessing transforms the training and validation inputs. `shap.LinearExplainer` is then applied to the fitted linear estimator in transformed feature space. Its background is a seeded sample of 200 training rows after the fitted preprocessing; the global summary uses all 1,808 validation rows. Numeric features retain their original names. One-hot encoded columns are summed back into their original `current_compound` or `current_team` feature group.

SHAP contributions and predictions are in seconds. The local reconstruction is checked as base value plus the sum of signed contributions. The observed prediction error is calculated separately and is not a SHAP contribution.

### Local explanation

For `pair_id = 2026-R9-23-L6`:

- Model prediction: **96.825754 s**.
- SHAP base value: **84.976871 s**.
- Sum of signed SHAP contributions: **+11.848883 s**.
- Base plus contributions: **96.825754 s** (absolute reconstruction difference below 0.000001 s).
- Observed next-lap time: **105.128000 s**; absolute prediction error: **8.302246 s**.

| Original input | Signed SHAP contribution (s) |
|---|---:|
| `recent_mean_s` | +7.536491 |
| `current_lap_s` | +4.694683 |
| `recent_std_s` | −0.340734 |
| `current_compound` (aggregated one-hot group) | +0.048113 |
| `current_tyre_life` | −0.037664 |
| `current_lap_number` | −0.035224 |
| `current_team` (aggregated one-hot group) | −0.016782 |

For this row, `recent_mean_s` and `current_lap_s` move the prediction above the background base value, while `recent_std_s` moves it down by approximately 0.341 seconds. These values explain the model prediction relative to the SHAP background; they do not explain the 8.302-second observed error and do not imply causal effects.

### Global summary

The summary reports the mean absolute value of each original feature's grouped SHAP contribution across all 1,808 validation rows. For categorical features, the encoded one-hot contributions are first summed within each original feature for each row, then the absolute value is averaged.

| Original feature | Mean absolute SHAP (s) | Encoded columns |
|---|---:|---:|
| `recent_mean_s` | 7.257214 | 1 |
| `current_lap_s` | 4.101725 | 1 |
| `recent_std_s` | 0.129883 | 1 |
| `current_team` | 0.059195 | 11 |
| `current_compound` | 0.032244 | 3 |
| `current_tyre_life` | 0.022917 | 1 |
| `current_lap_number` | 0.022813 | 1 |

The two pace-level variables, `recent_mean_s` and `current_lap_s`, dominate the model's prediction variation. `recent_std_s` has a smaller average absolute contribution, but it is the third-largest feature group in this summary and its inclusion improved the matched validation MAE. SHAP magnitude is not a measure of incremental predictive performance; the ablation is the evidence for that performance claim. This global summary describes only this fitted model and these two validation races.

Saved SHAP evidence: `outputs/shap_global_summary.csv`, `outputs/shap_global_summary.png`, `outputs/shap_local_case.csv` and `outputs/shap_local_case_metadata.csv`.

## 4. Reproducibility and decisions

- Data: the released 2026 NextLap training/validation files and original race snapshots in `base_NextLap/`. The reference helper checks snapshot SHA-256 hashes against `source_manifest_v2.json`.
- Split: training R1–R7, 5,772 rows; validation R8–R9, 1,808 rows. Validation row counts are R8 = 1,066 and R9 = 742.
- Seed: 414.
- Run command: `python run_hito2.py`.
- Environment used for the recorded run: Python 3.13.5, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, SHAP 0.50.0.
- The run audit records zero train/validation `pair_id` overlap, unique validation IDs, complete ID coverage, exact agreement with saved Hito 1 predictions within tolerance, and a SHAP reconstruction check.
- Final-evaluation data were not used.

### Decisions

We retain `recent_std_s` for the fixed comparison because it lowers overall MAE and improves both validation races. We do not claim that it helps every observation: it worsens the imputed short-history slice. The next investigation should test a defensible short-history strategy, for example a separate indicator for insufficient history or a control restricted to the same observed-feature rows, while clearly keeping restricted-population diagnostics separate from the common 1,808-row comparison.

### Individual explanations

**Member 1 · Álvaro Torres:**

> I contributed to checking the temporal validity of the recent pace history feature and preventing data leakage. I verified that recent_std_s uses only information available up to the end of the current lap and does not mix observations from different drivers, races or tyre stints. The experiment showed that the extended model reduced validation MAE from 0.386707 to 0.365223 seconds across 1,808 observations. However, the improvement was not uniform: the model performed worse on the 148 observations where the feature had to be imputed. This suggests that short historical sequences require further investigation. My next step would be to evaluate a separate short-history indicator while preserving the same validation population.

**Member 2 · Sebastián Ramorino:**

> I contributed to reproducing the OLS reference, checking that the ablation changes only the added feature, and evaluating the candidate on the same 1,808 validation IDs. The candidate reduces MAE from 0.386707 to 0.365223 seconds, and the refitted predictions match the saved Hito 1 outputs. SHAP shows that recent mean and current lap time dominate the candidate's predictions. A limitation is the worse performance when `recent_std_s` is imputed; I would next test a train-fitted short-history treatment without changing the common evaluation rows.

### AI assistance and verification

ChatGPT (OpenAI) was used on 19 November to help draft the Hito 2 execution script, implement SHAP and contextual error summaries, and prepare the written interpretation. With the Hito 1 ZIP and Hito 2 brief supplied as sources. The adopted workflow was verified by executing `run_hito2.py`, checking that the refitted reference and candidate predictions match the saved Hito 1 predictions, checking the common validation IDs and counts, and verifying that the SHAP base value plus signed contributions reconstructs the selected prediction. AI-generated suggestions were not accepted as evidence without these checks.
