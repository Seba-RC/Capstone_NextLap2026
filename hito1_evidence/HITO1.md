# Hito 1 · Problem framing, decision value and experiment plan

## 1. Problem and decision value

| Item | Definition |
|---|---|
| Target | `next_lap_s`: duration of the same driver's next lap (t+1). |
| Unit | Seconds. |
| Prediction time | The end of lap t, using only what is known by then. |
| Intended reader | A race engineer or strategist following one driver. |
| Decision informed | Whether the driver's pace on the coming lap is expected to hold or drop, as one input when judging tyre state or comparing against a rival's pace. |

What it cannot establish: a lap-time estimate does not say when a car should pit or how many positions a strategy gains.

Limit of the clean-pair setting: `clean_pair` selects rows retrospectively using lap t+1 conditions (no pit, green track, accurate timing). The results describe laps that turned out clean; they are not evidence about all race laps, and the flag cannot be used to decide in advance which live laps to predict.

## 2. Reproduced reference

- Data version: student data release v1 (30 October 2026), `RANDOM_SEED = 414`.
- Split: training = 5,772 clean pairs, rounds 1–7; validation = 1,808 clean pairs, rounds 8–9. Unique `pair_id` in each split, zero overlap. No reserved final-evaluation data were accessed.
- Features: `current_lap_s`, `recent_mean_s`, `current_lap_number`, `current_tyre_life` (numeric); `current_compound`, `current_team` (categorical).
- Fit/transform separation: imputer, scaler and one-hot encoder sit inside the helper's pipeline and are fitted by `.fit()` on training rows only. The training median is computed on training rows only. Validation rows are only predicted.
- Command: run all cells of `Hito1_Notebook.ipynb`, Part A.

| Model | Validation MAE (s), n = 1,808 | Round 8 Austria MAE (s), n = 1,066 | Round 9 Britain MAE (s), n = 742 |
|---|---:|---:|---:|
| Training median | 11.376111 | 11.715781 | 10.888121 |
| Persistence (`current_lap_s`) | 0.409289 | 0.400516 | 0.421892 |
| OLS reference | 0.386707 | 0.373571 | 0.405579 |

All three models are scored on the same 1,808 `pair_id` values, saved in `outputs/reference_predictions.csv`.

Comparison explained: OLS beats persistence by 0.0226 s (about 5.5%) and is lower in both races. The gap is small because the current lap already carries most of the information about the next one; the other five inputs add a modest correction. The median is about 11 s off because one constant cannot follow lap-time differences between circuits.

## 3. Primary hypothesis

Direction: recent pace history.

| Item | Definition |
|---|---|
| Proposed feature | `recent_std_s`: sample standard deviation (ddof = 1) of the driver's recent clean lap times. |
| What differs from the reference | The reference already has the recent mean (`recent_mean_s`). The mean describes the level of recent pace; the standard deviation describes how stable it was. It is not a linear combination of the six supplied inputs. |
| Expected mechanism | After steady laps, the next lap should resemble the current one. After erratic laps (traffic, a mistake, tyres changing behaviour), the current lap is a less reliable guide, so the model can adjust how far the next lap is expected to move. |
| Source | The released timing snapshots only. No external data, so no fallback source is needed. |
| Availability at t | At most three laps, ending at lap t, within the same season, round, driver and tyre stint. Nothing from lap t+1 or later. Built on the original timeline before the clean-pair selection, using the same clean-lap rule as `recent_mean_s`. |
| Short-history rule | Fewer than two valid laps in the window gives a missing value. Missing values are filled by a median imputer fitted on training rows, so no row is dropped. |

Availability checks (notebook Part B, each stops the run on failure):

| Check | Inspects | Result |
|---|---|---|
| L1 last source lap | Latest lap number and timestamp inside every window is lap t or earlier. | Passed on 9,300 laps. |
| L2 truncation | Every lap after t is deleted and the feature rebuilt; the value must not change. | Passed on 200 sampled validation rows. |
| L3 boundaries | The history restarts at each stint and never reaches another stint, driver or race. | Passed on 461 stint starts. |
| L4 model inputs | No `next_*` column, target or `clean_pair` flag among the inputs. | Passed. |

## 4. Experiment plan

| Stays fixed | Value |
|---|---|
| Target and unit | `next_lap_s`, seconds |
| Rows | Training 5,772 and validation 1,808 `pair_id` values from the release |
| Split | Rounds 1–7 train, rounds 8–9 validation |
| Model | OLS from `make_model('OLS')`, no hyperparameters |
| Preprocessing protocol | Median imputation, scaling and one-hot encoding fitted on training only |
| Metric | MAE in seconds, overall and per validation race, with counts |
| Seed | 414 |

| Comparison | What it shows |
|---|---|
| Reference OLS (six inputs) vs OLS + `recent_std_s` | Incremental value of recent pace variability. |
| Augmented feature list with `recent_std_s` removed | Must reproduce the reference predictions exactly, proving the added column is the only difference. |

Checks written before the extension (notebook Part C, each stops the run on failure):

| Check | Inspects | Failure |
|---|---|---|
| C1 timeline | `recent_mean_s` and `recent_count` rebuilt from the raw snapshots equal the released columns on every row. | The history was not rebuilt on the same timeline and boundaries. |
| C2 rows | Validation `pair_id` set, count (1,808) and order after adding the feature. | A changed, missing or duplicated validation row. |
| C3 short history | `recent_std_s` is missing exactly where `recent_count < 2`. | A value computed from one lap, or a gap elsewhere. |
| C4 ablation | Augmented list minus the feature reproduces the reference predictions. | Any other difference between the pipelines. |

What would support or contradict the hypothesis, fixed in advance. Supported only if all three hold: the augmented MAE is lower on the 1,808 rows; the paired difference is negative in both round 8 and round 9; and the 95% interval of the mean paired difference (bootstrap resampling whole driver-races, 2,000 draws, seed 414) lies entirely below zero. Anything else is reported as "no supported improvement" and kept in the trial log.

Search plan: one candidate, window fixed at 3 to match `recent_mean_s`, one model. No window or model search is planned. Any later trial is appended to `outputs/trial_log.csv`, including unsuccessful ones.

Validation and test: rounds 8–9 are used for this comparison only. Reserved final-evaluation data are not used to choose features, parameters or models.

Planned robustness checks for Hito 2: MAE by validation race with counts; rows where the feature is observed vs imputed, reported separately; the broader observed-pairs population reported apart from the fixed common comparison.

## 5. Traceability and next steps

| Item | Record |
|---|---|
| Contributions | See `DECISION_LOG.md`. |
| AI record | See `DECISION_LOG.md`. |
| Revision after feedback | Peer review (6 Nov) asked for an explicit availability time and identical evaluation rows. Both are now executable checks (L1–L4, C2), and "the MAE must improve" became the three-part rule above. |
| Member 1 before Hito 2 | [Álvaro to fill: e.g. contextual errors by stint position and compound] |
| Member 2 before Hito 2 | Full comparison analysis, short-history rows, SHAP for the augmented model. |
| Known blocker | None for Hito 1. Open question for Hito 2: the augmented model is worse on rows where the feature is imputed. |
