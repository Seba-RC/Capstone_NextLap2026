# Post-Hito 2 experiments

These experiments keep the Hito 2 evaluation fixed: training on rounds 1–7 (5,772 rows), validation on the same 1,808 `pair_id`s from rounds 8–9, MAE in seconds, and the same predeclared decision rule (lower pooled MAE, lower in both races, 95% driver-race cluster-bootstrap CI for the paired difference entirely below zero). The Hito 2 files in `outputs/` are not modified; each experiment writes to its own `outputs/exp*/` folder.

Run from the repository root with the pinned environment (`pip install -r requirements.txt`):

```bash
python experiments/run_short_history_flag.py
```

## Experiment 1 · short-history flag (`run_short_history_flag.py`) — rejected

**Change:** add one binary input, `short_history` = 1 when fewer than two valid recent laps exist (so `recent_std_s` is imputed with the training median). Everything else is identical to the Hito 2 candidate (OLS + `recent_std_s`).

| Version | Pooled MAE (s) | R8 Austria | R9 Britain | Short-history rows (n = 148) | Observed rows (n = 1,660) |
|---|---:|---:|---:|---:|---:|
| OLS reference | 0.386707 | 0.373571 | 0.405579 | 0.395886 | 0.385889 |
| OLS + `recent_std_s` (Hito 2) | **0.365223** | **0.357604** | **0.376169** | 0.416261 | **0.360673** |
| OLS + `recent_std_s` + `short_history` | 0.369728 | 0.358793 | 0.385439 | 0.453406 | 0.362268 |

Paired difference (flag − no flag): **+0.004505 s**, 95% cluster-bootstrap CI **[+0.000847, +0.008393] s** over 43 driver-race groups. All three rules fail, so the flag is **not kept**; OLS + `recent_std_s` stays the candidate.

**Why it fails:** in the training races, the lap after a short-history lap is on average 0.369 s faster than the current lap (median 0.252 s), so the flag learns a large negative shift (≈ −0.389 s per flagged row). In the two validation races that speed-up is much smaller (mean 0.127 s, median 0.100 s), so the flag over-corrects and the short-history slice gets worse (0.416 → 0.453 s). The effect of being early in a stint does not transfer well between circuits. This is a validation-informed observation, not a causal claim.

Outputs: `outputs/exp1_short_history_flag/` (`mae_by_slice.csv`, `decision.json`, `validation_predictions.csv`).

## Experiment 2 · median regression + traffic gap (`run_median_traffic_gap.py`, branch `experiment/median-traffic-gap`)

```bash
python experiments/run_median_traffic_gap.py
```

**Changes, tested one at a time from the Hito 2 candidate (OLS + `recent_std_s`):**

- **A · loss:** OLS → median regression (`QuantileRegressor`, quantile 0.5, no penalty), which minimises absolute error, the metric we are scored on. Same inputs.
- **B · feature:** add `gap_ahead_s`, the gap at the end of lap *t* to the car that completed the same lap number just before this driver. Built from `end_time_s` in the original snapshots (known at the end of lap *t*). Capped at 5 s; the race leader gets the cap. The cap was fixed before looking at results. Limitation: a lapped car physically ahead is not counted.
- **C · both** (the requested model).

| Version | Pooled MAE (s) | R8 Austria | R9 Britain | Short-history (148) | SOFT (133) |
|---|---:|---:|---:|---:|---:|
| OLS + `recent_std_s` (Hito 2) | 0.365223 | 0.357604 | 0.376169 | 0.416261 | **0.473944** |
| A · median + `recent_std_s` | 0.356141 | 0.348080 | **0.367721** | **0.403531** | 0.486941 |
| B · OLS + `recent_std_s` + `gap_ahead_s` | 0.365444 | 0.357391 | 0.377015 | 0.418398 | 0.473790 |
| C · median + `recent_std_s` + `gap_ahead_s` | **0.355671** | **0.347281** | 0.367725 | 0.408650 | 0.485180 |

| Comparison | Paired difference (s) | 95% CI (s), 43 driver-race groups | Both races lower | Passes rule |
|---|---:|---|---|---|
| A vs Hito 2 (loss only) | −0.009082 | [−0.013874, −0.004509] | yes | **yes** |
| B vs Hito 2 (gap only, OLS) | +0.000221 | [−0.000293, +0.000757] | no | no |
| C vs A (gap added to median) | −0.000470 | [−0.001263, +0.000323] | no (R9 +0.000004) | no |
| C vs Hito 2 (both) | −0.009552 | [−0.014548, −0.004735] | yes | yes |

**Reading:** the whole improvement comes from switching to median regression (about 2.5% lower MAE than the Hito 2 candidate, 7.6% lower than the OLS reference). It also helps the short-history slice the flag could not fix. `gap_ahead_s` does not add a reliable gain under either loss; its fitted effect in model C is tiny (−0.008 s of next-lap time per extra second of gap). C passes against the Hito 2 candidate only because it contains A. The SOFT slice (133 rows) is worse under median regression; it is small, so read it with caution.

**Suggested decision:** adopt median regression with the Hito 2 inputs (model A); do not add `gap_ahead_s` in this form. A different traffic definition (e.g. a `< 1 s` indicator, or counting lapped cars) would be a separate, predeclared experiment.

Outputs: `outputs/exp2_median_traffic_gap/` (`mae_by_slice.csv`, `comparisons.json`, `validation_predictions.csv`).

## Experiment 3 · tree-based models (`run_tree_models.py`, branch `experiment/median-traffic-gap`)

```bash
python experiments/run_tree_models.py
```

**Question:** does gradient boosting or a random forest beat OLS + `recent_std_s` with the same inputs and the same 1,808 validation IDs?

**Setup:** `HistGradientBoostingRegressor` (absolute-error and squared-error loss) and `RandomForestRegressor` (300 trees, `min_samples_leaf=5`), library defaults otherwise, seed 414. Inputs: the six published inputs, `recent_std_s`, and `current_lap_s − recent_mean_s`. Two targets:

- **raw:** predict `next_lap_s` directly, the same target as OLS.
- **delta:** predict `next_lap_s − recent_mean_s` and add `recent_mean_s` back, which removes each circuit's lap-time level.

No hyperparameters were tuned, so the validation races were not used to pick settings.

| Model | MAE (s) | R8 Austria | R9 Britain | vs Hito 2 (s) | 95% CI (s) | Passes rule |
|---|---:|---:|---:|---:|---|---|
| OLS + `recent_std_s` (Hito 2) | 0.365223 | 0.357604 | 0.376169 | — | — | — |
| Median regression (Exp. 2 A) | **0.356141** | **0.348080** | 0.367721 | −0.009082 | [−0.013874, −0.004509] | **yes** |
| Gradient boosting, absolute error · raw | 1.291460 | 1.920359 | 0.387948 | +0.926237 | [+0.581849, +1.263533] | no |
| Gradient boosting, squared error · raw | 1.445112 | 2.171615 | 0.401376 | +1.079889 | [+0.719826, +1.433513] | no |
| Random forest · raw | 1.173260 | 1.710682 | 0.401169 | +0.808037 | [+0.503262, +1.109637] | no |
| Gradient boosting, absolute error · delta | 0.365470 | 0.365992 | **0.364720** | +0.000247 | [−0.008855, +0.009417] | no (tie) |
| Gradient boosting, squared error · delta | 0.381808 | 0.379746 | 0.384770 | +0.016585 | [+0.007404, +0.025569] | no |
| Random forest · delta | 0.375531 | 0.370664 | 0.382522 | +0.010308 | [+0.002078, +0.018740] | no |

**Reading:**

1. **Raw-target trees fail on Austria because they cannot extrapolate.** Austrian laps (median 72.85 s, range 70.37–77.88 s) are faster than every training circuit (medians 77.1–98.1 s; fastest training next lap 73.48 s). A tree can only predict values in the range it saw in training; no raw-target tree prediction went below 74.32 s. OLS extrapolates because it scales the recent pace linearly. See `circuit_lap_levels.csv` and `audit.json`.
2. **Predicting the change fixes most of this, but does not beat OLS.** Absolute-error boosting on the delta target ties the Hito 2 model (+0.0002 s, CI crossing zero): better in Britain, worse in Austria. Squared-error boosting and the random forest are worse.
3. **The signal is mostly linear.** Consistent with the Hito 2 SHAP results, recent pace level explains most of each prediction; the large errors left over are sudden events that no current input signals. With 5,772 rows from 7 circuits, flexible models have little extra structure to learn.

**Decision:** keep a linear model; median regression (Exp. 2 A) remains the best candidate. Revisit tree models only if new information is added (e.g. a different traffic definition or driver-specific history), using absolute-error boosting on the delta target and tuning by leave-one-race-out within rounds 1–7 only.

Outputs: `outputs/exp3_tree_models/` (`model_comparison.csv`, `mae_by_slice.csv`, `circuit_lap_levels.csv`, `audit.json`, `validation_predictions.csv`).
