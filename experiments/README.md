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
