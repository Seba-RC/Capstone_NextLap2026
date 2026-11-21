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
