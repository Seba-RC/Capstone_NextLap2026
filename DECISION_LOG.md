# Decision log · Final report (Team 2)

This log carries forward the Hito 1 log (`hito1_evidence/DECISION_LOG.md`) and the Hito 2 decisions, then records the post-Hito 2 trials and the response to feedback. Final decision: **retain OLS + `recent_std_s` (the Hito 2 candidate) and do not add the short-history flag.**

## Experimental decisions and evidence

| Date | Decision | Evidence | Status |
|---|---|---|---|
| 6 Nov / Hito 1 | Keep recent pace history as the primary direction. | Hito 1 worksheet and `HITO1.md`. | Carried forward. |
| 11 Nov / Hito 1 | Use `recent_std_s` (sample SD, window 3), not a new rolling mean. | Hito 1 log: rolling mean duplicates the supplied `recent_mean_s`. | Carried forward. |
| 19 Nov | Refit the OLS reference and OLS + `recent_std_s` on the fixed training rows. | `outputs/run_audit.json`; both refitted prediction vectors match saved Hito 1 predictions within tolerance. | Verified. |
| 19 Nov | Treat OLS reference as the matched control because removing `recent_std_s` leaves the six reference inputs and same OLS pipeline. | Ablation refit and prediction match; `outputs/comparison_metrics.csv`. | Verified. |
| 19 Nov | Retain `recent_std_s` for the common validation comparison. | MAE 0.386707 → 0.365223 s; R8 and R9 both improve. | Retain, with limitation. |
| 19 Nov | Do not claim uniform benefit across all validation rows. | Observed-feature rows improve by 0.025216 s; 148 imputed rows worsen by 0.020375 s (control minus candidate sign). | Limitation recorded. |
| 19 Nov | Use the largest candidate residual as the inspected case. | `2026-R9-23-L6`; target 105.128 s, candidate prediction 96.825754 s, absolute error 8.302246 s. | Verified against saved validation prediction and source snapshot. |
| 19 Nov | Explain the fitted candidate with `shap.LinearExplainer`. | Training background n=200; validation sample n=1,808; reconstruction difference approximately 1.4e-14 s. | Verified. |

## Trial settings and results

| Trial | Date | Model | Inputs | Validation n | MAE (s) | vs comparison model (s), 95% CI | Decision | Evidence |
|---|---|---|---|---:|---:|---|---|---|
| T0 reference | 6 Nov | OLS | Six reference inputs | 1,808 | 0.386707 | — | Matched control | `outputs/comparison_metrics.csv` |
| T1 candidate | 11 / 19 Nov | OLS | + `recent_std_s` (window 3) | 1,808 | 0.365223 | −0.021484 vs T0, [−0.031503, −0.012860] | **Retain** (passes rule) | `hito1_evidence/outputs/trial_log.csv`; `outputs/comparison_metrics.csv` |
| T2 flag (feedback) | 21 Nov | OLS | T1 + `short_history` | 1,808 | 0.369728 | +0.004505 vs T1, [+0.000847, +0.008393] | Reject | `outputs/exp1_short_history_flag/decision.json` |
| T2b flag, training-only check | 21 Nov | OLS | T1 vs T2, leave one of R1–R7 out | 5,772 | — | pooled −0.002771; lower in 4 of 7 held-out rounds | Supports rejection (inconsistent) | `outputs/exp1_short_history_flag/loro_training_check.csv` |
| T2c flag under median loss | 21 Nov | Median regression | T3 + `short_history` | 1,808 | 0.355005 | −0.001136 vs T3, [−0.004063, +0.001823]; R9 worse | Reject | `outputs/exp1_short_history_flag/decision.json` |
| T3 median loss | 21 Nov | Median regression | Same as T1 | 1,808 | 0.356141 | −0.009082 vs T1, [−0.013874, −0.004509] | Exploratory; confirm before adopting | `outputs/exp2_median_traffic_gap/comparisons.json` |
| T4 traffic gap | 21 Nov | OLS | T1 + `gap_ahead_s` (cap 5 s) | 1,808 | 0.365444 | +0.000221 vs T1, [−0.000293, +0.000757] | Reject | same |
| T5 median + gap | 21 Nov | Median regression | T3 + `gap_ahead_s` | 1,808 | 0.355671 | −0.000470 vs T3, [−0.001263, +0.000323] | Reject the gap | same |
| T6 trees, raw target | 21 Nov | Gradient boosting ×2, random forest | T1 inputs + `current_lap_s − recent_mean_s` | 1,808 | 1.173–1.445 | +0.81 to +1.08 vs T1 | Reject (cannot extrapolate to Austria) | `outputs/exp3_tree_models/model_comparison.csv` |
| T7 trees, delta target | 21 Nov | Same | Same, target `next_lap_s − recent_mean_s` | 1,808 | 0.365470–0.381808 | best +0.000247 vs T1, [−0.008855, +0.009417] | Reject (no gain) | same |

CI = 95% driver-race cluster bootstrap over 43 groups, seed 414. "Passes rule" = lower pooled MAE, lower in both R8 and R9, CI entirely below zero. Unsuccessful trials are kept. T3–T7 were run after seeing the Hito 2 validation results, so they are exploratory: the validation races informed which ideas were tried.

## Verification

- `run_hito2.py` uses the released train/validation split and original snapshots.
- The course helper verifies the snapshot hashes using `source_manifest_v2.json`.
- Train and validation `pair_id` values are unique and have zero overlap.
- Both models are evaluated on the same 1,808 validation IDs; the script asserts full ID coverage.
- Preprocessing and OLS are fitted only on training rows.
- The script verifies that refitted reference and candidate predictions match the saved Hito 1 predictions.
- SHAP uses a background sample from training data; the local base value plus contributions reconstructs the candidate prediction.
- 21 Nov: all four scripts were re-run from the pinned environment; every Hito 2 CSV in `outputs/` was byte-identical to the committed version (only the recorded Python patch version and the PNG rendering differed, and these were not replaced).
- Every post-Hito 2 script asserts that its refitted OLS + `recent_std_s` predictions equal the saved Hito 2 predictions (tolerance 1e-8 s) before comparing anything.

## AI use

**Hito 2 (19 Nov).** ChatGPT (OpenAI) was used on 19 November to assist with the Hito 2 script, SHAP implementation, contextual error analysis and report drafting. The team accepted the suggested matched-control structure, grouped SHAP presentation and validation-case diagnosis after checking them through execution. Verification included matching the refitted predictions to Hito 1 outputs, checking the validation row identities/counts and checking the SHAP reconstruction arithmetic. The submitting members must confirm that this entry accurately records their actual AI use and add any other tools or prompts they used.

**Final report (20–21 Nov), Member 2 (Sebastián).** Claude (Anthropic) was used to summarise the Hito 2 package and draft the Demo Day slides; create this GitHub repository and push the Hito 2 commit; write `experiments/` (common helpers, Experiments 1–3 and the leave-one-race-out flag check); and draft the final report and this log. Suggestions adopted: testing the feedback flag with the predeclared rule, adding a training-only leave-one-race-out check, the median-regression loss, the delta target for trees, and the extrapolation diagnosis for raw-target trees. Checks performed: each experiment script asserts its refitted candidate equals the saved Hito 2 predictions; the full set of scripts was re-run and the Hito 2 outputs matched byte-for-byte; every number in the report was copied from a saved output file listed in the trial table; rejected suggestions (the flag, the traffic gap, tree models) are kept as unsuccessful trials. [Sebastián: confirm this matches your actual use and the checks you personally repeated.]

Hito 1's Claude use by Member 2 (Sebastián) remains documented in the Hito 1 decision log. The Hito 1 AI verification field should be completed truthfully; do not claim a manual hand calculation unless it was actually performed.

## Response to feedback (final report)

| Item | Record |
|---|---|
| Priority | Demo Day / Hito 2 feedback centred on flags for the short-history rows: the candidate is worse on the 148 validation rows where `recent_std_s` is imputed, and an explicit missingness flag was suggested. [Paste the exact feedback wording here.] |
| Change tested | T2: add `short_history` (1 when fewer than two valid recent laps) to OLS + `recent_std_s`, everything else fixed. |
| Checks and results | T2 on the common 1,808 IDs: MAE 0.365223 → 0.369728 s, CI [+0.000847, +0.008393], worse in both races, short-history slice 0.416261 → 0.453406 s. T2b (training rounds only): flag lower in 4 of 7 held-out rounds, pooled −0.002771 s; the short-history speed-up it relies on ranges from −0.703 s (Canada) to +0.149 s (Barcelona). T2c: under median regression the flag still fails (CI [−0.004063, +0.001823], R9 worse). |
| Decision | Retain the original decision: keep `recent_std_s`, do not add the flag. The early-stint effect the flag learns is circuit-dependent, so a single shift does not transfer to new circuits. The short-history limitation is kept in the scope statement. |
| Evidence | `experiments/run_short_history_flag.py`; `outputs/exp1_short_history_flag/` (`decision.json`, `mae_by_slice.csv`, `loro_training_check.csv`). |

## Member contributions and individual decisions

**Member 1 · Álvaro Torres:** Contributed to the validation of historical feature availability and leakage prevention. Reviewed the construction of recent_std_s, focusing on chronological ordering and separation by driver, race and tyre stint. Contributed to interpreting the limitations associated with short historical sequences and missing feature values.

Individual decision: Retain the additional pace-history feature for the common validation comparison, while recognizing that its performance is weaker when the feature is imputed. The next investigation should focus on improving the treatment of short histories without changing the common evaluation population.

**Member 2 · Sebastián Ramorino:** Hito 1 records reference reproduction, same-row checks, ablation checks, metric tables and the experiment decision rule. Hito 2 adds the refitted matched comparison, contextual error slices, inspected validation case and SHAP evidence. Final report: ran the feedback response (T2, T2b, T2c) and the exploratory trials T3–T7, decided to retain the Hito 2 candidate and reject the flag on the evidence in the trial table, and prepared the repository for submission.

## Next checks (after the final report)

1. Confirm median regression (T3) on data not used to choose it (the reserved final-evaluation races, or leave-one-race-out within R1–R7) before adopting it.
2. Short-history rows: try a circuit-independent treatment, e.g. a short-history indicator interacted with `current_tyre_life`, chosen with leave-one-race-out on R1–R7 only.
3. Inspect other large residuals such as `2026-R9-23-L6` for context observable by the end of lap t.
