# Decision log · Hito 2

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

| Trial | Model | Feature | Window | Validation n | MAE (s) | Decision |
|---|---|---|---:|---:|---:|---|
| Hito 1 reference | OLS | Six reference inputs | — | 1,808 | 0.386707 | Matched control |
| Hito 1 / Hito 2 candidate | OLS | Six reference inputs + `recent_std_s` | 3 | 1,808 | 0.365223 | Retain for this fixed comparison |

No additional feature-window or model-family search was conducted for Hito 2. The earlier Hito 1 trial log is retained as `hito1_evidence/outputs/trial_log.csv`. The candidate's short-history weakness is not hidden or treated as a failed run.

## Verification

- `run_hito2.py` uses the released train/validation split and original snapshots.
- The course helper verifies the snapshot hashes using `source_manifest_v2.json`.
- Train and validation `pair_id` values are unique and have zero overlap.
- Both models are evaluated on the same 1,808 validation IDs; the script asserts full ID coverage.
- Preprocessing and OLS are fitted only on training rows.
- The script verifies that refitted reference and candidate predictions match the saved Hito 1 predictions.
- SHAP uses a background sample from training data; the local base value plus contributions reconstructs the candidate prediction.

## AI use

ChatGPT (OpenAI) was used on 19 November to assist with the Hito 2 script, SHAP implementation, contextual error analysis and report drafting. The team accepted the suggested matched-control structure, grouped SHAP presentation and validation-case diagnosis after checking them through execution. Verification included matching the refitted predictions to Hito 1 outputs, checking the validation row identities/counts and checking the SHAP reconstruction arithmetic. The submitting members must confirm that this entry accurately records their actual AI use and add any other tools or prompts they used.

Hito 1's Claude use by Member 2 (Sebastián) remains documented in the Hito 1 decision log. The Hito 1 AI verification field should be completed truthfully; do not claim a manual hand calculation unless it was actually performed.

## Member contributions and individual decisions

**Member 1 · Álvaro Torres:** Contributed to the validation of historical feature availability and leakage prevention. Reviewed the construction of recent_std_s, focusing on chronological ordering and separation by driver, race and tyre stint. Contributed to interpreting the limitations associated with short historical sequences and missing feature values.

Individual decision: Retain the additional pace-history feature for the common validation comparison, while recognizing that its performance is weaker when the feature is imputed. The next investigation should focus on improving the treatment of short histories without changing the common evaluation population.

**Member 2 · Sebastián Ramorino:** Hito 1 records reference reproduction, same-row checks, ablation checks, metric tables and the experiment decision rule. Hito 2 adds the refitted matched comparison, contextual error slices, inspected validation case and SHAP evidence. Confirm that this attribution accurately reflects the work he personally completed and can defend.

## Next checks

1. Investigate why the model is worse on the 148 rows where `recent_std_s` is imputed; do not silently remove these rows from the common comparison.
2. Inspect other large residuals and test whether any additional context is observable by the end of lap t.
3. If any restricted-population comparison is added, label it separately and report its row count and coverage.
4. Record instructor feedback and the corresponding final correction or justified decision to retain the design in the final report due 22 November.

## Post-Hito 2 · Experiment 1: short-history flag (rejected)

Tested next check 1 by adding a binary `short_history` input (1 when `recent_std_s` is imputed) to OLS + `recent_std_s`, with the same split, validation IDs and decision rule. Pooled MAE rose from 0.365223 s to 0.369728 s (paired difference +0.004505 s, 95% driver-race cluster-bootstrap CI [+0.000847, +0.008393] s), it was worse in both races, and the short-history slice worsened from 0.416261 s to 0.453406 s. Decision: do not add the flag; OLS + `recent_std_s` remains the candidate. Likely reason: the early-stint speed-up learned from rounds 1–7 is larger than in rounds 8–9, so the flag over-corrects. Details and outputs: `experiments/README.md`, `outputs/exp1_short_history_flag/`.

## Post-Hito 2 · Experiment 2: median regression + traffic gap (branch `experiment/median-traffic-gap`)

Tested next check 2 and the loss mismatch (OLS minimises squared error, evaluation uses absolute error). On the same 1,808 IDs: median regression with the Hito 2 inputs lowers MAE from 0.365223 s to 0.356141 s (−0.009082 s, CI [−0.013874, −0.004509]), lower in both races. Adding `gap_ahead_s` (gap to the car ahead at the end of lap t, capped at 5 s) gives no reliable extra gain (−0.000470 s vs median alone, CI [−0.001263, +0.000323]). Proposed decision, pending team agreement before merging into `main`: adopt median regression; do not add `gap_ahead_s` in this form. Details: `experiments/README.md`.

## Post-Hito 2 · Experiment 3: tree-based models (branch `experiment/median-traffic-gap`)

Compared untuned gradient boosting (absolute- and squared-error loss) and a random forest with OLS + `recent_std_s` on the same 1,808 IDs, using both the raw target and a delta target (`next_lap_s − recent_mean_s`). Raw-target trees are far worse (MAE 1.17–1.45 s) because Austria's lap times (70.37–77.88 s) are below the training range and trees cannot extrapolate. On the delta target, absolute-error boosting ties the Hito 2 model (0.365470 vs 0.365223 s, CI [−0.008855, +0.009417]); the other trees are worse. No tree model passes the decision rule. Decision: stay with a linear model; median regression (Exp. 2 A, 0.356141 s) remains the best candidate. Details: `experiments/README.md`.
