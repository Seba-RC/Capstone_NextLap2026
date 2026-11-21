# Decision log · Hito 1

## Decisions

| Date | Who | Decision | Evidence | Status |
|---|---|---|---|---|
| 6 Nov | Team | Primary direction: recent pace history. | Kickoff worksheet, section 6. | Kept. |
| 6 Nov | Team | Baseline is the supplied OLS at 0.386707 s on 1,808 rows. | Own run of the class script on 6 Nov; reproduced by the notebook on 11 Nov with identical values. | Done. |
| 6 Nov | Member 2 | Compare models only on identical validation `pair_id` values. | Worksheet prior check; peer review asked for this to be explicit. | Check C2. |
| 11 Nov | Team | The extension is `recent_std_s`, not a rolling average. | The rolling average of the last three laps written in the worksheet's Build step is the supplied `recent_mean_s`, so it would not change the model. The standard deviation was already specified in worksheet section 3. | Changed. |
| 11 Nov | Member 1 | Verified the temporal availability of the historical pace feature. I inspected the lap sequence and confirmed that `recent_std_s` must use only completed laps available up to the end of the current lap t, within the same driver, race and tyre stint. I also checked that future lap observations were not included in the feature calculation. | `Hito1_Notebook.ipynb`, Part B: the four temporal-availability checks (L1–L4). L1 confirmed that each feature window ends at lap t or earlier (9,300 laps); L2 confirmed that deleting laps after t does not change the feature value (200 sampled validation rows); L3 confirmed that history does not cross stint, driver or race boundaries (461 stint starts); and L4 confirmed that no `next_*` columns, target or `clean_pair` flag are used as model inputs. | All checks passed. |
| 11 Nov | Member 2 | Revision after feedback: "the MAE must improve" replaced by a three-part rule fixed in advance. | One lower number cannot separate a real gain from noise or from one race driving the result. | HITO1.md section 4. |
| 11 Nov | Member 2 | The ablation is verified, not assumed. | Check C4. | Passed. |
| 11 Nov | Member 2 | First run of the comparison, logged as trial 1. | `outputs/trial_log.csv`: 0.386707 → 0.365223 s, difference −0.021484 s, interval [−0.031503, −0.012860] over 43 driver-races; round 8 −0.015967 s, round 9 −0.029410 s. | Preliminary; full analysis belongs to Hito 2. |
| 11 Nov | Member 2 | Open issue: on the 148 validation rows where the feature is imputed, the augmented model is worse by 0.020375 s; on the 1,660 observed rows it is better by 0.025216 s. | Same run. | For Hito 2. |

## AI use

| Member | Record |
|---|---|
| Member 2 (Sebastián) | Claude (Anthropic) was used on 11 November to draft `Hito1_Notebook.ipynb`, `README.md`, `HITO1.md` and this log from the brief, the data release and the team worksheet. Suggestions adopted: the leakage checks L1–L4 (in particular the truncation test), the C1 rebuild-and-compare check, the bootstrap by driver-race, the three-part decision rule, and the observation that the worksheet's rolling average duplicated `recent_mean_s`. How I checked them: [fill in after doing it — e.g. reran the notebook locally and confirmed the reference MAE values match my 6 November run to six decimals; recomputed `recent_std_s` by hand for one driver and stint from the raw race CSV; read each assertion and can explain what it stops]. |
| Member 1 (Álvaro) | ChatGpt was used to analyze the capstone and "Hito 1", as well to check the Hito 1 results. |

## What each member did and can explain

| Member | Contribution |
|---|---|
| Member 1 (Álvaro) | Reviewed the historical lap-time data and contributed to the temporal availability and data leakage checks. I focused on verifying that historical observations belonged to the same driver, race and tyre stint, and that no future lap information was used to construct the additional feature. I can explain the purpose of the L1–L4 leakage checks and their importance for a valid prediction at time t. |
| Member 2 (Sebastián) | Reference reproduction, same-rows check, ablation check, decision rule, metric tables. |

## Next steps before Hito 2 (19 Nov)

- Contextual errors of the comparison: short-history rows, by race, by compound.
- SHAP for the augmented model.
- Broader observed-pairs population, reported separately from the fixed comparison.
