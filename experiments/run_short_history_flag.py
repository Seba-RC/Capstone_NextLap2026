"""Experiment 1 (post-Hito 2): short-history indicator flag.

Hito 2 found that OLS + recent_std_s is worse than the control on the 148
validation rows where recent_std_s is imputed (fewer than two valid recent laps).
This run adds one binary input, `short_history`, so the model can learn a
separate adjustment for those rows. Everything else is held fixed.

Predeclared decision rule (same as Hito 1/2): keep the flag only if
  1. pooled validation MAE is lower than OLS + recent_std_s on all 1,808 IDs,
  2. MAE is lower in both validation races, and
  3. the 95% driver-race cluster-bootstrap CI for the paired difference
     (flag minus no-flag) lies entirely below zero.

Run from the repository root:  python experiments/run_short_history_flag.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import FEATURE, FLAG, ROOT, cluster_bootstrap_ci, fit_predict, load_fixed_split, ref, slice_table

OUT = ROOT / 'outputs' / 'exp1_short_history_flag'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    train_x, val_x, _ = load_fixed_split()
    y = val_x[ref.TARGET].to_numpy()

    reference_numeric = list(ref.NUMERIC)
    candidate_numeric = reference_numeric + [FEATURE]
    flagged_numeric = candidate_numeric + [FLAG]

    versions = {
        'OLS reference': reference_numeric,
        'OLS + recent_std_s (Hito 2 candidate)': candidate_numeric,
        'OLS + recent_std_s + short_history': flagged_numeric,
    }
    preds, errors = {}, {}
    for name, numeric in versions.items():
        model = ref.make_model('OLS', numeric=numeric)
        preds[name] = fit_predict(model, train_x, val_x, numeric + ref.CATEGORICAL)
        errors[name] = np.abs(y - preds[name])

    # The Hito 2 candidate must reproduce its saved predictions exactly.
    saved = pd.read_csv(ROOT / 'outputs' / 'validation_predictions.csv')
    saved = val_x[['pair_id']].merge(saved[['pair_id', 'pred_ols_plus_recent_std_s']], on='pair_id', validate='one_to_one')
    assert np.allclose(preds['OLS + recent_std_s (Hito 2 candidate)'], saved.pred_ols_plus_recent_std_s, atol=1e-8)

    slices = slice_table(val_x, {f'MAE_s · {k}': v for k, v in errors.items()})
    slices.to_csv(OUT / 'mae_by_slice.csv', index=False, float_format='%.6f')

    base, new = errors['OLS + recent_std_s (Hito 2 candidate)'], errors['OLS + recent_std_s + short_history']
    mean_diff, low, high, n_groups = cluster_bootstrap_ci(val_x, base, new)
    by_race = pd.DataFrame({'round': val_x['round'], 'base': base, 'new': new}).groupby('round').mean()
    decision = {
        'comparison': 'OLS + recent_std_s + short_history  vs  OLS + recent_std_s',
        'validation_n': int(len(val_x)),
        'MAE_without_flag_s': float(base.mean()),
        'MAE_with_flag_s': float(new.mean()),
        'paired_difference_flag_minus_no_flag_s': mean_diff,
        'bootstrap_95ci_s': [low, high],
        'bootstrap_groups_driver_race': n_groups,
        'lower_in_every_race': bool((by_race.new < by_race.base).all()),
        'rule_1_lower_pooled_MAE': bool(new.mean() < base.mean()),
        'rule_3_ci_below_zero': bool(high < 0),
    }
    decision['keep_flag'] = bool(decision['rule_1_lower_pooled_MAE'] and decision['lower_in_every_race']
                                 and decision['rule_3_ci_below_zero'])
    (OUT / 'decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')

    out = val_x[['pair_id', 'round', 'driver_number', 'current_compound', 'recent_count', FLAG, ref.TARGET]].copy()
    for name, p in preds.items():
        out[f'pred · {name}'] = p
    out.to_csv(OUT / 'validation_predictions.csv', index=False, float_format='%.6f')

    print(slices.to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
