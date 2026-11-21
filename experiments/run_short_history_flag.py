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

    # Check 2 (training rounds only, no validation data): leave one training round out.
    # Does the flag help consistently on unseen circuits, and is the early-stint speed-up it
    # learns stable across circuits?
    loro_rows = []
    for held_out in sorted(train_x['round'].unique()):
        fit_part, test_part = train_x[train_x['round'] != held_out], train_x[train_x['round'] == held_out]
        y_test = test_part[ref.TARGET].to_numpy()
        short = test_part[FLAG].to_numpy() == 1
        err = {}
        for label, numeric in [('no_flag', candidate_numeric), ('flag', flagged_numeric)]:
            model = ref.make_model('OLS', numeric=numeric)
            err[label] = np.abs(y_test - fit_predict(model, fit_part, test_part, numeric + ref.CATEGORICAL))
        loro_rows.append({
            'held_out_round': int(held_out), 'event': test_part.event_name.iloc[0], 'n': len(test_part),
            'n_short_history': int(short.sum()),
            'MAE_no_flag_s': err['no_flag'].mean(), 'MAE_flag_s': err['flag'].mean(),
            'flag_minus_no_flag_s': err['flag'].mean() - err['no_flag'].mean(),
            'short_rows_MAE_no_flag_s': err['no_flag'][short].mean(), 'short_rows_MAE_flag_s': err['flag'][short].mean(),
            'short_rows_mean_next_minus_current_s': float((test_part[ref.TARGET] - test_part.current_lap_s)[short].mean()),
        })
    loro = pd.DataFrame(loro_rows)
    loro.to_csv(OUT / 'loro_training_check.csv', index=False, float_format='%.6f')
    pooled_loro = float((loro.flag_minus_no_flag_s * loro.n).sum() / loro.n.sum())

    # Check 3: does the flag pass under median regression (Experiment 2's loss)?
    from sklearn.linear_model import QuantileRegressor
    med_err = {}
    for label, numeric in [('median_control', reference_numeric), ('median_std', candidate_numeric),
                           ('median_std_flag', flagged_numeric)]:
        model = ref.make_model('OLS', numeric=numeric).set_params(
            regression=QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs'))
        med_err[label] = np.abs(y - fit_predict(model, train_x, val_x, numeric + ref.CATEGORICAL))
    m_diff, m_low, m_high, _ = cluster_bootstrap_ci(val_x, med_err['median_std'], med_err['median_std_flag'])
    m_races = pd.DataFrame({'r': val_x['round'], 'a': med_err['median_std'], 'b': med_err['median_std_flag']}).groupby('r').mean()
    decision['training_loro_check'] = {
        'pooled_flag_minus_no_flag_s': pooled_loro,
        'held_out_rounds_where_flag_lower': int((loro.flag_minus_no_flag_s < 0).sum()),
        'held_out_rounds': int(len(loro)),
        'short_rows_speed_up_range_s': [float(loro.short_rows_mean_next_minus_current_s.min()),
                                        float(loro.short_rows_mean_next_minus_current_s.max())],
    }
    s_diff, s_low, s_high, _ = cluster_bootstrap_ci(val_x, med_err['median_control'], med_err['median_std'])
    decision['median_regression_check'] = {
        'MAE_median_control_without_recent_std_s': float(med_err['median_control'].mean()),
        'recent_std_s_effect_under_median_s': s_diff, 'recent_std_s_effect_bootstrap_95ci_s': [s_low, s_high],
        'MAE_median_std_s': float(med_err['median_std'].mean()),
        'MAE_median_std_flag_s': float(med_err['median_std_flag'].mean()),
        'paired_difference_flag_minus_no_flag_s': m_diff, 'bootstrap_95ci_s': [m_low, m_high],
        'lower_in_every_race': bool((m_races.b < m_races.a).all()),
        'passes_rule': bool(m_diff < 0 and m_high < 0 and (m_races.b < m_races.a).all()),
    }
    (OUT / 'decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')

    print(slices.to_string(index=False))
    print(loro.round(6).to_string(index=False))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
