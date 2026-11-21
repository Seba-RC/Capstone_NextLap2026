"""Experiment 2 (branch experiment/median-traffic-gap): median regression + traffic gap.

Two changes are tested one at a time, starting from the Hito 2 candidate
(OLS + recent_std_s), on the same 1,808 validation IDs:

  A. Loss: replace OLS (squared error) with median regression
     (sklearn QuantileRegressor, quantile=0.5, alpha=0), which minimises absolute
     error -- the metric we are scored on. Inputs unchanged.
  B. Feature: add `gap_ahead_s`, the time gap at the end of lap t to the car that
     completed the same lap number immediately before this driver.
  C. A + B together (the requested model).

gap_ahead_s is built from the original timing snapshots before clean-pair
selection, using only end_time_s of lap t, which every driver ahead has already
completed when this driver finishes lap t (no future information). Gaps are
capped at GAP_CAP_S seconds; the race leader (no car ahead) gets the cap. The cap
was fixed before looking at validation results: beyond a few seconds a car is
in clear air. Known limitation: a lapped car physically ahead on track is not
counted, because it completed a different lap number.

Decision rule (as in Hito 1/2): pooled MAE lower than the Hito 2 candidate,
lower in both races, and 95% driver-race cluster-bootstrap CI below zero.

Run from the repository root:  python experiments/run_median_traffic_gap.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from common import FEATURE, ROOT, cluster_bootstrap_ci, fit_predict, load_fixed_split, ref, slice_table

OUT = ROOT / 'outputs' / 'exp2_median_traffic_gap'
OUT.mkdir(parents=True, exist_ok=True)
GAP = 'gap_ahead_s'
GAP_CAP_S = 5.0


def add_gap(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    laps = raw.dropna(subset=['end_time_s']).sort_values(['season', 'round', 'lap_number', 'end_time_s']).copy()
    laps[GAP] = laps.end_time_s - laps.groupby(['season', 'round', 'lap_number']).end_time_s.shift(1)
    laps[GAP] = laps[GAP].fillna(GAP_CAP_S).clip(upper=GAP_CAP_S)  # leader -> clear air
    laps['pair_id'] = (laps.season.astype(str) + '-R' + laps['round'].astype(str) + '-'
                       + laps.driver_number.astype(str) + '-L' + laps.lap_number.astype(int).astype(str))
    merged = frame.merge(laps[['pair_id', GAP, 'end_time_s']].rename(columns={'end_time_s': '_end'}),
                         on='pair_id', how='left', validate='one_to_one', sort=False)
    assert merged.pair_id.tolist() == frame.pair_id.tolist()
    assert merged[GAP].notna().all()
    assert np.allclose(merged._end, merged.end_time_s)  # same lap t as the clean pair
    return merged.drop(columns='_end')


def median_model(numeric):
    model = ref.make_model('OLS', numeric=numeric)
    return model.set_params(regression=QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs'))


def main():
    train_x, val_x, raw = load_fixed_split()
    train_x, val_x = add_gap(train_x, raw), add_gap(val_x, raw)
    y = val_x[ref.TARGET].to_numpy()

    base_numeric = ref.NUMERIC + [FEATURE]
    gap_numeric = base_numeric + [GAP]
    versions = {
        'OLS + recent_std_s (Hito 2 candidate)': (ref.make_model('OLS', numeric=base_numeric), base_numeric),
        'A · median + recent_std_s': (median_model(base_numeric), base_numeric),
        'B · OLS + recent_std_s + gap_ahead_s': (ref.make_model('OLS', numeric=gap_numeric), gap_numeric),
        'C · median + recent_std_s + gap_ahead_s': (median_model(gap_numeric), gap_numeric),
    }
    preds, errors = {}, {}
    for name, (model, numeric) in versions.items():
        preds[name] = fit_predict(model, train_x, val_x, numeric + ref.CATEGORICAL)
        errors[name] = np.abs(y - preds[name])

    saved = pd.read_csv(ROOT / 'outputs' / 'validation_predictions.csv')
    saved = val_x[['pair_id']].merge(saved[['pair_id', 'pred_ols_plus_recent_std_s']], on='pair_id', validate='one_to_one')
    assert np.allclose(preds['OLS + recent_std_s (Hito 2 candidate)'], saved.pred_ols_plus_recent_std_s, atol=1e-8)

    slices = slice_table(val_x, {f'MAE_s · {k}': v for k, v in errors.items()})
    gap_band = pd.cut(val_x[GAP], [-0.01, 1.0, 2.0, GAP_CAP_S - 1e-9, GAP_CAP_S], labels=['<1s', '1-2s', '2-5s', 'clear air (5s cap)'])
    for band, idx in val_x.groupby(gap_band, observed=True).groups.items():
        pos = val_x.index.get_indexer(idx)
        slices.loc[len(slices)] = {'context': 'gap_ahead', 'group': str(band), 'n': len(pos),
                                   **{f'MAE_s · {k}': v[pos].mean() for k, v in errors.items()}}
    slices.to_csv(OUT / 'mae_by_slice.csv', index=False, float_format='%.6f')

    base_name = 'OLS + recent_std_s (Hito 2 candidate)'
    comparisons = []
    for a, b in [(base_name, 'A · median + recent_std_s'),
                 (base_name, 'B · OLS + recent_std_s + gap_ahead_s'),
                 ('A · median + recent_std_s', 'C · median + recent_std_s + gap_ahead_s'),
                 (base_name, 'C · median + recent_std_s + gap_ahead_s')]:
        diff, low, high, groups = cluster_bootstrap_ci(val_x, errors[a], errors[b])
        races = pd.DataFrame({'r': val_x['round'], 'a': errors[a], 'b': errors[b]}).groupby('r').mean()
        comparisons.append({
            'baseline': a, 'candidate': b, 'MAE_baseline_s': float(errors[a].mean()),
            'MAE_candidate_s': float(errors[b].mean()), 'paired_difference_candidate_minus_baseline_s': diff,
            'bootstrap_95ci_s': [low, high], 'bootstrap_groups': groups,
            'lower_in_every_race': bool((races.b < races.a).all()),
            'passes_rule': bool(diff < 0 and (races.b < races.a).all() and high < 0),
        })
    (OUT / 'comparisons.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')

    gap_model = versions['C · median + recent_std_s + gap_ahead_s'][0]
    names = gap_model.named_steps['prepare'].get_feature_names_out().tolist()
    coef = gap_model.named_steps['regression'].coef_
    scale = gap_model.named_steps['prepare'].named_transformers_['numeric'].named_steps['scale'].scale_
    per_second = float(coef[names.index(f'numeric__{GAP}')] / scale[gap_numeric.index(GAP)])

    out = val_x[['pair_id', 'round', 'driver_number', 'current_compound', 'recent_count', GAP, ref.TARGET]].copy()
    for name, p in preds.items():
        out[f'pred · {name}'] = p
    out.to_csv(OUT / 'validation_predictions.csv', index=False, float_format='%.6f')

    print(slices.to_string(index=False))
    print(json.dumps(comparisons, indent=2))
    print(f'Model C: next-lap change per +1 s of gap_ahead_s (up to {GAP_CAP_S} s): {per_second:+.4f} s')


if __name__ == '__main__':
    main()
