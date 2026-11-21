"""Experiment 3 (branch experiment/median-traffic-gap): tree-based models.

Question: does a non-linear model (gradient boosting or random forest) beat the
Hito 2 candidate, OLS + recent_std_s, with the same inputs and the same
1,808 validation IDs?

Two target definitions are compared, because tree models cannot predict outside
the range of targets seen in training:
  * raw:   predict next_lap_s directly (same target as OLS).
  * delta: predict next_lap_s - recent_mean_s, then add recent_mean_s back.
           This removes each circuit's lap-time level from what the trees learn.

Tree inputs are the six published inputs plus recent_std_s, with one derived
relative input, current_lap_s - recent_mean_s (computed from inputs already
available at the end of lap t). Categoricals are native categories for
HistGradientBoosting and integer codes for RandomForest; missing recent_std_s is
left as NaN for HistGradientBoosting and set to -1 for RandomForest (a value a
standard deviation can never take).

All models use library defaults (plus a fixed seed; RandomForest uses 300 trees
and min_samples_leaf=5). No hyperparameter was tuned, so the validation races
are not used for model selection. Any later tuning should use leave-one-race-out
within rounds 1-7 only.

Run from the repository root:  python experiments/run_tree_models.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import QuantileRegressor

from common import FEATURE, ROOT, SEED, cluster_bootstrap_ci, load_fixed_split, ref, slice_table

OUT = ROOT / 'outputs' / 'exp3_tree_models'
OUT.mkdir(parents=True, exist_ok=True)
NUMERIC = ref.NUMERIC + [FEATURE]
INPUTS = NUMERIC + ref.CATEGORICAL


def tree_inputs(train_x, val_x):
    frames = []
    for frame in (train_x, val_x):
        X = frame[INPUTS].copy()
        X['current_minus_recent_mean_s'] = frame.current_lap_s - frame.recent_mean_s
        frames.append(X)
    for column in ref.CATEGORICAL:  # identical category sets in train and validation
        levels = sorted(set(train_x[column]) | set(val_x[column]))
        for X in frames:
            X[column] = pd.Categorical(X[column], categories=levels)
    return frames


def as_codes(X):
    X = X.copy()
    for column in ref.CATEGORICAL:
        X[column] = X[column].cat.codes
    return X.fillna(-1.0)


def tree_models():
    return {
        'Gradient boosting (absolute error)': (
            HistGradientBoostingRegressor(loss='absolute_error', categorical_features='from_dtype', random_state=SEED), False),
        'Gradient boosting (squared error)': (
            HistGradientBoostingRegressor(categorical_features='from_dtype', random_state=SEED), False),
        'Random forest': (
            RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1), True),
    }


def main():
    train_x, val_x, _ = load_fixed_split()
    y = val_x[ref.TARGET].to_numpy()
    preds = {}

    ols = ref.make_model('OLS', numeric=NUMERIC).fit(train_x[INPUTS], train_x[ref.TARGET])
    preds['OLS + recent_std_s (Hito 2 candidate)'] = ols.predict(val_x[INPUTS])
    median = ref.make_model('OLS', numeric=NUMERIC).set_params(
        regression=QuantileRegressor(quantile=0.5, alpha=0.0, solver='highs'))
    preds['Median regression + recent_std_s (Exp. 2 A)'] = median.fit(train_x[INPUTS], train_x[ref.TARGET]).predict(val_x[INPUTS])

    saved = pd.read_csv(ROOT / 'outputs' / 'validation_predictions.csv')
    saved = val_x[['pair_id']].merge(saved[['pair_id', 'pred_ols_plus_recent_std_s']], on='pair_id', validate='one_to_one')
    assert np.allclose(preds['OLS + recent_std_s (Hito 2 candidate)'], saved.pred_ols_plus_recent_std_s, atol=1e-8)

    X_train, X_val = tree_inputs(train_x, val_x)
    for target in ['raw', 'delta']:
        offset_train = train_x.recent_mean_s.to_numpy() if target == 'delta' else 0.0
        offset_val = val_x.recent_mean_s.to_numpy() if target == 'delta' else 0.0
        for name, (model, needs_codes) in tree_models().items():
            A, B = (as_codes(X_train), as_codes(X_val)) if needs_codes else (X_train, X_val)
            model.fit(A, train_x[ref.TARGET].to_numpy() - offset_train)
            preds[f'{name} · target={target}'] = model.predict(B) + offset_val

    errors = {name: np.abs(y - p) for name, p in preds.items()}
    base_name = 'OLS + recent_std_s (Hito 2 candidate)'
    rows = []
    for name, err in errors.items():
        diff, low, high, groups = cluster_bootstrap_ci(val_x, errors[base_name], err)
        races = pd.Series(err).groupby(val_x['round'].to_numpy()).mean()
        base_races = pd.Series(errors[base_name]).groupby(val_x['round'].to_numpy()).mean()
        rows.append({'model': name, 'MAE_s': err.mean(), 'MAE_R8_s': races[8], 'MAE_R9_s': races[9],
                     'difference_vs_hito2_s': diff, 'ci95_low_s': low, 'ci95_high_s': high,
                     'lower_in_both_races': bool(name != base_name and (races < base_races).all()),
                     'passes_rule': bool(name != base_name and diff < 0 and high < 0 and (races < base_races).all())})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'model_comparison.csv', index=False, float_format='%.6f')
    slice_table(val_x, {f'MAE_s · {k}': v for k, v in errors.items()}).to_csv(
        OUT / 'mae_by_slice.csv', index=False, float_format='%.6f')

    # Evidence for the extrapolation problem: circuit lap-time levels.
    levels = pd.concat([train_x.assign(split='train'), val_x.assign(split='validation')]).groupby(
        ['split', 'round', 'event_name']).agg(n=('pair_id', 'size'), median_current_lap_s=('current_lap_s', 'median'),
                                              min_next_lap_s=(ref.TARGET, 'min'), max_next_lap_s=(ref.TARGET, 'max')).reset_index()
    levels.to_csv(OUT / 'circuit_lap_levels.csv', index=False, float_format='%.3f')
    raw_tree_preds = np.concatenate([preds[k] for k in preds if k.endswith('target=raw')])
    audit = {'training_target_range_s': [float(train_x[ref.TARGET].min()), float(train_x[ref.TARGET].max())],
             'validation_R8_target_range_s': [float(val_x.loc[val_x['round'].eq(8), ref.TARGET].min()),
                                              float(val_x.loc[val_x['round'].eq(8), ref.TARGET].max())],
             'min_raw_target_tree_prediction_s': float(raw_tree_preds.min()),
             'hyperparameters_tuned': False, 'seed': SEED}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

    out = val_x[['pair_id', 'round', 'driver_number', ref.TARGET]].copy()
    for name, p in preds.items():
        out[f'pred · {name}'] = p
    out.to_csv(OUT / 'validation_predictions.csv', index=False, float_format='%.6f')

    print(levels.to_string(index=False))
    print(summary.round(6).to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
