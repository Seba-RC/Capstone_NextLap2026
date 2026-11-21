from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import sklearn
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'base_NextLap'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 414
FEATURE = 'recent_std_s'
WINDOW = 3

sys.path.insert(0, str(DATA))
import nextlap_reference_v1 as ref


def add_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Build rolling sample SD on original timeline, within driver/race/stint."""
    laps = raw.sort_values(ref.KEYS + ['lap_number']).reset_index(drop=True).copy()
    clean = (laps.lap_number.gt(1) & laps.lap_time_s.gt(0) & laps.is_accurate
             & ~laps.pit_in & ~laps.pit_out & ~laps.deleted & laps.track_status.eq('1'))
    laps['history_value'] = laps.lap_time_s.where(clean)
    stint_keys = ref.KEYS + ['stint']

    def roll(column, how, min_periods=1):
        return laps.groupby(stint_keys, dropna=False, sort=False)[column].transform(
            lambda s: getattr(s.rolling(WINDOW, min_periods=min_periods), how)())

    laps['rebuilt_mean_s'] = roll('history_value', 'mean')
    laps['rebuilt_count'] = roll('history_value', 'count')
    laps[FEATURE] = roll('history_value', 'std', min_periods=2)
    laps['last_source_lap'] = roll('lap_number', 'max')
    # Only clean source laps count; invalid laps must not set the latest source marker.
    laps['source_lap'] = laps.lap_number.where(clean)
    laps['last_source_lap'] = laps.groupby(stint_keys, dropna=False, sort=False)['source_lap'].transform(
        lambda s: s.rolling(WINDOW, min_periods=1).max())
    laps['pair_id'] = (laps.season.astype(str) + '-R' + laps['round'].astype(str) + '-'
                       + laps.driver_number.astype(str) + '-L' + laps.lap_number.astype(int).astype(str))
    if not laps.pair_id.is_unique:
        raise AssertionError('Duplicate pair_id in original timeline')
    return laps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shap-sample', type=int, default=1808,
                        help='Number of validation rows in global SHAP summary (default: all 1808)')
    parser.add_argument('--background', type=int, default=200,
                        help='Training rows in SHAP background (default: 200)')
    args = parser.parse_args()

    # Load fixed split and original snapshots.
    train = pd.read_csv(DATA / 'train_clean_v1.csv', dtype={'driver_number': str})
    validation = pd.read_csv(DATA / 'validation_clean_v1.csv', dtype={'driver_number': str})
    raw, manifest = ref.load_snapshots(DATA)
    laps = add_history(raw)
    history = laps[['pair_id', 'rebuilt_mean_s', 'rebuilt_count', FEATURE, 'last_source_lap']].copy()

    assert len(train) == 5772 and len(validation) == 1808
    assert set(train['round']) == set(range(1, 8))
    assert set(validation['round']) == {8, 9}
    assert train.pair_id.is_unique and validation.pair_id.is_unique
    assert not set(train.pair_id).intersection(validation.pair_id)
    assert validation.groupby('round').size().to_dict() == {8: 1066, 9: 742}

    def attach(frame):
        merged = frame.merge(history, on='pair_id', how='left', validate='one_to_one', sort=False)
        assert merged.pair_id.tolist() == frame.pair_id.tolist()
        assert np.allclose(merged.rebuilt_mean_s, merged.recent_mean_s, atol=1e-6)
        assert merged.rebuilt_count.eq(merged.recent_count).all()
        assert merged[FEATURE].isna().eq(merged.recent_count.lt(2)).all()
        assert (merged.last_source_lap.dropna() <= merged.current_lap_number.dropna()).all()
        return merged

    train_x = attach(train)
    val_x = attach(validation)
    assert val_x.pair_id.is_unique and len(val_x) == 1808

    # Refit matched reference and candidate on exactly the same training/validation population.
    reference_inputs = ref.NUMERIC + ref.CATEGORICAL
    candidate_numeric = ref.NUMERIC + [FEATURE]
    candidate_inputs = candidate_numeric + ref.CATEGORICAL
    reference_model = ref.make_model('OLS').fit(train_x[reference_inputs], train_x[ref.TARGET])
    candidate_model = ref.make_model('OLS', numeric=candidate_numeric).fit(
        train_x[candidate_inputs], train_x[ref.TARGET])
    pred_ref = reference_model.predict(val_x[reference_inputs])
    pred_candidate = candidate_model.predict(val_x[candidate_inputs])

    # Verify the matched control reproduces the Hito 1 OLS predictions and the candidate metrics.
    h1_ref = pd.read_csv(ROOT / 'hito1_evidence' / 'outputs' / 'reference_predictions.csv')
    h1_ext = pd.read_csv(ROOT / 'hito1_evidence' / 'outputs' / 'extension_predictions.csv')
    check_ref = val_x[['pair_id']].merge(h1_ref[['pair_id', 'pred_ols_reference']], on='pair_id', validate='one_to_one')
    assert np.allclose(pred_ref, check_ref.pred_ols_reference.to_numpy(), atol=1e-8)
    check_ext = val_x[['pair_id']].merge(
        h1_ext[['pair_id', 'pred_ols_plus_recent_std_s']], on='pair_id', validate='one_to_one')
    assert np.allclose(pred_candidate, check_ext.pred_ols_plus_recent_std_s.to_numpy(), atol=1e-8)

    val_x = val_x.copy()
    val_x['pred_ols_reference'] = pred_ref
    val_x['pred_ols_plus_recent_std_s'] = pred_candidate
    val_x['abs_error_reference_s'] = np.abs(val_x[ref.TARGET] - pred_ref)
    val_x['abs_error_candidate_s'] = np.abs(val_x[ref.TARGET] - pred_candidate)
    val_x['candidate_minus_control_abs_error_s'] = val_x.abs_error_candidate_s - val_x.abs_error_reference_s
    val_x['feature_status'] = np.where(val_x[FEATURE].notna(), 'observed', 'imputed_short_history')

    # 1) Overall and race-level comparison.
    comp_rows = []
    for name, p in [('OLS reference / matched control', pred_ref), ('OLS + recent_std_s', pred_candidate)]:
        err = np.abs(val_x[ref.TARGET].to_numpy() - p)
        comp_rows.append({'version': name, 'population': 'all validation clean pairs', 'n': len(err), 'MAE_s': err.mean()})
        for rnd, part in val_x.assign(abs_error=err).groupby('round'):
            comp_rows.append({'version': name, 'population': f'round_{rnd}_{part.event_name.iloc[0]}',
                              'n': len(part), 'MAE_s': part.abs_error.mean()})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / 'comparison_metrics.csv', index=False, float_format='%.9f')

    race_rows = []
    for rnd, part in val_x.groupby('round'):
        control_mae = part.abs_error_reference_s.mean()
        candidate_mae = part.abs_error_candidate_s.mean()
        race_rows.append({'round': int(rnd), 'event': part.event_name.iloc[0], 'n': len(part),
                          'control_MAE_s': control_mae, 'candidate_MAE_s': candidate_mae,
                          'control_minus_candidate_s': control_mae - candidate_mae})
    race = pd.DataFrame(race_rows)
    race.to_csv(OUT / 'race_error_comparison.csv', index=False, float_format='%.9f')

    # Context diagnostics: observed vs imputed feature; tyre compound.
    context_rows = []
    for context_name, grouped in [
        ('feature_status', val_x.groupby('feature_status', dropna=False)),
        ('current_compound', val_x.groupby('current_compound', dropna=False)),
        ('round', val_x.groupby('round', dropna=False)),
    ]:
        for key, part in grouped:
            context_rows.append({'context': context_name, 'group': str(key), 'n': len(part),
                                 'control_MAE_s': part.abs_error_reference_s.mean(),
                                 'candidate_MAE_s': part.abs_error_candidate_s.mean(),
                                 'control_minus_candidate_s': part.abs_error_reference_s.mean() - part.abs_error_candidate_s.mean()})
    context = pd.DataFrame(context_rows)
    context.to_csv(OUT / 'context_error_slices.csv', index=False, float_format='%.9f')

    # 2) Select a real case: largest candidate absolute error (deterministic tie-break by pair_id).
    case = val_x.sort_values(['abs_error_candidate_s', 'pair_id'], ascending=[False, True]).iloc[0]
    case_df = pd.DataFrame([{
        'pair_id': case.pair_id, 'round': int(case['round']), 'event': case.event_name,
        'driver_number': case.driver_number, 'current_lap_number': int(case.current_lap_number),
        'current_compound': case.current_compound, 'recent_count': int(case.recent_count),
        'recent_std_s': case[FEATURE], 'observed_next_lap_s': case[ref.TARGET],
        'candidate_prediction_s': case.pred_ols_plus_recent_std_s,
        'candidate_absolute_error_s': case.abs_error_candidate_s,
        'control_prediction_s': case.pred_ols_reference,
        'control_absolute_error_s': case.abs_error_reference_s,
    }])
    case_df.to_csv(OUT / 'inspected_validation_case.csv', index=False, float_format='%.9f')
    case_timeline = raw[(raw['round'].eq(int(case['round'])))
                        & raw.driver_number.astype(str).eq(str(case.driver_number))
                        & raw.lap_number.between(max(1, int(case.current_lap_number) - 2), int(case.current_lap_number) + 2)].copy()
    case_timeline = case_timeline[['round', 'event_name', 'driver_number', 'lap_number', 'lap_time_s', 'end_time_s',
                                   'stint', 'compound', 'track_status', 'is_accurate', 'pit_in', 'pit_out', 'deleted']]
    case_timeline.to_csv(OUT / 'inspected_case_timeline.csv', index=False, float_format='%.9f')

    # 3) SHAP for the fitted candidate pipeline. Explain transformed linear model.
    prep = candidate_model.named_steps['prepare']
    estimator = candidate_model.named_steps['regression']
    X_train_trans = prep.transform(train_x[candidate_inputs])
    X_val_trans = prep.transform(val_x[candidate_inputs])
    encoded_names = prep.get_feature_names_out().tolist()
    rng = np.random.default_rng(SEED)
    bg_n = min(args.background, len(X_train_trans))
    bg_idx = np.sort(rng.choice(len(X_train_trans), size=bg_n, replace=False))
    background = X_train_trans[bg_idx]
    explainer = shap.LinearExplainer(estimator, background)

    shap_n = min(args.shap_sample, len(X_val_trans))
    shap_idx = np.arange(shap_n) if shap_n == len(X_val_trans) else np.sort(
        rng.choice(len(X_val_trans), size=shap_n, replace=False))
    shap_values_obj = explainer(X_val_trans[shap_idx])
    sv = np.asarray(shap_values_obj.values)
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv[:, :, 0]
    base_values = np.asarray(shap_values_obj.base_values)
    if base_values.ndim == 0:
        base_vec = np.full(shap_n, float(base_values))
    else:
        base_vec = base_values.reshape(-1)
    reconstructed = base_vec + sv.sum(axis=1)
    predicted_shap = candidate_model.predict(val_x.iloc[shap_idx][candidate_inputs])
    max_reconstruction_error = float(np.max(np.abs(reconstructed - predicted_shap)))
    assert max_reconstruction_error < 1e-5, f'SHAP reconstruction failed: {max_reconstruction_error}'

    # Map transformed columns back to their original source features and aggregate one-hot groups.
    mapping = []
    for encoded in encoded_names:
        source = encoded.split('__', 1)[-1]
        if source.startswith('current_compound_'):
            original = 'current_compound'
        elif source.startswith('current_team_'):
            original = 'current_team'
        else:
            original = source
        mapping.append(original)
    original_features = candidate_inputs

    global_rows = []
    for original in original_features:
        positions = [i for i, mapped in enumerate(mapping) if mapped == original]
        group_sv = sv[:, positions].sum(axis=1)
        global_rows.append({'feature': original, 'mean_abs_SHAP_s': np.abs(group_sv).mean(),
                            'mean_signed_SHAP_s': group_sv.mean(), 'encoded_columns': len(positions),
                            'validation_sample_n': shap_n})
    global_shap = pd.DataFrame(global_rows).sort_values('mean_abs_SHAP_s', ascending=False)
    global_shap.to_csv(OUT / 'shap_global_summary.csv', index=False, float_format='%.9f')

    # Compact global SHAP bar plot (mean absolute contributions in seconds).
    plot_data = global_shap.sort_values('mean_abs_SHAP_s', ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_data['feature'], plot_data['mean_abs_SHAP_s'])
    ax.set_xlabel('Mean absolute SHAP contribution (seconds)')
    ax.set_title(f'Global SHAP summary · validation sample n={shap_n}')
    fig.tight_layout()
    fig.savefig(OUT / 'shap_global_summary.png', dpi=180)
    plt.close(fig)

    # Explain the same selected case. If case isn't in sample, calculate it separately.
    case_pos = int(val_x.index[val_x.pair_id.eq(case.pair_id)][0])
    case_x = prep.transform(val_x.iloc[[case_pos]][candidate_inputs])
    case_shap = explainer(case_x)
    case_sv = np.asarray(case_shap.values)
    if case_sv.ndim == 3 and case_sv.shape[-1] == 1:
        case_sv = case_sv[:, :, 0]
    case_base = float(np.asarray(case_shap.base_values).reshape(-1)[0])
    local_rows = []
    for original in original_features:
        positions = [i for i, mapped in enumerate(mapping) if mapped == original]
        contribution = float(case_sv[0, positions].sum())
        local_rows.append({'feature': original, 'signed_SHAP_contribution_s': contribution})
    local = pd.DataFrame(local_rows)
    local = local.sort_values('signed_SHAP_contribution_s', key=lambda s: s.abs(), ascending=False)
    local.to_csv(OUT / 'shap_local_case.csv', index=False, float_format='%.9f')
    local_meta = pd.DataFrame([{
        'pair_id': case.pair_id, 'model': 'OLS + recent_std_s', 'prediction_s': float(case.pred_ols_plus_recent_std_s),
        'base_value_s': case_base, 'sum_SHAP_s': float(case_sv.sum()),
        'base_plus_SHAP_s': float(case_base + case_sv.sum()),
        'reconstruction_abs_difference_s': abs(float(case_base + case_sv.sum()) - float(case.pred_ols_plus_recent_std_s)),
        'observed_next_lap_s': float(case[ref.TARGET]), 'absolute_prediction_error_s': float(case.abs_error_candidate_s),
        'explainer': 'shap.LinearExplainer', 'background_rows': bg_n,
        'background_source': 'random sample without replacement from training rows after fitted preprocessing',
    }])
    local_meta.to_csv(OUT / 'shap_local_case_metadata.csv', index=False, float_format='%.9f')

    # Save all row-level predictions and audit checks.
    pred_cols = ['pair_id', 'round', 'event_name', 'driver_number', 'current_lap_number', 'current_compound',
                 'recent_count', FEATURE, ref.TARGET, 'pred_ols_reference', 'pred_ols_plus_recent_std_s',
                 'abs_error_reference_s', 'abs_error_candidate_s', 'candidate_minus_control_abs_error_s', 'feature_status']
    val_x[pred_cols].to_csv(OUT / 'validation_predictions.csv', index=False, float_format='%.9f')

    control_minus_candidate = float(val_x.abs_error_reference_s.mean() - val_x.abs_error_candidate_s.mean())
    audit = {
        'data_release': 'IIT414W_Capstone_NextLap_StudentData_v1 / StudentPack v2 files',
        'manifest_season': manifest.get('season'), 'training_rounds': sorted(train['round'].unique().tolist()),
        'validation_rounds': sorted(validation['round'].unique().tolist()), 'training_n': len(train),
        'validation_n': len(validation), 'validation_round_counts': validation.groupby('round').size().to_dict(),
        'train_validation_pair_id_overlap': 0, 'validation_pair_id_unique': bool(val_x.pair_id.is_unique),
        'validation_id_coverage': int(val_x.pair_id.isin(validation.pair_id).sum()),
        'reference_predictions_match_hito1': True, 'candidate_predictions_match_hito1': True,
        'candidate_control_MAE_difference_s_control_minus_candidate': control_minus_candidate,
        'candidate_rows_feature_observed': int(val_x[FEATURE].notna().sum()),
        'candidate_rows_feature_imputed': int(val_x[FEATURE].isna().sum()),
        'shap_explainer': 'shap.LinearExplainer', 'shap_background_n': bg_n,
        'shap_validation_sample_n': shap_n, 'shap_max_reconstruction_abs_error_s': max_reconstruction_error,
        'python': sys.version.split()[0], 'pandas': pd.__version__, 'numpy': np.__version__,
        'scikit_learn': sklearn.__version__, 'shap': shap.__version__,
        'random_seed': SEED,
    }
    (OUT / 'run_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

    print('\nCOMPARISON')
    print(comp.to_string(index=False))
    print('\nRACE COMPARISON')
    print(race.to_string(index=False))
    print('\nCONTEXT SLICES')
    print(context.to_string(index=False))
    print('\nINSPECTED CASE')
    print(case_df.to_string(index=False))
    print('\nGLOBAL SHAP')
    print(global_shap.to_string(index=False))
    print('\nLOCAL SHAP METADATA')
    print(local_meta.to_string(index=False))
    print('\nLOCAL SHAP CONTRIBUTIONS')
    print(local.to_string(index=False))
    print('\nAUDIT')
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
