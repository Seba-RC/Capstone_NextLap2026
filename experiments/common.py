"""Shared loading and evaluation helpers for post-Hito 2 experiments.

Every experiment reuses the fixed Hito 2 setup: training rounds 1-7, the same
1,808 validation pair_ids from rounds 8-9, MAE in seconds, and the
recent_std_s history rebuilt from the original timing snapshots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'base_NextLap'
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(ROOT))

import nextlap_reference_v1 as ref  # noqa: E402
from run_hito2 import FEATURE, SEED, add_history  # noqa: E402

FLAG = 'short_history'


def load_fixed_split():
    """Return (train_x, val_x, raw) with recent_std_s and the short-history flag attached."""
    train = pd.read_csv(DATA / 'train_clean_v1.csv', dtype={'driver_number': str})
    validation = pd.read_csv(DATA / 'validation_clean_v1.csv', dtype={'driver_number': str})
    raw, _ = ref.load_snapshots(DATA)
    history = add_history(raw)[['pair_id', FEATURE]]

    assert len(train) == 5772 and len(validation) == 1808
    assert not set(train.pair_id).intersection(validation.pair_id)

    def attach(frame):
        merged = frame.merge(history, on='pair_id', how='left', validate='one_to_one', sort=False)
        assert merged.pair_id.tolist() == frame.pair_id.tolist()
        assert merged[FEATURE].isna().eq(merged.recent_count.lt(2)).all()
        # 1 when fewer than two valid recent laps exist, i.e. recent_std_s must be imputed.
        merged[FLAG] = merged.recent_count.lt(2).astype(int)
        return merged

    return attach(train), attach(validation), raw


def fit_predict(model, train_x, val_x, inputs):
    model.fit(train_x[inputs], train_x[ref.TARGET])
    return model.predict(val_x[inputs])


def cluster_bootstrap_ci(val_x, err_a, err_b, n_boot=5000, seed=SEED):
    """95% driver-race cluster-bootstrap CI for mean(err_b - err_a).

    Negative values mean model b has lower absolute error than model a.
    """
    diff = pd.Series(np.asarray(err_b) - np.asarray(err_a), index=val_x.index)
    groups = val_x['round'].astype(str) + '-' + val_x.driver_number.astype(str)
    sums = diff.groupby(groups).sum().to_numpy()
    counts = diff.groupby(groups).size().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    stats = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high), int(len(sums))


def slice_table(val_x, errors: dict[str, np.ndarray]):
    """MAE per model for overall, each race, feature status and compound."""
    scored = val_x[['round', 'event_name', 'current_compound', FLAG]].copy()
    for name, err in errors.items():
        scored[name] = err
    scored['feature_status'] = np.where(scored[FLAG].eq(1), 'imputed_short_history', 'observed')
    rows = [{'context': 'all', 'group': 'all validation clean pairs', 'n': len(scored),
             **{name: scored[name].mean() for name in errors}}]
    for context in ['round', 'feature_status', 'current_compound']:
        for key, part in scored.groupby(context):
            rows.append({'context': context, 'group': str(key), 'n': len(part),
                         **{name: part[name].mean() for name in errors}})
    return pd.DataFrame(rows)
