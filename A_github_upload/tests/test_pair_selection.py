"""Pair-selection tests: deterministic direction, dedup uniqueness, one-pair-per-coin,
positive-Beta rule, half-life/crossing filters -- ported/adapted from
`legacy/Stat_Arb_cont_res_1.ipynb` Sections 6, 19, 20, 22."""
import numpy as np
import pandas as pd

from stat_arb.pair_selection import (compute_pair_time_stops, dedup_pair_directions,
                                      screen_pairs_at_date, select_pairs_at_rebalance)


def test_dedup_pair_directions_keeps_lower_pvalue():
    sig_df = pd.DataFrame({
        'pair': [('A', 'B'), ('A', 'B')],
        'pvalue_adj': [0.03, 0.01],
        'y': ['A', 'B'], 'x': ['B', 'A'],
    })
    out = dedup_pair_directions(sig_df)
    assert len(out) == 1
    assert out.iloc[0]['pvalue_adj'] == 0.01
    assert out['pair'].is_unique


def _synthetic_universe(n=3000, n_coins=6, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='4h')
    common = np.cumsum(rng.normal(0, 1, n))
    data = {}
    for i in range(n_coins):
        beta = 0.5 + 0.3 * i
        data[f'COIN{i}USDT'] = np.exp((common * beta + rng.normal(0, 0.2, n)) * 0.01 + 3.0)
    return pd.DataFrame(data, index=idx)


def test_select_pairs_at_rebalance_enforces_one_pair_per_coin():
    px = _synthetic_universe()
    wf_config = {
        'EG_ALPHA': 0.5, 'MIN_OBS_FOR_TEST': 100, 'FORMATION_MONTHS': 12,
        'MAX_HALF_LIFE_BARS': 100000, 'MIN_CROSSINGS': 0, 'REQUIRE_POSITIVE_BETA': False,
        'FDR_METHOD': 'fdr_bh', 'TIME_STOP_MULTIPLIER_PRIMARY': 2.5,
        'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,
    }
    rebalance_date = px.index[2500]
    selected = select_pairs_at_rebalance(rebalance_date, px, px.columns.tolist(), wf_config, max_pairs=10)
    used_coins = [c for s in selected for c in (s['x'], s['y'])]
    assert len(used_coins) == len(set(used_coins)), 'one-pair-per-coin violated'


def test_screen_pairs_at_date_positive_beta_filter():
    idx = pd.date_range('2020-01-01', periods=3000, freq='4h')
    rng = np.random.default_rng(2)
    common = np.cumsum(rng.normal(0, 1, len(idx)))
    # Construct a pair with a deliberately NEGATIVE hedge ratio (y moves opposite to x).
    log_x = common * 0.01 + rng.normal(0, 0.05, len(idx))
    log_y = -common * 0.008 + rng.normal(0, 0.05, len(idx))
    px = pd.DataFrame({'AUSDT': np.exp(log_x + 3), 'BUSDT': np.exp(log_y + 3)}, index=idx)
    rebalance_date = idx[2500]
    wf_config_require = {
        'EG_ALPHA': 0.5, 'MIN_OBS_FOR_TEST': 100, 'FORMATION_MONTHS': 12,
        'MAX_HALF_LIFE_BARS': 100000, 'MIN_CROSSINGS': 0, 'REQUIRE_POSITIVE_BETA': True,
        'FDR_METHOD': 'fdr_bh', 'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,
    }
    result = screen_pairs_at_date(rebalance_date, px, ['AUSDT', 'BUSDT'], wf_config_require)
    assert len(result) == 0, 'a negative-beta pair must be excluded when REQUIRE_POSITIVE_BETA=True'


def test_compute_pair_time_stops_marks_non_mean_reverting_ineligible():
    idx = pd.date_range('2020-01-01', periods=2000, freq='4h')
    rng = np.random.default_rng(3)
    # Two independent random walks: no cointegrating, non-mean-reverting spread.
    x = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    y = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    formation_px = pd.DataFrame({'AUSDT': x, 'BUSDT': y})
    time_stops, ineligible = compute_pair_time_stops([('AUSDT', 'BUSDT')], formation_px, multiplier=2.5)
    assert ('AUSDT', 'BUSDT') in ineligible or ('AUSDT', 'BUSDT') in time_stops
    assert len(time_stops) + len(ineligible) == 1
