"""No-look-ahead tests, adapted from `legacy/Stat_Arb_cont_res_1.ipynb` Section 7's rolling-
signal check and Section 19/20's point-in-time formation-window assertions.

The original notebook's rolling-signal check used real BTCUSDT/ETHUSDT price history; here it
is reproduced with synthetic data so the test is fast and deterministic while proving the
identical property: truncating the input series at bar t and recomputing must not change the
rolling signal's value at t (i.e. `pandas.rolling(..., min_periods=window)` only ever looks
backward).
"""
import numpy as np
import pandas as pd

from stat_arb.signals import gen_rolling_signal
from stat_arb.pair_selection import screen_pairs_at_date


def _synthetic_price_panel(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2020-01-01', periods=n, freq='4h')
    common = np.cumsum(rng.normal(0, 1, n))
    x = pd.Series(np.exp((common + rng.normal(0, 0.05, n)) * 0.01 + 4.0), index=idx)
    y = pd.Series(np.exp((common * 1.3 + rng.normal(0, 0.05, n)) * 0.01 + 3.0), index=idx)
    return idx, x, y


def test_rolling_signal_truncation_does_not_change_past_value():
    idx, x, y = _synthetic_price_panel()
    window = 100
    check_bar = 500

    _, _, _, z_full = gen_rolling_signal(x, y, window, rolling_beta=True)
    x_trunc = x.iloc[:check_bar + 1]
    y_trunc = y.iloc[:check_bar + 1]
    _, _, _, z_trunc = gen_rolling_signal(x_trunc, y_trunc, window, rolling_beta=True)

    val_full = z_full.iloc[check_bar]
    val_trunc = z_trunc.iloc[check_bar]
    assert np.isclose(val_full, val_trunc, equal_nan=True), (
        f'Look-ahead detected: z_score at bar {check_bar} differs when future data is present '
        f'({val_full} vs {val_trunc})')


def test_screen_pairs_at_date_window_ends_before_rebalance():
    idx, x, y = _synthetic_price_panel(n=3000, seed=1)
    price_data = pd.DataFrame({'AUSDT': x, 'BUSDT': y})
    rebalance_date = idx[2500]

    wf_config = {
        'EG_ALPHA': 0.05, 'MIN_OBS_FOR_TEST': 100, 'FORMATION_MONTHS': 12,
        'MAX_HALF_LIFE_BARS': 300, 'MIN_CROSSINGS': 0, 'REQUIRE_POSITIVE_BETA': False,
        'FDR_METHOD': 'fdr_bh', 'ACTIVE_AT_REBALANCE_LOOKBACK_BARS': 3,
    }
    # Plant an extreme, easily-detectable outlier strictly after rebalance_date. If the screen
    # leaked future data into its formation-window slice, this would change alpha/beta/spread
    # statistics for a pair whose window otherwise ends exactly at rebalance_date.
    price_data_leaked = price_data.copy()
    price_data_leaked.loc[rebalance_date:, 'AUSDT'] *= 1000.0

    result_clean = screen_pairs_at_date(rebalance_date, price_data, ['AUSDT', 'BUSDT'], wf_config)
    result_leaked = screen_pairs_at_date(rebalance_date, price_data_leaked, ['AUSDT', 'BUSDT'], wf_config)

    if len(result_clean) and len(result_leaked):
        pd.testing.assert_frame_equal(
            result_clean.drop(columns=['return_correlation']) if 'return_correlation' in result_clean else result_clean,
            result_leaked.drop(columns=['return_correlation']) if 'return_correlation' in result_leaked else result_leaked,
            check_exact=False, atol=1e-9,
        )
