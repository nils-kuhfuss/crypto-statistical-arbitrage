"""4h-to-daily reconciliation tests, ported from `legacy/Stat_Arb_cont_res_1.ipynb` Section 16's
inline reconciliation assertion (daily-compounded cumulative return must equal the 4h
cumulative return, and flat days must be kept, not dropped, in the daily-252 Sharpe)."""
import numpy as np
import pandas as pd

from stat_arb.metrics import aggregate_daily_returns, daily_sharpe_252


def test_daily_aggregation_reconciles_with_4h_equity_curve():
    idx = pd.date_range('2024-01-01', periods=6 * 30, freq='4h')  # 30 days of 4h bars
    rng = np.random.default_rng(0)
    bar_net_ret = pd.Series(rng.normal(0, 0.002, len(idx)), index=idx)
    # Zero out a random subset of bars to mimic flat (no-position) periods.
    bar_net_ret.iloc[::5] = 0.0

    daily_ret, n_bars = aggregate_daily_returns(bar_net_ret)
    cum_from_daily = (1.0 + daily_ret).prod() - 1.0
    cum_from_4h = (1.0 + bar_net_ret.fillna(0.0)).prod() - 1.0
    assert np.isclose(cum_from_daily, cum_from_4h, atol=1e-10)


def test_aggregate_daily_returns_keeps_flat_days_not_dropped():
    idx = pd.date_range('2024-01-01', periods=6 * 10, freq='4h')  # 10 days
    bar_net_ret = pd.Series(0.0, index=idx)
    bar_net_ret.iloc[0] = 0.01  # only the very first day has any activity
    daily_ret, n_bars = aggregate_daily_returns(bar_net_ret)
    assert len(daily_ret) == 10, 'flat days must remain in the series with return 0.0, not be dropped'
    assert (daily_ret.iloc[1:] == 0.0).all()


def test_daily_sharpe_252_uses_flat_days_in_mean_and_std():
    active = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    flat_padded = pd.concat([active, pd.Series([0.0] * 20)], ignore_index=True)
    sharpe_active_only = daily_sharpe_252(active)
    sharpe_with_flats = daily_sharpe_252(flat_padded)
    assert sharpe_active_only != sharpe_with_flats, (
        'including flat days must change the Sharpe (they dilute mean and reduce std), '
        'confirming flat days are not silently excluded'
    )


def test_daily_sharpe_252_nan_for_zero_variance():
    constant = pd.Series([0.0] * 10)
    assert np.isnan(daily_sharpe_252(constant))
