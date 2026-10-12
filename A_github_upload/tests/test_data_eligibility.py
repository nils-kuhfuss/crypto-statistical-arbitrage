"""Point-in-time eligibility tests, ported/adapted from `legacy/Stat_Arb_cont_res_1.ipynb`
Sections 3, 4, 19 (sample-split overlap, active-at-rebalance, tradable-price gap handling)."""
import numpy as np
import pandas as pd
import pytest

from stat_arb.data import get_tradable_price, split_samples, check_active_at_rebalance


def test_split_samples_no_overlap():
    idx = pd.date_range('2020-01-01', '2026-06-30', freq='4h')
    px = pd.DataFrame({'AUSDT': 1.0}, index=idx)
    config = {
        'FORMATION_START': '2020-01-01', 'FORMATION_END': '2022-12-31',
        'VALIDATION_START': '2023-01-01', 'VALIDATION_END': '2023-12-31',
        'OOS_START': '2024-01-01', 'OOS_END': '2026-06-30',
    }
    formation_px, validation_px, oos_px = split_samples(px, config)
    assert formation_px.index.max() < validation_px.index.min()
    assert validation_px.index.max() < oos_px.index.min()


def test_split_samples_overlap_raises():
    idx = pd.date_range('2020-01-01', '2026-06-30', freq='4h')
    px = pd.DataFrame({'AUSDT': 1.0}, index=idx)
    config = {
        'FORMATION_START': '2020-01-01', 'FORMATION_END': '2023-06-01',  # overlaps validation
        'VALIDATION_START': '2023-01-01', 'VALIDATION_END': '2023-12-31',
        'OOS_START': '2024-01-01', 'OOS_END': '2026-06-30',
    }
    with pytest.raises(AssertionError):
        split_samples(px, config)


def test_get_tradable_price_bridges_small_gaps_only():
    idx = pd.date_range('2021-01-01', periods=10, freq='4h')
    s = pd.Series([1.0, 1.1, np.nan, 1.2, np.nan, np.nan, np.nan, 1.3, 1.4, 1.5], index=idx)
    out = get_tradable_price(s, ffill_limit=2)
    assert out.iloc[2] == 1.1  # single-bar gap bridged
    assert pd.isna(out.iloc[4]) is False or pd.isna(out.iloc[5])  # 3-bar gap only partially bridged
    assert pd.isna(out.iloc[6])  # beyond ffill_limit, still NaN


def test_get_tradable_price_no_fill_after_delisting():
    idx = pd.date_range('2021-01-01', periods=6, freq='4h')
    s = pd.Series([1.0, 1.1, 1.2, np.nan, np.nan, np.nan], index=idx)
    out = get_tradable_price(s, ffill_limit=2)
    # last_valid_index is bar 2; nothing after it should ever be filled, no matter ffill_limit.
    assert out.iloc[3:].isna().all()


def test_check_active_at_rebalance_excludes_delisted_asset():
    idx = pd.date_range('2021-01-01', periods=20, freq='4h')
    px = pd.DataFrame(index=idx)
    px['ALIVE'] = 1.0
    px['DELISTED'] = [1.0] * 10 + [np.nan] * 10  # long trailing NaN tail = genuinely delisted
    rebalance_date = idx[15]
    active = check_active_at_rebalance(px, ['ALIVE', 'DELISTED'], rebalance_date, lookback_bars=3)
    assert active['ALIVE'] and not active['DELISTED']


def test_check_active_at_rebalance_tolerates_isolated_gap():
    idx = pd.date_range('2021-01-01', periods=20, freq='4h')
    px = pd.DataFrame(index=idx)
    prices = [1.0] * 20
    prices[13] = np.nan  # one isolated gap just before the lookback window
    px['COIN'] = prices
    rebalance_date = idx[15]
    active = check_active_at_rebalance(px, ['COIN'], rebalance_date, lookback_bars=3)
    assert active['COIN'], 'an isolated gap should not be mistaken for delisting'


def test_check_active_at_rebalance_is_point_in_time():
    idx = pd.date_range('2021-01-01', periods=20, freq='4h')
    px = pd.DataFrame(index=idx)
    px['COIN'] = [1.0] * 10 + [np.nan] * 10
    rebalance_date = idx[8]  # before the coin's last valid bar even ends
    active = check_active_at_rebalance(px, ['COIN'], rebalance_date, lookback_bars=3)
    assert active['COIN'], 'a coin must be eligible at rebalances before its own delisting'
